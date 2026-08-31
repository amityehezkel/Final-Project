from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DetectorConfig
from .detectors import (
    detect_vacuoles,
    prepare_intensity_response,
    threshold_intensity_response,
)
from .io import load_manifest, read_binary_mask, read_grayscale
from .masks import remove_small_components, sanitize_fiber_masks, scale_bar_region
from .metrics import segmentation_metrics


MIN_AREA_GRID_UM2 = (0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.0075, 0.01)
HIGH_THRESHOLD_OFFSET_GRID = (0.15, 0.20, 0.25)
LOW_THRESHOLD_OFFSET_GRID = (0.075, 0.10, 0.15, 0.20)
GAUSSIAN_SIGMA_GRID_UM = (0.02,)
MORPHOLOGY_RADIUS_GRID_UM = (0.01, 0.015, 0.02, 0.025, 0.03)


def tune_detector(manifest_path: str | Path, output_dir: str | Path) -> DetectorConfig:
    rows = [row for row in load_manifest(manifest_path) if row.split == "dev"]
    if not rows:
        raise ValueError("The manifest contains no development rows")
    if any(row.consensus_vacuole_mask_path is None for row in rows):
        raise ValueError("Every development row needs a consensus_vacuole_mask_path")

    candidates: list[DetectorConfig] = []
    for area in MIN_AREA_GRID_UM2:
        for high_offset in HIGH_THRESHOLD_OFFSET_GRID:
            for low_offset in LOW_THRESHOLD_OFFSET_GRID:
                if low_offset > high_offset:
                    continue
                for sigma_um in GAUSSIAN_SIGMA_GRID_UM:
                    for radius_um in MORPHOLOGY_RADIUS_GRID_UM:
                        candidates.append(
                            DetectorConfig(
                                detector="intensity",
                                min_area_um2=area,
                                intensity_threshold_offset=high_offset,
                                intensity_low_threshold_offset=low_offset,
                                gaussian_sigma_um=sigma_um,
                                morphology_radius_um=radius_um,
                            )
                        )
        if all(row.compact_myelin_mask_path is not None for row in rows):
            candidates.append(DetectorConfig(detector="geometry", min_area_um2=area))

    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]] = {}
    for row in rows:
        image = read_grayscale(row.image_path)
        axon = read_binary_mask(row.axon_mask_path, image.shape)
        outer = read_binary_mask(row.outer_fiber_mask_path, image.shape)
        excluded = (
            scale_bar_region(image.shape)
            if candidates[0].exclude_scale_bar
            else np.zeros(image.shape, dtype=bool)
        )
        axon, outer, flags = sanitize_fiber_masks(axon, outer, excluded)
        if {"empty_axon", "empty_outer_fiber", "empty_gross_sheath"}.intersection(flags):
            raise ValueError(f"Development row {row.id!r} has invalid masks: {flags}")
        compact = (
            read_binary_mask(row.compact_myelin_mask_path, image.shape) & outer & ~axon
            if row.compact_myelin_mask_path is not None
            else None
        )
        truth = read_binary_mask(row.consensus_vacuole_mask_path, image.shape)
        truth &= outer & ~axon & ~excluded
        cache[row.id] = (image, axon, outer, compact, truth)

    prepared: dict[
        tuple[str, float], tuple[np.ndarray, np.ndarray, float]
    ] = {}
    for row in rows:
        image, axon, outer, _, _ = cache[row.id]
        for sigma_um in GAUSSIAN_SIGMA_GRID_UM:
            prepared[(row.id, sigma_um)] = prepare_intensity_response(
                image,
                axon,
                outer,
                row.scale_nm_per_px,
                candidates[0].clahe_clip_limit,
                sigma_um,
            )

    unfiltered_predictions: dict[
        tuple[str, float, float, float, float], np.ndarray
    ] = {}
    records: list[dict[str, float | str]] = []
    for config in candidates:
        scores: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        compact_false_positives = 0
        compact_count = 0
        for row in rows:
            image, axon, outer, compact, truth = cache[row.id]
            if config.detector == "intensity":
                smoothed, gross_sheath, otsu_threshold = prepared[
                    (row.id, config.gaussian_sigma_um)
                ]
                prediction_key = (
                    row.id,
                    config.intensity_threshold_offset,
                    float(config.intensity_low_threshold_offset),
                    config.gaussian_sigma_um,
                    config.morphology_radius_um,
                )
                if prediction_key not in unfiltered_predictions:
                    unfiltered_predictions[prediction_key] = (
                        threshold_intensity_response(
                            smoothed,
                            gross_sheath,
                            otsu_threshold,
                            row.scale_nm_per_px,
                            replace(config, min_area_um2=0.0),
                        )
                    )
                prediction = remove_small_components(
                    unfiltered_predictions[prediction_key],
                    config.min_area_um2,
                    row.scale_nm_per_px,
                )
            else:
                prediction = detect_vacuoles(
                    image, axon, outer, row.scale_nm_per_px, config, compact
                )
            score = segmentation_metrics(prediction, truth)
            scores.append(score["dice"])
            precisions.append(score["precision"])
            recalls.append(score["recall"])
            if not truth.any():
                compact_count += 1
                compact_false_positives += int(prediction.any())
        records.append(
            {
                "detector": config.detector,
                "min_area_um2": config.min_area_um2,
                "intensity_threshold_offset": config.intensity_threshold_offset,
                "intensity_low_threshold_offset": (
                    config.intensity_threshold_offset
                    if config.intensity_low_threshold_offset is None
                    else config.intensity_low_threshold_offset
                ),
                "gaussian_sigma_um": config.gaussian_sigma_um,
                "morphology_radius_um": config.morphology_radius_um,
                "median_dice": float(np.median(scores)),
                "mean_dice": float(np.mean(scores)),
                "median_precision": float(np.median(precisions)),
                "median_recall": float(np.median(recalls)),
                "compact_false_positive_rate": (
                    float(compact_false_positives / compact_count)
                    if compact_count
                    else 0.0
                ),
            }
        )

    results = pd.DataFrame(records).sort_values(
        [
            "median_dice",
            "mean_dice",
            "compact_false_positive_rate",
            "detector",
            "min_area_um2",
            "intensity_threshold_offset",
            "intensity_low_threshold_offset",
            "gaussian_sigma_um",
            "morphology_radius_um",
        ],
        ascending=[False, False, True, True, True, True, True, True, True],
    )
    winner = results.iloc[0]
    frozen = replace(
        DetectorConfig(
            detector=str(winner["detector"]),
            min_area_um2=float(winner["min_area_um2"]),
            intensity_threshold_offset=float(winner["intensity_threshold_offset"]),
            intensity_low_threshold_offset=float(
                winner["intensity_low_threshold_offset"]
            ),
            gaussian_sigma_um=float(winner["gaussian_sigma_um"]),
            morphology_radius_um=float(winner["morphology_radius_um"]),
        ),
        tuned_on_split="dev",
        development_median_dice=float(winner["median_dice"]),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "tuning_results.csv", index=False)
    frozen.to_json(output / "detector_config.json")
    status = {
        "selected": frozen.__dict__,
        "manual_correction_required": bool(winner["median_dice"] < 0.50),
    }
    with (output / "tuning_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(status, stream, indent=2)
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune only on the development split")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = tune_detector(args.manifest, args.output)
    print(json.dumps(config.__dict__, indent=2))


if __name__ == "__main__":
    main()
