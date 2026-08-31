from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DetectorConfig
from .detectors import (
    prepare_intensity_response,
    refine_vacuole_boundaries,
    threshold_intensity_response,
)
from .io import load_manifest, read_binary_mask, read_grayscale
from .masks import sanitize_fiber_masks, scale_bar_region
from .metrics import segmentation_metrics


MAX_DISTANCE_GRID_UM = (0.02, 0.03, 0.05)
GROWTH_OFFSET_GRID = (-0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05)
MAX_AREA_RATIO_GRID = (1.5, 2.0, 3.0)


def tune_boundary_refinement(
    manifest_path: str | Path,
    base_config_path: str | Path,
    output_dir: str | Path,
) -> DetectorConfig:
    """Tune seed expansion on development data while keeping detection fixed."""

    rows = [row for row in load_manifest(manifest_path) if row.split == "dev"]
    if not rows:
        raise ValueError("The manifest contains no development rows")
    base = replace(
        DetectorConfig.from_json(base_config_path),
        boundary_refinement=False,
    )
    cached: list[dict[str, Any]] = []
    for row in rows:
        if row.consensus_vacuole_mask_path is None:
            raise ValueError(f"Development row {row.id!r} has no consensus mask")
        image = read_grayscale(row.image_path)
        axon = read_binary_mask(row.axon_mask_path, image.shape)
        outer = read_binary_mask(row.outer_fiber_mask_path, image.shape)
        excluded = (
            scale_bar_region(
                image.shape,
                base.scale_bar_right_fraction,
                base.scale_bar_bottom_fraction,
            )
            if base.exclude_scale_bar
            else np.zeros(image.shape, dtype=bool)
        )
        axon, outer, flags = sanitize_fiber_masks(axon, outer, excluded)
        if {"empty_axon", "empty_outer_fiber", "empty_gross_sheath"}.intersection(flags):
            raise ValueError(f"Development row {row.id!r} has invalid masks: {flags}")
        truth = read_binary_mask(row.consensus_vacuole_mask_path, image.shape)
        truth &= outer & ~axon & ~excluded
        response, gross, otsu = prepare_intensity_response(
            image,
            axon,
            outer,
            row.scale_nm_per_px,
            base.clahe_clip_limit,
            base.gaussian_sigma_um,
        )
        seed = threshold_intensity_response(
            response,
            gross,
            otsu,
            row.scale_nm_per_px,
            base,
        )
        cached.append(
            {
                "row": row,
                "truth": truth,
                "response": response,
                "gross": gross,
                "otsu": otsu,
                "seed": seed,
            }
        )

    records: list[dict[str, Any]] = []
    for max_distance in MAX_DISTANCE_GRID_UM:
        for growth_offset in GROWTH_OFFSET_GRID:
            for max_area_ratio in MAX_AREA_RATIO_GRID:
                scores: list[dict[str, float]] = []
                growth_ratios: list[float] = []
                compact_count = 0
                compact_false_positives = 0
                for item in cached:
                    row = item["row"]
                    seed = item["seed"]
                    prediction = refine_vacuole_boundaries(
                        item["response"],
                        item["gross"],
                        seed,
                        item["otsu"],
                        row.scale_nm_per_px,
                        max_distance,
                        growth_offset,
                        max_area_ratio,
                    )
                    truth = item["truth"]
                    scores.append(segmentation_metrics(prediction, truth))
                    if seed.any():
                        growth_ratios.append(float(prediction.sum() / seed.sum()))
                    if not truth.any():
                        compact_count += 1
                        compact_false_positives += int(prediction.any())
                records.append(
                    {
                        "refinement_max_distance_um": max_distance,
                        "refinement_growth_offset": growth_offset,
                        "refinement_max_area_ratio": max_area_ratio,
                        "median_dice": float(np.median([x["dice"] for x in scores])),
                        "mean_dice": float(np.mean([x["dice"] for x in scores])),
                        "median_precision": float(
                            np.median([x["precision"] for x in scores])
                        ),
                        "median_recall": float(np.median([x["recall"] for x in scores])),
                        "median_prediction_growth_ratio": (
                            float(np.median(growth_ratios)) if growth_ratios else 1.0
                        ),
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
            "refinement_max_distance_um",
            "refinement_growth_offset",
            "refinement_max_area_ratio",
        ],
        ascending=[False, False, True, True, False, True],
    )
    winner = results.iloc[0]
    frozen = replace(
        base,
        boundary_refinement=True,
        refinement_max_distance_um=float(winner["refinement_max_distance_um"]),
        refinement_growth_offset=float(winner["refinement_growth_offset"]),
        refinement_max_area_ratio=float(winner["refinement_max_area_ratio"]),
        tuned_on_split="dev",
        development_median_dice=float(winner["median_dice"]),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "refinement_tuning_results.csv", index=False)
    frozen.to_json(output / "detector_config.json")
    summary = {
        "base_config": base.__dict__,
        "selected": frozen.__dict__,
        "selection_split": "dev",
        "candidate_count": int(len(results)),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune seed-based vacuole boundary refinement on development data"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True, help="Frozen base detector config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = tune_boundary_refinement(args.manifest, args.config, args.output)
    print(json.dumps(config.__dict__, indent=2))


if __name__ == "__main__":
    main()
