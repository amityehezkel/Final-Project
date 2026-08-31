from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .config import DetectorConfig
from .detectors import detect_vacuoles
from .io import load_manifest, read_binary_mask, read_grayscale
from .masks import remove_small_components, sanitize_fiber_masks, scale_bar_region
from .metrics import segmentation_metrics


DEFAULT_CANDIDATES_UM2 = (0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.0075, 0.01)
EXCLUSION_FLAGS = {
    "scale_bar_overlap",
    "border_touching",
    "empty_axon",
    "empty_outer_fiber",
    "empty_gross_sheath",
}


def analyze_minimum_area(
    manifest_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    candidates_um2: tuple[float, ...] = DEFAULT_CANDIDATES_UM2,
) -> dict[str, Any]:
    """Isolate the minimum-area filter while keeping detector response fixed."""

    rows = load_manifest(manifest_path)
    config = DetectorConfig.from_json(config_path)
    unfiltered_config = replace(config, min_area_um2=0.0)
    component_records: list[dict[str, Any]] = []
    cached: list[dict[str, Any]] = []

    for row in rows:
        if row.consensus_vacuole_mask_path is None:
            raise ValueError(f"Row {row.id!r} has no consensus vacuole mask")
        image = read_grayscale(row.image_path)
        axon = read_binary_mask(row.axon_mask_path, image.shape)
        outer = read_binary_mask(row.outer_fiber_mask_path, image.shape)
        excluded = (
            scale_bar_region(
                image.shape,
                config.scale_bar_right_fraction,
                config.scale_bar_bottom_fraction,
            )
            if config.exclude_scale_bar
            else np.zeros(image.shape, dtype=bool)
        )
        axon, outer, flags = sanitize_fiber_masks(axon, outer, excluded)
        truth = read_binary_mask(row.consensus_vacuole_mask_path, image.shape)
        truth &= outer & ~axon & ~excluded
        raw_prediction = detect_vacuoles(
            image,
            axon,
            outer,
            row.scale_nm_per_px,
            unfiltered_config,
        )
        raw_prediction &= outer & ~axon & ~excluded
        excluded_from_summary = bool(EXCLUSION_FLAGS.intersection(flags))
        cached.append(
            {
                "row": row,
                "truth": truth,
                "raw_prediction": raw_prediction,
                "excluded_from_summary": excluded_from_summary,
            }
        )

        labels, count = ndi.label(truth, structure=np.ones((3, 3), dtype=bool))
        pixel_area_um2 = (row.scale_nm_per_px / 1000.0) ** 2
        for component_index in range(1, count + 1):
            pixels = int((labels == component_index).sum())
            component_records.append(
                {
                    "id": row.id,
                    "split": row.split,
                    "component_index": component_index,
                    "pixels": pixels,
                    "area_um2": pixels * pixel_area_um2,
                }
            )

    components = pd.DataFrame(component_records)
    retention_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    for area in candidates_um2:
        for split in sorted({row.split for row in rows}):
            split_components = components.loc[components["split"] == split]
            retained = split_components.loc[split_components["area_um2"] >= area]
            retention_records.append(
                {
                    "min_area_um2": area,
                    "split": split,
                    "annotated_components": int(len(split_components)),
                    "retained_components": int(len(retained)),
                    "retained_component_fraction": (
                        float(len(retained) / len(split_components))
                        if len(split_components)
                        else float("nan")
                    ),
                    "annotated_area_um2": float(split_components["area_um2"].sum()),
                    "retained_area_um2": float(retained["area_um2"].sum()),
                    "retained_area_fraction": (
                        float(retained["area_um2"].sum() / split_components["area_um2"].sum())
                        if split_components["area_um2"].sum() > 0
                        else float("nan")
                    ),
                }
            )

            per_fiber: list[dict[str, Any]] = []
            for item in cached:
                row = item["row"]
                if row.split != split or item["excluded_from_summary"]:
                    continue
                prediction = remove_small_components(
                    item["raw_prediction"], area, row.scale_nm_per_px
                )
                truth = item["truth"]
                per_fiber.append(
                    {
                        **segmentation_metrics(prediction, truth),
                        "truth_positive": bool(truth.any()),
                        "prediction_positive": bool(prediction.any()),
                    }
                )
            frame = pd.DataFrame(per_fiber)
            positives = frame["truth_positive"].astype(bool)
            predictions = frame["prediction_positive"].astype(bool)
            compact = ~positives
            metric_records.append(
                {
                    "min_area_um2": area,
                    "split": split,
                    "n_evaluated": int(len(frame)),
                    "median_dice": float(frame["dice"].median()),
                    "mean_dice": float(frame["dice"].mean()),
                    "median_precision": float(frame["precision"].median()),
                    "median_recall": float(frame["recall"].median()),
                    "positive_fibers": int(positives.sum()),
                    "complete_false_negatives": int((positives & ~predictions).sum()),
                    "compact_fibers": int(compact.sum()),
                    "compact_false_positives": int((compact & predictions).sum()),
                    "compact_false_positive_rate": (
                        float((compact & predictions).sum() / compact.sum())
                        if compact.sum()
                        else float("nan")
                    ),
                }
            )

    retention = pd.DataFrame(retention_records)
    metrics = pd.DataFrame(metric_records)
    development_retention = retention.loc[retention["split"] == "dev"]
    logical = development_retention.loc[
        development_retention["retained_area_fraction"] >= 0.95
    ].sort_values("min_area_um2", ascending=False)
    logical_minimum = float(logical.iloc[0]["min_area_um2"]) if len(logical) else None
    development = metrics.loc[metrics["split"] == "dev"].sort_values(
        ["median_dice", "mean_dice", "compact_false_positive_rate", "min_area_um2"],
        ascending=[False, False, True, True],
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    components.to_csv(output / "annotation_components.csv", index=False)
    retention.to_csv(output / "annotation_retention_by_candidate.csv", index=False)
    metrics.to_csv(output / "fixed_response_candidate_metrics.csv", index=False)
    summary = {
        "selection_data": "development split only",
        "base_detector_config": config.__dict__,
        "candidates_um2": list(candidates_um2),
        "logical_minimum_um2": logical_minimum,
        "logical_minimum_rule": (
            "largest tested cutoff retaining at least 95% of annotated development vacuole area"
        ),
        "development_metric_winner_um2": float(development.iloc[0]["min_area_um2"]),
        "development_metric_winner": development.iloc[0].to_dict(),
        "note": (
            "Minimum area is a detector noise-control parameter, not a biological "
            "definition of whether a cleft is a vacuole."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze annotated component areas and isolate the detector minimum-area filter"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--candidates",
        type=float,
        nargs="+",
        default=list(DEFAULT_CANDIDATES_UM2),
        help="Candidate minimum component areas in square micrometers",
    )
    args = parser.parse_args()
    summary = analyze_minimum_area(
        args.manifest,
        args.config,
        args.output,
        tuple(args.candidates),
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
