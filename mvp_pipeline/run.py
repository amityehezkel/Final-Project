from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DetectorConfig
from .detectors import detect_vacuoles
from .evaluation import (
    copy_ranked_examples,
    evaluate_vacuoles,
    summarize_evaluation,
    write_evaluation_plot,
)
from .io import load_manifest, read_binary_mask, read_grayscale, write_binary_mask
from .masks import sanitize_fiber_masks, scale_bar_region
from .metrics import compute_fiber_metrics
from .overlay import create_overlay, write_overlay


EXCLUSION_FLAGS = {
    "scale_bar_overlap",
    "border_touching",
    "empty_axon",
    "empty_outer_fiber",
    "empty_gross_sheath",
}


def run_batch(
    manifest_path: str | Path,
    output_dir: str | Path,
    config: DetectorConfig,
    split: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = load_manifest(manifest_path)
    if split:
        rows = [row for row in rows if row.split == split.lower()]
    if not rows:
        raise ValueError("No manifest rows match the requested split")

    output = Path(output_dir)
    mask_dir = output / "masks"
    overlay_dir = output / "overlays"
    metric_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []

    for row in rows:
        image = read_grayscale(row.image_path)
        shape = image.shape
        axon = read_binary_mask(row.axon_mask_path, shape)
        outer = read_binary_mask(row.outer_fiber_mask_path, shape)
        excluded = (
            scale_bar_region(
                shape,
                config.scale_bar_right_fraction,
                config.scale_bar_bottom_fraction,
            )
            if config.exclude_scale_bar
            else np.zeros(shape, dtype=bool)
        )
        axon, outer, flags = sanitize_fiber_masks(axon, outer, excluded)
        compact = (
            read_binary_mask(row.compact_myelin_mask_path, shape)
            if row.compact_myelin_mask_path is not None
            else None
        )
        if compact is not None:
            compact = compact & outer & ~axon & ~excluded

        prediction = detect_vacuoles(
            image,
            axon,
            outer,
            row.scale_nm_per_px,
            config,
            compact,
        )
        prediction &= outer & ~axon & ~excluded
        write_binary_mask(mask_dir / f"{row.id}_vacuole.png", prediction)
        write_overlay(
            overlay_dir / f"{row.id}_overlay.png",
            create_overlay(image, axon, outer, prediction),
        )

        excluded_from_summary = bool(EXCLUSION_FLAGS.intersection(flags))
        metric_rows.append(
            {
                "id": row.id,
                "source_image_id": row.source_image_id or row.image_path.stem,
                "split": row.split,
                "detector": config.detector,
                "min_area_um2": config.min_area_um2,
                "mask_source": row.mask_source,
                "correction_minutes": row.correction_minutes,
                "quality_control_flags": ";".join(flags) if flags else "pass",
                "excluded_from_summary": excluded_from_summary,
                **compute_fiber_metrics(
                    axon, outer, prediction, row.scale_nm_per_px
                ),
            }
        )

        if row.consensus_vacuole_mask_path is not None:
            truth = read_binary_mask(row.consensus_vacuole_mask_path, shape)
            truth &= outer & ~axon & ~excluded
            evaluation_rows.append(
                {
                    "id": row.id,
                    "source_image_id": row.source_image_id or row.image_path.stem,
                    "split": row.split,
                    "excluded_from_summary": excluded_from_summary,
                    **evaluate_vacuoles(
                        prediction, truth, axon, outer, row.scale_nm_per_px
                    ),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    evaluation = pd.DataFrame(evaluation_rows)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)

    summary: dict[str, Any] = {
        "manifest": str(Path(manifest_path).resolve()),
        "detector_config": config.__dict__,
        "n_processed": int(len(metrics)),
        "n_excluded_by_qc": int(metrics["excluded_from_summary"].sum()),
    }
    if not evaluation.empty:
        evaluation.to_csv(output / "evaluation_per_axon.csv", index=False)
        evaluation_summary: dict[str, Any] = {}
        for split_name, group in evaluation.groupby("split"):
            evaluation_summary[str(split_name)] = summarize_evaluation(group)
        summary["evaluation"] = evaluation_summary
        write_evaluation_plot(evaluation, output / "evaluation_overview.png")
        copy_ranked_examples(evaluation, overlay_dir, output / "examples")

    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=True)
    return metrics, evaluation, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Per-axon benchmark CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", help="Frozen detector JSON from mvp_pipeline.tune")
    parser.add_argument("--split", choices=("dev", "test"), help="Optional split filter")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = DetectorConfig.from_json(args.config) if args.config else DetectorConfig()
    _, _, summary = run_batch(args.manifest, args.output, config, args.split)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
