from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import summarize_evaluation, write_evaluation_plot
from .io import load_manifest, read_binary_mask
from .metrics import segmentation_metrics


def _copy_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_overlays(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*_overlay.png")):
        shutil.copy2(path, destination / path.name)


def _whole_mode_evaluation(
    benchmark_path: Path,
    whole_results: Path,
    cropped_evaluation: pd.DataFrame,
    automatic_mask_evaluation: pd.DataFrame,
) -> pd.DataFrame:
    crop_by_id = cropped_evaluation.set_index("id")
    masks_by_id = automatic_mask_evaluation.set_index("id")
    records: list[dict[str, Any]] = []
    for row in load_manifest(benchmark_path):
        truth = read_binary_mask(row.consensus_vacuole_mask_path)
        prediction = read_binary_mask(
            whole_results / "masks" / f"{row.id}_vacuole.png",
            truth.shape,
        )
        scores = segmentation_metrics(prediction, truth)
        pixel_area_um2 = (row.scale_nm_per_px / 1000.0) ** 2
        predicted_area = float(prediction.sum() * pixel_area_um2)
        true_area = float(truth.sum() * pixel_area_um2)
        absolute_error = abs(predicted_area - true_area)
        percentage_error = (
            0.0
            if true_area == 0 and predicted_area == 0
            else (float("nan") if true_area == 0 else 100.0 * absolute_error / true_area)
        )
        mask_row = masks_by_id.loc[row.id]
        records.append(
            {
                "id": row.id,
                "source_image_id": row.source_image_id or row.image_path.stem,
                "split": row.split,
                # Use the same reference-image QC eligibility as cropped mode,
                # but never hide an automatic extraction miss.
                "excluded_from_summary": bool(
                    crop_by_id.loc[row.id, "excluded_from_summary"]
                ),
                "automatic_fiber_recovered": bool(
                    mask_row["selection_status"] != "complete_miss"
                ),
                "automatic_selection_status": mask_row["selection_status"],
                "automatic_axon_dice": float(mask_row["axon_dice"]),
                "automatic_outer_fiber_dice": float(mask_row["outer_fiber_dice"]),
                "automatic_both_masks_pass": bool(mask_row["both_masks_pass"]),
                **scores,
                "predicted_vacuole_area_um2": predicted_area,
                "true_vacuole_area_um2": true_area,
                "vacuole_area_absolute_error_um2": absolute_error,
                "vacuole_area_percentage_error": percentage_error,
            }
        )
    return pd.DataFrame(records)


def _comparison_rows(
    cropped: pd.DataFrame,
    whole: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for mode, frame in (("cropped_mode", cropped), ("whole_image_mode", whole)):
        for split, group in frame.groupby("split"):
            summary = summarize_evaluation(group)
            record: dict[str, Any] = {"mode": mode, "split": split, **summary}
            if mode == "whole_image_mode":
                eligible = group.loc[~group["excluded_from_summary"].astype(bool)]
                record["automatic_fibers_recovered"] = int(
                    eligible["automatic_fiber_recovered"].sum()
                )
                record["automatic_complete_misses"] = int(
                    (~eligible["automatic_fiber_recovered"]).sum()
                )
                record["automatic_both_masks_pass"] = int(
                    eligible["automatic_both_masks_pass"].sum()
                )
            records.append(record)
    return pd.DataFrame(records)


def build_separate_mode_results(
    benchmark_path: str | Path,
    cropped_results: str | Path,
    whole_results: str | Path,
    automatic_mask_validation: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    benchmark = Path(benchmark_path).resolve()
    cropped = Path(cropped_results).resolve()
    whole = Path(whole_results).resolve()
    validation = Path(automatic_mask_validation).resolve()
    output = Path(output_dir).resolve()
    crop_output = output / "cropped_mode"
    whole_output = output / "whole_image_mode"
    crop_output.mkdir(parents=True, exist_ok=True)
    whole_output.mkdir(parents=True, exist_ok=True)

    cropped_evaluation = pd.read_csv(cropped / "evaluation_per_axon.csv")
    automatic_evaluation = pd.read_csv(validation / "per_crop_metrics.csv")
    whole_evaluation = _whole_mode_evaluation(
        benchmark,
        whole,
        cropped_evaluation,
        automatic_evaluation,
    )

    for name in ("metrics.csv", "evaluation_per_axon.csv", "summary.json"):
        _copy_file(cropped / name, crop_output / name)
    if (cropped / "evaluation_overview.png").exists():
        _copy_file(
            cropped / "evaluation_overview.png",
            crop_output / "evaluation_overview.png",
        )
    _copy_overlays(cropped / "overlays", crop_output / "overlays")

    whole_evaluation.to_csv(
        whole_output / "end_to_end_evaluation_per_fiber.csv",
        index=False,
    )
    automatic_evaluation.to_csv(
        whole_output / "automatic_mask_evaluation_per_fiber.csv",
        index=False,
    )
    _copy_file(whole / "metrics.csv", whole_output / "metrics.csv")
    _copy_overlays(whole / "overlays", whole_output / "overlays")
    write_evaluation_plot(
        whole_evaluation,
        whole_output / "end_to_end_evaluation_overview.png",
    )

    validation_summary = json.loads(
        (validation / "summary.json").read_text(encoding="utf-8")
    )
    whole_summary: dict[str, Any] = {
        "scope": (
            "Controlled end-to-end automatic-mask evaluation on the fixed benchmark "
            "fields. It uses the AxonDeepSeg front end and frozen vacuole detector, "
            "but is not a rematching experiment on the original full laboratory TIFFs."
        ),
        "n_processed": int(len(whole_evaluation)),
        "automatic_mask_front_end": validation_summary,
        "evaluation": {
            split: summarize_evaluation(group)
            for split, group in whole_evaluation.groupby("split")
        },
    }
    (whole_output / "summary.json").write_text(
        json.dumps(whole_summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    comparison = _comparison_rows(cropped_evaluation, whole_evaluation)
    comparison.to_csv(output / "mode_comparison_summary.csv", index=False)
    (output / "README.md").write_text(
        """# Separate results by input mode

The two modes answer different scientific questions and must not share a headline score.

## `cropped_mode/`

The image, axon mask, and outer-fiber mask are supplied. Its Dice and area errors primarily measure the frozen vacuole detector. These are the appropriate results when reporting the masked fiber-crop workflow.

## `whole_image_mode/`

AxonDeepSeg supplies the axon and outer-fiber masks before the same frozen vacuole detector runs. Its end-to-end results include missed fibers and segmentation errors. The benchmark comparison holds the field of view fixed for one-to-one scoring; it is not a full-TIFF rematching study. Therefore, describe this as the controlled automatic-mask or whole-image-front-end benchmark, while describing deployment on full TIFFs as an experimental proposal workflow.

`mode_comparison_summary.csv` places the split-level results side by side. Do not average the two modes together.
""",
        encoding="utf-8",
    )
    return {
        "cropped_mode_rows": int(len(cropped_evaluation)),
        "whole_image_mode_rows": int(len(whole_evaluation)),
        "comparison_rows": int(len(comparison)),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Package cropped and whole-image mode results separately")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--cropped-results", required=True)
    parser.add_argument("--whole-results", required=True)
    parser.add_argument("--automatic-mask-validation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_separate_mode_results(
        args.benchmark,
        args.cropped_results,
        args.whole_results,
        args.automatic_mask_validation,
        args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
