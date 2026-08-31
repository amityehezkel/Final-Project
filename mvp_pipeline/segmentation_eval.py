from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .io import read_binary_mask
from .metrics import segmentation_metrics


REQUIRED_COLUMNS = {
    "id",
    "variant",
    "axon_prediction_path",
    "outer_prediction_path",
    "axon_truth_path",
    "outer_truth_path",
    "correction_minutes",
}


def evaluate_variants(table_path: str | Path, output_dir: str | Path) -> pd.DataFrame:
    table_path = Path(table_path).resolve()
    table = pd.read_csv(table_path)
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"Segmentation comparison is missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for row in table.to_dict(orient="records"):
        resolved = {}
        for column in REQUIRED_COLUMNS - {"id", "variant", "correction_minutes"}:
            path = Path(str(row[column]))
            resolved[column] = path if path.is_absolute() else (table_path.parent / path).resolve()
        axon_truth = read_binary_mask(resolved["axon_truth_path"])
        outer_truth = read_binary_mask(resolved["outer_truth_path"], axon_truth.shape)
        axon_pred = read_binary_mask(resolved["axon_prediction_path"], axon_truth.shape)
        outer_pred = read_binary_mask(resolved["outer_prediction_path"], axon_truth.shape)
        records.append(
            {
                "id": row["id"],
                "variant": row["variant"],
                "correction_minutes": float(row["correction_minutes"]),
                "axon_dice": segmentation_metrics(axon_pred, axon_truth)["dice"],
                "outer_fiber_dice": segmentation_metrics(outer_pred, outer_truth)["dice"],
            }
        )
    frame = pd.DataFrame(records)
    summary = frame.groupby("variant").agg(
        median_axon_dice=("axon_dice", "median"),
        median_outer_fiber_dice=("outer_fiber_dice", "median"),
        median_correction_minutes=("correction_minutes", "median"),
    )
    summary["automatic_masks_acceptable"] = (
        (summary["median_axon_dice"] >= 0.75)
        & (summary["median_outer_fiber_dice"] >= 0.75)
        & (summary["median_correction_minutes"] <= 3.0)
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "segmentation_per_axon.csv", index=False)
    summary.to_csv(output / "segmentation_summary.csv")
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw, 2.36, and 4.93 nm/px masks")
    parser.add_argument("--table", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(evaluate_variants(args.table, args.output).to_string(index=False))


if __name__ == "__main__":
    main()

