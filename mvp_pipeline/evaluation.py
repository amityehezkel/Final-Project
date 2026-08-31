from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import compute_fiber_metrics, segmentation_metrics


def evaluate_vacuoles(
    prediction: np.ndarray,
    truth: np.ndarray,
    axon: np.ndarray,
    outer_fiber: np.ndarray,
    scale_nm_per_px: float,
) -> dict[str, Any]:
    scores = segmentation_metrics(prediction, truth)
    predicted = compute_fiber_metrics(axon, outer_fiber, prediction, scale_nm_per_px)
    expected = compute_fiber_metrics(axon, outer_fiber, truth, scale_nm_per_px)
    pred_area = predicted["vacuole_area_um2"]
    true_area = expected["vacuole_area_um2"]
    area_abs_error = abs(pred_area - true_area)
    area_pct_error = (
        0.0 if true_area == 0 and pred_area == 0 else math_nan_if_zero(true_area, area_abs_error)
    )
    return {
        **scores,
        "predicted_vacuole_area_um2": pred_area,
        "true_vacuole_area_um2": true_area,
        "vacuole_area_absolute_error_um2": area_abs_error,
        "vacuole_area_percentage_error": area_pct_error,
    }


def math_nan_if_zero(denominator: float, numerator: float) -> float:
    return float("nan") if denominator == 0 else 100.0 * numerator / denominator


def interpretation(median_dice: float, median_area_pct_error: float) -> str:
    if median_dice >= 0.65 and median_area_pct_error <= 30:
        return "successful proof of concept against reference annotations"
    if median_dice >= 0.40:
        return "preliminary detector with documented limitations"
    return "semi-automatic proposal workflow; automated detection underperformed"


def summarize_evaluation(frame: pd.DataFrame) -> dict[str, Any]:
    valid = frame.loc[~frame["excluded_from_summary"].astype(bool)].copy()
    if valid.empty:
        return {"n_evaluated": 0, "interpretation": "no eligible consensus-labeled rows"}
    summary: dict[str, Any] = {
        "n_evaluated": int(len(valid)),
        "median_dice": float(valid["dice"].median()),
        "median_iou": float(valid["iou"].median()),
        "median_precision": float(valid["precision"].median()),
        "median_recall": float(valid["recall"].median()),
        "median_vacuole_area_absolute_error_um2": float(
            valid["vacuole_area_absolute_error_um2"].median()
        ),
    }
    finite_area = valid["vacuole_area_percentage_error"].replace([np.inf, -np.inf], np.nan).dropna()
    median_pct = float(finite_area.median()) if not finite_area.empty else float("nan")
    summary["median_vacuole_area_percentage_error"] = median_pct
    summary["interpretation"] = interpretation(summary["median_dice"], median_pct)
    return summary


def write_evaluation_plot(frame: pd.DataFrame, path: str | Path) -> None:
    valid = frame.loc[~frame["excluded_from_summary"].astype(bool)].copy()
    if valid.empty:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].hist(valid["dice"], bins=np.linspace(0, 1, 11), color="#4c78a8", edgecolor="white")
    axes[0].axvline(valid["dice"].median(), color="#d62728", linestyle="--", label="median")
    axes[0].set(xlabel="Vacuole Dice", ylabel="Axons", xlim=(0, 1), title="Pixel overlap")
    axes[0].legend()

    axes[1].scatter(
        valid["true_vacuole_area_um2"],
        valid["predicted_vacuole_area_um2"],
        c=valid["dice"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        edgecolors="black",
        linewidths=0.4,
    )
    limit = float(
        max(
            valid["true_vacuole_area_um2"].max(),
            valid["predicted_vacuole_area_um2"].max(),
            1e-6,
        )
    )
    axes[1].plot([0, limit], [0, limit], color="#d62728", linestyle="--")
    axes[1].set(
        xlabel="Reference vacuole area (µm²)",
        ylabel="Predicted vacuole area (µm²)",
        title="Area agreement",
    )
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def copy_ranked_examples(
    evaluation: pd.DataFrame, overlay_dir: str | Path, output_dir: str | Path
) -> dict[str, list[str]]:
    valid = evaluation.loc[~evaluation["excluded_from_summary"].astype(bool)].sort_values("dice")
    if valid.empty:
        return {"success": [], "failure": []}
    source_dir = Path(overlay_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    # Prefer genuinely vacuolated reference fibers so a trivial true-negative
    # compact fiber is not presented as the detector's best visual success.
    positive = valid.loc[valid["true_vacuole_area_um2"] > 0]
    ranked = positive if len(positive) >= 3 else valid
    selected = {
        "failure": ranked.head(min(3, len(ranked))),
        "success": ranked.tail(min(3, len(ranked))).sort_values(
            "dice", ascending=False
        ),
    }
    result: dict[str, list[str]] = {"success": [], "failure": []}
    for kind, subset in selected.items():
        for rank, row in enumerate(subset.itertuples(index=False), start=1):
            source = source_dir / f"{row.id}_overlay.png"
            if not source.exists():
                continue
            target = destination / f"{kind}_{rank}_{row.id}_dice-{row.dice:.3f}.png"
            shutil.copy2(source, target)
            result[kind].append(str(target))
    with (destination / "examples.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result
