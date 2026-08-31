from __future__ import annotations

import math
from typing import Any

import numpy as np


def pixel_area_um2(scale_nm_per_px: float) -> float:
    if scale_nm_per_px <= 0:
        raise ValueError("scale_nm_per_px must be positive")
    return (scale_nm_per_px / 1000.0) ** 2


def compute_fiber_metrics(
    axon: np.ndarray,
    outer_fiber: np.ndarray,
    vacuoles: np.ndarray,
    scale_nm_per_px: float,
) -> dict[str, Any]:
    axon = np.asarray(axon, dtype=bool)
    outer = np.asarray(outer_fiber, dtype=bool)
    vacuoles = np.asarray(vacuoles, dtype=bool)
    if axon.shape != outer.shape or axon.shape != vacuoles.shape:
        raise ValueError("All masks must have the same shape")

    outer = outer | axon
    gross_sheath = outer & ~axon
    vacuoles = vacuoles & gross_sheath
    intact = gross_sheath & ~vacuoles
    unit = pixel_area_um2(scale_nm_per_px)

    axon_area = float(axon.sum() * unit)
    outer_area = float(outer.sum() * unit)
    gross_area = float(gross_sheath.sum() * unit)
    vacuole_area = float(vacuoles.sum() * unit)
    intact_area = float(intact.sum() * unit)
    standard_g = math.sqrt(axon_area / outer_area) if outer_area > 0 else math.nan
    effective_denominator = outer_area - vacuole_area
    intact_g = (
        math.sqrt(axon_area / effective_denominator)
        if effective_denominator > 0
        else math.nan
    )
    burden = vacuole_area / gross_area if gross_area > 0 else math.nan

    return {
        "scale_nm_per_px": float(scale_nm_per_px),
        "axon_area_um2": axon_area,
        "outer_fiber_area_um2": outer_area,
        "gross_sheath_area_um2": gross_area,
        "intact_myelin_area_um2": intact_area,
        "vacuole_area_um2": vacuole_area,
        "vacuole_burden": burden,
        "g_ratio": standard_g,
        "intact_equivalent_g_ratio": intact_g,
    }


def segmentation_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(truth, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError("Prediction and truth must have the same shape")
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    union = tp + fp + fn
    denom = 2 * tp + fp + fn
    return {
        "dice": 1.0 if denom == 0 else (2.0 * tp) / denom,
        "iou": 1.0 if union == 0 else tp / union,
        "precision": 1.0 if tp + fp == 0 else tp / (tp + fp),
        "recall": 1.0 if tp + fn == 0 else tp / (tp + fn),
    }
