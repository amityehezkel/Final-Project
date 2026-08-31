import math

import numpy as np

from mvp_pipeline.masks import (
    area_um2_to_pixels,
    sanitize_fiber_masks,
    scale_bar_region,
    um_to_pixels,
)
from mvp_pipeline.metrics import compute_fiber_metrics, segmentation_metrics


def test_physical_conversions():
    assert um_to_pixels(0.1, 5.0) == 20
    assert area_um2_to_pixels(0.01, 5.0) == 400


def test_metrics_and_g_ratios():
    axon = np.zeros((20, 20), dtype=bool)
    outer = np.zeros_like(axon)
    vacuoles = np.zeros_like(axon)
    axon[6:14, 6:14] = True
    outer[4:16, 4:16] = True
    vacuoles[4:6, 8:12] = True
    metrics = compute_fiber_metrics(axon, outer, vacuoles, scale_nm_per_px=10)
    unit = 0.0001
    assert math.isclose(metrics["axon_area_um2"], 64 * unit)
    assert math.isclose(metrics["outer_fiber_area_um2"], 144 * unit)
    assert math.isclose(metrics["vacuole_area_um2"], 8 * unit)
    assert math.isclose(metrics["g_ratio"], math.sqrt(64 / 144))
    assert math.isclose(metrics["intact_equivalent_g_ratio"], math.sqrt(64 / 136))
    assert metrics["intact_equivalent_g_ratio"] > metrics["g_ratio"]


def test_overlap_scores():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[5:7, 5:7] = True
    scores = segmentation_metrics(mask, mask)
    assert scores == {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}


def test_scale_bar_and_border_quality_control():
    axon = np.zeros((100, 100), dtype=bool)
    outer = np.zeros_like(axon)
    axon[40:60, 40:60] = True
    outer[30:70, 30:70] = True
    excluded = scale_bar_region(outer.shape)
    clean_axon, clean_outer, flags = sanitize_fiber_masks(axon, outer, excluded)
    assert not flags
    assert clean_axon.sum() == axon.sum()
    assert clean_outer.sum() == outer.sum()

    outer[0, 50] = True
    _, _, flags = sanitize_fiber_masks(axon, outer)
    assert "border_touching" in flags
