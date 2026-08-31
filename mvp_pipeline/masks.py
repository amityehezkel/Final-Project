from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi


def um_to_pixels(length_um: float, scale_nm_per_px: float) -> int:
    if length_um <= 0:
        return 0
    return max(1, int(round(length_um * 1000.0 / scale_nm_per_px)))


def area_um2_to_pixels(area_um2: float, scale_nm_per_px: float) -> int:
    pixel_area_um2 = (scale_nm_per_px / 1000.0) ** 2
    if area_um2 <= 0:
        return 0
    return max(1, int(math.ceil(area_um2 / pixel_area_um2)))


def scale_bar_region(
    shape: tuple[int, int], right_fraction: float = 0.25, bottom_fraction: float = 0.20
) -> np.ndarray:
    height, width = shape
    result = np.zeros(shape, dtype=bool)
    x0 = max(0, min(width, int(round(width * (1.0 - right_fraction)))))
    y0 = max(0, min(height, int(round(height * (1.0 - bottom_fraction)))))
    result[y0:, x0:] = True
    return result


def remove_small_components(
    mask: np.ndarray, min_area_um2: float, scale_nm_per_px: float
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    min_pixels = area_um2_to_pixels(min_area_um2, scale_nm_per_px)
    if min_pixels <= 1:
        return mask.copy()
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_pixels
    keep[0] = False
    return keep[labels]


def remove_border_components(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3)))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    border_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    result = np.asarray(mask, dtype=bool).copy()
    result[np.isin(labels, border_labels[border_labels != 0])] = False
    return result


def touches_image_border(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def sanitize_fiber_masks(
    axon: np.ndarray,
    outer_fiber: np.ndarray,
    excluded_region: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    axon = np.asarray(axon, dtype=bool).copy()
    outer = np.asarray(outer_fiber, dtype=bool).copy()
    if axon.shape != outer.shape:
        raise ValueError("Axon and outer-fiber masks must have the same shape")
    flags: list[str] = []
    if excluded_region is not None:
        if excluded_region.shape != axon.shape:
            raise ValueError("Excluded region shape does not match masks")
        if (outer & excluded_region).any():
            flags.append("scale_bar_overlap")
        axon[excluded_region] = False
        outer[excluded_region] = False
    if touches_image_border(outer):
        flags.append("border_touching")
    if (axon & ~outer).any():
        flags.append("axon_outside_outer_fiber")
        outer |= axon
    if not axon.any():
        flags.append("empty_axon")
    if not outer.any():
        flags.append("empty_outer_fiber")
    if not (outer & ~axon).any():
        flags.append("empty_gross_sheath")
    return axon, outer, flags

