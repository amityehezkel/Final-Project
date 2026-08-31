from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from .masks import area_um2_to_pixels, um_to_pixels


CONNECTIVITY = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class FiberInstance:
    """One automatically separated fiber, stored in crop coordinates."""

    number: int
    bbox: tuple[int, int, int, int]
    axon: np.ndarray
    outer_fiber: np.ndarray
    extraction_flags: tuple[str, ...]
    source_cluster_axon_count: int
    myelin_coverage: float
    axon_area_um2: float
    axon_solidity: float


@dataclass(frozen=True)
class ExtractionResult:
    fibers: tuple[FiberInstance, ...]
    axon_components_found: int
    rejection_counts: dict[str, int]


def _touches_border(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def _bounding_box(mask: np.ndarray, margin_px: int) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        raise ValueError("Cannot create a bounding box for an empty mask")
    height, width = mask.shape
    y0 = max(0, int(yy.min()) - margin_px)
    y1 = min(height, int(yy.max()) + 1 + margin_px)
    x0 = max(0, int(xx.min()) - margin_px)
    x1 = min(width, int(xx.max()) + 1 + margin_px)
    return x0, y0, x1, y1


def extract_fiber_instances(
    axon_mask: np.ndarray,
    myelin_mask: np.ndarray,
    scale_nm_per_px: float,
    *,
    excluded_region: np.ndarray | None = None,
    min_axon_area_um2: float = 0.01,
    crop_margin_um: float = 0.25,
) -> ExtractionResult:
    """Separate whole-image AxonDeepSeg masks into per-fiber crops.

    Each connected axon is used as a watershed seed. This allows adjacent
    myelin predictions that touch one another to be split into one region per
    axon. Enclosed gaps are filled so that vacuoles remain part of the gross
    outer-fiber envelope searched by the downstream detector.
    """

    if scale_nm_per_px <= 0:
        raise ValueError("scale_nm_per_px must be positive")
    if min_axon_area_um2 < 0 or crop_margin_um < 0:
        raise ValueError("Physical area and margin settings must be non-negative")

    axon = np.asarray(axon_mask, dtype=bool).copy()
    myelin = np.asarray(myelin_mask, dtype=bool).copy()
    if axon.shape != myelin.shape:
        raise ValueError("Axon and myelin masks must have the same shape")

    if excluded_region is None:
        excluded = np.zeros_like(axon)
    else:
        excluded = np.asarray(excluded_region, dtype=bool)
        if excluded.shape != axon.shape:
            raise ValueError("Excluded region must match the segmentation shape")

    # Preserve the fact that a segmentation touched the excluded scale-bar
    # region before removing those pixels.
    exclusion_edge = ndi.binary_dilation(excluded, structure=CONNECTIVITY) & ~excluded
    axon_in_excluded = axon & excluded
    axon[excluded] = False
    myelin[excluded] = False
    myelin &= ~axon

    initial_labels, initial_count = ndi.label(axon, structure=CONNECTIVITY)
    rejection_counts: Counter[str] = Counter()
    if axon_in_excluded.any():
        _, count = ndi.label(axon_in_excluded, structure=CONNECTIVITY)
        rejection_counts["scale_bar_overlap"] += int(count)

    if initial_count == 0:
        return ExtractionResult((), 0, dict(rejection_counts))

    minimum_pixels = area_um2_to_pixels(min_axon_area_um2, scale_nm_per_px)
    sizes = np.bincount(initial_labels.ravel())
    keep_ids = np.flatnonzero(sizes >= minimum_pixels)
    keep_ids = keep_ids[keep_ids != 0]
    rejection_counts["axon_below_minimum_area"] += int(initial_count - len(keep_ids))
    axon = np.isin(initial_labels, keep_ids)

    axon_labels, axon_count = ndi.label(axon, structure=CONNECTIVITY)
    if axon_count == 0:
        return ExtractionResult((), int(initial_count), dict(rejection_counts))

    # A one-pixel closing tolerates small class-boundary gaps in the external
    # segmentation. Any resulting touching clusters are resolved by watershed.
    combined = ndi.binary_closing(axon | myelin, structure=CONNECTIVITY)
    combined |= axon | myelin
    distance = ndi.distance_transform_edt(combined)
    zones = watershed(-distance, markers=axon_labels, mask=combined)

    cluster_labels, _ = ndi.label(combined, structure=CONNECTIVITY)
    axons_per_cluster: Counter[int] = Counter()
    cluster_for_axon: dict[int, int] = {}
    for label_id in range(1, axon_count + 1):
        cluster_values = cluster_labels[axon_labels == label_id]
        cluster_values = cluster_values[cluster_values > 0]
        if cluster_values.size == 0:
            continue
        cluster_id = int(np.bincount(cluster_values).argmax())
        cluster_for_axon[label_id] = cluster_id
        axons_per_cluster[cluster_id] += 1

    margin_px = um_to_pixels(crop_margin_um, scale_nm_per_px)
    fibers: list[FiberInstance] = []
    for label_id in range(1, axon_count + 1):
        current_axon = axon_labels == label_id
        assigned = zones == label_id
        if not assigned.any():
            rejection_counts["unassigned_axon"] += 1
            continue

        outer = ndi.binary_fill_holes(assigned) | current_axon
        foreign_axons = (axon_labels > 0) & ~current_axon
        if (outer & foreign_axons).any():
            rejection_counts["ambiguous_foreign_axon"] += 1
            continue
        if _touches_border(outer):
            rejection_counts["border_touching"] += 1
            continue
        if (outer & exclusion_edge).any():
            rejection_counts["scale_bar_boundary"] += 1
            continue

        gross_sheath = outer & ~current_axon
        if not gross_sheath.any():
            rejection_counts["empty_gross_sheath"] += 1
            continue

        covered = int((myelin & gross_sheath).sum())
        myelin_coverage = covered / int(gross_sheath.sum())
        axon_area_um2 = float(
            current_axon.sum() * (scale_nm_per_px / 1000.0) ** 2
        )
        convex = ndi.binary_fill_holes(current_axon)
        try:
            from skimage.morphology import convex_hull_image

            convex = convex_hull_image(convex)
        except ValueError:
            # A degenerate component is still exported but receives a low
            # solidity and is therefore recommended for review.
            pass
        axon_solidity = float(current_axon.sum() / max(1, convex.sum()))
        cluster_id = cluster_for_axon.get(label_id, 0)
        cluster_size = int(axons_per_cluster.get(cluster_id, 1))
        flags: list[str] = []
        if cluster_size > 1:
            flags.append("watershed_split_touching_cluster")
        if myelin_coverage < 0.5:
            flags.append("low_myelin_coverage")
        if axon_solidity < 0.75:
            flags.append("irregular_axon_shape")

        x0, y0, x1, y1 = _bounding_box(outer, margin_px)
        fibers.append(
            FiberInstance(
                number=len(fibers) + 1,
                bbox=(x0, y0, x1, y1),
                axon=current_axon[y0:y1, x0:x1].copy(),
                outer_fiber=outer[y0:y1, x0:x1].copy(),
                extraction_flags=tuple(flags),
                source_cluster_axon_count=cluster_size,
                myelin_coverage=float(myelin_coverage),
                axon_area_um2=axon_area_um2,
                axon_solidity=axon_solidity,
            )
        )

    # Grossly merged axon predictions can otherwise look like a single valid
    # component. Mark relative area outliers without assuming a fixed species-
    # specific axon diameter.
    if fibers:
        median_axon_area = float(np.median([fiber.axon_area_um2 for fiber in fibers]))
        if median_axon_area > 0:
            revised: list[FiberInstance] = []
            for fiber in fibers:
                if fiber.axon_area_um2 > 5.0 * median_axon_area:
                    revised.append(
                        replace(
                            fiber,
                            extraction_flags=tuple(
                                (*fiber.extraction_flags, "large_axon_area_outlier")
                            ),
                        )
                    )
                else:
                    revised.append(fiber)
            fibers = revised

    return ExtractionResult(
        tuple(fibers), int(initial_count), dict(sorted(rejection_counts.items()))
    )
