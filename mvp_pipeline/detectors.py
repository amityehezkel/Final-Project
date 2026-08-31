from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology

from .config import DetectorConfig
from .masks import remove_small_components, um_to_pixels


def _normalize_to_unit(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    values = image[np.asarray(region, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def geometry_detector(
    axon: np.ndarray,
    outer_fiber: np.ndarray,
    compact_myelin: np.ndarray,
    scale_nm_per_px: float,
    min_area_um2: float,
) -> np.ndarray:
    """Find enclosed gaps not explained by axon or compact myelin."""

    axon = np.asarray(axon, dtype=bool)
    outer = np.asarray(outer_fiber, dtype=bool) | axon
    compact = np.asarray(compact_myelin, dtype=bool) & outer & ~axon
    combined = axon | compact
    filled = ndi.binary_fill_holes(combined)
    gross_sheath = outer & ~axon
    candidate = filled & gross_sheath & ~compact
    return remove_small_components(candidate, min_area_um2, scale_nm_per_px)


def intensity_detector(
    image: np.ndarray,
    axon: np.ndarray,
    outer_fiber: np.ndarray,
    scale_nm_per_px: float,
    config: DetectorConfig,
) -> np.ndarray:
    """Find bright anomaly candidates inside the gross myelin envelope."""

    smoothed, gross_sheath, otsu_threshold = prepare_intensity_response(
        image,
        axon,
        outer_fiber,
        scale_nm_per_px,
        config.clahe_clip_limit,
        config.gaussian_sigma_um,
    )
    seed = threshold_intensity_response(
        smoothed,
        gross_sheath,
        otsu_threshold,
        scale_nm_per_px,
        config,
    )
    if config.boundary_refinement and seed.any():
        seed = refine_vacuole_boundaries(
            smoothed,
            gross_sheath,
            seed,
            otsu_threshold,
            scale_nm_per_px,
            config.refinement_max_distance_um,
            config.refinement_growth_offset,
            config.refinement_max_area_ratio,
        )
    if not config.thin_seed_rescue:
        return seed
    raw_candidate = raw_intensity_candidate(
        smoothed,
        gross_sheath,
        otsu_threshold,
        config,
    )
    rescued = rescue_thin_seed_candidates(
        raw_candidate,
        axon,
        outer_fiber,
        scale_nm_per_px,
        config,
    )
    return (seed | rescued) & gross_sheath


def refine_vacuole_boundaries(
    response: np.ndarray,
    gross_sheath: np.ndarray,
    seeds: np.ndarray,
    otsu_threshold: float,
    scale_nm_per_px: float,
    max_distance_um: float,
    growth_offset: float,
    max_area_ratio: float,
) -> np.ndarray:
    """Grow detected seeds through locally similar pixels without creating objects."""

    gross = np.asarray(gross_sheath, dtype=bool)
    seed = np.asarray(seeds, dtype=bool) & gross
    if not seed.any() or max_distance_um <= 0:
        return seed

    labels = measure.label(seed, connectivity=2)
    max_distance_px = max_distance_um * 1000.0 / scale_nm_per_px
    minimum_response = np.clip(otsu_threshold + growth_offset, 0.0, 1.0)
    refined = np.zeros(seed.shape, dtype=bool)
    for component in measure.regionprops(labels):
        seed_component = labels == component.label
        distance = ndi.distance_transform_edt(~seed_component)
        allowed = gross & (distance <= max_distance_px) & (
            response >= minimum_response
        )
        allowed |= seed_component
        grown = ndi.binary_propagation(seed_component, mask=allowed)
        if grown.sum() <= max_area_ratio * seed_component.sum():
            refined |= grown
        else:
            # A region that exceeds the guardrail likely leaked through a weak edge.
            refined |= seed_component
    return refined & gross


def prepare_intensity_response(
    image: np.ndarray,
    axon: np.ndarray,
    outer_fiber: np.ndarray,
    scale_nm_per_px: float,
    clahe_clip_limit: float,
    gaussian_sigma_um: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Prepare the scale-aware intensity response shared by many thresholds."""

    axon = np.asarray(axon, dtype=bool)
    outer = np.asarray(outer_fiber, dtype=bool) | axon
    gross_sheath = outer & ~axon
    if not gross_sheath.any():
        return np.zeros_like(image, dtype=np.float32), gross_sheath, 1.0

    normalized = _normalize_to_unit(image, gross_sheath)
    kernel_px = min(min(image.shape), max(8, um_to_pixels(0.25, scale_nm_per_px)))
    equalized = exposure.equalize_adapthist(
        normalized,
        kernel_size=(kernel_px, kernel_px),
        clip_limit=clahe_clip_limit,
        nbins=256,
    )
    sigma_px = gaussian_sigma_um * 1000.0 / scale_nm_per_px
    smoothed = filters.gaussian(equalized, sigma=sigma_px, preserve_range=True)
    values = smoothed[gross_sheath]
    otsu_threshold = (
        filters.threshold_otsu(values)
        if np.unique(values).size > 1
        else float(values[0])
    )
    return smoothed, gross_sheath, float(otsu_threshold)


def threshold_intensity_response(
    smoothed: np.ndarray,
    gross_sheath: np.ndarray,
    otsu_threshold: float,
    scale_nm_per_px: float,
    config: DetectorConfig,
) -> np.ndarray:
    """Threshold and physically filter a prepared intensity response."""

    candidate = raw_intensity_candidate(
        smoothed,
        gross_sheath,
        otsu_threshold,
        config,
    )
    if not candidate.any():
        return candidate
    radius_px = um_to_pixels(config.morphology_radius_um, scale_nm_per_px)
    if radius_px > 0:
        if radius_px >= 5:
            # Distance-transform morphology is equivalent to a circular
            # footprint and is dramatically faster for highly magnified data.
            candidate = morphology.isotropic_closing(candidate, radius_px)
            candidate = morphology.isotropic_opening(candidate, radius_px)
        else:
            footprint = morphology.disk(radius_px)
            candidate = morphology.closing(candidate, footprint)
            candidate = morphology.opening(candidate, footprint)
    return remove_small_components(candidate, config.min_area_um2, scale_nm_per_px)


def raw_intensity_candidate(
    smoothed: np.ndarray,
    gross_sheath: np.ndarray,
    otsu_threshold: float,
    config: DetectorConfig,
) -> np.ndarray:
    """Create the connected high/low-threshold candidate before morphology."""

    if not gross_sheath.any():
        return np.zeros_like(gross_sheath)
    high_threshold = min(
        1.0, otsu_threshold + config.intensity_threshold_offset
    )
    low_offset = (
        config.intensity_threshold_offset
        if config.intensity_low_threshold_offset is None
        else config.intensity_low_threshold_offset
    )
    low_threshold = min(1.0, otsu_threshold + low_offset)
    if low_threshold < high_threshold:
        # Outside-sheath pixels cannot connect two candidate regions during
        # hysteresis growth.
        masked_response = np.where(gross_sheath, smoothed, -np.inf)
        candidate = filters.apply_hysteresis_threshold(
            masked_response, low_threshold, high_threshold
        )
    else:
        candidate = (smoothed > high_threshold) & gross_sheath
    return candidate & gross_sheath


def rescue_thin_seed_candidates(
    raw_candidate: np.ndarray,
    axon: np.ndarray,
    outer_fiber: np.ndarray,
    scale_nm_per_px: float,
    config: DetectorConfig,
) -> np.ndarray:
    """Preserve plausible inner-sheath clefts removed by standard morphology."""

    axon = np.asarray(axon, dtype=bool)
    outer = np.asarray(outer_fiber, dtype=bool) | axon
    gross = outer & ~axon
    candidate = np.asarray(raw_candidate, dtype=bool) & gross
    radius_px = um_to_pixels(config.rescue_morphology_radius_um, scale_nm_per_px)
    if radius_px > 0:
        if radius_px >= 5:
            candidate = morphology.isotropic_closing(candidate, radius_px)
            candidate = morphology.isotropic_opening(candidate, radius_px)
        else:
            footprint = morphology.disk(radius_px)
            candidate = morphology.closing(candidate, footprint)
            candidate = morphology.opening(candidate, footprint)
    labels = measure.label(candidate, connectivity=2)
    if labels.max() == 0:
        return np.zeros_like(candidate)

    distance_to_axon = ndi.distance_transform_edt(~axon)
    distance_to_outer = ndi.distance_transform_edt(outer)
    pixel_area_um2 = (scale_nm_per_px / 1000.0) ** 2
    rescued = np.zeros_like(candidate)
    for component in measure.regionprops(labels):
        component_mask = labels == component.label
        area_um2 = component.area * pixel_area_um2
        thickness_um = (
            2.0
            * ndi.distance_transform_edt(component_mask).max()
            * scale_nm_per_px
            / 1000.0
        )
        denominator = (
            distance_to_axon[component_mask]
            + distance_to_outer[component_mask]
            + 1e-9
        )
        radial_position = float(
            np.median(distance_to_axon[component_mask] / denominator)
        )
        if (
            area_um2 >= config.rescue_min_area_um2
            and thickness_um >= config.rescue_min_thickness_um
            and radial_position <= config.rescue_max_radial_position
            and component.eccentricity <= config.rescue_max_eccentricity
            and component.solidity >= config.rescue_min_solidity
        ):
            rescued |= component_mask
    return rescued


def detect_vacuoles(
    image: np.ndarray,
    axon: np.ndarray,
    outer_fiber: np.ndarray,
    scale_nm_per_px: float,
    config: DetectorConfig,
    compact_myelin: np.ndarray | None = None,
) -> np.ndarray:
    if config.detector == "geometry":
        if compact_myelin is None:
            raise ValueError("The geometry detector requires compact_myelin_mask_path")
        return geometry_detector(
            axon,
            outer_fiber,
            compact_myelin,
            scale_nm_per_px,
            config.min_area_um2,
        )
    return intensity_detector(image, axon, outer_fiber, scale_nm_per_px, config)
