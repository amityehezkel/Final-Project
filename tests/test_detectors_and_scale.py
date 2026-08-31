import numpy as np
import pytest

from mvp_pipeline.config import DetectorConfig
from mvp_pipeline.detectors import (
    detect_vacuoles,
    refine_vacuole_boundaries,
    rescue_thin_seed_candidates,
)
from mvp_pipeline.metrics import segmentation_metrics
from mvp_pipeline.scale import resample_image, restore_mask, target_shape


def synthetic_fiber():
    yy, xx = np.ogrid[:128, :128]
    axon = (xx - 64) ** 2 + (yy - 64) ** 2 <= 25**2
    outer = (xx - 64) ** 2 + (yy - 64) ** 2 <= 45**2
    vacuole = ((xx - 87) / 9) ** 2 + ((yy - 64) / 5) ** 2 <= 1
    vacuole &= outer & ~axon
    compact = outer & ~axon & ~vacuole
    image = np.full((128, 128), 110, dtype=np.float32)
    image[outer & ~axon] = 25
    image[vacuole] = 235
    image[axon] = 150
    return image, axon, outer, compact, vacuole


def test_geometry_detector_recovers_enclosed_gap():
    image, axon, outer, compact, truth = synthetic_fiber()
    config = DetectorConfig(detector="geometry", min_area_um2=0.00001)
    pred = detect_vacuoles(image, axon, outer, 5.0, config, compact)
    assert segmentation_metrics(pred, truth)["dice"] == 1.0


def test_intensity_detector_finds_bright_region():
    image, axon, outer, _, truth = synthetic_fiber()
    config = DetectorConfig(
        detector="intensity",
        min_area_um2=0.00001,
        gaussian_sigma_um=0,
        morphology_radius_um=0,
    )
    pred = detect_vacuoles(image, axon, outer, 5.0, config)
    assert (pred & truth).sum() > 0
    assert pred.sum() <= (outer & ~axon).sum()


def test_high_intensity_threshold_offset_removes_candidates():
    image, axon, outer, _, _ = synthetic_fiber()
    config = DetectorConfig(
        detector="intensity",
        min_area_um2=0.00001,
        intensity_threshold_offset=1.0,
        gaussian_sigma_um=0,
        morphology_radius_um=0,
    )
    pred = detect_vacuoles(image, axon, outer, 5.0, config)
    assert not pred.any()


def test_hysteresis_low_threshold_cannot_exceed_high_threshold():
    with pytest.raises(ValueError):
        DetectorConfig(
            intensity_threshold_offset=0.05,
            intensity_low_threshold_offset=0.10,
        )


def test_seeded_refinement_expands_without_creating_new_objects():
    yy, xx = np.ogrid[:80, :80]
    gross = (xx - 40) ** 2 + (yy - 40) ** 2 <= 30**2
    truth = ((xx - 52) / 10) ** 2 + ((yy - 40) / 7) ** 2 <= 1
    seed = ((xx - 52) / 6) ** 2 + ((yy - 40) / 4) ** 2 <= 1
    distractor = (xx - 24) ** 2 + (yy - 24) ** 2 <= 3**2
    response = np.full((80, 80), 0.1, dtype=np.float32)
    response[truth] = 0.6
    response[seed] = 0.9
    response[distractor] = 0.9

    refined = refine_vacuole_boundaries(
        response,
        gross,
        seed,
        otsu_threshold=0.4,
        scale_nm_per_px=5.0,
        max_distance_um=0.05,
        growth_offset=-0.05,
        max_area_ratio=3.0,
    )

    assert np.array_equal(refined, truth)
    assert not (refined & distractor).any()


def test_thin_seed_rescue_keeps_inner_cleft_and_rejects_outer_halo():
    yy, xx = np.ogrid[:160, :160]
    axon = (xx - 80) ** 2 + (yy - 80) ** 2 <= 30**2
    outer = (xx - 80) ** 2 + (yy - 80) ** 2 <= 60**2
    inner_cleft = ((xx - 119) / 7) ** 2 + ((yy - 80) / 11) ** 2 <= 1
    outer_halo = ((xx - 80) / 20) ** 2 + ((yy - 132) / 4) ** 2 <= 1
    raw_candidate = (inner_cleft | outer_halo) & outer & ~axon
    config = DetectorConfig(
        thin_seed_rescue=True,
        rescue_morphology_radius_um=0,
        rescue_min_area_um2=0.0015,
        rescue_min_thickness_um=0.045,
        rescue_max_radial_position=0.55,
        rescue_max_eccentricity=0.95,
        rescue_min_solidity=0.85,
    )

    rescued = rescue_thin_seed_candidates(
        raw_candidate,
        axon,
        outer,
        scale_nm_per_px=5.0,
        config=config,
    )

    assert (rescued & inner_cleft).any()
    assert not (rescued & outer_halo).any()


def test_resample_and_restore_mask():
    image = np.arange(100 * 80, dtype=np.float32).reshape(100, 80)
    assert target_shape(image.shape, 1.0, 2.0) == (50, 40)
    resampled = resample_image(image, 1.0, 2.0)
    assert resampled.shape == (50, 40)
    mask = np.zeros((50, 40), dtype=bool)
    mask[10:20, 10:20] = True
    restored = restore_mask(mask, image.shape)
    assert restored.shape == image.shape
    assert restored.dtype == bool
