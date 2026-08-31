from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from mvp_pipeline.auto_run import run_automatic_whole_image
from mvp_pipeline.config import DetectorConfig
from mvp_pipeline.instances import extract_fiber_instances


def _circle(shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def _write_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def test_extract_instances_splits_touching_myelin_and_rejects_border() -> None:
    shape = (150, 180)
    axon_a = _circle(shape, (70, 65), 11)
    axon_b = _circle(shape, (70, 105), 11)
    axon_border = _circle(shape, (70, 5), 5)
    outer_a = _circle(shape, (70, 65), 24)
    outer_b = _circle(shape, (70, 105), 24)
    outer_border = _circle(shape, (70, 5), 12)
    axon = axon_a | axon_b | axon_border
    myelin = (outer_a | outer_b | outer_border) & ~axon

    result = extract_fiber_instances(
        axon,
        myelin,
        5.0,
        min_axon_area_um2=0.0001,
        crop_margin_um=0.05,
    )

    assert result.axon_components_found == 3
    assert len(result.fibers) == 2
    assert result.rejection_counts["border_touching"] == 1
    assert all(fiber.source_cluster_axon_count == 2 for fiber in result.fibers)
    assert all(
        "watershed_split_touching_cluster" in fiber.extraction_flags
        for fiber in result.fibers
    )
    assert all((fiber.axon & ~fiber.outer_fiber).sum() == 0 for fiber in result.fibers)


def test_automatic_whole_image_with_precomputed_segmentation(tmp_path: Path) -> None:
    shape = (180, 240)
    axon_a = _circle(shape, (80, 65), 11)
    axon_b = _circle(shape, (80, 160), 12)
    outer_a = _circle(shape, (80, 65), 27)
    outer_b = _circle(shape, (80, 160), 29)
    axon = axon_a | axon_b
    myelin = (outer_a | outer_b) & ~axon

    yy, xx = np.ogrid[: shape[0], : shape[1]]
    vacuole = ((xx - 84) / 7) ** 2 + ((yy - 80) / 5) ** 2 <= 1
    vacuole &= outer_a & ~axon_a
    image = np.full(shape, 105, dtype=np.uint8)
    image[myelin] = 25
    image[axon] = 145
    image[vacuole] = 240

    image_path = tmp_path / "whole.png"
    axon_path = tmp_path / "axon.png"
    myelin_path = tmp_path / "myelin.png"
    Image.fromarray(image).save(image_path)
    _write_mask(axon_path, axon)
    _write_mask(myelin_path, myelin)

    config = DetectorConfig(
        detector="intensity",
        min_area_um2=0.001,
        clahe_clip_limit=0.01,
        intensity_threshold_offset=0.20,
        intensity_low_threshold_offset=0.075,
        gaussian_sigma_um=0.02,
        morphology_radius_um=0.01,
    )
    output = tmp_path / "automatic"
    summary = run_automatic_whole_image(
        image_path,
        5.0,
        output,
        config,
        axon_mask_path=axon_path,
        myelin_mask_path=myelin_path,
        exclude_scale_bar=False,
        min_axon_area_um2=0.001,
        crop_margin_um=0.10,
    )

    automatic = summary["automatic_whole_image"]
    assert automatic["axon_components_found"] == 2
    assert automatic["fibers_processed"] == 2
    assert (output / "automatic_manifest.csv").exists()
    assert (output / "metrics.csv").exists()
    assert (output / "whole_image_overlay.png").exists()
    assert len(list((output / "overlays").glob("*_overlay.png"))) == 2
    assert len(list((output / "masks").glob("*_vacuole.png"))) == 2

    manifest = pd.read_csv(output / "automatic_manifest.csv")
    metrics = pd.read_csv(output / "metrics.csv")
    assert len(manifest) == len(metrics) == 2
    assert set(metrics["automatic_extraction_flags"]) == {"pass"}
    assert metrics["vacuole_area_um2"].max() > 0
