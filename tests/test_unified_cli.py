from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mvp_pipeline import cli
from mvp_pipeline.cli import run_fiber_crop, run_fiber_folder, run_whole_image
from mvp_pipeline.config import DetectorConfig
from mvp_pipeline.io import (
    read_binary_mask,
    read_grayscale,
    write_binary_mask,
    write_grayscale,
)


def _write_fiber_inputs(directory: Path, mismatched_outer: bool = False):
    yy, xx = np.ogrid[:128, :128]
    axon = (xx - 64) ** 2 + (yy - 64) ** 2 <= 25**2
    outer = (xx - 64) ** 2 + (yy - 64) ** 2 <= 42**2
    vacuole = ((xx - 91) / 9) ** 2 + ((yy - 64) / 6) ** 2 <= 1
    vacuole &= outer & ~axon
    image = np.full((128, 128), 90, dtype=np.uint8)
    image[outer & ~axon] = 35
    image[axon] = 125
    image[vacuole] = 240

    image_path = directory / "fiber.tif"
    axon_path = directory / "axon.png"
    outer_path = directory / "outer.png"
    write_grayscale(image_path, image)
    write_binary_mask(axon_path, axon)
    write_binary_mask(
        outer_path,
        outer[:-1] if mismatched_outer else outer,
    )
    return image_path, axon_path, outer_path


def test_direct_fiber_crop_builds_manifest_and_runs_shared_detector(
    tmp_path: Path,
) -> None:
    image, axon, outer = _write_fiber_inputs(tmp_path)
    output = tmp_path / "output"
    config = DetectorConfig(
        detector="intensity",
        min_area_um2=0.00001,
        intensity_threshold_offset=0.05,
        gaussian_sigma_um=0.0,
        morphology_radius_um=0.0,
    )

    summary = run_fiber_crop(
        image,
        5.0,
        axon,
        outer,
        output,
        config,
        fiber_id="fiber 01",
    )

    manifest = pd.read_csv(output / "input_manifest.csv")
    metrics = pd.read_csv(output / "metrics.csv")
    assert summary["input_mode"] == "fiber_crop"
    assert manifest.loc[0, "id"] == "fiber_01"
    assert manifest.loc[0, "mask_source"] == "user_provided"
    assert metrics.loc[0, "vacuole_area_um2"] > 0
    assert (output / "masks" / "fiber_01_vacuole.png").exists()
    assert (output / "overlays" / "fiber_01_overlay.png").exists()


def test_direct_fiber_crop_rejects_mask_dimension_mismatch(tmp_path: Path) -> None:
    image, axon, outer = _write_fiber_inputs(tmp_path, mismatched_outer=True)
    with pytest.raises(ValueError, match="expected"):
        run_fiber_crop(
            image,
            5.0,
            axon,
            outer,
            tmp_path / "output",
            DetectorConfig(),
        )


def test_fiber_folder_matches_common_mask_names_and_runs_once(tmp_path: Path) -> None:
    images = tmp_path / "images"
    axons = tmp_path / "axon_masks"
    outers = tmp_path / "outer_masks"
    for directory in (images, axons, outers):
        directory.mkdir()

    yy, xx = np.ogrid[:96, :96]
    axon = (xx - 48) ** 2 + (yy - 48) ** 2 <= 18**2
    outer = (xx - 48) ** 2 + (yy - 48) ** 2 <= 31**2
    image = np.full((96, 96), 100, dtype=np.uint8)
    image[outer & ~axon] = 35
    image[axon] = 125
    image[45:51, 72:78] = 240

    write_grayscale(images / "fiber_a.tif", image)
    write_binary_mask(axons / "fiber_a.png", axon)
    write_binary_mask(outers / "fiber_a.png", outer)
    write_grayscale(images / "fiber_b.png", image)
    write_binary_mask(axons / "fiber_b_axon_mask.tif", axon)
    write_binary_mask(outers / "fiber_b_outer_fiber.png", outer)

    output = tmp_path / "folder_output"
    summary = run_fiber_folder(
        images,
        5.0,
        axons,
        outers,
        output,
        DetectorConfig(min_area_um2=0.00001),
    )

    manifest = pd.read_csv(output / "input_manifest.csv")
    metrics = pd.read_csv(output / "metrics.csv")
    assert summary["input_mode"] == "fiber_folder"
    assert summary["inputs"]["matched_fibers"] == 2
    assert set(manifest["id"]) == {"fiber_a", "fiber_b"}
    assert len(metrics) == 2
    assert len(list((output / "overlays").glob("*_overlay.png"))) == 2


def test_fiber_folder_reports_missing_matching_mask(tmp_path: Path) -> None:
    images = tmp_path / "images"
    axons = tmp_path / "axon_masks"
    outers = tmp_path / "outer_masks"
    for directory in (images, axons, outers):
        directory.mkdir()
    image, axon, _ = _write_fiber_inputs(tmp_path)
    write_grayscale(images / "wanted.tif", read_grayscale(image))
    write_binary_mask(axons / "wanted_axon.png", read_binary_mask(axon))
    write_binary_mask(outers / "different_outer.png", read_binary_mask(axon))

    with pytest.raises(FileNotFoundError, match="No outer-fiber mask matches"):
        run_fiber_folder(
            images,
            5.0,
            axons,
            outers,
            tmp_path / "output",
            DetectorConfig(),
        )


def test_whole_image_wrapper_chooses_scale_and_marks_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: dict[str, object] = {}

    def fake_automatic(*args, **kwargs):
        received.update(kwargs)
        output = Path(args[2])
        output.mkdir(parents=True)
        return {"automatic_whole_image": {"fibers_processed": 1}}

    monkeypatch.setattr(cli, "run_automatic_whole_image", fake_automatic)
    summary = run_whole_image(
        "image.tif",
        1.09,
        tmp_path / "whole",
        DetectorConfig(),
        model_path="model",
    )

    assert received["target_scale_nm_per_px"] == 2.36
    assert summary["input_mode"] == "whole_image"
    assert (tmp_path / "whole" / "summary.json").exists()
