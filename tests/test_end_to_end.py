from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import mvp_pipeline.tune as tune_module
from mvp_pipeline.config import DetectorConfig
from mvp_pipeline.run import run_batch
from mvp_pipeline.tune import tune_detector


def write_mask(path: Path, mask: np.ndarray):
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def make_manifest(tmp_path: Path) -> Path:
    yy, xx = np.ogrid[:96, :96]
    rows = []
    for index, split in enumerate(("dev", "dev", "test"), start=1):
        axon = (xx - 48) ** 2 + (yy - 48) ** 2 <= 18**2
        outer = (xx - 48) ** 2 + (yy - 48) ** 2 <= 34**2
        vacuole = ((xx - 69) / 7) ** 2 + ((yy - 48) / 4) ** 2 <= 1
        vacuole &= outer & ~axon
        compact = outer & ~axon & ~vacuole
        image = np.full((96, 96), 110, dtype=np.uint8)
        image[outer & ~axon] = 25
        image[vacuole] = 235
        image[axon] = 150
        stem = f"fiber_{index}"
        image_path = tmp_path / f"{stem}.png"
        axon_path = tmp_path / f"{stem}_axon.png"
        outer_path = tmp_path / f"{stem}_outer.png"
        compact_path = tmp_path / f"{stem}_compact.png"
        truth_path = tmp_path / f"{stem}_truth.png"
        Image.fromarray(image).save(image_path)
        for path, mask in (
            (axon_path, axon),
            (outer_path, outer),
            (compact_path, compact),
            (truth_path, vacuole),
        ):
            write_mask(path, mask)
        rows.append(
            {
                "id": stem,
                "source_image_id": f"source_{index}",
                "image_path": image_path,
                "scale_nm_per_px": 5.0,
                "axon_mask_path": axon_path,
                "outer_fiber_mask_path": outer_path,
                "compact_myelin_mask_path": compact_path,
                "consensus_vacuole_mask_path": truth_path,
                "split": split,
                "mask_source": "manual",
                "correction_minutes": 0,
            }
        )
    manifest = tmp_path / "benchmark.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def test_tuning_and_batch_outputs(tmp_path, monkeypatch: pytest.MonkeyPatch):
    # Exercise both detector families without running the full production tuning
    # grid. The complete search is validated by the frozen benchmark artifacts;
    # unit tests should remain quick enough to run during workspace setup.
    monkeypatch.setattr(tune_module, "MIN_AREA_GRID_UM2", (0.001,))
    monkeypatch.setattr(tune_module, "HIGH_THRESHOLD_OFFSET_GRID", (0.0,))
    monkeypatch.setattr(tune_module, "LOW_THRESHOLD_OFFSET_GRID", (0.0,))
    monkeypatch.setattr(tune_module, "GAUSSIAN_SIGMA_GRID_UM", (0.01,))
    monkeypatch.setattr(tune_module, "MORPHOLOGY_RADIUS_GRID_UM", (0.0,))

    manifest = make_manifest(tmp_path)
    tuning = tmp_path / "tuning"
    config = tune_detector(manifest, tuning)
    assert config.detector == "geometry"
    assert config.development_median_dice == 1.0
    assert (tuning / "detector_config.json").exists()

    output = tmp_path / "outputs"
    metrics, evaluation, summary = run_batch(manifest, output, config)
    assert len(metrics) == 3
    assert len(evaluation) == 3
    assert evaluation["dice"].min() == 1.0
    assert summary["evaluation"]["test"]["median_dice"] == 1.0
    assert (output / "metrics.csv").exists()
    assert (output / "evaluation_overview.png").exists()
    assert len(list((output / "overlays").glob("*_overlay.png"))) == 3
