from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from mvp_pipeline.crops import create_crops
from mvp_pipeline.segmentation_eval import evaluate_variants


def test_crop_creation_enforces_and_builds_benchmark(tmp_path: Path):
    source_image = np.arange(80 * 80, dtype=np.uint8).reshape(80, 80)
    source_path = tmp_path / "source.png"
    Image.fromarray(source_image).save(source_path)
    rows = []
    source_specs = [
        ("dev_a", "dev", 4),
        ("dev_b", "dev", 4),
        ("test_a", "test", 4),
        ("test_b", "test", 3),
        ("test_c", "test", 3),
        ("test_d", "test", 2),
        ("test_e", "test", 2),
        ("test_f", "test", 2),
    ]
    index = 0
    for source_id, split, count in source_specs:
        for _ in range(count):
            index += 1
            rows.append(
                {
                    "id": f"axon_{index:02d}",
                    "source_image_id": source_id,
                    "image_path": source_path,
                    "scale_nm_per_px": 5.0,
                    "split": split,
                    "apparent_class": "vacuolated" if index <= 12 else "compact",
                    "x0": 10,
                    "y0": 10,
                    "x1": 60,
                    "y1": 60,
                }
            )
    plan = tmp_path / "crop_plan.csv"
    pd.DataFrame(rows).to_csv(plan, index=False)
    manifest = create_crops(plan, tmp_path / "benchmark")
    frame = pd.read_csv(manifest)
    assert len(frame) == 24
    assert frame["split"].value_counts().to_dict() == {"test": 16, "dev": 8}
    assert not Path(frame.iloc[0]["image_path"]).is_absolute()
    assert (manifest.parent / frame.iloc[0]["image_path"]).exists()
    assert (manifest.parent / frame.iloc[0]["axon_mask_path"]).exists()


def test_segmentation_variant_stop_rule(tmp_path: Path):
    truth = np.zeros((20, 20), dtype=np.uint8)
    truth[5:15, 5:15] = 255
    truth_path = tmp_path / "truth.png"
    Image.fromarray(truth).save(truth_path)
    table = pd.DataFrame(
        [
            {
                "id": "axon_1",
                "variant": "scale_4.93",
                "axon_prediction_path": truth_path,
                "outer_prediction_path": truth_path,
                "axon_truth_path": truth_path,
                "outer_truth_path": truth_path,
                "correction_minutes": 2.0,
            },
            {
                "id": "axon_2",
                "variant": "scale_4.93",
                "axon_prediction_path": truth_path,
                "outer_prediction_path": truth_path,
                "axon_truth_path": truth_path,
                "outer_truth_path": truth_path,
                "correction_minutes": 3.0,
            },
        ]
    )
    table_path = tmp_path / "comparison.csv"
    table.to_csv(table_path, index=False)
    summary = evaluate_variants(table_path, tmp_path / "comparison_output")
    assert bool(summary.iloc[0]["automatic_masks_acceptable"])
