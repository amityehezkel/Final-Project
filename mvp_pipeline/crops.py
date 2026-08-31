from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .io import write_binary_mask, write_grayscale


CROP_PLAN_COLUMNS = {
    "id",
    "source_image_id",
    "image_path",
    "scale_nm_per_px",
    "split",
    "x0",
    "y0",
    "x1",
    "y1",
}


def create_crops(plan_path: str | Path, output_dir: str | Path) -> Path:
    plan_path = Path(plan_path).resolve()
    plan = pd.read_csv(plan_path)
    missing = CROP_PLAN_COLUMNS - set(plan.columns)
    if missing:
        raise ValueError(f"Crop plan is missing columns: {sorted(missing)}")
    if len(plan) != 24:
        raise ValueError(f"The benchmark crop plan must contain exactly 24 rows, got {len(plan)}")
    if plan.loc[plan["split"].astype(str).str.lower() == "dev", "source_image_id"].nunique() < 2:
        raise ValueError("Development axons must come from at least two source images")
    if plan.loc[plan["split"].astype(str).str.lower() == "test", "source_image_id"].nunique() < 4:
        raise ValueError("Test axons must come from at least four source images")
    if plan.groupby("source_image_id")["split"].nunique().max() != 1:
        raise ValueError("A source image cannot appear in both development and test splits")
    if "apparent_class" in plan.columns:
        class_counts = plan["apparent_class"].astype(str).str.lower().value_counts().to_dict()
        if class_counts != {"vacuolated": 12, "compact": 12}:
            raise ValueError(
                f"Expected 12 apparently vacuolated and 12 compact axons, got {class_counts}"
            )
    split_counts = plan["split"].astype(str).str.lower().value_counts().to_dict()
    if split_counts.get("dev", 0) != 8 or split_counts.get("test", 0) != 16:
        raise ValueError(f"Expected 8 dev and 16 test rows, got {split_counts}")
    if plan.groupby("source_image_id").size().max() > 4:
        raise ValueError("No source image may contribute more than four fibers")

    output = Path(output_dir).resolve()
    image_dir = output / "images"
    mask_dir = output / "masks" / "consensus"
    manifest_rows: list[dict[str, object]] = []
    for record in plan.to_dict(orient="records"):
        source = Path(str(record["image_path"]))
        if not source.is_absolute():
            source = (plan_path.parent / source).resolve()
        with Image.open(source) as image:
            grayscale = np.asarray(image.convert("L"))
        x0, y0, x1, y1 = (int(record[name]) for name in ("x0", "y0", "x1", "y1"))
        if not (0 <= x0 < x1 <= grayscale.shape[1] and 0 <= y0 < y1 <= grayscale.shape[0]):
            raise ValueError(
                f"Invalid crop for {record['id']!r}: {(x0, y0, x1, y1)} in {grayscale.shape}"
            )
        crop = grayscale[y0:y1, x0:x1]
        crop_path = image_dir / f"{record['id']}.png"
        write_grayscale(crop_path, crop)
        blank_paths = {
            name: mask_dir / f"{record['id']}_{name}.png"
            for name in ("axon", "outer_fiber", "vacuole")
        }
        for path in blank_paths.values():
            write_binary_mask(path, np.zeros(crop.shape, dtype=bool))
        manifest_rows.append(
            {
                "id": record["id"],
                "source_image_id": record["source_image_id"],
                "image_path": crop_path.relative_to(output),
                "scale_nm_per_px": float(record["scale_nm_per_px"]),
                "axon_mask_path": blank_paths["axon"].relative_to(output),
                "outer_fiber_mask_path": blank_paths["outer_fiber"].relative_to(output),
                "compact_myelin_mask_path": "",
                "consensus_vacuole_mask_path": blank_paths["vacuole"].relative_to(output),
                "split": str(record["split"]).lower(),
                "mask_source": "manual",
                "correction_minutes": "",
            }
        )
    manifest_path = output / "benchmark.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 24 per-axon crops and blank masks")
    parser.add_argument("--plan", required=True, help="Crop-plan CSV")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(create_crops(args.plan, args.output))


if __name__ == "__main__":
    main()
