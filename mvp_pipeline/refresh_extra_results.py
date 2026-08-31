from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .cli import run_fiber_crop
from .config import DetectorConfig


def refresh_extra_results(
    source_table: str | Path,
    config_path: str | Path,
) -> int:
    table_path = Path(source_table).resolve()
    project_root = table_path.parent.parent
    config = DetectorConfig.from_json(config_path)
    rows = pd.read_csv(table_path, dtype=str).fillna("")
    for record in rows.to_dict(orient="records"):
        image_path = project_root / record["image_path"]
        run_fiber_crop(
            image_path=image_path,
            scale_nm_per_px=float(record["scale_nm_per_px"]),
            axon_mask_path=project_root / record["axon_mask_path"],
            outer_fiber_mask_path=project_root / record["outer_fiber_mask_path"],
            output_dir=image_path.parent / "res",
            config=config,
            fiber_id=image_path.stem,
        )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the per-crop example outputs with a frozen detector"
    )
    parser.add_argument("--sources", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    count = refresh_extra_results(args.sources, args.config)
    print(f"Refreshed {count} crop result folders")


if __name__ == "__main__":
    main()
