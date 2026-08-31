from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_binary_mask, read_grayscale, write_binary_mask


PATH_COLUMNS = (
    "image_path",
    "axon_mask_path",
    "outer_fiber_mask_path",
    "compact_myelin_mask_path",
    "consensus_vacuole_mask_path",
)


def _relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path.resolve(), parent.resolve())


def build_expanded_benchmark(
    base_manifest: str | Path,
    extra_sources: str | Path,
    output_manifest: str | Path,
) -> Path:
    base_path = Path(base_manifest).resolve()
    sources_path = Path(extra_sources).resolve()
    output_path = Path(output_manifest).resolve()
    project_root = sources_path.parent.parent
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(base_path, dtype=str).fillna("")
    if output_path.parent != base_path.parent:
        for column in PATH_COLUMNS:
            if column not in base.columns:
                continue
            base[column] = [
                _relative(base_path.parent / value, output_path.parent)
                if value
                else ""
                for value in base[column]
            ]

    sources = pd.read_csv(sources_path, dtype=str).fillna("")
    required = {
        "id",
        "source_image_id",
        "image_path",
        "scale_nm_per_px",
        "axon_mask_path",
        "outer_fiber_mask_path",
        "vacuole_mask_path",
        "split",
    }
    missing = required.difference(sources.columns)
    if missing:
        raise ValueError(f"Extra-source table is missing columns: {sorted(missing)}")
    if not set(sources["split"]).issubset({"dev", "test"}):
        raise ValueError("Every extra crop split must be 'dev' or 'test'")

    extra_truth_dir = output_path.parent / "masks" / "extra_student"
    extra_truth_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for record in sources.to_dict(orient="records"):
        image_path = project_root / record["image_path"]
        axon_path = project_root / record["axon_mask_path"]
        outer_path = project_root / record["outer_fiber_mask_path"]
        image = read_grayscale(image_path)
        axon = read_binary_mask(axon_path, image.shape)
        outer = read_binary_mask(outer_path, image.shape) | axon

        source_truth = record["vacuole_mask_path"]
        if source_truth == "EMPTY":
            truth = np.zeros(image.shape, dtype=bool)
        else:
            truth = read_binary_mask(project_root / source_truth, image.shape)
        truth = truth & outer & ~axon
        truth_path = extra_truth_dir / f"{record['id']}_vacuole.png"
        write_binary_mask(truth_path, truth)

        rows.append(
            {
                "id": record["id"],
                "source_image_id": record["source_image_id"],
                "image_path": _relative(image_path, output_path.parent),
                "scale_nm_per_px": record["scale_nm_per_px"],
                "axon_mask_path": _relative(axon_path, output_path.parent),
                "outer_fiber_mask_path": _relative(outer_path, output_path.parent),
                "compact_myelin_mask_path": "",
                "consensus_vacuole_mask_path": _relative(
                    truth_path, output_path.parent
                ),
                "split": record["split"],
                "mask_source": "single_student_extra",
                "correction_minutes": "",
                "label_note": record.get("label_note", ""),
            }
        )

    expanded = pd.concat([base, pd.DataFrame(rows)], ignore_index=True).fillna("")
    if expanded["id"].duplicated().any():
        duplicates = expanded.loc[expanded["id"].duplicated(), "id"].tolist()
        raise ValueError(f"Duplicate benchmark ids: {duplicates}")
    source_splits = expanded.groupby("source_image_id")["split"].nunique()
    leaked = source_splits[source_splits > 1].index.tolist()
    if leaked:
        raise ValueError(f"Source images occur in both splits: {leaked}")
    expanded.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append curated extra fiber annotations to a benchmark"
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--extra-sources", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build_expanded_benchmark(args.base, args.extra_sources, args.output))


if __name__ == "__main__":
    main()
