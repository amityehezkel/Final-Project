from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .io import load_manifest, read_binary_mask, read_grayscale
from .overlay import create_overlay, write_overlay


MASK_NAMES = ("axon", "outer_fiber", "vacuole")


def visualize_annotations(
    manifest_path: str | Path,
    annotation_dir: str | Path | None,
    output_dir: str | Path,
) -> Path:
    """Render manual annotation masks over every benchmark crop."""

    rows = load_manifest(manifest_path)
    annotations = Path(annotation_dir).resolve() if annotation_dir else None
    output = Path(output_dir).resolve()
    overlay_root = output / "overlays"
    index_rows: list[dict[str, object]] = []

    for row in rows:
        image = read_grayscale(row.image_path)
        shape = image.shape
        if annotations is None:
            if row.consensus_vacuole_mask_path is None:
                raise ValueError(f"Benchmark row {row.id!r} has no vacuole mask")
            paths = {
                "axon": row.axon_mask_path,
                "outer_fiber": row.outer_fiber_mask_path,
                "vacuole": row.consensus_vacuole_mask_path,
            }
        else:
            paths = {
                name: annotations / f"{row.id}_{name}.png" for name in MASK_NAMES
            }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing annotation masks for {row.id!r}: {missing}"
            )

        axon = read_binary_mask(paths["axon"], shape)
        outer = read_binary_mask(paths["outer_fiber"], shape)
        vacuole = read_binary_mask(paths["vacuole"], shape)

        # Keep the rendered annotation anatomically consistent even when a few
        # hand-painted boundary pixels overlap.
        outer |= axon
        vacuole &= outer & ~axon

        split_dir = overlay_root / row.split
        overlay_path = split_dir / f"{row.id}_annotation_overlay.png"
        write_overlay(
            overlay_path,
            create_overlay(image, axon, outer, vacuole),
        )
        index_rows.append(
            {
                "id": row.id,
                "source_image_id": row.source_image_id or row.image_path.stem,
                "split": row.split,
                "overlay_path": str(overlay_path),
                "has_annotated_vacuole": bool(vacuole.any()),
                "annotated_vacuole_pixels": int(np.count_nonzero(vacuole)),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "annotation_overlay_index.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render manual axon, outer-fiber, and vacuole annotation overlays"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--annotations",
        help="Optional mask folder; omit to render masks referenced by the manifest",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(visualize_annotations(args.manifest, args.annotations, args.output))


if __name__ == "__main__":
    main()
