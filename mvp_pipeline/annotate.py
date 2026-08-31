from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .io import load_manifest, read_binary_mask, read_grayscale, write_binary_mask


def annotate_row(
    manifest_path: str | Path,
    row_id: str,
    output_dir: str | Path,
    load_consensus: bool = False,
    reference_a: str | Path | None = None,
    reference_b: str | Path | None = None,
) -> None:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("Napari is not installed in this Python environment") from exc

    matches = [row for row in load_manifest(manifest_path) if row.id == row_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest row with id {row_id!r}, found {len(matches)}")
    row = matches[0]
    image = read_grayscale(row.image_path)
    shape = image.shape
    output = Path(output_dir)
    saved_paths = {
        name: output / f"{row.id}_{name}.png"
        for name in ("axon", "outer_fiber", "vacuole")
    }

    def initial_mask(name: str) -> np.ndarray:
        # Independent annotators can safely close Napari and resume the same
        # fiber later without losing work. Consensus review still explicitly
        # starts from the manifest's consensus masks.
        if load_consensus:
            manifest_path_for_name = {
                "axon": row.axon_mask_path,
                "outer_fiber": row.outer_fiber_mask_path,
                "vacuole": row.consensus_vacuole_mask_path,
            }[name]
            if manifest_path_for_name is not None:
                return read_binary_mask(manifest_path_for_name, shape)
        if saved_paths[name].exists():
            return read_binary_mask(saved_paths[name], shape)
        return np.zeros(shape, bool)

    layers = {
        "axon": initial_mask("axon"),
        "outer_fiber": initial_mask("outer_fiber"),
        "vacuole": initial_mask("vacuole"),
    }
    viewer = napari.Viewer(title=f"Vacuole benchmark annotation: {row.id}")
    viewer.add_image(image, name="image", colormap="gray")
    viewer.add_labels(layers["outer_fiber"].astype(np.uint8), name="outer_fiber", opacity=0.35)
    viewer.add_labels(layers["axon"].astype(np.uint8), name="axon", opacity=0.35)
    viewer.add_labels(layers["vacuole"].astype(np.uint8), name="vacuole", opacity=0.55)
    for label, directory in (("A", reference_a), ("B", reference_b)):
        if directory is None:
            continue
        directory = Path(directory)
        for layer_name in ("outer_fiber", "axon", "vacuole"):
            path = directory / f"{row.id}_{layer_name}.png"
            reference = read_binary_mask(path, shape)
            viewer.add_labels(
                reference.astype(np.uint8),
                name=f"reference_{label}_{layer_name}",
                opacity=0.18,
                visible=layer_name == "vacuole",
            )
    napari.run()

    output.mkdir(parents=True, exist_ok=True)
    for name in ("axon", "outer_fiber", "vacuole"):
        write_binary_mask(output / f"{row.id}_{name}.png", np.asarray(viewer.layers[name].data) > 0)


def annotate_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    row_id: str | None = None,
    split: str | None = None,
    load_consensus: bool = False,
    reference_a: str | Path | None = None,
    reference_b: str | Path | None = None,
) -> None:
    rows = load_manifest(manifest_path)
    output = Path(output_dir)
    selected = [
        row
        for row in rows
        if (row_id is None or row.id == row_id)
        and (split is None or row.split == split.lower())
    ]
    if not selected:
        raise ValueError("No manifest rows match --id/--split")

    # Batch runs are resumable: a fiber is complete once all three masks were
    # saved. An explicit --id always reopens that fiber for review/correction.
    if row_id is None:
        selected = [
            row
            for row in selected
            if not all(
                (output / f"{row.id}_{name}.png").exists()
                for name in ("axon", "outer_fiber", "vacuole")
            )
        ]
        if not selected:
            print("All requested annotations are already complete.")
            return

    for row in selected:
        annotate_row(
            manifest_path,
            row.id,
            output_dir,
            load_consensus,
            reference_a,
            reference_b,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate one benchmark crop in Napari")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--id", dest="row_id", help="One row id; omit to annotate all rows")
    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument("--output", required=True, help="Annotator-specific output directory")
    parser.add_argument("--load-consensus", action="store_true")
    parser.add_argument("--reference-a", help="First annotator directory for consensus review")
    parser.add_argument("--reference-b", help="Second annotator directory for consensus review")
    args = parser.parse_args()
    annotate_manifest(
        args.manifest,
        args.output,
        args.row_id,
        args.split,
        args.load_consensus,
        args.reference_a,
        args.reference_b,
    )


if __name__ == "__main__":
    main()
