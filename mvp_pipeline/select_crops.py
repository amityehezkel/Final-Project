from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_grayscale


SOURCE_COLUMNS = {
    "source_image_id",
    "image_path",
    "scale_nm_per_px",
    "split",
    "max_crops",
}
TARGET_SPLIT_COUNTS = {"dev": 8, "test": 16}
TARGET_CLASS_COUNTS = {"vacuolated": 12, "compact": 12}


def _counts(crops: list[dict[str, object]]) -> tuple[dict[str, int], dict[str, int]]:
    split_counts = {name: 0 for name in TARGET_SPLIT_COUNTS}
    class_counts = {name: 0 for name in TARGET_CLASS_COUNTS}
    for crop in crops:
        split_counts[str(crop["split"])] += 1
        class_counts[str(crop["apparent_class"])] += 1
    return split_counts, class_counts


def _save_progress(
    state_path: Path,
    output_path: Path,
    completed_sources: list[str],
    crops: list[dict[str, object]],
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {"completed_sources": completed_sources, "crops": crops},
            stream,
            indent=2,
            ensure_ascii=False,
        )
    pd.DataFrame(crops).to_csv(output_path, index=False)


def select_crops(source_csv: str | Path, output_csv: str | Path) -> Path:
    try:
        import napari
    except ImportError as exc:
        raise RuntimeError("Napari is required for interactive crop selection") from exc

    source_path = Path(source_csv).resolve()
    sources = pd.read_csv(source_path)
    missing = SOURCE_COLUMNS - set(sources.columns)
    if missing:
        raise ValueError(f"Source CSV is missing columns: {sorted(missing)}")
    sources["split"] = sources["split"].astype(str).str.lower()
    if not set(sources["split"]).issubset(TARGET_SPLIT_COUNTS):
        raise ValueError("Every source split must be 'dev' or 'test'")
    if sources.loc[sources["split"] == "dev", "source_image_id"].nunique() < 2:
        raise ValueError("Provide at least two development candidate images")
    if sources.loc[sources["split"] == "test", "source_image_id"].nunique() < 4:
        raise ValueError("Provide at least four test candidate images")
    for split, target in TARGET_SPLIT_COUNTS.items():
        available = int(sources.loc[sources["split"] == split, "max_crops"].sum())
        if available < target:
            raise ValueError(f"Candidate sources allow only {available} {split} crops; need {target}")

    output_path = Path(output_csv).resolve()
    state_path = output_path.with_suffix(".progress.json")
    completed_sources: list[str] = []
    crop_rows: list[dict[str, object]] = []
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as stream:
            state = json.load(stream)
        completed_sources = [str(value) for value in state.get("completed_sources", [])]
        crop_rows = list(state.get("crops", []))
        print(f"Resuming {len(crop_rows)} saved crops from {state_path}")

    for source in sources.to_dict(orient="records"):
        source_id = str(source["source_image_id"])
        if source_id in completed_sources:
            continue
        split = str(source["split"])
        split_counts, class_counts = _counts(crop_rows)
        split_remaining = TARGET_SPLIT_COUNTS[split] - split_counts[split]
        vac_remaining = TARGET_CLASS_COUNTS["vacuolated"] - class_counts["vacuolated"]
        compact_remaining = TARGET_CLASS_COUNTS["compact"] - class_counts["compact"]
        allowed = max(0, min(int(source["max_crops"]), split_remaining))

        if allowed == 0:
            completed_sources.append(source_id)
            _save_progress(state_path, output_path, completed_sources, crop_rows)
            continue

        image_path = Path(str(source["image_path"]))
        if not image_path.is_absolute():
            image_path = (source_path.parent / image_path).resolve()
        image = read_grayscale(image_path)
        viewer = napari.Viewer(
            title=(
                f"{source_id}: draw 0-{allowed} complete fibers, or close to skip | "
                f"{split} remaining {split_remaining}; vac remaining {vac_remaining}; "
                f"compact remaining {compact_remaining}"
            )
        )
        viewer.add_image(image, name="image", colormap="gray")
        # Do not expose a class whose benchmark quota is already full.  Besides
        # making the next action clearer, this prevents valid work from ending
        # in an avoidable "would exceed the 12/12 targets" error.
        if vac_remaining > 0:
            viewer.add_shapes(
                name="vacuolated_crops",
                shape_type="rectangle",
                edge_color="yellow",
            )
        if compact_remaining > 0:
            viewer.add_shapes(
                name="compact_crops",
                shape_type="rectangle",
                edge_color="cyan",
            )
        napari.run()

        selections: list[tuple[str, np.ndarray]] = []
        for apparent_class in ("vacuolated", "compact"):
            layer_name = f"{apparent_class}_crops"
            if layer_name not in viewer.layers:
                continue
            layer = viewer.layers[layer_name]
            selections.extend((apparent_class, np.asarray(vertices)) for vertices in layer.data)
        if len(selections) > allowed:
            raise ValueError(
                f"{source_id} allows at most {allowed} crops at this point, got {len(selections)}. "
                "Earlier images are saved; rerun the command and this image will reopen."
            )
        new_class_counts = dict(class_counts)
        for apparent_class, _ in selections:
            new_class_counts[apparent_class] += 1
        if any(new_class_counts[name] > target for name, target in TARGET_CLASS_COUNTS.items()):
            raise ValueError(
                f"Selections would exceed the 12/12 class targets: {new_class_counts}. "
                "Earlier images are saved; rerun and select fewer from this image."
            )

        new_rows: list[dict[str, object]] = []
        for index, (apparent_class, vertices) in enumerate(selections, start=1):
            y0, x0 = np.floor(vertices.min(axis=0)).astype(int)
            y1, x1 = np.ceil(vertices.max(axis=0)).astype(int)
            y0, x0 = max(0, y0), max(0, x0)
            y1, x1 = min(image.shape[0], y1), min(image.shape[1], x1)
            new_rows.append(
                {
                    "id": f"{source_id}_axon-{index:02d}",
                    "source_image_id": source_id,
                    "image_path": os.path.relpath(image_path, output_path.parent),
                    "scale_nm_per_px": float(source["scale_nm_per_px"]),
                    "split": split,
                    "apparent_class": apparent_class,
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(x1),
                    "y1": int(y1),
                }
            )
        crop_rows.extend(new_rows)
        completed_sources.append(source_id)
        _save_progress(state_path, output_path, completed_sources, crop_rows)

    split_counts, class_counts = _counts(crop_rows)
    if split_counts != TARGET_SPLIT_COUNTS or class_counts != TARGET_CLASS_COUNTS:
        raise ValueError(
            "Candidate pool finished before the benchmark targets were met. "
            f"Current split counts: {split_counts}; class counts: {class_counts}. "
            f"Progress is saved in {state_path}. Add more source rows and rerun."
        )
    print(f"Completed benchmark selection. Progress state: {state_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a resumable 24-axon benchmark in Napari"
    )
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(select_crops(args.sources, args.output))


if __name__ == "__main__":
    main()
