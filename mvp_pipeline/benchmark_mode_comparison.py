from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .config import DetectorConfig
from .io import load_manifest
from .run import run_batch


VIEW_NAMES = (
    "01_annotation_overlay.png",
    "02_cropped_mode_overlay.png",
    "03_whole_image_mode_overlay.png",
)


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def _write_automatic_manifest(
    benchmark_path: Path,
    automatic_mask_dir: Path,
    destination: Path,
) -> Path:
    rows = load_manifest(benchmark_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for row in rows:
        records.append(
            {
                "id": row.id,
                "source_image_id": row.source_image_id or row.image_path.stem,
                "image_path": _relative(row.image_path, destination.parent),
                "scale_nm_per_px": row.scale_nm_per_px,
                "axon_mask_path": _relative(
                    automatic_mask_dir / f"{row.id}_axon.png",
                    destination.parent,
                ),
                "outer_fiber_mask_path": _relative(
                    automatic_mask_dir / f"{row.id}_outer_fiber.png",
                    destination.parent,
                ),
                "compact_myelin_mask_path": "",
                "consensus_vacuole_mask_path": "",
                "split": row.split,
                "mask_source": "axondeepseg_benchmark_field",
                "correction_minutes": "",
            }
        )
    pd.DataFrame(records).to_csv(destination, index=False)
    return destination


def _labelled_panel(path: Path, label: str, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size[0], size[1] + 30), "white")
    x = (size[0] - image.width) // 2
    y = 30 + (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    draw.text(((size[0] - (box[2] - box[0])) // 2, 9), label, fill="black", font=font)
    return canvas


def _write_comparison(paths: list[Path], output_path: Path, title: str) -> None:
    panels = [
        _labelled_panel(paths[0], "Annotation", (320, 300)),
        _labelled_panel(paths[1], "Cropped mode", (320, 300)),
        _labelled_panel(paths[2], "Whole-image automatic-mask route", (320, 300)),
    ]
    title_height = 28
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), panels[0].height + title_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), title, font=font)
    draw.text(
        ((canvas.width - (box[2] - box[0])) // 2, 7),
        title,
        fill="black",
        font=font,
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, title_height))
        x += panel.width
    canvas.save(output_path)


def build_benchmark_mode_comparison(
    benchmark_path: str | Path,
    annotation_root: str | Path,
    cropped_results: str | Path,
    automatic_validation: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    benchmark_path = Path(benchmark_path).resolve()
    annotation_root = Path(annotation_root).resolve()
    cropped_results = Path(cropped_results).resolve()
    automatic_validation = Path(automatic_validation).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    automatic_manifest = _write_automatic_manifest(
        benchmark_path,
        automatic_validation / "predicted_masks",
        output / "whole_mode_input_manifest.csv",
    )
    whole_results = output / "_whole_mode_results"
    run_batch(
        automatic_manifest,
        whole_results,
        DetectorConfig.from_json(config_path),
    )

    validation = pd.read_csv(automatic_validation / "per_crop_metrics.csv")
    validation_by_id = validation.set_index("id")
    records: list[dict[str, object]] = []
    for row in load_manifest(benchmark_path):
        fiber_dir = output / row.id
        fiber_dir.mkdir(parents=True, exist_ok=True)
        sources = [
            annotation_root / "overlays" / row.split / f"{row.id}_annotation_overlay.png",
            cropped_results / "overlays" / f"{row.id}_overlay.png",
            whole_results / "overlays" / f"{row.id}_overlay.png",
        ]
        destinations = [fiber_dir / name for name in VIEW_NAMES]
        for source, destination in zip(sources, destinations, strict=True):
            if not source.exists():
                raise FileNotFoundError(f"Missing overlay for {row.id}: {source}")
            shutil.copy2(source, destination)
        _write_comparison(
            destinations,
            fiber_dir / "00_three_way_comparison.png",
            row.id,
        )

        validation_row = validation_by_id.loc[row.id]
        records.append(
            {
                "id": row.id,
                "split": row.split,
                "whole_mode_selection_status": validation_row["selection_status"],
                "whole_mode_candidate_fibers": int(validation_row["candidate_fibers"]),
                "whole_mode_axon_dice": float(validation_row["axon_dice"]),
                "whole_mode_outer_fiber_dice": float(validation_row["outer_fiber_dice"]),
                "whole_mode_both_masks_pass": bool(validation_row["both_masks_pass"]),
                "folder": row.id,
            }
        )

    status = pd.DataFrame(records)
    status.to_csv(output / "comparison_status.csv", index=False)
    summary = {
        "benchmark_fibers": int(len(status)),
        "folders_created": int(len(status)),
        "whole_mode_complete_misses": int(
            (status["whole_mode_selection_status"] == "complete_miss").sum()
        ),
        "whole_mode_fibers_recovered": int(
            (status["whole_mode_selection_status"] != "complete_miss").sum()
        ),
        "whole_mode_both_masks_pass": int(status["whole_mode_both_masks_pass"].sum()),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    (output / "README.md").write_text(
        """# Benchmark overlay comparison

Each fiber folder contains:

- the original benchmark **fiber crop** used to draw the masks and score the detector;
- the corresponding **whole laboratory image with an arrow pointing to that fiber**, added for source traceability and anatomical context;
- `01_annotation_overlay.png`: the manual benchmark reference;
- `02_cropped_mode_overlay.png`: the frozen vacuole detector using the supplied manual axon and outer-fiber masks;
- `03_whole_image_mode_overlay.png`: the same frozen detector using masks generated automatically by the AxonDeepSeg front end;
- `00_three_way_comparison.png`: the three views arranged side by side.

The crop and whole-image files retain their source filenames, so their names are not uniform. The crop is the smaller field centered on one fiber. The contextual whole image is normally 1872 × 1872 pixels and contains the manually added arrow. In `extra_p1202_neoview12_01`, both source files originally had the name `NeoView_12.tif`; the crop is therefore preserved as `fiber_crop.tif` to prevent it from being overwritten by the arrow-marked whole image.

The arrow is for human orientation only. The arrow-marked image was not used for model inference, mask creation, tuning, or scoring, and the arrow must not be interpreted as an automatic detection. It connects the benchmark crop to its location in the source field.

For a controlled one-to-one comparison, the whole-image automatic-mask route was run on each benchmark field of view and the central automatic fiber was selected. This tests the automatic-mask stage used by whole-image mode while keeping the exact image fixed. It is not a rematch of every crop to a fresh run on its original full laboratory TIFF. A plain whole-mode image means AxonDeepSeg did not recover a central fiber; these failures are listed as `complete_miss` in `comparison_status.csv`.

Colors are blue for axon, red for the outer-fiber boundary, and yellow for predicted or annotated vacuole.
""",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-fiber annotation, cropped-mode, and automatic-mask comparisons"
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--cropped-results", required=True)
    parser.add_argument("--automatic-validation", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = build_benchmark_mode_comparison(
        args.benchmark,
        args.annotations,
        args.cropped_results,
        args.automatic_validation,
        args.config,
        args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
