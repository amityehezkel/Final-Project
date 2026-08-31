from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DetectorConfig
from .instances import ExtractionResult, extract_fiber_instances
from .io import (
    read_binary_mask,
    read_grayscale,
    write_binary_mask,
    write_grayscale,
)
from .masks import scale_bar_region
from .overlay import create_overlay, write_overlay
from .run import run_batch
from .segment_scale import segment_at_scale


REVIEW_FLAGS = {
    "low_myelin_coverage",
    "irregular_axon_shape",
    "large_axon_area_outlier",
}


def _prepare_external_segmentation(
    image_path: Path,
    source_scale_nm_per_px: float,
    output_dir: Path,
    *,
    model_path: str | Path | None,
    target_scale_nm_per_px: float,
    gpu_id: int,
    axon_mask_path: str | Path | None,
    myelin_mask_path: str | Path | None,
) -> tuple[Path, Path, str]:
    if model_path is not None:
        if axon_mask_path is not None or myelin_mask_path is not None:
            raise ValueError(
                "Use either --model-path or precomputed --axon-mask/--myelin-mask, not both"
            )
        outputs = segment_at_scale(
            image_path,
            source_scale_nm_per_px,
            target_scale_nm_per_px,
            model_path,
            output_dir / "segmentation",
            gpu_id,
        )
        if "axon" not in outputs:
            raise RuntimeError("AxonDeepSeg did not produce an axon mask")
        if "myelin" in outputs:
            myelin = outputs["myelin"]
        elif "axonmyelin" in outputs:
            combined = read_binary_mask(outputs["axonmyelin"])
            axon = read_binary_mask(outputs["axon"], combined.shape)
            myelin = output_dir / "segmentation" / "derived_seg-myelin.png"
            write_binary_mask(myelin, combined & ~axon)
        else:
            raise RuntimeError("AxonDeepSeg did not produce a myelin or combined mask")
        return outputs["axon"], myelin, "axondeepseg_scale_normalized"

    if axon_mask_path is None or myelin_mask_path is None:
        raise ValueError(
            "Provide --model-path, or provide both --axon-mask and --myelin-mask"
        )
    return Path(axon_mask_path), Path(myelin_mask_path), "precomputed_automatic"


def _write_generated_benchmark(
    image: np.ndarray,
    image_path: Path,
    scale_nm_per_px: float,
    extraction: ExtractionResult,
    output_dir: Path,
    mask_source: str,
) -> tuple[Path, pd.DataFrame]:
    image_dir = output_dir / "crops" / "images"
    axon_dir = output_dir / "crops" / "axon_masks"
    outer_dir = output_dir / "crops" / "outer_fiber_masks"
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    source_id = image_path.stem

    for fiber in extraction.fibers:
        fiber_id = f"{source_id}_fiber-{fiber.number:04d}"
        x0, y0, x1, y1 = fiber.bbox
        crop_path = image_dir / f"{fiber_id}.png"
        axon_path = axon_dir / f"{fiber_id}_axon.png"
        outer_path = outer_dir / f"{fiber_id}_outer_fiber.png"
        write_grayscale(crop_path, image[y0:y1, x0:x1])
        write_binary_mask(axon_path, fiber.axon)
        write_binary_mask(outer_path, fiber.outer_fiber)
        rows.append(
            {
                "id": fiber_id,
                "source_image_id": source_id,
                "image_path": crop_path.relative_to(output_dir),
                "scale_nm_per_px": scale_nm_per_px,
                "axon_mask_path": axon_path.relative_to(output_dir),
                "outer_fiber_mask_path": outer_path.relative_to(output_dir),
                "compact_myelin_mask_path": "",
                "consensus_vacuole_mask_path": "",
                "split": "inference",
                "mask_source": mask_source,
                "correction_minutes": "",
            }
        )
        flags = ";".join(fiber.extraction_flags) if fiber.extraction_flags else "pass"
        review_recommended = bool(REVIEW_FLAGS.intersection(fiber.extraction_flags))
        metadata.append(
            {
                "id": fiber_id,
                "bbox_x0": x0,
                "bbox_y0": y0,
                "bbox_x1": x1,
                "bbox_y1": y1,
                "automatic_extraction_flags": flags,
                "manual_review_recommended": review_recommended,
                "source_cluster_axon_count": fiber.source_cluster_axon_count,
                "myelin_coverage": fiber.myelin_coverage,
                "automatic_axon_area_um2": fiber.axon_area_um2,
                "automatic_axon_solidity": fiber.axon_solidity,
            }
        )

    manifest = output_dir / "automatic_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest, pd.DataFrame(metadata)


def _write_whole_image_overlay(
    image: np.ndarray,
    extraction: ExtractionResult,
    metadata: pd.DataFrame,
    vacuole_dir: Path,
    output_path: Path,
) -> None:
    all_axons = np.zeros(image.shape, dtype=bool)
    all_outer = np.zeros(image.shape, dtype=bool)
    all_vacuoles = np.zeros(image.shape, dtype=bool)
    by_number = {fiber.number: fiber for fiber in extraction.fibers}
    for record in metadata.to_dict(orient="records"):
        fiber_id = str(record["id"])
        number = int(fiber_id.rsplit("-", 1)[1])
        fiber = by_number[number]
        x0, y0, x1, y1 = fiber.bbox
        all_axons[y0:y1, x0:x1] |= fiber.axon
        all_outer[y0:y1, x0:x1] |= fiber.outer_fiber
        vacuole_path = vacuole_dir / f"{fiber_id}_vacuole.png"
        vacuole = read_binary_mask(vacuole_path, fiber.axon.shape)
        all_vacuoles[y0:y1, x0:x1] |= vacuole
    write_overlay(
        output_path,
        create_overlay(image, all_axons, all_outer, all_vacuoles),
    )


def run_automatic_whole_image(
    image_path: str | Path,
    source_scale_nm_per_px: float,
    output_dir: str | Path,
    config: DetectorConfig,
    *,
    model_path: str | Path | None = None,
    target_scale_nm_per_px: float = 4.93,
    gpu_id: int = -1,
    axon_mask_path: str | Path | None = None,
    myelin_mask_path: str | Path | None = None,
    min_axon_area_um2: float = 0.01,
    crop_margin_um: float = 0.25,
    exclude_scale_bar: bool = True,
) -> dict[str, Any]:
    """Run automatic whole-image fiber extraction and vacuole measurement."""

    image_path = Path(image_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    image = read_grayscale(image_path)
    axon_path, myelin_path, mask_source = _prepare_external_segmentation(
        image_path,
        source_scale_nm_per_px,
        output,
        model_path=model_path,
        target_scale_nm_per_px=target_scale_nm_per_px,
        gpu_id=gpu_id,
        axon_mask_path=axon_mask_path,
        myelin_mask_path=myelin_mask_path,
    )
    axon = read_binary_mask(axon_path, image.shape)
    myelin = read_binary_mask(myelin_path, image.shape)
    excluded = scale_bar_region(image.shape) if exclude_scale_bar else None
    extraction = extract_fiber_instances(
        axon,
        myelin,
        source_scale_nm_per_px,
        excluded_region=excluded,
        min_axon_area_um2=min_axon_area_um2,
        crop_margin_um=crop_margin_um,
    )
    if not extraction.fibers:
        raise RuntimeError(
            "No complete fibers survived automatic extraction. Inspect the segmentation "
            "masks or reduce --min-axon-area."
        )

    manifest, metadata = _write_generated_benchmark(
        image,
        image_path,
        source_scale_nm_per_px,
        extraction,
        output,
        mask_source,
    )
    metrics, _, summary = run_batch(manifest, output, config)
    metrics = metrics.merge(metadata, on="id", how="left", validate="one_to_one")
    metrics.to_csv(output / "metrics.csv", index=False)
    _write_whole_image_overlay(
        image,
        extraction,
        metadata,
        output / "masks",
        output / "whole_image_overlay.png",
    )

    summary["automatic_whole_image"] = {
        "source_image": str(image_path),
        "source_scale_nm_per_px": float(source_scale_nm_per_px),
        "mask_source": mask_source,
        "axon_mask": str(Path(axon_path).resolve()),
        "myelin_mask": str(Path(myelin_path).resolve()),
        "axon_components_found": extraction.axon_components_found,
        "fibers_processed": len(extraction.fibers),
        "rejection_counts": extraction.rejection_counts,
        "fibers_requiring_review": int(metadata["manual_review_recommended"].sum()),
        "fibers_with_informational_flags": int(
            (metadata["automatic_extraction_flags"] != "pass").sum()
        ),
        "scale_bar_excluded": bool(exclude_scale_bar),
        "warning": (
            "Automatic measurements are only as reliable as the AxonDeepSeg masks; "
            "validate axon and outer-fiber segmentation before biological use."
        ),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run scale-normalized AxonDeepSeg, automatically extract fibers, "
            "detect vacuoles, and export measurements"
        )
    )
    parser.add_argument("--image", required=True, help="Whole laboratory TIFF or PNG")
    parser.add_argument("--source-scale", required=True, type=float, dest="source_scale")
    parser.add_argument("--config", required=True, help="Frozen vacuole-detector JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", help="AxonDeepSeg model folder")
    parser.add_argument(
        "--target-scale", type=float, default=4.93, choices=(2.36, 4.93)
    )
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--axon-mask", help="Use a precomputed full-image axon mask")
    parser.add_argument("--myelin-mask", help="Use a precomputed full-image myelin mask")
    parser.add_argument("--min-axon-area", type=float, default=0.01)
    parser.add_argument("--crop-margin", type=float, default=0.25)
    parser.add_argument(
        "--no-scale-bar-exclusion",
        action="store_true",
        help="Do not exclude the bottom-right scale-bar region",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = DetectorConfig.from_json(args.config)
    summary = run_automatic_whole_image(
        args.image,
        args.source_scale,
        args.output,
        config,
        model_path=args.model_path,
        target_scale_nm_per_px=args.target_scale,
        gpu_id=args.gpu_id,
        axon_mask_path=args.axon_mask,
        myelin_mask_path=args.myelin_mask,
        min_axon_area_um2=args.min_axon_area,
        crop_margin_um=args.crop_margin,
        exclude_scale_bar=not args.no_scale_bar_exclusion,
    )
    print(json.dumps(summary["automatic_whole_image"], indent=2))


if __name__ == "__main__":
    main()
