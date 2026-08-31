from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .instances import ExtractionResult, FiberInstance, extract_fiber_instances
from .io import (
    load_manifest,
    read_binary_mask,
    read_grayscale,
    write_binary_mask,
    write_grayscale,
)
from .metrics import segmentation_metrics
from .overlay import create_overlay, write_overlay
from .scale import resample_image, restore_mask


def target_scale_for_crop(source_scale_nm_per_px: float) -> float:
    """Select the prescribed AxonDeepSeg scale for a crop."""

    return 2.36 if source_scale_nm_per_px < 2.0 else 4.93


def select_central_fiber(
    extraction: ExtractionResult, image_shape: tuple[int, int]
) -> FiberInstance | None:
    """Select the extracted axon whose centroid is nearest the crop center."""

    if not extraction.fibers:
        return None
    center_y = (image_shape[0] - 1) / 2.0
    center_x = (image_shape[1] - 1) / 2.0

    def distance(fiber: FiberInstance) -> float:
        yy, xx = np.nonzero(fiber.axon)
        x0, y0, _, _ = fiber.bbox
        centroid_y = y0 + float(yy.mean())
        centroid_x = x0 + float(xx.mean())
        return (centroid_y - center_y) ** 2 + (centroid_x - center_x) ** 2

    return min(extraction.fibers, key=distance)


def _restore_selected_masks(
    selected: FiberInstance | None, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    axon = np.zeros(shape, dtype=bool)
    outer = np.zeros(shape, dtype=bool)
    if selected is None:
        return axon, outer
    x0, y0, x1, y1 = selected.bbox
    axon[y0:y1, x0:x1] = selected.axon
    outer[y0:y1, x0:x1] = selected.outer_fiber
    return axon, outer


def _write_overview(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(frame["axon_dice"], frame["outer_fiber_dice"], alpha=0.8)
    axes[0].axvline(0.75, color="tab:red", linestyle="--", linewidth=1)
    axes[0].axhline(0.75, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="Axon Dice", ylabel="Outer-fiber Dice")
    axes[0].set_title("Per-crop mask agreement")
    axes[0].grid(alpha=0.2)

    ordered = frame.sort_values(["both_masks_pass", "axon_dice", "outer_fiber_dice"])
    positions = np.arange(len(ordered))
    axes[1].plot(positions, ordered["axon_dice"], "o", label="axon", markersize=4)
    axes[1].plot(
        positions,
        ordered["outer_fiber_dice"],
        "o",
        label="outer fiber",
        markersize=4,
    )
    axes[1].axhline(0.75, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Crops sorted by performance", ylabel="Dice", ylim=(0, 1.02))
    axes[1].set_title("Individual results")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def validate_cropped_images(
    manifest_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    *,
    gpu_id: int = -1,
    minimum_axon_area_um2: float = 0.001,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run AxonDeepSeg on every crop and compare with supplied manual masks."""

    try:
        from AxonDeepSeg.apply_model import axon_segmentation
        from AxonDeepSeg.segment import get_model_input_format, get_model_type
    except ImportError as exc:
        raise RuntimeError(
            "AxonDeepSeg is not importable. Run with the astih conda Python."
        ) from exc

    rows = load_manifest(manifest_path)
    if not rows:
        raise ValueError("The crop manifest is empty")
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"AxonDeepSeg model folder does not exist: {model_path}")
    file_format, channels = get_model_input_format(model_path)
    if channels != 1 or file_format.lower() != ".png":
        raise ValueError("Crop validation requires a one-channel PNG AxonDeepSeg model")

    output = Path(output_dir).resolve()
    prediction_dir = output / "predicted_masks"
    overlay_dir = output / "overlays"
    records: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="ads-crop-validation-") as temporary:
        temporary_path = Path(temporary)
        inputs: list[Path] = []
        originals: dict[str, np.ndarray] = {}
        for row in rows:
            image = read_grayscale(row.image_path)
            originals[row.id] = image
            target_scale = target_scale_for_crop(row.scale_nm_per_px)
            prepared = resample_image(image, row.scale_nm_per_px, target_scale)
            input_path = temporary_path / f"{row.id}.png"
            write_grayscale(input_path, prepared)
            inputs.append(input_path)

        # Load the model once and let nnU-Net pad small inputs internally.
        axon_segmentation(
            path_inputs=inputs,
            path_model=model_path,
            model_type=get_model_type(model_path),
            gpu_id=gpu_id,
            verbosity_level=0,
        )

        for row in rows:
            image = originals[row.id]
            shape = image.shape
            predicted_axon_path = temporary_path / f"{row.id}_seg-axon.png"
            predicted_myelin_path = temporary_path / f"{row.id}_seg-myelin.png"
            if not predicted_axon_path.exists() or not predicted_myelin_path.exists():
                raise RuntimeError(f"AxonDeepSeg did not produce both masks for {row.id}")
            axon_all = restore_mask(read_binary_mask(predicted_axon_path), shape)
            myelin_all = restore_mask(read_binary_mask(predicted_myelin_path), shape)
            extraction = extract_fiber_instances(
                axon_all,
                myelin_all,
                row.scale_nm_per_px,
                min_axon_area_um2=minimum_axon_area_um2,
                crop_margin_um=0.0,
            )
            selected = select_central_fiber(extraction, shape)
            predicted_axon, predicted_outer = _restore_selected_masks(selected, shape)
            truth_axon = read_binary_mask(row.axon_mask_path, shape)
            truth_outer = read_binary_mask(row.outer_fiber_mask_path, shape)
            axon_scores = segmentation_metrics(predicted_axon, truth_axon)
            outer_scores = segmentation_metrics(predicted_outer, truth_outer)
            status = "complete_miss" if selected is None else "pass"
            flags = ""
            if selected is not None and selected.extraction_flags:
                flags = ";".join(selected.extraction_flags)
            if selected is not None and len(extraction.fibers) > 1:
                status = "multiple_candidates_central_selected"

            write_binary_mask(
                prediction_dir / f"{row.id}_axon.png", predicted_axon
            )
            write_binary_mask(
                prediction_dir / f"{row.id}_outer_fiber.png", predicted_outer
            )
            write_overlay(
                overlay_dir / f"{row.id}_prediction.png",
                create_overlay(
                    image,
                    predicted_axon,
                    predicted_outer,
                    np.zeros(shape, dtype=bool),
                ),
            )
            write_overlay(
                overlay_dir / f"{row.id}_reference.png",
                create_overlay(
                    image,
                    truth_axon,
                    truth_outer,
                    np.zeros(shape, dtype=bool),
                ),
            )
            records.append(
                {
                    "id": row.id,
                    "source_image_id": row.source_image_id or row.image_path.stem,
                    "split": row.split,
                    "source_scale_nm_per_px": row.scale_nm_per_px,
                    "target_scale_nm_per_px": target_scale_for_crop(
                        row.scale_nm_per_px
                    ),
                    "crop_height_px": shape[0],
                    "crop_width_px": shape[1],
                    "candidate_fibers": len(extraction.fibers),
                    "selection_status": status,
                    "extraction_flags": flags or "pass",
                    "axon_dice": axon_scores["dice"],
                    "axon_iou": axon_scores["iou"],
                    "axon_precision": axon_scores["precision"],
                    "axon_recall": axon_scores["recall"],
                    "outer_fiber_dice": outer_scores["dice"],
                    "outer_fiber_iou": outer_scores["iou"],
                    "outer_fiber_precision": outer_scores["precision"],
                    "outer_fiber_recall": outer_scores["recall"],
                    "both_masks_pass": bool(
                        axon_scores["dice"] >= 0.75
                        and outer_scores["dice"] >= 0.75
                    ),
                }
            )

    frame = pd.DataFrame(records)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_crop_metrics.csv", index=False)
    pass_rate = float(frame["both_masks_pass"].mean())
    complete_misses = int((frame["selection_status"] == "complete_miss").sum())
    median_axon = float(frame["axon_dice"].median())
    median_outer = float(frame["outer_fiber_dice"].median())
    summary: dict[str, object] = {
        "manifest": str(Path(manifest_path).resolve()),
        "model": str(model_path),
        "n_crops": len(frame),
        "median_axon_dice": median_axon,
        "median_outer_fiber_dice": median_outer,
        "individual_both_masks_pass_count": int(frame["both_masks_pass"].sum()),
        "individual_both_masks_pass_rate": pass_rate,
        "complete_misses": complete_misses,
        "acceptance_requirements": {
            "median_axon_dice_at_least": 0.75,
            "median_outer_fiber_dice_at_least": 0.75,
            "individual_pass_rate_at_least": 0.80,
            "complete_misses_at_most": 1,
        },
        "accepted": bool(
            median_axon >= 0.75
            and median_outer >= 0.75
            and pass_rate >= 0.80
            and complete_misses <= 1
        ),
        "reference_limitations": (
            "The masks are single-student project references, not expert ground truth."
        ),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    _write_overview(frame, output / "overview.png")
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate AxonDeepSeg directly on annotated per-fiber crops"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--min-axon-area", type=float, default=0.001)
    args = parser.parse_args()
    _, summary = validate_cropped_images(
        args.manifest,
        args.model_path,
        args.output,
        gpu_id=args.gpu_id,
        minimum_axon_area_um2=args.min_axon_area,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
