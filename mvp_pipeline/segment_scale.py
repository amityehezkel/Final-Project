from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np
from scipy import ndimage as ndi

from .io import read_binary_mask, read_grayscale, write_binary_mask, write_grayscale
from .masks import scale_bar_region
from .scale import mask_scale_bar_in_image, resample_image, restore_mask


MASK_SUFFIXES = ("axon", "myelin", "axonmyelin")


def segment_at_scale(
    image_path: str | Path,
    source_scale_nm_per_px: float,
    target_scale_nm_per_px: float,
    model_path: str | Path,
    output_dir: str | Path,
    gpu_id: int = -1,
) -> dict[str, Path]:
    """Run an external AxonDeepSeg model on a rescaled copy and restore its masks."""

    try:
        from AxonDeepSeg.apply_model import axon_segmentation
        from AxonDeepSeg.segment import get_model_input_format, get_model_type
    except ImportError as exc:
        raise RuntimeError(
            "AxonDeepSeg is not importable. Run this command with the astih conda Python."
        ) from exc

    image_path = Path(image_path).resolve()
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"AxonDeepSeg model folder does not exist: {model_path}")
    file_format, channels = get_model_input_format(model_path)
    if channels != 1:
        raise ValueError("The MVP scale wrapper currently supports one-channel models only")
    if file_format.lower() != ".png":
        raise ValueError(f"Expected a PNG AxonDeepSeg model, got {file_format!r}")

    original = read_grayscale(image_path)
    prepared = resample_image(
        mask_scale_bar_in_image(original), source_scale_nm_per_px, target_scale_nm_per_px
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ads-scale-") as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "input.png"
        write_grayscale(input_path, prepared)
        axon_segmentation(
            path_inputs=[input_path],
            path_model=model_path,
            model_type=get_model_type(model_path),
            gpu_id=gpu_id,
            verbosity_level=0,
        )
        outputs: dict[str, Path] = {}
        restored_masks: dict[str, np.ndarray] = {}
        excluded = scale_bar_region(original.shape)
        for suffix in MASK_SUFFIXES:
            predicted_path = temporary_path / f"input_seg-{suffix}.png"
            if not predicted_path.exists():
                continue
            predicted = read_binary_mask(predicted_path)
            restored = restore_mask(predicted, original.shape)
            restored[excluded] = False
            restored_masks[suffix] = restored
            destination = output / (
                f"{image_path.stem}_scale-{target_scale_nm_per_px:g}_seg-{suffix}.png"
            )
            write_binary_mask(destination, restored)
            outputs[suffix] = destination
        if "axonmyelin" in restored_masks:
            outer_fiber = ndi.binary_fill_holes(restored_masks["axonmyelin"])
            outer_fiber[excluded] = False
            destination = output / (
                f"{image_path.stem}_scale-{target_scale_nm_per_px:g}_seg-outer_fiber.png"
            )
            write_binary_mask(destination, outer_fiber)
            outputs["outer_fiber"] = destination
    if not outputs:
        raise RuntimeError("AxonDeepSeg completed without producing recognized masks")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AxonDeepSeg through a physical-scale-normalization wrapper"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-scale", type=float, required=True)
    parser.add_argument("--target-scale", type=float, required=True, choices=(2.36, 4.93))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", type=int, default=-1)
    args = parser.parse_args()
    paths = segment_at_scale(
        args.image,
        args.source_scale,
        args.target_scale,
        args.model_path,
        args.output,
        args.gpu_id,
    )
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
