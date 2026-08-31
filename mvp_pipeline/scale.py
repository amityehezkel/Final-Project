from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from skimage.transform import resize

from .io import read_grayscale, write_grayscale
from .masks import scale_bar_region


def target_shape(
    shape: tuple[int, int], source_scale_nm_per_px: float, target_scale_nm_per_px: float
) -> tuple[int, int]:
    if source_scale_nm_per_px <= 0 or target_scale_nm_per_px <= 0:
        raise ValueError("Physical scales must be positive")
    factor = source_scale_nm_per_px / target_scale_nm_per_px
    return tuple(max(1, int(round(axis * factor))) for axis in shape)


def resample_image(
    image: np.ndarray, source_scale_nm_per_px: float, target_scale_nm_per_px: float
) -> np.ndarray:
    shape = target_shape(image.shape, source_scale_nm_per_px, target_scale_nm_per_px)
    return resize(
        image,
        shape,
        order=1,
        mode="reflect",
        anti_aliasing=shape[0] < image.shape[0] or shape[1] < image.shape[1],
        preserve_range=True,
    ).astype(np.float32)


def restore_mask(mask: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    return resize(
        np.asarray(mask, dtype=np.uint8),
        original_shape,
        order=0,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    ) > 0


def mask_scale_bar_in_image(image: np.ndarray) -> np.ndarray:
    result = np.asarray(image).copy()
    excluded = scale_bar_region(result.shape)
    available = result[~excluded]
    fill = float(np.median(available)) if available.size else 0.0
    result[excluded] = fill
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a physical-scale-normalized image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-scale", required=True, type=float, dest="source_scale")
    parser.add_argument("--target-scale", required=True, type=float, dest="target_scale")
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-scale-bar", action="store_true")
    args = parser.parse_args()
    image = read_grayscale(args.image)
    if args.exclude_scale_bar:
        image = mask_scale_bar_in_image(image)
    normalized = resample_image(image, args.source_scale, args.target_scale)
    write_grayscale(args.output, normalized)
    print(f"Wrote {Path(args.output).resolve()} with shape {normalized.shape}")


if __name__ == "__main__":
    main()
