from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries


def create_overlay(
    image: np.ndarray,
    axon: np.ndarray,
    outer_fiber: np.ndarray,
    vacuoles: np.ndarray,
) -> np.ndarray:
    base = np.asarray(image, dtype=np.float32)
    finite = base[np.isfinite(base)]
    if finite.size == 0:
        base = np.zeros_like(base, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [1, 99])
        base = (
            np.zeros_like(base, dtype=np.uint8)
            if high <= low
            else (255 * np.clip((base - low) / (high - low), 0, 1)).astype(np.uint8)
        )
    rgb = np.repeat(base[..., None], 3, axis=2).astype(np.float32)

    axon = np.asarray(axon, dtype=bool)
    outer = np.asarray(outer_fiber, dtype=bool)
    vac = np.asarray(vacuoles, dtype=bool)
    rgb[axon] = 0.55 * rgb[axon] + 0.45 * np.array([40, 100, 255])
    rgb[vac] = 0.25 * rgb[vac] + 0.75 * np.array([255, 225, 20])
    rgb[find_boundaries(outer, mode="outer")] = np.array([255, 40, 40])
    rgb[find_boundaries(axon, mode="outer")] = np.array([30, 120, 255])
    rgb[find_boundaries(vac, mode="outer")] = np.array([255, 255, 0])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def write_overlay(path: str | Path, overlay: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(overlay, dtype=np.uint8), mode="RGB").save(target)
