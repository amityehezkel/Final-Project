from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


REQUIRED_MANIFEST_COLUMNS = {
    "id",
    "image_path",
    "scale_nm_per_px",
    "axon_mask_path",
    "outer_fiber_mask_path",
    "split",
}

OPTIONAL_PATH_COLUMNS = {
    "compact_myelin_mask_path",
    "consensus_vacuole_mask_path",
}


@dataclass(frozen=True)
class ManifestRow:
    id: str
    image_path: Path
    scale_nm_per_px: float
    axon_mask_path: Path
    outer_fiber_mask_path: Path
    split: str
    compact_myelin_mask_path: Path | None = None
    consensus_vacuole_mask_path: Path | None = None
    mask_source: str = "manual"
    correction_minutes: float | None = None
    source_image_id: str | None = None


def _optional_path(value: Any, base: Path) -> Path | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    # Manifests committed from Windows often contain backslashes. Forward
    # slashes are understood on Windows too, so normalize for portable clones.
    path = Path(text.replace("\\", "/"))
    return path if path.is_absolute() else (base / path).resolve()


def _required_path(value: Any, base: Path, column: str) -> Path:
    path = _optional_path(value, base)
    if path is None:
        raise ValueError(f"Manifest column {column!r} contains an empty path")
    return path


def load_manifest(path: str | Path, require_files: bool = True) -> list[ManifestRow]:
    manifest_path = Path(path).resolve()
    frame = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if frame["id"].astype(str).duplicated().any():
        duplicates = frame.loc[frame["id"].astype(str).duplicated(), "id"].tolist()
        raise ValueError(f"Manifest ids must be unique; duplicates: {duplicates}")

    rows: list[ManifestRow] = []
    for record in frame.to_dict(orient="records"):
        scale = float(record["scale_nm_per_px"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid scale_nm_per_px for {record['id']!r}: {scale}")
        correction = record.get("correction_minutes")
        if correction is not None and not pd.isna(correction):
            correction = float(correction)
        else:
            correction = None

        row = ManifestRow(
            id=str(record["id"]),
            image_path=_required_path(record["image_path"], manifest_path.parent, "image_path"),
            scale_nm_per_px=scale,
            axon_mask_path=_required_path(
                record["axon_mask_path"], manifest_path.parent, "axon_mask_path"
            ),
            outer_fiber_mask_path=_required_path(
                record["outer_fiber_mask_path"],
                manifest_path.parent,
                "outer_fiber_mask_path",
            ),
            split=str(record["split"]).strip().lower(),
            compact_myelin_mask_path=_optional_path(
                record.get("compact_myelin_mask_path"), manifest_path.parent
            ),
            consensus_vacuole_mask_path=_optional_path(
                record.get("consensus_vacuole_mask_path"), manifest_path.parent
            ),
            mask_source=str(record.get("mask_source", "manual") or "manual"),
            correction_minutes=correction,
            source_image_id=(
                None
                if pd.isna(record.get("source_image_id"))
                else str(record.get("source_image_id"))
            ),
        )
        if require_files:
            paths = [row.image_path, row.axon_mask_path, row.outer_fiber_mask_path]
            paths.extend(
                p
                for p in (
                    row.compact_myelin_mask_path,
                    row.consensus_vacuole_mask_path,
                )
                if p is not None
            )
            missing_paths = [str(p) for p in paths if not p.exists()]
            if missing_paths:
                raise FileNotFoundError(
                    f"Files for manifest id {row.id!r} do not exist: {missing_paths}"
                )
        rows.append(row)
    return rows


def read_grayscale(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"Image is empty: {path}")
    return array


def read_binary_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L")) > 0
    if shape is not None and array.shape != shape:
        raise ValueError(f"Mask {path} has shape {array.shape}, expected {shape}")
    return array


def write_binary_mask(path: str | Path, mask: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8)).save(target)


def write_grayscale(path: str | Path, image: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if array.dtype != np.uint8:
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise ValueError("Cannot write an image with no finite values")
        low, high = float(finite.min()), float(finite.max())
        array = np.zeros_like(array, dtype=np.uint8) if high <= low else (
            255 * (array - low) / (high - low)
        ).clip(0, 255).astype(np.uint8)
    Image.fromarray(array).save(target)
