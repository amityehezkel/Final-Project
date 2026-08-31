from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import DetectorConfig
from .io import read_binary_mask, read_grayscale, write_binary_mask, write_grayscale
from .run import run_batch


STATE_VERSION = 1
STATE_FILENAME = "workflow_state.json"
MANIFEST_FILENAME = "input_manifest.csv"
INPUT_STATUS_FILENAME = "input_status.csv"
IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def _safe_id(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    cleaned = cleaned.strip("._-")
    return cleaned or "image"


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, path)


def _state_path(output_dir: str | Path) -> Path:
    return Path(output_dir).resolve() / STATE_FILENAME


def load_guided_state(state_or_output: str | Path) -> dict[str, Any]:
    supplied = Path(state_or_output).resolve()
    path = supplied if supplied.is_file() else supplied / STATE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Guided-workflow state does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        state = json.load(stream)
    if int(state.get("version", -1)) != STATE_VERSION:
        raise ValueError(
            f"Unsupported guided-workflow state version: {state.get('version')!r}"
        )
    state["state_path"] = str(path)
    return state


def save_guided_state(state: dict[str, Any]) -> Path:
    path = Path(state.get("state_path") or _state_path(state["output_dir"])).resolve()
    serializable = dict(state)
    serializable.pop("state_path", None)
    _atomic_write_json(path, serializable)
    state["state_path"] = str(path)
    _write_input_status(state)
    return path


def _input_status_records(state: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    crops_by_source: dict[str, list[dict[str, Any]]] = {}
    for crop in state.get("crops", []):
        crops_by_source.setdefault(str(crop.get("source_id", "")), []).append(crop)

    if state.get("mode") == "guided_whole_folder":
        for source in state.get("sources", []):
            source_crops = crops_by_source.get(str(source["id"]), [])
            saved_status = str(source.get("status", "pending_crops"))
            if saved_status in {"skipped", "archiving_skipped"}:
                status = "skipped"
            elif saved_status in {"crops_complete", "archiving_processed"}:
                status = "processed"
            else:
                status = "pending"
            original_path = Path(str(source["original_path"]))
            records.append(
                {
                    "input_id": source["id"],
                    "input_filename": original_path.name,
                    "input_path": str(original_path),
                    "status": status,
                    "skip_reason": source.get("skip_reason") or "",
                    "crop_count": len(source_crops),
                    "predicted_crop_count": sum(
                        crop.get("status") == "prediction_complete"
                        for crop in source_crops
                    ),
                }
            )
        return records

    for crop in state.get("crops", []):
        input_path = Path(str(crop["input_path"]))
        records.append(
            {
                "input_id": crop["id"],
                "input_filename": input_path.name,
                "input_path": str(input_path),
                "status": (
                    "processed"
                    if crop.get("status") == "prediction_complete"
                    else "pending"
                ),
                "skip_reason": "",
                "crop_count": 1,
                "predicted_crop_count": int(
                    crop.get("status") == "prediction_complete"
                ),
            }
        )
    return records


def _write_input_status(state: dict[str, Any]) -> Path:
    destination = Path(state["output_dir"]).resolve() / "results" / INPUT_STATUS_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "input_id",
        "input_filename",
        "input_path",
        "status",
        "skip_reason",
        "crop_count",
        "predicted_crop_count",
    ]
    pd.DataFrame(_input_status_records(state), columns=columns).to_csv(
        destination, index=False
    )
    return destination


def _input_status_counts(state: dict[str, Any]) -> dict[str, int]:
    statuses = [record["status"] for record in _input_status_records(state)]
    return {
        "n_inputs_processed": statuses.count("processed"),
        "n_inputs_skipped": statuses.count("skipped"),
        "n_inputs_pending": statuses.count("pending"),
    }


def _ensure_output_layout(output: Path) -> None:
    for relative in (
        "crops/images",
        "crops/axon_masks",
        "crops/outer_fiber_masks",
        "results",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)


def _image_files(
    folder: str | Path, description: str, *, allow_empty: bool = False
) -> list[Path]:
    directory = Path(folder).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {directory}")
    files = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    if not files and not allow_empty:
        raise ValueError(f"{description} contains no supported images: {directory}")
    return files


def _read_scale_mapping(scales_csv: str | Path | None) -> dict[str, float]:
    if scales_csv is None:
        return {}
    path = Path(scales_csv).resolve()
    frame = pd.read_csv(path)
    required = {"filename", "scale_nm_per_px"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Scale CSV is missing columns: {sorted(missing)}")
    mapping: dict[str, float] = {}
    for record in frame.to_dict(orient="records"):
        name = str(record["filename"]).strip().lower()
        scale = float(record["scale_nm_per_px"])
        if not name or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid scale row: {record}")
        if name in mapping:
            raise ValueError(f"Scale CSV contains duplicate filename: {name}")
        mapping[name] = scale
    return mapping


def _scale_for(path: Path, common_scale: float | None, mapping: dict[str, float]) -> float:
    if common_scale is not None:
        scale = float(common_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("nm/pixel must be greater than zero")
        return scale
    key = path.name.lower()
    if key not in mapping:
        raise ValueError(f"Scale CSV has no row for {path.name!r}")
    return mapping[key]


def _unique_id(base: str, used: set[str], identity: str) -> str:
    candidate = _safe_id(base)
    if candidate not in used:
        return candidate
    suffix = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:8]
    candidate = f"{candidate}_{suffix}"
    number = 2
    while candidate in used:
        candidate = f"{_safe_id(base)}_{suffix}_{number}"
        number += 1
    return candidate


def _new_state(mode: str, input_dir: Path, output: Path, archive_mode: str) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "mode": mode,
        "input_dir": str(input_dir),
        "output_dir": str(output),
        "archive_mode": "leave",
        "source_handling": "leave_in_place",
        "status": "in_progress",
        "sources": [],
        "crops": [],
        "results": None,
        "state_path": str(output / STATE_FILENAME),
    }


def _load_or_create_state(
    mode: str,
    input_dir: Path,
    output: Path,
    archive_mode: str,
) -> dict[str, Any]:
    state_path = output / STATE_FILENAME
    if state_path.exists():
        state = load_guided_state(state_path)
        if state["mode"] != mode:
            raise ValueError(
                f"Output already contains a {state['mode']!r} workflow, not {mode!r}"
            )
        if Path(state["input_dir"]).resolve() != input_dir:
            raise ValueError(
                "Resume with the same input folder used to create this workflow: "
                f"{state['input_dir']}"
            )
        return state
    return _new_state(mode, input_dir, output, archive_mode)


def _reconcile_interrupted_archives(state: dict[str, Any]) -> None:
    """Normalize an interrupted legacy archive without moving or copying inputs."""

    changed = False
    for source in state["sources"]:
        status = source.get("status")
        if status not in {"archiving_processed", "archiving_skipped"}:
            continue
        original = Path(source["original_path"])
        current = Path(source["current_path"])
        destination = Path(source["archive_path"]) if source.get("archive_path") else None
        available = next(
            (
                candidate
                for candidate in (original, current, destination)
                if candidate is not None and candidate.exists()
            ),
            None,
        )
        if available is None:
            raise FileNotFoundError(
                f"Cannot resume source {source['id']!r}; its recorded image no longer exists"
            )
        source["current_path"] = str(available.resolve())
        source["status"] = (
            "skipped" if status == "archiving_skipped" else "crops_complete"
        )
        changed = True
    if changed:
        save_guided_state(state)


def prepare_guided_whole_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    scale_nm_per_px: float | None = None,
    scales_csv: str | Path | None = None,
    archive_mode: str = "leave",
) -> Path:
    """Create or resume the non-GUI state for a whole-image guided workflow."""

    if archive_mode not in {"leave", "move", "copy"}:
        raise ValueError("archive_mode must be 'leave', 'move', or 'copy'")
    if (scale_nm_per_px is None) == (scales_csv is None):
        raise ValueError("Provide exactly one of scale_nm_per_px or scales_csv")
    input_path = Path(input_dir).resolve()
    output = Path(output_dir).resolve()
    if output == input_path or output.is_relative_to(input_path):
        raise ValueError("Output must be outside the whole-image input queue")
    mapping = _read_scale_mapping(scales_csv)
    _ensure_output_layout(output)
    state = _load_or_create_state(
        "guided_whole_folder", input_path, output, "leave"
    )
    state["archive_mode"] = "leave"
    state["source_handling"] = "leave_in_place"
    _reconcile_interrupted_archives(state)
    files = _image_files(
        input_path,
        "Whole-image input folder",
        allow_empty=bool(state["sources"]),
    )
    known = {str(Path(row["original_path"]).resolve()) for row in state["sources"]}
    used = {str(row["id"]) for row in state["sources"]}
    added = False
    for image_path in files:
        identity = str(image_path.resolve())
        if identity in known:
            continue
        source_id = _unique_id(image_path.stem, used, identity)
        used.add(source_id)
        state["sources"].append(
            {
                "id": source_id,
                "original_path": identity,
                "current_path": identity,
                "archive_path": None,
                "scale_nm_per_px": _scale_for(image_path, scale_nm_per_px, mapping),
                "status": "pending_crops",
                "draft_rectangles": [],
                "crop_ids": [],
                "skip_reason": None,
            }
        )
        added = True
    if added and state["status"] in {"complete", "complete_no_crops"}:
        state["status"] = "in_progress"
        state["results"] = None
    save_guided_state(state)
    return output / STATE_FILENAME


def prepare_guided_crop_folder(
    images_dir: str | Path,
    output_dir: str | Path,
    *,
    scale_nm_per_px: float | None = None,
    scales_csv: str | Path | None = None,
) -> Path:
    """Reference existing crops from a resumable annotation workspace."""

    if (scale_nm_per_px is None) == (scales_csv is None):
        raise ValueError("Provide exactly one of scale_nm_per_px or scales_csv")
    images_path = Path(images_dir).resolve()
    output = Path(output_dir).resolve()
    if output == images_path or output.is_relative_to(images_path):
        raise ValueError("Output must be outside the crop input folder")
    mapping = _read_scale_mapping(scales_csv)
    _ensure_output_layout(output)
    state = _load_or_create_state(
        "guided_crop_folder", images_path, output, "leave"
    )
    state["archive_mode"] = "leave"
    state["source_handling"] = "leave_in_place"
    files = _image_files(
        images_path,
        "Fiber-crop input folder",
        allow_empty=bool(state["crops"]),
    )
    known = {str(Path(row["input_path"]).resolve()) for row in state["crops"]}
    used = {str(row["id"]) for row in state["crops"]}
    added = False
    for image_path in files:
        identity = str(image_path.resolve())
        if identity in known:
            continue
        crop_id = _unique_id(image_path.stem, used, identity)
        used.add(crop_id)
        state["crops"].append(
            _crop_record(
                crop_id,
                source_id=image_path.stem,
                input_path=identity,
                image_path=image_path,
                output=output,
                scale_nm_per_px=_scale_for(image_path, scale_nm_per_px, mapping),
            )
        )
        added = True
    if added and state["status"] in {"complete", "complete_no_crops"}:
        state["status"] = "in_progress"
        state["results"] = None
    save_guided_state(state)
    return output / STATE_FILENAME


def _crop_record(
    crop_id: str,
    *,
    source_id: str,
    input_path: str,
    image_path: Path,
    output: Path,
    scale_nm_per_px: float,
) -> dict[str, Any]:
    return {
        "id": crop_id,
        "source_id": source_id,
        "input_path": input_path,
        "image_path": str(image_path.resolve()),
        "scale_nm_per_px": float(scale_nm_per_px),
        "axon_mask_path": str((output / "crops" / "axon_masks" / f"{crop_id}_axon.png").resolve()),
        "outer_fiber_mask_path": str(
            (output / "crops" / "outer_fiber_masks" / f"{crop_id}_outer_fiber.png").resolve()
        ),
        "status": "pending_masks",
        "validation_warnings": [],
    }


def save_source_draft(
    state_path: str | Path,
    source_id: str,
    rectangles: Iterable[Iterable[float]],
) -> None:
    state = load_guided_state(state_path)
    source = _find(state["sources"], source_id, "source")
    source["draft_rectangles"] = [list(map(float, rectangle)) for rectangle in rectangles]
    save_guided_state(state)


def _normalized_rectangles(
    rectangles: Iterable[Iterable[float]], shape: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    height, width = shape
    normalized: list[tuple[int, int, int, int]] = []
    for raw in rectangles:
        values = [float(value) for value in raw]
        if len(values) != 4:
            raise ValueError(f"Each crop rectangle must be x0,y0,x1,y1; got {values}")
        x0, y0, x1, y1 = values
        x0, x1 = sorted((int(np.floor(x0)), int(np.ceil(x1))))
        y0, y1 = sorted((int(np.floor(y0)), int(np.ceil(y1))))
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Crop rectangle is empty after clipping: {values}")
        normalized.append((x0, y0, x1, y1))
    return normalized


def _archive_source(state: dict[str, Any], source: dict[str, Any], skipped: bool) -> None:
    original = Path(source["original_path"])
    if not original.exists():
        raise FileNotFoundError(f"Source image no longer exists: {original}")
    source["current_path"] = str(original.resolve())
    source["archive_path"] = None
    source["status"] = "skipped" if skipped else "crops_complete"
    save_guided_state(state)


def finish_source_crops(
    state_path: str | Path,
    source_id: str,
    rectangles: Iterable[Iterable[float]],
) -> list[str]:
    """Extract any number of crops while leaving the whole image in place."""

    state = load_guided_state(state_path)
    source = _find(state["sources"], source_id, "source")
    if source["status"] not in {"pending_crops", "archiving_processed"}:
        raise ValueError(f"Source {source_id!r} is already {source['status']}")
    image_path = Path(source["current_path"])
    image = read_grayscale(image_path)
    normalized = _normalized_rectangles(rectangles, image.shape)
    output = Path(state["output_dir"])
    created: list[str] = []
    used = {str(row["id"]) for row in state["crops"]}
    for index, (x0, y0, x1, y1) in enumerate(normalized, start=1):
        base = f"{source_id}_fiber-{index:03d}"
        crop_id = _unique_id(base, used, f"{image_path}:{x0},{y0},{x1},{y1}")
        used.add(crop_id)
        destination = output / "crops" / "images" / f"{crop_id}.png"
        write_grayscale(destination, image[y0:y1, x0:x1])
        state["crops"].append(
            _crop_record(
                crop_id,
                source_id=source_id,
                input_path=str(image_path),
                image_path=destination,
                output=output,
                scale_nm_per_px=float(source["scale_nm_per_px"]),
            )
        )
        created.append(crop_id)
    source["crop_ids"] = created
    source["draft_rectangles"] = [list(rectangle) for rectangle in normalized]
    _archive_source(state, source, skipped=False)
    return created


def skip_source(
    state_path: str | Path, source_id: str, reason: str = "user_skipped"
) -> None:
    state = load_guided_state(state_path)
    source = _find(state["sources"], source_id, "source")
    source["skip_reason"] = reason
    source["draft_rectangles"] = []
    _archive_source(state, source, skipped=True)


def _find(rows: list[dict[str, Any]], identifier: str, description: str) -> dict[str, Any]:
    matches = [row for row in rows if str(row["id"]) == identifier]
    if len(matches) != 1:
        raise ValueError(f"Expected one {description} {identifier!r}, found {len(matches)}")
    return matches[0]


def validate_manual_masks(
    axon: np.ndarray, outer_fiber: np.ndarray
) -> tuple[list[str], list[str]]:
    axon = np.asarray(axon, dtype=bool)
    outer = np.asarray(outer_fiber, dtype=bool)
    errors: list[str] = []
    warnings: list[str] = []
    if axon.shape != outer.shape:
        errors.append("Axon and outer-fiber masks have different dimensions.")
        return errors, warnings
    if not axon.any():
        errors.append("The axon mask is empty.")
    if not outer.any():
        errors.append("The outer-fiber mask is empty.")
    if (axon & ~outer).any():
        errors.append("Every axon pixel must be inside the outer-fiber mask.")
    if outer.any() and not (outer & ~axon).any():
        errors.append("No gross sheath remains between the axon and outer boundary.")
    if outer.any() and (
        outer[0, :].any()
        or outer[-1, :].any()
        or outer[:, 0].any()
        or outer[:, -1].any()
    ):
        warnings.append("The outer-fiber mask touches the crop border; review the crop.")
    return errors, warnings


def save_mask_draft(
    state_path: str | Path,
    crop_id: str,
    axon: np.ndarray,
    outer_fiber: np.ndarray,
) -> None:
    state = load_guided_state(state_path)
    crop = _find(state["crops"], crop_id, "crop")
    image_shape = read_grayscale(crop["image_path"]).shape
    axon_array = np.asarray(axon, dtype=bool)
    outer_array = np.asarray(outer_fiber, dtype=bool)
    if axon_array.shape != image_shape or outer_array.shape != image_shape:
        raise ValueError(
            f"Mask dimensions must match crop {crop_id}: expected {image_shape}"
        )
    write_binary_mask(crop["axon_mask_path"], axon_array)
    write_binary_mask(crop["outer_fiber_mask_path"], outer_array)
    crop["status"] = "pending_masks"
    save_guided_state(state)


def complete_crop_masks(
    state_path: str | Path,
    crop_id: str,
    axon: np.ndarray,
    outer_fiber: np.ndarray,
) -> list[str]:
    state = load_guided_state(state_path)
    crop = _find(state["crops"], crop_id, "crop")
    image_shape = read_grayscale(crop["image_path"]).shape
    axon_array = np.asarray(axon, dtype=bool)
    outer_array = np.asarray(outer_fiber, dtype=bool)
    if axon_array.shape != image_shape or outer_array.shape != image_shape:
        raise ValueError(
            f"Mask dimensions must match crop {crop_id}: expected {image_shape}"
        )
    errors, warnings = validate_manual_masks(axon_array, outer_array)
    if errors:
        raise ValueError(" ".join(errors))
    write_binary_mask(crop["axon_mask_path"], axon_array)
    write_binary_mask(crop["outer_fiber_mask_path"], outer_array)
    crop["status"] = "masks_complete"
    crop["validation_warnings"] = warnings
    state["status"] = "in_progress"
    save_guided_state(state)
    return warnings


def write_guided_manifest(state_path: str | Path) -> Path:
    state = load_guided_state(state_path)
    output = Path(state["output_dir"])
    eligible = [
        crop
        for crop in state["crops"]
        if crop["status"] in {"masks_complete", "prediction_complete"}
    ]
    if not eligible:
        raise ValueError("No crops have complete masks")
    records: list[dict[str, Any]] = []
    for crop in eligible:
        records.append(
            {
                "id": crop["id"],
                "source_image_id": crop["source_id"],
                "image_path": os.path.relpath(crop["image_path"], output),
                "scale_nm_per_px": float(crop["scale_nm_per_px"]),
                "axon_mask_path": os.path.relpath(crop["axon_mask_path"], output),
                "outer_fiber_mask_path": os.path.relpath(
                    crop["outer_fiber_mask_path"], output
                ),
                "compact_myelin_mask_path": "",
                "consensus_vacuole_mask_path": "",
                "split": "inference",
                "mask_source": "guided_user",
                "correction_minutes": "",
            }
        )
    manifest = output / MANIFEST_FILENAME
    pd.DataFrame(records).to_csv(manifest, index=False)
    return manifest


def run_guided_detection(
    state_path: str | Path, config: DetectorConfig
) -> dict[str, Any]:
    state = load_guided_state(state_path)
    output = Path(state["output_dir"])
    incomplete_sources = [
        row["id"]
        for row in state["sources"]
        if row["status"] not in {"crops_complete", "skipped"}
    ]
    if incomplete_sources:
        raise ValueError(
            f"Whole-image crop selection is incomplete: {incomplete_sources}"
        )
    incomplete_crops = [
        row["id"]
        for row in state["crops"]
        if row["status"] not in {"masks_complete", "prediction_complete"}
    ]
    if incomplete_crops:
        raise ValueError(f"Mask annotation is incomplete: {incomplete_crops}")
    if not state["crops"]:
        state["status"] = "complete_no_crops"
        state["results"] = {
            "results_dir": str((output / "results").resolve()),
            "input_status": str((output / "results" / INPUT_STATUS_FILENAME).resolve()),
            "n_processed": 0,
            "n_excluded_by_qc": 0,
            **_input_status_counts(state),
        }
        save_guided_state(state)
        with (output / "workflow_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(state["results"], stream, indent=2)
        return state["results"]

    manifest = write_guided_manifest(state_path)
    _, _, summary = run_batch(manifest, output / "results", config)
    state = load_guided_state(state_path)
    for crop in state["crops"]:
        crop["status"] = "prediction_complete"
    state["status"] = "complete"
    state["results"] = {
        "manifest": str(manifest),
        "results_dir": str((output / "results").resolve()),
        "input_status": str((output / "results" / INPUT_STATUS_FILENAME).resolve()),
        "n_processed": int(summary["n_processed"]),
        "n_excluded_by_qc": int(summary["n_excluded_by_qc"]),
        **_input_status_counts(state),
    }
    save_guided_state(state)
    with (output / "workflow_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(state["results"], stream, indent=2)
    return state["results"]


def _require_gui():
    try:
        import napari
        from napari.qt import get_qapp
        from qtpy.QtWidgets import (
            QLabel,
            QMessageBox,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The guided workflow requires Napari and Qt. Install the full project "
            "environment with: pip install -e '.[full]'"
        ) from exc
    # The first guided action is an instructional QMessageBox, before a
    # napari.Viewer exists. Explicitly create/reuse Qt's application object so
    # that popup cannot be the first QWidget in the process.
    get_qapp()
    return napari, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget


def _message(kind: str, title: str, body: str) -> None:
    _, _, QMessageBox, _, _, _ = _require_gui()
    method = getattr(QMessageBox, kind)
    method(None, title, body)


def _rectangle_vertices(rectangle: Iterable[float]) -> np.ndarray:
    x0, y0, x1, y1 = map(float, rectangle)
    return np.asarray([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])


def _rectangles_from_layer(data: Iterable[np.ndarray]) -> list[list[float]]:
    rectangles: list[list[float]] = []
    for vertices in data:
        array = np.asarray(vertices, dtype=float)
        y0, x0 = array.min(axis=0)
        y1, x1 = array.max(axis=0)
        rectangles.append([float(x0), float(y0), float(x1), float(y1)])
    return rectangles


def _crop_selection_window(
    source: dict[str, Any], position: int, total: int
) -> tuple[str, list[list[float]]]:
    napari, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget = _require_gui()
    image = read_grayscale(source["current_path"])
    viewer = napari.Viewer(
        title=f"Guided crop selection {position}/{total}: {source['id']}"
    )
    viewer.add_image(image, name="whole_image", colormap="gray")
    draft = [_rectangle_vertices(row) for row in source.get("draft_rectangles", [])]
    crop_layer = viewer.add_shapes(
        draft,
        name="fiber_crops",
        shape_type="rectangle",
        edge_color="yellow",
        face_color="transparent",
        edge_width=2,
    )
    action: dict[str, Any] = {"name": "pause", "rectangles": None}
    widget = QWidget()
    layout = QVBoxLayout(widget)
    instructions = QLabel(
        "Draw one rectangle around every complete fiber you want to analyze.\n\n"
        "There is no crop limit. Keep the entire outer myelin boundary inside each rectangle.\n\n"
        "Finish image saves all rectangles and opens the next whole image. Closing the window pauses safely."
    )
    instructions.setWordWrap(True)
    layout.addWidget(instructions)
    finish_button = QPushButton("Finish image and continue")
    skip_button = QPushButton("Skip this whole image")
    pause_button = QPushButton("Save draft and pause")
    layout.addWidget(finish_button)
    layout.addWidget(skip_button)
    layout.addWidget(pause_button)
    layout.addStretch()
    viewer.window.add_dock_widget(widget, area="right", name="Guided workflow")

    def finish() -> None:
        rectangles = _rectangles_from_layer(crop_layer.data)
        try:
            _normalized_rectangles(rectangles, image.shape)
        except ValueError as exc:
            QMessageBox.warning(None, "Invalid crop", str(exc))
            return
        action["name"] = "finish"
        action["rectangles"] = rectangles
        viewer.close()

    def skip() -> None:
        response = QMessageBox.question(
            None,
            "Skip whole image?",
            "This image will remain in the input folder and be recorded as skipped. "
            "No crops will be created.",
        )
        if response == QMessageBox.Yes:
            action["name"] = "skip"
            action["rectangles"] = _rectangles_from_layer(crop_layer.data)
            viewer.close()

    def pause() -> None:
        action["name"] = "pause"
        action["rectangles"] = _rectangles_from_layer(crop_layer.data)
        viewer.close()

    finish_button.clicked.connect(finish)
    skip_button.clicked.connect(skip)
    pause_button.clicked.connect(pause)
    napari.run()
    rectangles = action["rectangles"]
    if rectangles is None:
        # Closing the title-bar X removes layers from viewer.layers in Napari
        # 0.9, but the direct layer reference remains readable.
        rectangles = _rectangles_from_layer(crop_layer.data)
    return str(action["name"]), rectangles


def _initial_crop_masks(crop: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    image = read_grayscale(crop["image_path"])
    axon_path = Path(crop["axon_mask_path"])
    outer_path = Path(crop["outer_fiber_mask_path"])
    axon = read_binary_mask(axon_path, image.shape) if axon_path.exists() else np.zeros(image.shape, bool)
    outer = read_binary_mask(outer_path, image.shape) if outer_path.exists() else np.zeros(image.shape, bool)
    return axon, outer


def _mask_annotation_window(
    crop: dict[str, Any], position: int, total: int
) -> tuple[str, np.ndarray, np.ndarray]:
    napari, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget = _require_gui()
    image = read_grayscale(crop["image_path"])
    axon, outer = _initial_crop_masks(crop)
    viewer = napari.Viewer(
        title=f"Guided mask annotation {position}/{total}: {crop['id']}"
    )
    viewer.add_image(image, name="image", colormap="gray")
    outer_layer = viewer.add_labels(
        outer.astype(np.uint8), name="outer_fiber", opacity=0.35
    )
    axon_layer = viewer.add_labels(
        axon.astype(np.uint8), name="axon", opacity=0.45
    )
    action: dict[str, Any] = {"name": "pause", "arrays": None}
    widget = QWidget()
    layout = QVBoxLayout(widget)
    instructions = QLabel(
        "Draw two filled masks:\n"
        "1. outer_fiber: the complete filled envelope inside the outer myelin boundary.\n"
        "2. axon: the complete axoplasm inside the innermost boundary.\n\n"
        "Do not draw vacuoles. They will be predicted automatically after all masks are complete."
    )
    instructions.setWordWrap(True)
    layout.addWidget(instructions)
    save_button = QPushButton("Validate, save, and next")
    back_button = QPushButton("Save draft and previous crop")
    pause_button = QPushButton("Save draft and pause")
    layout.addWidget(save_button)
    layout.addWidget(back_button)
    layout.addWidget(pause_button)
    layout.addStretch()
    viewer.window.add_dock_widget(widget, area="right", name="Guided workflow")

    def arrays() -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(axon_layer.data) > 0,
            np.asarray(outer_layer.data) > 0,
        )

    def save() -> None:
        axon_data, outer_data = arrays()
        errors, warnings = validate_manual_masks(axon_data, outer_data)
        if errors:
            QMessageBox.warning(None, "Masks need correction", "\n".join(errors))
            return
        if warnings:
            response = QMessageBox.question(
                None,
                "Mask warning",
                "\n".join(warnings) + "\n\nSave anyway?",
            )
            if response != QMessageBox.Yes:
                return
        action["name"] = "save"
        action["arrays"] = (axon_data.copy(), outer_data.copy())
        viewer.close()

    def back() -> None:
        action["name"] = "back"
        axon_data, outer_data = arrays()
        action["arrays"] = (axon_data.copy(), outer_data.copy())
        viewer.close()

    def pause() -> None:
        action["name"] = "pause"
        axon_data, outer_data = arrays()
        action["arrays"] = (axon_data.copy(), outer_data.copy())
        viewer.close()

    save_button.clicked.connect(save)
    back_button.clicked.connect(back)
    pause_button.clicked.connect(pause)
    napari.run()
    saved_arrays = action["arrays"]
    if saved_arrays is None:
        # Title-bar close: preserve the draft through retained layer objects.
        saved_arrays = arrays()
    axon_data, outer_data = saved_arrays
    return str(action["name"]), axon_data, outer_data


def _run_mask_windows(state_path: Path, config: DetectorConfig) -> dict[str, Any]:
    state = load_guided_state(state_path)
    if not state["crops"]:
        return run_guided_detection(state_path, config)
    _message(
        "information",
        "Mask-annotation step",
        f"The workflow will now create axon and outer-fiber masks for {len(state['crops'])} crops.",
    )
    crops = state["crops"]
    pending_indices = [
        index
        for index, crop in enumerate(crops)
        if crop["status"] not in {"masks_complete", "prediction_complete"}
    ]
    index = pending_indices[0] if pending_indices else len(crops)
    while index < len(crops):
        state = load_guided_state(state_path)
        crop = state["crops"][index]
        action, axon, outer = _mask_annotation_window(crop, index + 1, len(crops))
        if action == "save":
            complete_crop_masks(state_path, crop["id"], axon, outer)
            index += 1
        elif action == "back":
            save_mask_draft(state_path, crop["id"], axon, outer)
            index = max(0, index - 1)
        else:
            save_mask_draft(state_path, crop["id"], axon, outer)
            return {"status": "paused", "state_path": str(state_path)}
    _message(
        "information",
        "Running vacuole recognition",
        "All masks are complete. The frozen vacuole detector will now create predictions, overlays, and measurements.",
    )
    results = run_guided_detection(state_path, config)
    _message(
        "information",
        "Workflow complete",
        f"Processed {results['n_processed']} crops. Results were saved in:\n{results.get('results_dir', Path(state_path).parent)}",
    )
    return results


def run_guided_whole_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    config: DetectorConfig,
    *,
    scale_nm_per_px: float | None = None,
    scales_csv: str | Path | None = None,
    archive_mode: str = "leave",
) -> dict[str, Any]:
    state_path = prepare_guided_whole_folder(
        input_dir,
        output_dir,
        scale_nm_per_px=scale_nm_per_px,
        scales_csv=scales_csv,
        archive_mode=archive_mode,
    )
    state = load_guided_state(state_path)
    if state["status"] in {"complete", "complete_no_crops"}:
        return state["results"] or {"n_processed": 0}
    pending = [row for row in state["sources"] if row["status"] == "pending_crops"]
    if pending:
        _message(
            "information",
            "Guided whole-image workflow",
            "The program will open each whole image in sequence. Draw any number of "
            "complete-fiber crops, then use Finish image and continue. Source images "
            "remain unchanged in the input folder.",
        )
    for position, source in enumerate(pending, start=1):
        action, rectangles = _crop_selection_window(source, position, len(pending))
        save_source_draft(state_path, source["id"], rectangles)
        if action == "finish":
            finish_source_crops(state_path, source["id"], rectangles)
        elif action == "skip":
            skip_source(state_path, source["id"])
        else:
            return {"status": "paused", "state_path": str(state_path)}
    return _run_mask_windows(state_path, config)


def run_guided_crop_folder(
    images_dir: str | Path,
    output_dir: str | Path,
    config: DetectorConfig,
    *,
    scale_nm_per_px: float | None = None,
    scales_csv: str | Path | None = None,
) -> dict[str, Any]:
    state_path = prepare_guided_crop_folder(
        images_dir,
        output_dir,
        scale_nm_per_px=scale_nm_per_px,
        scales_csv=scales_csv,
    )
    state = load_guided_state(state_path)
    if state["status"] in {"complete", "complete_no_crops"}:
        return state["results"] or {"n_processed": 0}
    return _run_mask_windows(state_path, config)
