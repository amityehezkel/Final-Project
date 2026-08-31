from __future__ import annotations

import importlib.util
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DetectorConfig
from .io import OPTIONAL_PATH_COLUMNS, REQUIRED_MANIFEST_COLUMNS, load_manifest
from .resources import MODEL_RELATIVE_PATH, axondeepseg_model_status


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DISTRIBUTIONS = (
    "numpy",
    "pandas",
    "Pillow",
    "scipy",
    "scikit-image",
    "matplotlib",
)


def _distribution_versions(names: tuple[str, ...]) -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    return versions, missing


def check_workspace(project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    """Check whether a copied project root can run every supported mode."""

    root = Path(project_root).resolve()
    config_path = root / "work" / "best_model_results" / "detector_config.json"
    model_path = root / MODEL_RELATIVE_PATH
    benchmark_path = root / "work" / "benchmark" / "benchmark.csv"

    versions, missing_core = _distribution_versions(CORE_DISTRIBUTIONS)
    config_error: str | None = None
    if config_path.exists():
        try:
            DetectorConfig.from_json(config_path)
        except Exception as exc:  # configuration details belong in the report
            config_error = str(exc)
    else:
        config_error = "file is missing"

    model_status = axondeepseg_model_status(model_path)
    model_files = model_status["files"]
    ads_available = importlib.util.find_spec("AxonDeepSeg") is not None
    ads_version: str | None = None
    if ads_available:
        try:
            ads_version = metadata.version("AxonDeepSeg")
        except metadata.PackageNotFoundError:
            ads_version = "unknown"

    benchmark_error: str | None = None
    absolute_manifest_paths: list[str] = []
    benchmark_rows = 0
    if benchmark_path.exists():
        try:
            frame = pd.read_csv(benchmark_path)
            benchmark_rows = len(frame)
            path_columns = (
                {"image_path", "axon_mask_path", "outer_fiber_mask_path"}
                | OPTIONAL_PATH_COLUMNS
            ) & set(frame.columns)
            for column in sorted(path_columns):
                for value in frame[column].dropna().astype(str):
                    if value.strip() and Path(value).is_absolute():
                        absolute_manifest_paths.append(f"{column}: {value}")
            load_manifest(benchmark_path)
        except Exception as exc:
            benchmark_error = str(exc)
    else:
        benchmark_error = "file is missing"

    crop_modes_ready = not missing_core and config_error is None
    whole_image_ready = (
        crop_modes_ready and ads_available and all(model_files.values())
    )
    benchmark_ready = (
        benchmark_error is None
        and not absolute_manifest_paths
        and benchmark_rows > 0
    )
    napari_available = importlib.util.find_spec("napari") is not None
    qtpy_available = importlib.util.find_spec("qtpy") is not None
    qt_bindings = {
        "PyQt5": importlib.util.find_spec("PyQt5") is not None,
        "PyQt6": importlib.util.find_spec("PyQt6") is not None,
        "PySide6": importlib.util.find_spec("PySide6") is not None,
    }
    guided_workflow_ready = bool(
        crop_modes_ready
        and napari_available
        and qtpy_available
        and any(qt_bindings.values())
    )
    return {
        "project_root": str(root),
        "crop_modes_ready": crop_modes_ready,
        "whole_image_ready": whole_image_ready,
        "benchmark_ready_and_portable": benchmark_ready,
        "guided_workflow_ready": guided_workflow_ready,
        "complete_workspace_ready": bool(
            crop_modes_ready and whole_image_ready and benchmark_ready
        ),
        "core_dependencies": {
            "versions": versions,
            "missing": missing_core,
        },
        "detector_config": {
            "path": str(config_path),
            "valid": config_error is None,
            "error": config_error,
        },
        "axondeepseg": {
            "package_available": ads_available,
            "version": ads_version,
            "model_path": str(model_path),
            "model_files": model_files,
            "model_valid": model_status["valid"],
            "setup_command": "python -m mvp_pipeline setup",
        },
        "interactive_annotation": {
            "napari_available": napari_available,
            "qtpy_available": qtpy_available,
            "qt_bindings": qt_bindings,
        },
        "benchmark": {
            "path": str(benchmark_path),
            "rows": benchmark_rows,
            "error": benchmark_error,
            "absolute_path_count": len(absolute_manifest_paths),
            "absolute_paths": absolute_manifest_paths,
            "required_columns": sorted(REQUIRED_MANIFEST_COLUMNS),
        },
    }
