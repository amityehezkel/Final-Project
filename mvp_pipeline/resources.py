from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterator


MODEL_FOLDER_NAME = "model_seg_generalist_light"
MODEL_RELATIVE_PATH = Path("work") / "models" / MODEL_FOLDER_NAME
MINIMUM_CHECKPOINT_BYTES = 100_000_000


def axondeepseg_model_status(model_path: str | Path) -> dict[str, Any]:
    """Validate the files required by the frozen AxonDeepSeg front end."""

    path = Path(model_path).resolve()
    checkpoint_paths = sorted(path.rglob("*.pth")) if path.exists() else []
    usable_checkpoints = [
        checkpoint
        for checkpoint in checkpoint_paths
        if checkpoint.is_file() and checkpoint.stat().st_size >= MINIMUM_CHECKPOINT_BYTES
    ]
    files = {
        "dataset.json": (path / "dataset.json").is_file(),
        "plans.json": (path / "plans.json").is_file(),
        "checkpoint": bool(usable_checkpoints),
    }
    return {
        "path": str(path),
        "valid": all(files.values()),
        "files": files,
        "checkpoint_paths": [str(item) for item in usable_checkpoints],
    }


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _official_model_downloader(destination: Path) -> Path:
    try:
        from AxonDeepSeg.download_model import download_model
    except ImportError as exc:
        raise RuntimeError(
            "AxonDeepSeg is not installed. Create or update the declared conda "
            "environment before downloading its model."
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    # AxonDeepSeg 5.3 extracts into the current directory before moving the
    # model to `destination`, so use a separate disposable working directory.
    working = destination.parent / "extract"
    working.mkdir(parents=True, exist_ok=True)
    with _working_directory(working):
        downloaded = download_model(
            model="generalist",
            model_type="light",
            destination=str(destination),
            overwrite=True,
        )
    return Path(downloaded).resolve()


def install_axondeepseg_model(
    project_root: str | Path,
    *,
    force: bool = False,
    downloader: Callable[[Path], Path] | None = None,
) -> dict[str, Any]:
    """Install the official generalist-light model in the project runtime path."""

    root = Path(project_root).resolve()
    target = root / MODEL_RELATIVE_PATH
    current = axondeepseg_model_status(target)
    if current["valid"] and not force:
        return {**current, "action": "already_present"}

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".axondeepseg-download-", dir=target.parent)
    )
    backup = target.parent / f".{MODEL_FOLDER_NAME}.previous"
    fetch = downloader or _official_model_downloader
    try:
        downloaded = fetch(temporary_root / "destination")
        downloaded_status = axondeepseg_model_status(downloaded)
        if not downloaded_status["valid"]:
            raise RuntimeError(
                "The AxonDeepSeg download completed but the model is incomplete: "
                f"{downloaded_status['files']}"
            )

        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            target.replace(backup)
        try:
            downloaded.replace(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.replace(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    return {**axondeepseg_model_status(target), "action": "downloaded"}
