from pathlib import Path

from mvp_pipeline.resources import (
    MODEL_FOLDER_NAME,
    axondeepseg_model_status,
    install_axondeepseg_model,
)
from mvp_pipeline.cli import build_parser


def _write_fake_model(path: Path) -> Path:
    model = path / MODEL_FOLDER_NAME
    (model / "fold_all").mkdir(parents=True)
    (model / "dataset.json").write_text("{}", encoding="utf-8")
    (model / "plans.json").write_text("{}", encoding="utf-8")
    checkpoint = model / "fold_all" / "checkpoint_final.pth"
    with checkpoint.open("wb") as stream:
        stream.truncate(100_000_000)
    return model


def test_model_status_rejects_tiny_placeholder_checkpoint(tmp_path: Path) -> None:
    model = tmp_path / MODEL_FOLDER_NAME
    (model / "fold_all").mkdir(parents=True)
    (model / "dataset.json").write_text("{}", encoding="utf-8")
    (model / "plans.json").write_text("{}", encoding="utf-8")
    (model / "fold_all" / "checkpoint_final.pth").write_bytes(b"placeholder")

    status = axondeepseg_model_status(model)

    assert not status["valid"]
    assert not status["files"]["checkpoint"]


def test_cli_exposes_workspace_setup_command() -> None:
    args = build_parser().parse_args(["setup", "--force-model"])

    assert args.mode == "setup"
    assert args.force_model


def test_setup_installs_download_into_runtime_path(tmp_path: Path) -> None:
    def fake_download(destination: Path) -> Path:
        return _write_fake_model(destination)

    status = install_axondeepseg_model(tmp_path, downloader=fake_download)
    installed = tmp_path / "work" / "models" / MODEL_FOLDER_NAME

    assert status["action"] == "downloaded"
    assert status["valid"]
    assert installed.is_dir()


def test_setup_does_not_redownload_valid_model(tmp_path: Path) -> None:
    model_parent = tmp_path / "work" / "models"
    _write_fake_model(model_parent)

    def unexpected_download(destination: Path) -> Path:
        raise AssertionError("valid model should not be downloaded again")

    status = install_axondeepseg_model(tmp_path, downloader=unexpected_download)

    assert status["action"] == "already_present"
