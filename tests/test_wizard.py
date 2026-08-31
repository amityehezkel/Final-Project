from __future__ import annotations

import json
from pathlib import Path

from mvp_pipeline.cli import build_parser
from mvp_pipeline.wizard import build_wizard_arguments


def _scripted(values: list[str]):
    iterator = iter(values)
    return lambda _prompt: next(iterator)


def test_wizard_selects_one_existing_masked_crop(tmp_path: Path) -> None:
    image = tmp_path / "fiber.tif"
    axon = tmp_path / "fiber_axon.png"
    outer = tmp_path / "fiber_outer.png"
    for path in (image, axon, outer):
        path.write_bytes(b"placeholder")
    output = tmp_path / "results"
    messages: list[str] = []

    arguments = build_wizard_arguments(
        input_fn=_scripted(
            [
                "1",  # existing crops
                "yes",  # masks exist
                "1",  # one crop
                str(image),
                str(axon),
                str(outer),
                "5.523",
                str(output),
                "",  # confirm
            ]
        ),
        output_fn=messages.append,
    )

    assert arguments == [
        "fiber-crop",
        "--image",
        str(image.resolve()),
        "--nm-per-pixel",
        "5.523",
        "--axon-mask",
        str(axon.resolve()),
        "--outer-fiber-mask",
        str(outer.resolve()),
        "--output",
        str(output.resolve()),
    ]
    assert any("Equivalent explicit command" in message for message in messages)


def test_wizard_selects_guided_whole_folder_with_scale_csv(
    tmp_path: Path,
) -> None:
    images = tmp_path / "whole images"
    images.mkdir()
    scales = tmp_path / "scales.csv"
    scales.write_text("filename,scale_nm_per_px\nimage.tif,5.523\n", encoding="utf-8")
    output = tmp_path / "guided"

    arguments = build_wizard_arguments(
        input_fn=_scripted(
            [
                "2",  # guided whole images
                str(images),
                "2",  # mixed scales
                str(scales),
                str(output),
                "",  # confirm
            ]
        ),
        output_fn=lambda _message: None,
    )

    assert arguments == [
        "guided-whole-folder",
        "--input",
        str(images.resolve()),
        "--scales-csv",
        str(scales.resolve()),
        "--output",
        str(output.resolve()),
    ]


def test_wizard_requires_explicit_opt_in_for_experimental_mode() -> None:
    messages: list[str] = []
    arguments = build_wizard_arguments(
        input_fn=_scripted(["3", ""]),
        output_fn=messages.append,
    )

    assert arguments is None
    assert any("EXPERIMENTAL MODE" in message for message in messages)
    assert any("not started" in message for message in messages)


def test_wizard_resumes_saved_guided_crop_session(tmp_path: Path) -> None:
    images = tmp_path / "crops"
    images.mkdir()
    output = tmp_path / "guided"
    output.mkdir()
    (output / "workflow_state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "guided_crop_folder",
                "input_dir": str(images),
                "output_dir": str(output),
                "archive_mode": "copy",
                "status": "in_progress",
                "sources": [],
                "crops": [{"id": "fiber", "scale_nm_per_px": 5.0}],
                "results": None,
            }
        ),
        encoding="utf-8",
    )

    arguments = build_wizard_arguments(
        input_fn=_scripted(
            [
                "4",
                str(output),
                "",  # one common scale
                "",  # reuse saved 5 nm/pixel value
                "",  # confirm
            ]
        ),
        output_fn=lambda _message: None,
    )

    assert arguments == [
        "guided-crop-folder",
        "--images",
        str(images.resolve()),
        "--nm-per-pixel",
        "5",
        "--output",
        str(output.resolve()),
    ]


def test_parser_supports_explicit_wizard_and_no_argument_launcher() -> None:
    parser = build_parser()

    assert parser.parse_args([]).mode is None
    assert parser.parse_args(["wizard"]).mode == "wizard"
