from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Callable


InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]
STATE_FILENAME = "workflow_state.json"


class WizardCancelled(Exception):
    """Internal signal used when the user safely leaves the wizard."""


def _clean(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def _read(prompt: str, input_fn: InputFunction) -> str:
    try:
        value = _clean(input_fn(prompt))
    except (EOFError, KeyboardInterrupt) as error:
        raise WizardCancelled from error
    if value.lower() in {"q", "quit", "cancel", "exit"}:
        raise WizardCancelled
    return value


def _choice(
    title: str,
    options: list[tuple[str, str]],
    input_fn: InputFunction,
    output_fn: OutputFunction,
    *,
    default: str | None = None,
) -> str:
    output_fn("")
    output_fn(title)
    for key, label in options:
        suffix = " [recommended]" if key == default else ""
        output_fn(f"  {key}. {label}{suffix}")
    while True:
        default_hint = f" [{default}]" if default is not None else ""
        answer = _read(f"Choose an option{default_hint}: ", input_fn)
        if not answer and default is not None:
            return default
        normalized = answer.lower()
        for key, label in options:
            if normalized in {key.lower(), label.lower()}:
                return key
        output_fn("Please enter one of the listed option numbers, or q to cancel.")


def _yes_no(
    prompt: str,
    input_fn: InputFunction,
    output_fn: OutputFunction,
    *,
    default: bool,
) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        answer = _read(f"{prompt} [{hint}]: ", input_fn).lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        output_fn("Please answer yes or no, or q to cancel.")


def _existing_path(
    prompt: str,
    input_fn: InputFunction,
    output_fn: OutputFunction,
    *,
    directory: bool,
) -> Path:
    expected = "folder" if directory else "file"
    while True:
        raw = _read(f"{prompt}: ", input_fn)
        if not raw:
            output_fn(f"Please enter a {expected} path.")
            continue
        path = Path(raw).expanduser().resolve()
        valid = path.is_dir() if directory else path.is_file()
        if valid:
            return path
        output_fn(f"That {expected} does not exist: {path}")


def _output_path(
    input_fn: InputFunction,
    output_fn: OutputFunction,
    *,
    outside: Path | None = None,
) -> Path:
    while True:
        raw = _read("Output folder: ", input_fn)
        if not raw:
            output_fn("Please enter an output-folder path.")
            continue
        path = Path(raw).expanduser().resolve()
        if path.exists() and not path.is_dir():
            output_fn(f"The output path is an existing file, not a folder: {path}")
            continue
        if outside is not None and (path == outside or path.is_relative_to(outside)):
            output_fn(
                "The guided output folder must be outside the input folder so that "
                "generated files are not mistaken for new inputs."
            )
            continue
        return path


def _scale_value(
    input_fn: InputFunction,
    output_fn: OutputFunction,
    *,
    default: float | None = None,
) -> str:
    while True:
        hint = f" [{default:g}]" if default is not None else ""
        raw = _read(f"Physical scale in nm/pixel{hint}: ", input_fn)
        if not raw and default is not None:
            return f"{default:g}"
        try:
            scale = float(raw)
        except ValueError:
            output_fn("Scale must be a number, for example 5.523.")
            continue
        if scale <= 0:
            output_fn("Scale must be greater than zero.")
            continue
        return f"{scale:g}"


def _guided_scale_arguments(
    input_fn: InputFunction,
    output_fn: OutputFunction,
    *,
    default_scale: float | None = None,
    force_csv: bool = False,
) -> list[str]:
    if force_csv:
        scales = _existing_path(
            "Scale CSV (filename, scale_nm_per_px)",
            input_fn,
            output_fn,
            directory=False,
        )
        return ["--scales-csv", str(scales)]
    scale_mode = _choice(
        "Do all input images have the same physical scale?",
        [("1", "Yes, use one nm/pixel value"), ("2", "No, use a scale CSV")],
        input_fn,
        output_fn,
        default="1",
    )
    if scale_mode == "1":
        return [
            "--nm-per-pixel",
            _scale_value(input_fn, output_fn, default=default_scale),
        ]
    scales = _existing_path(
        "Scale CSV (filename, scale_nm_per_px)",
        input_fn,
        output_fn,
        directory=False,
    )
    return ["--scales-csv", str(scales)]


def _existing_crops(
    input_fn: InputFunction, output_fn: OutputFunction
) -> list[str]:
    have_masks = _yes_no(
        "Do the crops already have axon and outer-fiber masks?",
        input_fn,
        output_fn,
        default=True,
    )
    if not have_masks:
        images = _existing_path(
            "Folder containing the fiber crops", input_fn, output_fn, directory=True
        )
        scale_args = _guided_scale_arguments(input_fn, output_fn)
        output = _output_path(input_fn, output_fn, outside=images)
        return [
            "guided-crop-folder",
            "--images",
            str(images),
            *scale_args,
            "--output",
            str(output),
        ]

    amount = _choice(
        "How many masked crops do you want to analyze?",
        [("1", "One crop"), ("2", "A folder of crops")],
        input_fn,
        output_fn,
        default="2",
    )
    if amount == "1":
        image = _existing_path("Fiber crop", input_fn, output_fn, directory=False)
        axon = _existing_path("Axon mask", input_fn, output_fn, directory=False)
        outer = _existing_path(
            "Outer-fiber mask", input_fn, output_fn, directory=False
        )
        scale = _scale_value(input_fn, output_fn)
        output = _output_path(input_fn, output_fn)
        return [
            "fiber-crop",
            "--image",
            str(image),
            "--nm-per-pixel",
            scale,
            "--axon-mask",
            str(axon),
            "--outer-fiber-mask",
            str(outer),
            "--output",
            str(output),
        ]

    images = _existing_path(
        "Folder containing the fiber crops", input_fn, output_fn, directory=True
    )
    axons = _existing_path(
        "Folder containing the axon masks", input_fn, output_fn, directory=True
    )
    outers = _existing_path(
        "Folder containing the outer-fiber masks",
        input_fn,
        output_fn,
        directory=True,
    )
    output_fn("All crops in this supplied-mask folder run must share one scale.")
    scale = _scale_value(input_fn, output_fn)
    output = _output_path(input_fn, output_fn)
    return [
        "fiber-folder",
        "--images",
        str(images),
        "--nm-per-pixel",
        scale,
        "--axon-masks",
        str(axons),
        "--outer-fiber-masks",
        str(outers),
        "--output",
        str(output),
    ]


def _guided_whole(
    input_fn: InputFunction, output_fn: OutputFunction
) -> list[str]:
    images = _existing_path(
        "Folder containing whole images", input_fn, output_fn, directory=True
    )
    scale_args = _guided_scale_arguments(input_fn, output_fn)
    output = _output_path(input_fn, output_fn, outside=images)
    return [
        "guided-whole-folder",
        "--input",
        str(images),
        *scale_args,
        "--output",
        str(output),
    ]


def _automatic_whole(
    input_fn: InputFunction, output_fn: OutputFunction
) -> list[str] | None:
    output_fn("")
    output_fn("EXPERIMENTAL MODE")
    output_fn(
        "AxonDeepSeg will propose the fiber boundaries automatically. This front end "
        "is less reliable than reviewed masks, so every overlay and QC flag must be "
        "inspected before measurements are used."
    )
    if not _yes_no(
        "Continue with automatic whole-image analysis?",
        input_fn,
        output_fn,
        default=False,
    ):
        return None
    image = _existing_path("Whole image", input_fn, output_fn, directory=False)
    scale = _scale_value(input_fn, output_fn)
    output = _output_path(input_fn, output_fn)
    return [
        "whole-image",
        "--image",
        str(image),
        "--nm-per-pixel",
        scale,
        "--output",
        str(output),
    ]


def _resume(
    input_fn: InputFunction, output_fn: OutputFunction
) -> list[str]:
    while True:
        output = _existing_path(
            "Existing guided output folder", input_fn, output_fn, directory=True
        )
        state_path = output / STATE_FILENAME
        if not state_path.is_file():
            output_fn(f"That folder does not contain {STATE_FILENAME}: {output}")
            continue
        try:
            with state_path.open("r", encoding="utf-8") as stream:
                state = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            output_fn(f"The saved workflow state cannot be read: {error}")
            continue
        mode = state.get("mode")
        if mode not in {"guided_whole_folder", "guided_crop_folder"}:
            output_fn(f"Unsupported guided workflow mode in state: {mode!r}")
            continue
        input_dir = Path(str(state.get("input_dir", ""))).expanduser().resolve()
        if not input_dir.is_dir():
            output_fn(
                "The original input folder recorded by this session no longer exists: "
                f"{input_dir}"
            )
            continue
        break

    records = state.get("sources") or state.get("crops") or []
    scales = {
        float(record["scale_nm_per_px"])
        for record in records
        if record.get("scale_nm_per_px") is not None
    }
    default_scale = next(iter(scales)) if len(scales) == 1 else None
    output_fn(f"Found {mode.replace('_', ' ')} with status {state.get('status', 'unknown')}.")
    scale_args = _guided_scale_arguments(
        input_fn,
        output_fn,
        default_scale=default_scale,
        force_csv=len(scales) > 1,
    )
    if mode == "guided_whole_folder":
        return [
            "guided-whole-folder",
            "--input",
            str(input_dir),
            *scale_args,
            "--output",
            str(output),
        ]
    return [
        "guided-crop-folder",
        "--images",
        str(input_dir),
        *scale_args,
        "--output",
        str(output),
    ]


def _summary(arguments: list[str], output_fn: OutputFunction) -> None:
    mode = arguments[0]
    values = dict(zip(arguments[1::2], arguments[2::2]))
    descriptions = {
        "fiber-crop": "one supplied-mask crop",
        "fiber-folder": "supplied-mask crop folder",
        "guided-crop-folder": "guided masks for existing crops",
        "guided-whole-folder": "guided crops and masks from whole images",
        "whole-image": "automatic whole-image analysis (experimental)",
    }
    output_fn("")
    output_fn("Ready to start")
    output_fn(f"  Workflow: {descriptions[mode]}")
    for key in ("--image", "--images", "--input", "--nm-per-pixel", "--scales-csv", "--output"):
        if key in values:
            output_fn(f"  {key.removeprefix('--').replace('-', ' ')}: {values[key]}")
    preview = subprocess.list2cmdline(["python", "-m", "mvp_pipeline", *arguments])
    output_fn(f"  Equivalent explicit command: {preview}")


def build_wizard_arguments(
    *,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> list[str] | None:
    """Interactively select an existing CLI mode and return its arguments."""

    output_fn("Vacuole-aware myelin workflow wizard")
    output_fn("Enter q at any prompt to cancel safely.")
    try:
        goal = _choice(
            "What would you like to do?",
            [
                ("1", "Analyze existing fiber crops"),
                ("2", "Prepare crops and masks manually from whole images"),
                ("3", "Analyze one whole image automatically (experimental)"),
                ("4", "Resume a guided session"),
            ],
            input_fn,
            output_fn,
            default="2",
        )
        if goal == "1":
            arguments = _existing_crops(input_fn, output_fn)
        elif goal == "2":
            arguments = _guided_whole(input_fn, output_fn)
        elif goal == "3":
            arguments = _automatic_whole(input_fn, output_fn)
            if arguments is None:
                output_fn("Automatic analysis was not started.")
                return None
        else:
            arguments = _resume(input_fn, output_fn)
        _summary(arguments, output_fn)
        if not _yes_no("Start this workflow now?", input_fn, output_fn, default=True):
            output_fn("No workflow was started. Run the wizard again to change a choice.")
            return None
        return arguments
    except WizardCancelled:
        output_fn("")
        output_fn("Wizard cancelled. No workflow was started.")
        return None
