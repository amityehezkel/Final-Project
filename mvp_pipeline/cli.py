from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

from .auto_run import run_automatic_whole_image
from .config import DetectorConfig
from .guided import run_guided_crop_folder, run_guided_whole_folder
from .io import read_binary_mask, read_grayscale
from .run import run_batch
from .resources import install_axondeepseg_model
from .workspace import check_workspace


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "work" / "best_model_results" / "detector_config.json"
DEFAULT_AXONDEEPSEG_MODEL = (
    PROJECT_ROOT / "work" / "models" / "model_seg_generalist_light"
)
IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
MASK_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp"}


def recommended_target_scale(source_scale_nm_per_px: float) -> float:
    """Choose the closest validated AxonDeepSeg inference scale."""

    return 2.36 if source_scale_nm_per_px < 2.0 else 4.93


def _positive_scale(value: str) -> float:
    scale = float(value)
    if scale <= 0:
        raise argparse.ArgumentTypeError("nm/pixel must be greater than zero")
    return scale


def _existing_resource(
    supplied: str | Path | None, default: Path, description: str
) -> Path:
    candidate = Path(supplied).expanduser() if supplied is not None else default
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"{description} does not exist: {candidate}. Supply its path explicitly."
        )
    return candidate


def _safe_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not result:
        raise ValueError("The fiber id must contain at least one letter or number")
    return result


def _portable_path(path: Path, base: Path) -> str:
    """Prefer a relocatable relative path, falling back across Windows drives."""

    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)


def _folder_files(folder: str | Path, extensions: set[str], description: str) -> list[Path]:
    directory = Path(folder).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {directory}")
    files = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ),
        key=lambda path: path.name.lower(),
    )
    if not files:
        raise ValueError(f"{description} contains no supported image files: {directory}")
    return files


def _match_mask(
    image: Path,
    mask_files: list[Path],
    suffixes: tuple[str, ...],
    description: str,
) -> Path:
    accepted_stems = {f"{image.stem}{suffix}".lower() for suffix in suffixes}
    matches = [path for path in mask_files if path.stem.lower() in accepted_stems]
    if not matches:
        expected = ", ".join(sorted(accepted_stems))
        raise FileNotFoundError(
            f"No {description} matches {image.name}. Expected a mask stem such as: "
            f"{expected}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple {description} files match {image.name}: "
            f"{[path.name for path in matches]}"
        )
    return matches[0]


def _write_inference_summary(
    output: Path, summary: dict[str, Any], mode: str, inputs: dict[str, Any]
) -> dict[str, Any]:
    summary["input_mode"] = mode
    summary["inputs"] = inputs
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=True)
    return summary


def run_fiber_crop(
    image_path: str | Path,
    scale_nm_per_px: float,
    axon_mask_path: str | Path,
    outer_fiber_mask_path: str | Path,
    output_dir: str | Path,
    config: DetectorConfig,
    *,
    fiber_id: str | None = None,
) -> dict[str, Any]:
    """Run the shared vacuole detector on one user-masked fiber crop."""

    image_path = Path(image_path).resolve()
    axon_mask_path = Path(axon_mask_path).resolve()
    outer_fiber_mask_path = Path(outer_fiber_mask_path).resolve()
    missing = [
        str(path)
        for path in (image_path, axon_mask_path, outer_fiber_mask_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Fiber-crop input files do not exist: {missing}")
    if scale_nm_per_px <= 0:
        raise ValueError("scale_nm_per_px must be greater than zero")
    if config.detector != "intensity":
        raise ValueError(
            "The direct fiber-crop interface requires the intensity detector because "
            "no compact-myelin mask is supplied."
        )

    image = read_grayscale(image_path)
    axon = read_binary_mask(axon_mask_path, image.shape)
    outer = read_binary_mask(outer_fiber_mask_path, image.shape)
    if not axon.any():
        raise ValueError("The supplied axon mask is empty")
    if not outer.any():
        raise ValueError("The supplied outer-fiber mask is empty")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identifier = _safe_id(fiber_id or image_path.stem)
    manifest = output / "input_manifest.csv"
    pd.DataFrame(
        [
            {
                "id": identifier,
                "source_image_id": image_path.stem,
                "image_path": _portable_path(image_path, output),
                "scale_nm_per_px": float(scale_nm_per_px),
                "axon_mask_path": _portable_path(axon_mask_path, output),
                "outer_fiber_mask_path": _portable_path(outer_fiber_mask_path, output),
                "compact_myelin_mask_path": "",
                "consensus_vacuole_mask_path": "",
                "split": "inference",
                "mask_source": "user_provided",
                "correction_minutes": "",
            }
        ]
    ).to_csv(manifest, index=False)

    _, _, summary = run_batch(manifest, output, config)
    return _write_inference_summary(
        output,
        summary,
        "fiber_crop",
        {
            "image": str(image_path),
            "scale_nm_per_px": float(scale_nm_per_px),
            "axon_mask": str(axon_mask_path),
            "outer_fiber_mask": str(outer_fiber_mask_path),
        },
    )


def run_fiber_folder(
    images_dir: str | Path,
    scale_nm_per_px: float,
    axon_masks_dir: str | Path,
    outer_fiber_masks_dir: str | Path,
    output_dir: str | Path,
    config: DetectorConfig,
) -> dict[str, Any]:
    """Run the shared detector on every matched fiber crop in three folders."""

    if scale_nm_per_px <= 0:
        raise ValueError("scale_nm_per_px must be greater than zero")
    if config.detector != "intensity":
        raise ValueError(
            "The fiber-folder interface requires the intensity detector because "
            "no compact-myelin masks are supplied."
        )

    images_directory = Path(images_dir).resolve()
    axon_directory = Path(axon_masks_dir).resolve()
    outer_directory = Path(outer_fiber_masks_dir).resolve()
    images = _folder_files(images_directory, IMAGE_EXTENSIONS, "Images folder")
    axon_masks = _folder_files(axon_directory, MASK_EXTENSIONS, "Axon-masks folder")
    outer_masks = _folder_files(
        outer_directory, MASK_EXTENSIONS, "Outer-fiber-masks folder"
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for image_path in images:
        axon_path = _match_mask(
            image_path,
            axon_masks,
            ("", "_axon", "_axon_mask"),
            "axon mask",
        )
        outer_path = _match_mask(
            image_path,
            outer_masks,
            ("", "_outer", "_outer_mask", "_outer_fiber", "_outer_fiber_mask"),
            "outer-fiber mask",
        )
        image = read_grayscale(image_path)
        axon = read_binary_mask(axon_path, image.shape)
        outer = read_binary_mask(outer_path, image.shape)
        if not axon.any():
            raise ValueError(f"The axon mask for {image_path.name} is empty: {axon_path}")
        if not outer.any():
            raise ValueError(
                f"The outer-fiber mask for {image_path.name} is empty: {outer_path}"
            )

        identifier = _safe_id(image_path.stem)
        if identifier in identifiers:
            raise ValueError(
                f"Image filenames produce the duplicate output id {identifier!r}"
            )
        identifiers.add(identifier)
        records.append(
            {
                "id": identifier,
                "source_image_id": image_path.stem,
                "image_path": _portable_path(image_path, output),
                "scale_nm_per_px": float(scale_nm_per_px),
                "axon_mask_path": _portable_path(axon_path, output),
                "outer_fiber_mask_path": _portable_path(outer_path, output),
                "compact_myelin_mask_path": "",
                "consensus_vacuole_mask_path": "",
                "split": "inference",
                "mask_source": "user_provided",
                "correction_minutes": "",
            }
        )

    manifest = output / "input_manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    _, _, summary = run_batch(manifest, output, config)
    return _write_inference_summary(
        output,
        summary,
        "fiber_folder",
        {
            "images_dir": str(images_directory),
            "axon_masks_dir": str(axon_directory),
            "outer_fiber_masks_dir": str(outer_directory),
            "scale_nm_per_px": float(scale_nm_per_px),
            "matched_fibers": len(records),
        },
    )


def run_whole_image(
    image_path: str | Path,
    scale_nm_per_px: float,
    output_dir: str | Path,
    config: DetectorConfig,
    *,
    model_path: str | Path | None,
    target_scale_nm_per_px: float | None = None,
    gpu_id: int = -1,
    axon_mask_path: str | Path | None = None,
    myelin_mask_path: str | Path | None = None,
    exclude_scale_bar: bool = True,
) -> dict[str, Any]:
    """Run the whole-image branch and the shared downstream detector."""

    if scale_nm_per_px <= 0:
        raise ValueError("scale_nm_per_px must be greater than zero")
    one_precomputed = (axon_mask_path is None) != (myelin_mask_path is None)
    if one_precomputed:
        raise ValueError("Provide both full-image masks or neither of them")
    target = (
        recommended_target_scale(scale_nm_per_px)
        if target_scale_nm_per_px is None
        else target_scale_nm_per_px
    )
    summary = run_automatic_whole_image(
        image_path,
        scale_nm_per_px,
        output_dir,
        config,
        model_path=model_path,
        target_scale_nm_per_px=target,
        gpu_id=gpu_id,
        axon_mask_path=axon_mask_path,
        myelin_mask_path=myelin_mask_path,
        exclude_scale_bar=exclude_scale_bar,
    )
    summary["input_mode"] = "whole_image"
    summary_path = Path(output_dir).resolve() / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mvp_pipeline",
        description=(
            "Run the vacuole-aware myelin MVP using an automatic, supplied-mask, "
            "or guided interactive workflow."
        ),
    )
    subparsers = parser.add_subparsers(dest="mode")

    subparsers.add_parser(
        "wizard",
        help="Interactively choose a workflow and collect its required inputs",
    )

    doctor = subparsers.add_parser(
        "doctor",
        help="Check whether this copied workspace can run every mode",
    )
    doctor.add_argument(
        "--project-root",
        help="Folder to inspect; defaults to the root containing this package",
    )

    setup = subparsers.add_parser(
        "setup",
        help="Download required external model files into this project workspace",
    )
    setup.add_argument(
        "--force-model",
        action="store_true",
        help="Download and replace the AxonDeepSeg model even when it is valid",
    )

    whole = subparsers.add_parser(
        "whole-image",
        help="Automatic proposal workflow for a whole laboratory TIFF",
    )
    whole.add_argument("--image", required=True, help="Whole laboratory TIFF or PNG")
    whole.add_argument(
        "--nm-per-pixel", required=True, type=_positive_scale, dest="scale"
    )
    whole.add_argument("--output", required=True)
    whole.add_argument("--config", help=f"Default: {DEFAULT_CONFIG}")
    whole.add_argument(
        "--model-path", help=f"Default: {DEFAULT_AXONDEEPSEG_MODEL}"
    )
    whole.add_argument(
        "--target-nm-per-pixel", type=float, choices=(2.36, 4.93), dest="target"
    )
    whole.add_argument("--gpu-id", type=int, default=-1)
    whole.add_argument("--axon-mask", help="Optional precomputed full-image axon mask")
    whole.add_argument(
        "--myelin-mask", help="Optional precomputed full-image compact-myelin mask"
    )
    whole.add_argument("--no-scale-bar-exclusion", action="store_true")

    crop = subparsers.add_parser(
        "fiber-crop",
        help="Analyze one cropped fiber with user-provided masks",
    )
    crop.add_argument("--image", required=True, help="Single-fiber TIFF or PNG crop")
    crop.add_argument(
        "--nm-per-pixel", required=True, type=_positive_scale, dest="scale"
    )
    crop.add_argument("--axon-mask", required=True)
    crop.add_argument("--outer-fiber-mask", required=True)
    crop.add_argument("--output", required=True)
    crop.add_argument("--id", help="Optional identifier used in output filenames")
    crop.add_argument("--config", help=f"Default: {DEFAULT_CONFIG}")

    folder = subparsers.add_parser(
        "fiber-folder",
        help="Analyze a folder of fiber crops and matching mask folders",
    )
    folder.add_argument("--images", required=True, help="Folder containing fiber crops")
    folder.add_argument(
        "--nm-per-pixel", required=True, type=_positive_scale, dest="scale"
    )
    folder.add_argument("--axon-masks", required=True)
    folder.add_argument("--outer-fiber-masks", required=True)
    folder.add_argument("--output", required=True)
    folder.add_argument("--config", help=f"Default: {DEFAULT_CONFIG}")

    guided_whole = subparsers.add_parser(
        "guided-whole-folder",
        help=(
            "Interactively crop every whole image, draw fiber masks, then run "
            "vacuole recognition"
        ),
    )
    guided_whole.add_argument(
        "--input", required=True, help="Folder used as the whole-image input queue"
    )
    guided_whole_scale = guided_whole.add_mutually_exclusive_group(required=True)
    guided_whole_scale.add_argument(
        "--nm-per-pixel", type=_positive_scale, dest="scale"
    )
    guided_whole_scale.add_argument(
        "--scales-csv",
        help="CSV with filename and scale_nm_per_px columns for mixed-scale inputs",
    )
    guided_whole.add_argument("--output", required=True)
    guided_whole.add_argument(
        "--archive-mode",
        choices=("leave", "move", "copy"),
        default="leave",
        help=argparse.SUPPRESS,
    )
    guided_whole.add_argument("--config", help=f"Default: {DEFAULT_CONFIG}")

    guided_crops = subparsers.add_parser(
        "guided-crop-folder",
        help=(
            "Interactively draw masks for existing fiber crops, then run vacuole "
            "recognition"
        ),
    )
    guided_crops.add_argument(
        "--images", required=True, help="Folder containing existing fiber crops"
    )
    guided_crop_scale = guided_crops.add_mutually_exclusive_group(required=True)
    guided_crop_scale.add_argument(
        "--nm-per-pixel", type=_positive_scale, dest="scale"
    )
    guided_crop_scale.add_argument(
        "--scales-csv",
        help="CSV with filename and scale_nm_per_px columns for mixed-scale inputs",
    )
    guided_crops.add_argument("--output", required=True)
    guided_crops.add_argument("--config", help=f"Default: {DEFAULT_CONFIG}")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    supplied = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(supplied)
    used_wizard = args.mode in {None, "wizard"}
    if used_wizard:
        from .wizard import build_wizard_arguments

        selected = build_wizard_arguments()
        if selected is None:
            return
        args = parser.parse_args(selected)

        readiness_key = (
            "guided_workflow_ready"
            if args.mode in {"guided-whole-folder", "guided-crop-folder"}
            else "whole_image_ready"
            if args.mode == "whole-image"
            else "crop_modes_ready"
        )
        print("Checking the local environment before starting...")
        readiness = check_workspace(PROJECT_ROOT)
        if not readiness[readiness_key]:
            print(
                f"The selected workflow is not ready ({readiness_key}=false). "
                "Run 'python -m mvp_pipeline doctor' for details."
            )
            raise SystemExit(1)
        print("Environment check passed.")

    if args.mode == "doctor":
        report = check_workspace(args.project_root or PROJECT_ROOT)
        print(json.dumps(report, indent=2, allow_nan=True))
        if not (
            report["complete_workspace_ready"]
            and report["guided_workflow_ready"]
        ):
            raise SystemExit(1)
        return

    if args.mode == "setup":
        model = install_axondeepseg_model(
            PROJECT_ROOT,
            force=args.force_model,
        )
        report = check_workspace(PROJECT_ROOT)
        print(
            json.dumps(
                {"axondeepseg_model": model, "workspace": report},
                indent=2,
                allow_nan=True,
            )
        )
        if not (
            report["complete_workspace_ready"]
            and report["guided_workflow_ready"]
        ):
            raise SystemExit(1)
        return

    config_path = _existing_resource(args.config, DEFAULT_CONFIG, "Detector config")
    config = DetectorConfig.from_json(config_path)

    if args.mode == "fiber-crop":
        summary = run_fiber_crop(
            args.image,
            args.scale,
            args.axon_mask,
            args.outer_fiber_mask,
            args.output,
            config,
            fiber_id=args.id,
        )
    elif args.mode == "fiber-folder":
        summary = run_fiber_folder(
            args.images,
            args.scale,
            args.axon_masks,
            args.outer_fiber_masks,
            args.output,
            config,
        )
    elif args.mode == "guided-whole-folder":
        summary = run_guided_whole_folder(
            args.input,
            args.output,
            config,
            scale_nm_per_px=args.scale,
            scales_csv=args.scales_csv,
            archive_mode=args.archive_mode,
        )
    elif args.mode == "guided-crop-folder":
        summary = run_guided_crop_folder(
            args.images,
            args.output,
            config,
            scale_nm_per_px=args.scale,
            scales_csv=args.scales_csv,
        )
    elif args.mode == "whole-image":
        using_precomputed = args.axon_mask is not None or args.myelin_mask is not None
        model_path = None
        if not using_precomputed:
            model_path = _existing_resource(
                args.model_path, DEFAULT_AXONDEEPSEG_MODEL, "AxonDeepSeg model"
            )
        elif args.model_path is not None:
            raise ValueError(
                "Do not combine --model-path with precomputed full-image masks"
            )
        summary = run_whole_image(
            args.image,
            args.scale,
            args.output,
            config,
            model_path=model_path,
            target_scale_nm_per_px=args.target,
            gpu_id=args.gpu_id,
            axon_mask_path=args.axon_mask,
            myelin_mask_path=args.myelin_mask,
            exclude_scale_bar=not args.no_scale_bar_exclusion,
        )
    else:  # pragma: no cover - argparse limits this value
        raise ValueError(f"Unsupported mode: {args.mode}")
    print(json.dumps(summary, indent=2, allow_nan=True))
