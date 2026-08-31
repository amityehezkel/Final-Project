from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .io import load_manifest, read_binary_mask
from .metrics import segmentation_metrics


def compute_agreement(
    manifest_path: str | Path,
    annotator_a: str | Path,
    annotator_b: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    rows = load_manifest(manifest_path)
    a_dir, b_dir = Path(annotator_a), Path(annotator_b)
    records: list[dict[str, object]] = []
    for row in rows:
        for layer in ("axon", "outer_fiber", "vacuole"):
            a_path = a_dir / f"{row.id}_{layer}.png"
            b_path = b_dir / f"{row.id}_{layer}.png"
            if not a_path.exists() or not b_path.exists():
                raise FileNotFoundError(f"Missing independent annotations: {a_path} or {b_path}")
            a = read_binary_mask(a_path)
            b = read_binary_mask(b_path, a.shape)
            records.append(
                {
                    "id": row.id,
                    "source_image_id": row.source_image_id or row.image_path.stem,
                    "split": row.split,
                    "layer": layer,
                    **segmentation_metrics(a, b),
                }
            )
    frame = pd.DataFrame(records)
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    summary = frame.groupby("layer")[["dice", "iou", "precision", "recall"]].median()
    summary.to_csv(target.with_name(f"{target.stem}_summary.csv"))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure independent student annotation agreement")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotator-a", required=True)
    parser.add_argument("--annotator-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = compute_agreement(
        args.manifest, args.annotator_a, args.annotator_b, args.output
    )
    print(frame.groupby("layer")["dice"].median().to_string())


if __name__ == "__main__":
    main()

