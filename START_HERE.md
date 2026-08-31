# Start here

## What this project does

This project measures vacuole-like bright spaces inside the myelin sheath of
electron-microscopy fibers. It produces a vacuole mask, a colored overlay, and
physical measurements such as vacuole area, vacuole burden, and g-ratio.

The vacuole detector is the same in every mode. What changes is how the axon
and outer-fiber masks are obtained.

## The workflow in one minute

```text
Input image + nm/pixel
        |
        +-- Whole image: AxonDeepSeg proposes axon/myelin masks
        |                 -> fibers are separated automatically
        |
        +-- One crop: user supplies axon and outer-fiber masks
        |
        +-- Crop folder: user supplies matching folders of both masks
        |
        +-- Guided whole folder: user draws unlimited crops, then two masks
        |
        +-- Guided crop folder: user starts by drawing the two masks
        |
        v
Shared vacuole detector
        |
        v
Vacuole mask + overlay + metrics.csv + summary.json
```

Whole-image mode is convenient but depends strongly on AxonDeepSeg. The two
crop modes are more reliable because the user supplies the fiber boundaries.

## First setup on another computer

Copy the entire `Vacuole-Aware-Myelin-MVP` folder. Open PowerShell inside that
folder and run:

```powershell
conda env create -f environment.yml
conda activate myelin-mvp
python -m mvp_pipeline setup
python -m mvp_pipeline doctor
```

`setup` downloads the official AxonDeepSeg generalist-light checkpoint into
the exact project-relative path used by whole-image mode. The checkpoint is
not committed to GitHub because it is approximately 268 MB. The final command
should report `complete_workspace_ready: true`. The two interactive workflows
also require `guided_workflow_ready: true`.

## Recommended way to start

Run one simple command from the project root:

```powershell
python -m mvp_pipeline
```

The terminal wizard will ask whether you have whole images or fiber crops,
whether masks already exist, whether the scale is shared or supplied in a CSV,
and where results should be written. It clearly marks automatic whole-image
analysis as experimental and requires confirmation before starting it. At the
final screen, verify the displayed workflow and paths, then press Enter to
start.

Choose **Resume a guided session** and select its output folder to continue
from `workflow_state.json`. You can also run the wizard explicitly with
`python -m mvp_pipeline wizard`.

The commands in the next section are the equivalent non-interactive interface.
Keep using them in scripts and when recording an exact reproducible command.

## Choose one input mode

If the crops or masks do not exist yet, use the guided workflow. It opens each
image in sequence, displays instructions, saves progress, and runs detection at
the end. Guided workflows do not move or copy the original input images; input
status is recorded in `results/input_status.csv`. See `GUIDED_WORKFLOW.md` for
the controls and output layout.

### Guided: start with a folder of whole images

```powershell
python -m mvp_pipeline guided-whole-folder `
  --input "path\to\incoming_whole_images" `
  --nm-per-pixel 1.0908 `
  --output "outputs\guided_run"
```

### Guided: start with a folder of crops

```powershell
python -m mvp_pipeline guided-crop-folder `
  --images "path\to\fiber_crops" `
  --nm-per-pixel 1.0908 `
  --output "outputs\guided_run"
```

### Whole laboratory image

```powershell
python -m mvp_pipeline whole-image `
  --image "path\to\image.tif" `
  --nm-per-pixel 1.0908 `
  --output "outputs\whole_image_run"
```

### One cropped fiber

```powershell
python -m mvp_pipeline fiber-crop `
  --image "path\to\fiber.tif" `
  --nm-per-pixel 1.0908 `
  --axon-mask "path\to\fiber_axon.png" `
  --outer-fiber-mask "path\to\fiber_outer_fiber.png" `
  --output "outputs\single_fiber_run"
```

### A folder of cropped fibers

```powershell
python -m mvp_pipeline fiber-folder `
  --images "path\to\images" `
  --nm-per-pixel 1.0908 `
  --axon-masks "path\to\axon_masks" `
  --outer-fiber-masks "path\to\outer_fiber_masks" `
  --output "outputs\fiber_folder_run"
```

For folder mode, matching files should share the image stem, for example
`fiber_01.tif`, `fiber_01_axon.png`, and `fiber_01_outer_fiber.png`.

## What to inspect after a run

- `overlays/`: blue is axon, red is the outer-fiber boundary, yellow is the
  predicted vacuole.
- `masks/`: binary vacuole masks for later analysis.
- `metrics.csv`: one row per fiber with physical measurements.
- `summary.json`: run settings, provenance, and evaluation information when
  reference annotations are available.
- `input_status.csv`: for guided runs, shows which original inputs were
  processed, skipped, or are still pending.
- `whole_image_overlay.png`: combined overview produced by whole-image mode.

Always visually inspect the overlays. These are candidate measurements, not a
clinical or expert-validated classifier.

## Where the important project material lives

- `mvp_pipeline/`: program code.
- `work/best_model_results/`: active detector configuration and benchmark
  results.
- `work/benchmark/`: 43-fiber benchmark and reference masks.
- `work/benchmark_three_mode_overlays/`: annotation, crop-mode, and
  whole-image-mode comparison for every benchmark fiber.
- `work/model_history/`: intentionally preserved previous official models.
- `work/models/`: locally downloaded AxonDeepSeg model needed by whole-image
  mode; `python -m mvp_pipeline setup` creates it after a GitHub clone.
- `TECHNICAL_MODEL_GUIDE.md`: detailed algorithm and model-development logic.
- `GUIDED_WORKFLOW.md`: step-by-step interactive crop-and-mask instructions.
- `PROJECT_NAVIGATION_AND_SHARING_GUIDE.md`: complete folder map and sharing
  instructions.

## Current result in plain language

On 21 eligible test fibers in cropped mode, the current detector reached
median Dice 0.752, median precision 0.930, and median recall 0.714. It reduced
complete missed positive fibers from five to one without increasing the
compact-fiber false-positive count. Whole-image performance is lower because
automatic fiber-mask extraction is currently the main bottleneck.
