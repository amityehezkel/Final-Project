# Project Navigation and Sharing Guide

This guide explains where everything is, which files are important, where commands must be run, and what to share with another group member.

## 1. Project root

On Yaniv’s computer, the project root is:

```text
C:\Users\yaniv\OneDrive - mail.tau.ac.il\שנה ג\פרוייקט\Vacuole-Aware-Myelin-MVP
```

The **project root** means the folder that directly contains:

```text
README.md
pyproject.toml
mvp_pipeline/
tests/
examples/
work/
```

All documented commands should be run from this directory. A partner’s absolute path does not have to be the same. For example, if a partner copies the project to `D:\University\MyelinProject`, that folder becomes their project root.

PowerShell example:

```powershell
cd "C:\path\to\פרוייקט"
conda activate astih
python -m mvp_pipeline --help
```

Always put a path containing spaces or Hebrew characters inside quotation marks.

## 2. Top-level folder map

```text
פרוייקט/
├── mvp_pipeline/              Python implementation
├── tests/                     Automated tests
├── examples/                  Example source-list and comparison templates
├── work/                      Annotations, models, final results, and evidence
├── דאטה מהמעבדה/              Original laboratory TIFF images
├── מיילין פגוע/               Curated damaged-myelin images
├── ASTIH/                     External ASTIH dataset and its own files
├── README.md                  Commands and concise technical instructions
├── START_HERE.md              One-page orientation and first commands
├── TECHNICAL_MODEL_GUIDE.md   Detailed algorithm and Methods explanation
├── GUIDED_WORKFLOW.md        Graphical crop/mask workflow instructions
├── GROUP_WORKFLOW_GUIDE.md    Detailed explanation of the pipeline logic
├── LABELING_GUIDE.md          Rules for drawing the three masks
├── pyproject.toml             Python package and dependency definition
├── Project Proposal - Noy, Roni, Amit.pdf
├── sciadv.adl4573.pdf         Main vacuolation article
├── ENEURO.0558-20.2021.full.pdf
├── תיוג תמונות.pdf
└── תיוג תמונות.docx
```

The complete local root is currently approximately 4.8 GB. Most of that size
comes from laboratory data, ASTIH, generated experiments, and the AxonDeepSeg
model. These large local resources are deliberately excluded from GitHub.

## Portable workspace status

The existing project root is the complete local workspace; we do not maintain
a second duplicate workspace. A GitHub clone becomes a complete runnable
workspace after its one-time setup command. Portability is handled as follows:

- runtime defaults locate `work/best_model_results/detector_config.json` and `work/models/model_seg_generalist_light/` relative to the copied project root;
- the benchmark manifest and crop-plan paths are relative rather than tied to Yaniv's Windows account;
- `environment.yml` declares the complete Conda/Python environment;
- `pyproject.toml` declares AxonDeepSeg 5.3.0 in the `whole-image` and `full` optional dependency groups;
- `python -m mvp_pipeline setup` downloads the official generalist-light model
  to the expected project-relative path;
- `python -m mvp_pipeline doctor` checks the software dependencies and runtime
  resources after setup.

Result `summary.json` files may display the computer path on which an old experiment ran. Those paths are provenance only and are not read by new inference commands.

## 3. Code folders

### `mvp_pipeline/`

This is our program. Important files include:

- `cli.py`: all command-line entry points and input validation;
- `guided.py`: resumable whole-image crop selection, two-mask annotation,
  source archiving, and automatic handoff to vacuole recognition;
- `auto_run.py`: automatic whole-image route;
- `segment_scale.py`: scale-normalized AxonDeepSeg wrapper;
- `instances.py`: separation and extraction of individual fibers;
- `detectors.py`: geometry and intensity vacuole detectors;
- `run.py`: shared batch detector and output writer;
- `metrics.py`: physical areas, burden, and g-ratio formulas;
- `masks.py`: mask validation and physical-unit utilities;
- `overlay.py`: visual result generation;
- `evaluation.py`: Dice, IoU, error metrics, plots, and examples;
- `annotate.py`, `select_crops.py`, and `crops.py`: benchmark preparation tools.

Ordinary users should not need to open these files to run the program.

### `tests/`

These tests check formulas, scale conversion, detectors, automatic extraction,
crop selection helpers, the three analysis modes, and the guided state/output
workflow.

Run them from the project root:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q
```

### `examples/`

- `source_images.csv`: images offered to the interactive crop selector;
- templates for scale-segmentation comparisons.

These are examples and preparation records, not final detector results.

## 4. Data folders

### `דאטה מהמעבדה/`

Original JEOL laboratory TIFFs. These files contain microscope calibration metadata and embedded scale bars. Treat this folder as read-only.

### `מיילין פגוע/`

Curated images showing damaged or vacuolated myelin. Treat this folder as read-only.

### `ASTIH/`

External ASTIH data and its accompanying repository files. ASTIH was useful for understanding axon/myelin segmentation, but it is not vacuole ground truth. It is not required for routine inference with our final detector.

Do not place our generated outputs in any of these three folders. New outputs belong under `work/`.

## 5. The `work/` folder

`work/` contains project-created artifacts. After cleanup, the important items are:

### Source annotations and selection records

- `benchmark/`: expanded 43-fiber manifest and masks; `benchmark_v1_24.csv` preserves the original split;
- `annotations/`: saved annotation masks;
- `annotation_results/`: visual representations of the annotations;
- `crop_plan.csv`: selected crop coordinates and class assignments;
- `crop_plan.progress.json`: completed selection state.

These are important project data and should be backed up.

### Current detector

- `best_model_results/detector_config.json`: the frozen vacuole-detector
  parameters used by every analysis and guided mode;
- `best_model_results/`: final test masks, overlays, measurements, and summary;
- `tuning_expanded_v2/`: development-only 275-configuration tuning record for the current detector;
- `tuning_hysteresis_v3/`: archived tuning record for the former detector;
- `min_area_analysis_v5/`, `boundary_refinement_v6/`, and `thin_rescue_v7/`: preserved development evidence for the paper;
- `model_history/`: previous official configurations with their masks and overlays;
- `model_comparison.csv`: comparison of explored detector versions.

Do not replace `detector_config.json` with an older tuning configuration.

### AxonDeepSeg

- `models/model_seg_generalist_light/`: downloaded AxonDeepSeg model used only by whole-image mode.

This folder is large, approximately 256 MB. Crop and folder modes do not need it because their masks are supplied by the user.

### Final validation evidence

- `crop_ads_validation/`: evidence that direct AxonDeepSeg inference on tight crops was rejected;
- `automatic_multi_image/`: final whole-image automatic tests, overlays, comparison table, and report.
- `benchmark_three_mode_overlays/`: one subfolder per benchmark fiber, containing the crop, the arrow-marked whole source image, the annotation overlay, and both mode results;
- `separate_mode_results/`: independent cropped-mode and whole-image-front-end summaries, tables, plots, and overlays.

These folders are not required to run new images, but they support our conclusions and presentation.

The arrow-marked whole images in `benchmark_three_mode_overlays/` are provenance aids. The arrows were added manually to show where each benchmark crop came from; these images were not model inputs and the arrows are not detections. Source filenames vary, so distinguish the small crop from the normally 1872 × 1872 contextual image by dimensions and by the visible arrow.

### Annotation assistance

- `candidate_sheets/`;
- `classification_guide/`;
- `external_myelin_examples/`.

These contain image sheets and examples used to understand compact versus vacuolated myelin.

## 6. Where to start for each task

| Task | Start here |
|---|---|
| Get a one-page orientation | `START_HERE.md` |
| Understand the algorithm and paper Methods | `TECHNICAL_MODEL_GUIDE.md` |
| Create crops and masks through the graphical workflow | `GUIDED_WORKFLOW.md` |
| Read the longer internal workflow history | `GROUP_WORKFLOW_GUIDE.md` |
| See all run commands | `README.md` |
| Understand mask definitions | `LABELING_GUIDE.md` |
| Run one masked crop | `python -m mvp_pipeline fiber-crop ...` |
| Run a folder of crops | `python -m mvp_pipeline fiber-folder ...` |
| Create unlimited crops and masks from whole images | `python -m mvp_pipeline guided-whole-folder ...` |
| Create masks for an existing crop folder | `python -m mvp_pipeline guided-crop-folder ...` |
| Try automatic whole-image processing | `python -m mvp_pipeline whole-image ...` |
| Inspect final detector parameters | `work/best_model_results/detector_config.json` |
| Inspect benchmark performance | `work/best_model_results/summary.json` and its overlays |
| Compare both modes fiber by fiber | `work/benchmark_three_mode_overlays/` |
| Compare cropped and automatic results quantitatively | `work/separate_mode_results/mode_comparison_summary.csv` |
| Inspect AxonDeepSeg limitations | `work/crop_ads_validation/REPORT.md` |
| Inspect whole-image tests | `work/automatic_multi_image/REPORT.md` |
| Modify implementation | `mvp_pipeline/` |
| Verify changes | `tests/` |

## 7. What to share

### Option A: complete project copy

If the goal is for a partner to access **everything exactly as it exists now**, share the entire project-root folder:

```text
פרוייקט/
```

This is the simplest and least error-prone option. It includes code, data, models, annotations, papers, and final evidence. The current size is approximately 4.8 GB, so use OneDrive, Google Drive, or another shared-storage service rather than email.

If sharing through OneDrive, make sure partners receive permission for the root and all children. They may need to select “Always keep on this device” before running code so TIFFs and model files are downloaded rather than online-only placeholders.

### Option B: recommended split between GitHub and shared storage

GitHub is useful for code and documentation, but the raw TIFF collections and model files are large. A clean division is:

#### Put in GitHub

```text
mvp_pipeline/
tests/
examples/
README.md
START_HERE.md
TECHNICAL_MODEL_GUIDE.md
GUIDED_WORKFLOW.md
GROUP_WORKFLOW_GUIDE.md
PROJECT_NAVIGATION_AND_SHARING_GUIDE.md
LABELING_GUIDE.md
pyproject.toml
.gitignore
work/best_model_results/
work/tuning_hysteresis_v3/
work/model_comparison.csv
```

If project policy permits sharing the annotated crops, also include:

```text
work/benchmark/
work/annotations/
work/annotation_results/
work/crop_plan.csv
```

#### Put in shared Drive/OneDrive storage

```text
דאטה מהמעבדה/
מיילין פגוע/
ASTIH/
work/models/
work/automatic_multi_image/
work/crop_ads_validation/
work/candidate_sheets/
work/classification_guide/
work/external_myelin_examples/
PDF and DOCX reference documents
```

The repository README should then tell partners where to place the downloaded large folders relative to the project root.

Before sharing laboratory data or student annotations outside the group, confirm that the laboratory and university permit it. Do not upload restricted or unpublished data to a public GitHub repository.

## 8. Minimum files for each operating mode

### Single-crop and folder modes

Required project files:

```text
mvp_pipeline/
pyproject.toml
work/best_model_results/detector_config.json
```

The user must additionally have the crop images, axon masks, outer-fiber masks, and nm/pixel value. `tests/` and the guides are strongly recommended but not required at runtime.

### Whole-image mode

Required project files:

```text
mvp_pipeline/
pyproject.toml
work/best_model_results/detector_config.json
```

The user must also have the input TIFF. The declared environment installs the
AxonDeepSeg package, and `python -m mvp_pipeline setup` downloads its model into
`work/models/model_seg_generalist_light/`. The entire ASTIH folder is not
required.

### Full reproduction of our reported results

Also share:

```text
work/benchmark/
work/annotations/
work/best_model_results/
work/tuning_hysteresis_v3/
work/crop_ads_validation/
work/automatic_multi_image/
work/benchmark_three_mode_overlays/
work/separate_mode_results/
```

## 9. Setup on another computer

From the partner’s copy of the project root:

```powershell
conda env create -f environment.yml
conda activate myelin-mvp
python -m mvp_pipeline setup
python -m mvp_pipeline doctor
python -m pytest -q
```

When `doctor` prints `complete_workspace_ready: true`, the three analysis modes
and benchmark resources are available. The graphical workflows additionally
require `guided_workflow_ready: true`. If only noninteractive crop modes are
needed, `crop_modes_ready: true` is sufficient.

As a lighter alternative without Conda:

```powershell
python -m pip install -e ".[full]"
python -m mvp_pipeline setup
python -m mvp_pipeline doctor
```

For crop-only interactive Napari annotation without AxonDeepSeg, install the
annotation extras:

```powershell
python -m pip install -e ".[annotation,test]"
```

Whole-image mode additionally requires the AxonDeepSeg package and downloaded
model folder. Crop modes do not import AxonDeepSeg.

## 10. Folder hygiene rules

- Treat the three source-data folders and `work/models/` as read-only after
  setup.
- Put every new experiment in a clearly named subfolder under `work/`.
- Do not overwrite `work/best_model_results/`.
- Do not replace the frozen detector config unless the group intentionally starts a new evaluated model version.
- Do not commit `__pycache__`, `.pytest_cache`, temporary renders, or one-off smoke outputs.
- Keep paths relative when creating shared CSV manifests whenever possible.
- Inspect overlays before using any measurements.
