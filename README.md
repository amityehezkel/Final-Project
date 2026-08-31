# Vacuole-Aware Myelin Measurement MVP

This repository contains a small, reproducible pipeline for candidate-vacuole detection and vacuole-aware myelin measurements in electron-microscopy images. It includes an automatic whole-image proposal workflow using scale-normalized AxonDeepSeg, plus the manual benchmark workflow needed to evaluate it honestly.

New group members should begin with [START_HERE.md](START_HERE.md). For the
algorithm and paper-oriented Methods details, read
[TECHNICAL_MODEL_GUIDE.md](TECHNICAL_MODEL_GUIDE.md). The longer
[PROJECT_NAVIGATION_AND_SHARING_GUIDE.md](PROJECT_NAVIGATION_AND_SHARING_GUIDE.md)
explains the complete folder layout and sharing options.
For the new resumable graphical workflow, use
[GUIDED_WORKFLOW.md](GUIDED_WORKFLOW.md).

The output is validated against student-consensus annotations. It is not expert, biological, clinical, or diagnostic validation.

## What is implemented

- Original interactive selection of 24 axon crops, subsequently expanded to 43 labeled fibers (20 development and 23 test), always split by source image.
- Independent Napari annotation and consensus-review workflow.
- Inter-annotator agreement measurement.
- Physical-scale image resampling and an optional AxonDeepSeg wrapper for 2.36 and 4.93 nm/pixel.
- Raw-versus-rescaled segmentation comparison with the 0.75 Dice / 3-minute stop rule.
- Geometry and CLAHE/Otsu intensity vacuole detectors.
- Development-only parameter selection and frozen test evaluation.
- Per-axon physical measurements, masks, overlays, summary JSON, evaluation CSV, plots, and ranked examples.
- Synthetic unit and end-to-end tests.
- One-command whole-image inference: AxonDeepSeg, axon-seeded watershed fiber separation, automatic internal crops, vacuole detection, QC, overlays, and measurements.
- One user-facing command with automatic, supplied-mask, and guided graphical
  input modes; all converge on the same frozen vacuole detector and measurement
  code.
- Resumable sequential Napari workflow for unlimited crops per whole image,
  validated axon/outer-fiber masks, source archiving, and automatic detector
  execution.

## Environment

### Recommended setup on a new computer

The project root is now the portable workspace. Copy the entire root, then create the declared environment from inside the copied folder:

```powershell
conda env create -f environment.yml
conda activate myelin-mvp
python -m mvp_pipeline setup
python -m mvp_pipeline doctor
```

The environment installs AxonDeepSeg 5.3.0, Napari 0.9.0, and a supported
PyQt6 interface backend. AxonDeepSeg's neural-network checkpoint is a separate
268-MB download and is intentionally not stored in GitHub. The `setup` command
downloads the official generalist-light model into
`work/models/model_seg_generalist_light/`, which is the default path used by
the program. It is safe to run `setup` again; an already valid model is kept.

`doctor` checks the core Python dependencies, frozen detector configuration,
AxonDeepSeg installation, Napari/Qt installation, model files, and relocatable
benchmark paths. A complete clone should report both
`complete_workspace_ready: true` and `guided_workflow_ready: true`.

The repository's `.gitignore` deliberately excludes raw laboratory data,
personal test runs, generated overlays, local environments, and the large
AxonDeepSeg checkpoint. It keeps the source code, tests, documentation, frozen
configuration, compact 43-fiber benchmark, examples, and final course
deliverables.

The user-facing inference commands do not contain Yaniv's absolute project
path. Their default detector configuration and AxonDeepSeg model are resolved
relative to the copied project root. Existing result summaries may retain
original absolute paths as provenance text, but they are not runtime inputs.

### Existing development environment

Run commands from this project directory. The existing `astih` conda environment already contains the required packages:

```powershell
conda activate astih
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest
```

Alternatively, create a clean environment and install the project:

```powershell
python -m pip install -e ".[full]"
python -m mvp_pipeline setup
```

The DANDI package installed in `astih` registers an unrelated pytest plugin. `-p no:dandi` or `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` prevents that plugin from creating external cache files during these tests.

## Unified MVP command

For a normal interactive run, start the workflow wizard from the project root:

```powershell
python -m mvp_pipeline
```

The wizard asks what material is available, requests only the paths and scale
information needed for that route, warns before the experimental automatic
mode, and shows the equivalent explicit command before anything starts. It can
also resume a saved guided session. `python -m mvp_pipeline wizard` starts the
same interface explicitly.

The established subcommands below remain available for scripts, reproducible
commands, and advanced use. List them without starting the wizard with:

```powershell
python -m mvp_pipeline --help
```

### Option 1: whole laboratory image

From this project folder, the frozen detector configuration and downloaded AxonDeepSeg model are selected automatically:

```powershell
python -m mvp_pipeline whole-image `
  --image "path\to\laboratory_image.tif" `
  --nm-per-pixel 5.523 `
  --output "work\whole_image_result"
```

The program chooses 2.36 nm/pixel for source scales below 2 nm/pixel and 4.93 nm/pixel otherwise. `--target-nm-per-pixel`, `--model-path`, and `--config` are available as advanced overrides. This route is an experimental proposal workflow because AxonDeepSeg mask generation did not pass the frozen cropped-image acceptance test. Inspect its QC flags and overlays before using measurements.

### Option 2: one masked fiber crop

```powershell
python -m mvp_pipeline fiber-crop `
  --image "path\to\fiber_crop.tif" `
  --nm-per-pixel 5.523 `
  --axon-mask "path\to\fiber_axon_mask.png" `
  --outer-fiber-mask "path\to\fiber_outer_mask.png" `
  --output "work\fiber_result"
```

The image and both binary masks must have identical dimensions. The outer-fiber mask is the complete gross fiber envelope, including the axon; the program defines the searched sheath as `outer_fiber & ~axon`. Empty masks and dimension mismatches stop with a clear error.

### Option 3: folder of masked fiber crops

Put the crops and masks in three separate folders, then run:

```powershell
python -m mvp_pipeline fiber-folder `
  --images "input\images" `
  --nm-per-pixel 5.523 `
  --axon-masks "input\axon_masks" `
  --outer-fiber-masks "input\outer_fiber_masks" `
  --output "work\folder_result"
```

Every file in `images` must have exactly one matching mask in each mask folder. Matching is case-insensitive and accepts either the same stem in separate folders or these suffixes:

- axon: `fiber01.png`, `fiber01_axon.png`, or `fiber01_axon_mask.png`;
- outer fiber: `fiber01.png`, `fiber01_outer.png`, `fiber01_outer_mask.png`, `fiber01_outer_fiber.png`, or `fiber01_outer_fiber_mask.png`.

Supported crop formats are PNG, TIFF, JPEG, and BMP; supported mask formats are PNG, TIFF, and BMP. Only files directly inside each folder are scanned. All crops in one folder run must have the supplied common physical scale. Missing masks, ambiguous matches, empty masks, and dimension mismatches stop the batch before detector execution.

All three analysis modes call the same frozen intensity/hysteresis vacuole
detector. They write `metrics.csv`, `summary.json`, predicted vacuole masks,
and overlays. Both supplied-mask crop modes also write `input_manifest.csv`
internally, so the user does not need to create a CSV. The guided modes create
the same manifest after mask annotation and place detector files under their
`results/` folder.

After installing the project, `myelin-mvp` is an equivalent short command, for example `myelin-mvp fiber-crop ...`.

### Options 4 and 5: guided crop-and-mask workflows

Use these when the user does not already have both masks. The whole-folder
route first opens every source image for unlimited rectangular crop selection;
the crop-folder route begins directly with mask creation. Both save progress
and start vacuole recognition after the final annotation:

```powershell
python -m mvp_pipeline guided-whole-folder `
  --input "path\to\whole_images" `
  --nm-per-pixel 5.523 `
  --output "work\guided_run"

python -m mvp_pipeline guided-crop-folder `
  --images "path\to\fiber_crops" `
  --nm-per-pixel 5.523 `
  --output "work\guided_crop_run"
```

The graphical application is Napari; AxonDeepSeg is an inference library, not
the annotation GUI. See `GUIDED_WORKFLOW.md` for controls, mixed-scale CSVs,
safe pause/resume, read-only input handling, and the output tree. Guided runs
leave original whole images and crop images in their input folders and record
processed/skipped status in `results/input_status.csv`.

## Automatic whole-image inference

This is the deployment command for a new laboratory image. It does **not** ask the user to select crops or draw masks:

```powershell
python -m mvp_pipeline.auto_run `
  --image "דאטה מהמעבדה\p120-2\NeoView_2.tif" `
  --source-scale 5.523 `
  --target-scale 4.93 `
  --model-path "work\models\model_seg_generalist_light" `
  --config "work\best_model_results\detector_config.json" `
  --output "work\automatic_NeoView_2"
```

The command:

1. creates a scale-normalized inference copy and runs AxonDeepSeg;
2. restores axon and myelin predictions to the original image resolution;
3. separates touching fibers with axon-seeded watershed;
4. rejects small axon components, border fibers, and scale-bar candidates;
5. creates per-fiber crops and masks internally;
6. runs the frozen vacuole detector on the original-resolution crops;
7. writes `metrics.csv`, `whole_image_overlay.png`, per-fiber overlays and masks, an automatically generated manifest, and `summary.json`.

`watershed_split_touching_cluster` is informational: it records that adjacent fibers were separated automatically. `low_myelin_coverage`, `irregular_axon_shape`, and `large_axon_area_outlier` set `manual_review_recommended=true`. Manual review is therefore targeted to suspicious outputs instead of being required for every fiber.

If full-image AxonDeepSeg masks already exist, skip inference and supply them directly:

```powershell
python -m mvp_pipeline.auto_run `
  --image "path\to\image.tif" `
  --source-scale 5.523 `
  --axon-mask "path\to\axon_mask.png" `
  --myelin-mask "path\to\myelin_mask.png" `
  --config "work\best_model_results\detector_config.json" `
  --output "work\automatic_output"
```

The automatic path is a proposal workflow until its AxonDeepSeg axon and outer-fiber masks pass the segmentation acceptance experiment described below. Incorrect upstream masks directly affect vacuole areas and g-ratios.

## 1. Select and create the benchmark

The supplied source list is a larger pool of 5.523 nm/pixel images from at least three lab folders. The 1.0908 nm/pixel close-ups remain part of the scale-normalization experiment and qualitative examples, but are not forced into the 24-axon benchmark because they often contain too few complete fibers:

```powershell
python -m mvp_pipeline.select_crops `
  --sources examples/source_images.csv `
  --output work/crop_plan.csv
```

For every source image, draw between zero and the maximum shown in the title. Close the window without drawing to skip an unsuitable image. Put rectangles in the `vacuolated_crops` or `compact_crops` layer. Progress is saved after every window, and rerunning the same command resumes automatically. The selector enforces:

- 24 crops total;
- 12 apparently vacuolated and 12 apparently compact;
- 8 development crops and 16 test crops with no image shared across splits;
- at least two development and four test source images;
- no more than four crops from one source image.

Create the crops, blank consensus masks, and batch manifest:

```powershell
python -m mvp_pipeline.crops `
  --plan work/crop_plan.csv `
  --output work/benchmark
```

The original resulting manifest contained 24 fibers. The current official
`work/benchmark/benchmark.csv` appends 19 additional labeled crops from
`few_tests/`, giving 43 fibers: 20 development and 23 test. The untouched
original manifest is preserved as `work/benchmark/benchmark_v1_24.csv`, and
`examples/extra_labeled_crops.csv` records exactly how the expansion was made.

## 2. Annotate independently and resolve consensus

Read [LABELING_GUIDE.md](LABELING_GUIDE.md) before annotating. These historical commands walk through the original 24 crops, opening one Napari window at a time. The 19 later single-student annotations are already represented in the expanded manifest:

```powershell
python -m mvp_pipeline.annotate `
  --manifest work/benchmark/benchmark.csv `
  --output work/annotations/annotator_A

python -m mvp_pipeline.annotate `
  --manifest work/benchmark/benchmark.csv `
  --output work/annotations/annotator_B
```

Compute agreement before consensus resolution:

```powershell
python -m mvp_pipeline.agreement `
  --manifest work/benchmark/benchmark.csv `
  --annotator-a work/annotations/annotator_A `
  --annotator-b work/annotations/annotator_B `
  --output work/annotation_agreement.csv
```

The third student creates the consensus masks while viewing both independent references:

```powershell
python -m mvp_pipeline.annotate `
  --manifest work/benchmark/benchmark.csv `
  --output work/benchmark/masks/consensus `
  --reference-a work/annotations/annotator_A `
  --reference-b work/annotations/annotator_B
```

Do not inspect detector predictions during consensus labeling.

## 3. Optional AxonDeepSeg scale experiment

The wrapper rescales only the inference copy, runs an external AxonDeepSeg model, restores masks to the original dimensions with nearest-neighbor interpolation, removes the bottom-right scale-bar region, and creates a filled `outer_fiber` proposal from the combined axon/myelin mask. It does not edit AxonDeepSeg.

```powershell
python -m mvp_pipeline.segment_scale `
  --image "path\to\NeoView.tif" `
  --source-scale 1.0908 `
  --target-scale 4.93 `
  --model-path "path\to\model_seg_generalist_light" `
  --output work/scale_experiment
```

Repeat at 2.36 nm/pixel and for the raw image. Populate `examples/segmentation_comparison_template.csv`, then evaluate:

```powershell
python -m mvp_pipeline.segmentation_eval `
  --table work/segmentation_comparison.csv `
  --output work/segmentation_evaluation
```

Use automatic masks only when both median axon and outer-fiber Dice are at least 0.75 and median correction time is at most three minutes. Otherwise use the manual consensus masks and stop spending time on AxonDeepSeg.

## 4. Tune on development data and freeze

The final expanded tuner evaluates 440 intensity configurations: physical
minimum areas of 0.0015–0.01 µm², high offsets of 0.15–0.25, low offsets of
0.075–0.20, and morphology radii of 0.01–0.03 µm. Expensive normalization is
cached, and large circular morphology uses a distance transform so the
1.0908 nm/pixel images remain practical.

```powershell
python -m mvp_pipeline.tune `
  --manifest work/benchmark/benchmark.csv `
  --output work/tuning
```

This writes `work/tuning/detector_config.json`. If development median Dice is below 0.50, `tuning_summary.json` sets `manual_correction_required` to true.

The selected official configuration is stored in
`work/best_model_results/detector_config.json`: minimum area 0.01 µm², high
offset 0.20, low offset 0.10, Gaussian sigma 0.02 µm, and morphology radius
0.02 µm. A development-tuned boundary-refinement stage then grows each
existing detection by at most 0.05 µm through connected pixels down to
`Otsu - 0.05`, with a maximum area ratio of 1.5.

A second development-selected branch rescues plausible thin seeds that the
main 0.02 µm morphology removes. It uses gentler 0.005 µm morphology and only
accepts components with area at least 0.0015 µm², thickness at least 0.045 µm,
radial position at most 0.55 from the axon toward the outer boundary,
eccentricity at most 0.95, and solidity at least 0.85. Unlike boundary
refinement, this branch can restore a missing vacuole object; its radial and
shape guardrails are intended to reject bright outer-boundary halos.

The refinement grid is reproducible with:

```powershell
python -m mvp_pipeline.tune_refinement `
  --manifest work\benchmark\benchmark.csv `
  --config work\model_history\pre_boundary_refinement_v5\detector_config.json `
  --output work\boundary_refinement_v6\base_0p01
```

### Thin-cleft minimum-area analysis

Thin clefts smaller than 0.01 µm² are biologically included in this project.
The detector's minimum area is only a noise-control parameter; it is not the
definition of a vacuole. Connected-component measurements from the manual
development masks show that 0.0015 µm² is a logical permissive cutoff: it is
the largest tested cutoff that retains at least 95% of annotated development
vacuole area (95.7%).

Lowering only this filter to 0.0015 µm² changed the held-out test
trade-off from 5 complete false negatives and 1/7 compact false positives to
0 complete false negatives and 5/7 compact false positives. The intermediate
0.005 µm² candidate gave 2 complete false negatives, 3/7 compact false
positives, median Dice 0.678, and median recall 0.600 in a descriptive test
comparison. Therefore the main filter remains 0.01 µm². The new thin-seed
branch instead uses the logical 0.0015 µm² cutoff together with thickness,
radial-position, eccentricity, and solidity constraints; it reduced complete
test misses from 5 to 1 without increasing the 1/7 compact false-positive
count.

The auditable component table and isolated sweep are in
`work/min_area_analysis_v5/isolated_area_sweep/`. Reproduce them with:

```powershell
python -m mvp_pipeline.min_area_analysis `
  --manifest work\benchmark\benchmark.csv `
  --config work\model_history\pre_boundary_refinement_v5\detector_config.json `
  --output work\min_area_analysis_v5\isolated_area_sweep
```

## 5. Run the frozen test evaluation

```powershell
python -m mvp_pipeline.run `
  --manifest work/benchmark/benchmark.csv `
  --config work/tuning/detector_config.json `
  --split test `
  --output work/test_outputs
```

On the expanded held-out split, 21 fibers passed QC. The current detector
achieved median Dice 0.752, median precision 0.930, median recall 0.714, and
median total-vacuole-area percentage error 20.3%. This is a successful proof
of concept against the student reference annotations, not an expert-validated
biological classifier.

The originally requested command also works, using the documented intensity defaults:

```powershell
python -m mvp_pipeline.run --manifest benchmark.csv --output outputs
```

Important outputs:

- `metrics.csv`: per-axon measurements and QC flags;
- `evaluation_per_axon.csv`: Dice, IoU, precision, recall, and vacuole-area error;
- `summary.json`: split-level medians and interpretation;
- `evaluation_overview.png`: overlap and area-agreement plots;
- `masks/` and `overlays/`: inspectable outputs;
- `examples/`: three highest- and three lowest-Dice overlays.

Rows touching an image border are marked `excluded_from_summary` and do not affect headline metrics. Batch crop configurations leave bottom-right exclusion disabled because crop selection already rejects scale-bar-obscured axons. Set `exclude_scale_bar` to `true` only when a manifest contains uncropped images with the embedded scale bar.

## Separate evaluation by input mode

Do not combine the cropped-mode and whole-image-mode headline scores. They answer different questions:

- **Cropped mode** uses the supplied axon and outer-fiber masks, so its evaluation primarily measures the frozen vacuole detector.
- **Whole-image mode** generates those masks with AxonDeepSeg, so its end-to-end result includes fiber-extraction misses and mask errors as well as vacuole-detector errors.

The packaged results are in `work/separate_mode_results/`, with independent `cropped_mode/` and `whole_image_mode/` folders and a side-by-side `mode_comparison_summary.csv`. The whole-image benchmark holds each annotated field of view fixed so all 43 fibers can be scored one-to-one. It therefore validates the automatic-mask front end used by whole-image mode, but is not a rematching study on the original full laboratory TIFFs; full-TIFF deployment remains an experimental proposal workflow.

For visual review and source traceability, `work/benchmark_three_mode_overlays/` contains one subfolder per benchmark fiber. Each subfolder includes the original fiber crop, its arrow-marked location in the whole laboratory image, the manual annotation overlay, the cropped-mode result, the controlled whole-image-front-end result, and a three-way comparison. The arrow-marked whole image is contextual documentation only: it was not supplied to the controlled benchmark inference or used for scoring.

Rebuild the separate packages with:

```powershell
python -m mvp_pipeline.mode_results `
  --benchmark work\benchmark\benchmark.csv `
  --cropped-results work\best_model_results `
  --whole-results work\benchmark_three_mode_overlays\_whole_mode_results `
  --automatic-mask-validation work\benchmark_whole_mode_masks `
  --output work\separate_mode_results
```

## Measurement definitions

For each axon:

```text
gross_sheath = outer_fiber - axon
intact_myelin = gross_sheath - vacuoles
vacuole_burden = vacuole_area / gross_sheath_area
g_ratio = sqrt(axon_area / outer_fiber_area)
intact_equivalent_g_ratio = sqrt(axon_area / (outer_fiber_area - vacuole_area))
```

`intact_equivalent_g_ratio` is a project-defined intact-myelin-equivalent measurement. It must not be presented as the conventional g-ratio or as an established biological correction.

## Interpretation thresholds

- Test Dice at least 0.65 and median vacuole-area error at most 30%: proof of concept against student consensus.
- Dice from 0.40 to 0.65: preliminary detector with limitations.
- Dice below 0.40: retain the detector only as a proposal generator and present the manual-correction workflow and failure analysis.
