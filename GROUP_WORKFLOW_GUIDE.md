# Vacuole-Aware Myelin MVP: Internal Workflow Guide

This document is an internal explanation for our group. It describes what the program actually does, why it does it, what was evaluated, and how to run it. It is intentionally more direct than a submission report.

For a faster introduction, begin with `START_HERE.md`. For the current
algorithm in paper-oriented technical detail, use `TECHNICAL_MODEL_GUIDE.md`.
For the resumable graphical crop-and-mask interface, use
`GUIDED_WORKFLOW.md`.

## 1. What we are trying to measure

Our goal is to identify candidate vacuoles inside the myelin sheath of a myelinated axon and calculate per-fiber measurements. The program needs three pieces of information before it can search for vacuoles:

1. the electron-microscopy image;
2. the axon mask;
3. the gross outer-fiber mask.

The **axon mask** covers the axoplasm inside the inner myelin boundary. The **outer-fiber mask** covers the entire gross fiber envelope, including the axon and all spaces inside the swollen sheath. The program defines:

```text
gross_sheath = outer_fiber AND NOT axon
```

Candidate vacuoles are allowed only inside this gross sheath. This prevents bright axoplasm and extracellular background from being classified as vacuoles.

## 2. The two different segmentation problems

It is important not to confuse these stages:

1. **Fiber segmentation:** finding the axon and gross outer-fiber boundaries.
2. **Vacuole detection:** finding bright candidate spaces inside an already defined gross sheath.

AxonDeepSeg is used only as an optional proposal method for the first problem. Our vacuole detector is a separate, deterministic image-processing algorithm. It is not AxonDeepSeg and it is not a trained neural network.

All three user modes eventually produce the same per-fiber inputs and then call exactly the same vacuole detector and measurement code.

Two additional guided entry points automate the manual preparation around
these modes: `guided-whole-folder` creates any number of crops per source and
then the two masks, while `guided-crop-folder` starts with the two masks. They
use Napari for interaction and automatically run the same detector at the end.

The recommended user-facing command is now simply
`python -m mvp_pipeline`. Its terminal wizard asks what inputs are available,
selects one of these existing entry points, validates paths and scale values,
shows the equivalent explicit command, and then hands execution to that same
entry point. The wizard is a routing and safety layer; it is not another model
and does not change any detector parameter.

## 3. Complete pipeline overview

```text
WHOLE-IMAGE MODE
whole TIFF + nm/pixel
        |
        v
scale-normalized AxonDeepSeg
        |
        v
automatic fiber separation and per-fiber masks
        |
        +-----------------------------+
                                      |
SINGLE-CROP MODE                      |
crop + nm/pixel + two masks ----------+
                                      |
FOLDER MODE                           |
many crops + common scale + masks ----+
                                      v
                         shared vacuole detector
                                      |
                                      v
                    masks + overlays + CSV measurements
```

## 4. Logic of the current vacuole detector

The frozen model configuration is:

```text
work/best_model_results/detector_config.json
```

Although we sometimes call it a “model,” it is a classical intensity-based segmentation algorithm with frozen parameters. Given one image and its two masks, it performs these steps:

1. **Restrict the search region.** Calculate `gross_sheath = outer_fiber & ~axon`. Pixels outside this region cannot become vacuoles.
2. **Normalize locally.** Within the gross sheath, map the 1st–99th intensity percentiles to the range 0–1. This reduces differences in overall brightness between images.
3. **Apply CLAHE.** Local contrast enhancement makes bright spaces more distinguishable from compact myelin. The neighborhood size is calculated from a physical length of 0.25 µm using the supplied nm/pixel value.
4. **Smooth noise.** Apply a Gaussian filter with a physical sigma of 0.02 µm.
5. **Calculate an Otsu threshold.** Otsu gives a data-dependent base threshold from the intensity distribution inside that fiber’s sheath.
6. **Use hysteresis thresholding.** Strong bright seeds must exceed `Otsu + 0.20`; connected weaker pixels may grow from those seeds if they exceed `Otsu + 0.10`. This change was introduced because a lower simple threshold incorrectly classified darker myelin as vacuoles.
7. **Clean the mask.** Morphological closing and opening use a physical radius of 0.02 µm.
8. **Remove tiny components.** Regions smaller than 0.01 µm² are removed.
9. **Constrain again.** The final prediction is intersected with the gross sheath so it cannot leak into the axon or extracellular area.

The same frozen parameters are used in every input mode. Physical sizes are converted to pixels separately for each image using its nm/pixel scale.

## 5. Mode 1: whole laboratory TIFF

### User input

- whole TIFF or PNG;
- physical scale in nm/pixel.

### Internal steps

1. Choose the recommended AxonDeepSeg inference scale: 2.36 nm/pixel when the source is below 2 nm/pixel, otherwise 4.93 nm/pixel.
2. Resample an inference copy without changing the original analysis image.
3. Run the downloaded AxonDeepSeg generalist-light model.
4. Restore predicted masks to the original image dimensions with nearest-neighbor interpolation.
5. Ignore the bottom-right scale-bar region.
6. Find axon components and associate surrounding myelin.
7. Use axon-seeded watershed when adjacent fibers form a touching cluster.
8. Reject border-touching fibers and very small axon components.
9. Create a crop, axon mask, and gross outer-fiber mask for every surviving fiber.
10. Run the shared vacuole detector on every generated crop.
11. Combine results into per-fiber files and a whole-image overlay.

### Command

```powershell
python -m mvp_pipeline whole-image `
  --image "path\to\laboratory_image.tif" `
  --nm-per-pixel 5.523 `
  --output "work\whole_image_result"
```

### Reliability

This is an **experimental proposal workflow**, not the reliable primary route. On seven manually referenced fibers from independent whole-image tests, median overlap was good, but only four of seven passed both mask thresholds and one fiber was completely missed. Any error in the automatic axon or outer boundary directly changes vacuole area and g-ratio measurements.

The output therefore contains extraction flags and `manual_review_recommended`. Fibers with `low_myelin_coverage`, `irregular_axon_shape`, or `large_axon_area_outlier` should be inspected. A watershed flag is informational: it says a touching cluster was separated automatically.

## 6. Mode 2: one masked fiber crop

### User input

- one fiber crop;
- nm/pixel;
- binary axon mask;
- binary outer-fiber mask.

The image and masks must have identical pixel dimensions. The crop should contain one complete fiber. The outer-fiber mask must be the complete envelope, not only a thin myelin ring.

### Internal steps

1. Validate file existence, image dimensions, and nonempty masks.
2. Create a one-row internal manifest.
3. Sanitize the masks and record QC flags.
4. Run the shared vacuole detector.
5. Calculate measurements and write the overlay and masks.

### Command

```powershell
python -m mvp_pipeline fiber-crop `
  --image "path\to\fiber_crop.tif" `
  --nm-per-pixel 5.523 `
  --axon-mask "path\to\axon_mask.png" `
  --outer-fiber-mask "path\to\outer_fiber_mask.png" `
  --output "work\fiber_result"
```

This is the most reliable MVP route because the user supplies the two masks instead of depending on inconsistent automatic fiber segmentation.

## 7. Mode 3: folder of masked fiber crops

This mode is the batch version of mode 2. All crops in one run must share the supplied nm/pixel scale.

### Recommended folder structure

```text
input/
  images/
    fiber01.tif
    fiber02.tif
  axon_masks/
    fiber01_axon.png
    fiber02_axon.png
  outer_fiber_masks/
    fiber01_outer_fiber.png
    fiber02_outer_fiber.png
```

The masks may instead use exactly the same filename stem as the image because they are in separate directories. Common `_axon`, `_axon_mask`, `_outer`, `_outer_mask`, `_outer_fiber`, and `_outer_fiber_mask` suffixes are recognized.

### Internal steps

1. Scan the three folders.
2. Match each image to exactly one axon mask and one outer-fiber mask.
3. Validate every triplet before starting detector execution.
4. Build one internal batch manifest.
5. Run the shared vacuole detector once across the batch.
6. Write one metrics table plus per-fiber masks and overlays.

### Command

```powershell
python -m mvp_pipeline fiber-folder `
  --images "input\images" `
  --nm-per-pixel 5.523 `
  --axon-masks "input\axon_masks" `
  --outer-fiber-masks "input\outer_fiber_masks" `
  --output "work\folder_result"
```

## 8. Physical scale

The program never guesses scale from axon appearance. Original JEOL laboratory TIFFs contain calibration metadata. For example, a TIFF with 181.061 pixels/µm has:

```text
1000 nm/µm / 181.061 pixels/µm = 5.523 nm/pixel
```

A crop made without resizing keeps its source TIFF’s scale. Exported PNGs may lose microscope metadata, so the command still requires the user to provide nm/pixel explicitly. If an image was resized, its scale must be adjusted by the resize ratio.

The scale matters because all reported areas and all detector size thresholds are physical rather than raw pixel quantities.

## 9. Output measurements

For every fiber, the program reports:

- axon area;
- gross-sheath area;
- intact-myelin area;
- outer-fiber area;
- total vacuole area;
- vacuole burden;
- conventional area-based g-ratio;
- project-defined intact-myelin-equivalent g-ratio;
- mask source and QC status.

Definitions:

```text
vacuole_burden = vacuole_area / gross_sheath_area
g_ratio = sqrt(axon_area / outer_fiber_area)
intact_equivalent_g_ratio = sqrt(axon_area / (outer_fiber_area - vacuole_area))
```

The last quantity is our project-defined metric. It must not be called a standard corrected g-ratio or an established biological measure.

## 10. Detector evaluation

The official benchmark now contains the original 24 fibers plus 19 additional
student-labeled crops: 20 development and 23 test fibers from source-disjoint
groups. The new crops add 1.0908, 5.523, and 9.0987 nm/pixel examples, although
the only 9.0987 nm/pixel source had to remain in development because it had
already been used during false-positive investigation. Two original test rows
were excluded by border-touching QC, leaving 21 evaluated test fibers.

The selected parameters are minimum area 0.01 µm², high threshold offset 0.20,
low threshold offset 0.10, Gaussian sigma 0.02 µm, and morphology radius
0.02 µm. Existing detections are then refined by connected local-intensity
growth: maximum distance 0.05 µm, growth threshold `Otsu - 0.05`, and maximum
area ratio 1.5. Expanded test performance is:

- median vacuole Dice: 0.752;
- median IoU: 0.603;
- median precision: 0.930;
- median recall: 0.714;
- median total-area absolute error: 0.00628 µm²;
- median total-area percentage error: 20.29%.

Boundary refinement does not create new components. The detector therefore
also has a thin-seed rescue branch that re-examines the raw intensity
candidate with 0.005 µm morphology. It keeps a component only when its area is
at least 0.0015 µm², thickness is at least 0.045 µm, radial position is at most
0.55, eccentricity is at most 0.95, and solidity is at least 0.85. This reduced
the eligible test set from five complete misses to one while leaving the
compact-fiber false-positive count at one of seven. The cost is a modest
precision decrease because recovered predictions contain more area.

A marker-controlled watershed was tested first, but internal texture produced
premature gradient ridges. Seeded connected-intensity growth was more robust.
Growth stays inside the supplied gross-myelin mask, within a physical distance,
and is rejected when it exceeds the area-ratio guardrail. The rescue branch is
what can solve a complete false negative by restoring a plausible thin seed.

These annotations are not expert biological ground truth. We cannot claim clinical, diagnostic, or biological validation.

### Thin clefts and the area filter

For this project, a thin annotated cleft can count as a vacuole even when its
area is below 0.01 µm². Do not interpret `min_area_um2` as a biological
definition. It is a post-processing filter that trades small-vacuole recall
against bright-artifact false positives.

The manual development masks support 0.0015 µm² as a permissive logical
minimum because it retains 95.7% of their total annotated vacuole area. An
isolated candidate sweep showed that this setting recovered every positive
test fiber, but also marked 5/7 compact test fibers. The main 0.01 µm²
configuration marked only 1/7 compact test fibers but completely missed 5
positive fibers. The active model keeps that conservative main filter and adds
the shape-aware thin-seed rescue described above. It missed 1 positive fiber
and still marked only 1/7 compact test fibers. The area-only evidence is in
`work/min_area_analysis_v5/isolated_area_sweep/`; the rescue evidence is in
`work/thin_rescue_v7/`.

## 11. Why direct AxonDeepSeg on tight crops was rejected

We tested the official AxonDeepSeg model directly on all 24 tightly cropped benchmark fibers. Manual masks were used only afterward for scoring. The results were:

- median axon Dice: 0.374;
- median outer-fiber Dice: 0.231;
- only 8/24 crops passed both 0.75 thresholds;
- eight complete misses.

Some individual predictions were excellent, but performance was too inconsistent. Therefore mode 2 and mode 3 require user-provided masks; they do not run AxonDeepSeg on each tight crop.

## 12. Important output files

Every inference mode writes:

- `metrics.csv`: per-fiber measurements and QC;
- `summary.json`: configuration and run summary;
- `masks/*_vacuole.png`: predicted candidate-vacuole masks;
- `overlays/*_overlay.png`: image, axon, outer boundary, and vacuole visualization.

The folder and crop modes also write `input_manifest.csv` automatically. Whole-image mode writes `automatic_manifest.csv`, generated per-fiber crops, a whole-image overlay, and extraction metadata.

### Benchmark visual traceability

`work/benchmark_three_mode_overlays/` contains one subfolder for every benchmark fiber. Each subfolder combines six human-review items:

1. the original cropped fiber;
2. the whole laboratory image with a manually added arrow pointing to the crop location;
3. the manual annotation overlay;
4. the cropped-mode prediction overlay;
5. the controlled whole-image-front-end prediction overlay;
6. a three-way overlay comparison.

The crop and whole-image filenames vary because the original source names were retained; use the arrow and image dimensions to distinguish them. The arrow-marked whole image establishes provenance and tissue context only. It is not a model output and was not used for inference or scoring. In particular, `03_whole_image_mode_overlay.png` is the fixed-field automatic-mask benchmark described above, not a crop rematched from the arrow-marked full TIFF.

## 13. Current files that matter

- `mvp_pipeline/`: implementation.
- `tests/`: automated tests.
- `work/benchmark/`: the expanded 43-fiber manifest, original consensus masks, added student masks, and preserved 24-fiber manifest.
- `work/annotations/`: saved annotation layers.
- `work/best_model_results/`: frozen detector configuration and final benchmark outputs.
- `work/tuning_expanded_v2/`: full 275-configuration development-only search.
- `work/tuning_corrected_axon_v4/`: repeated development search after correcting the hollow `p126_neo1_axon-01` axon mask; the selected parameters were unchanged.
- `work/min_area_analysis_v5/`: annotation-derived minimum-area analysis, candidate sweeps, and preserved comparison runs for thin clefts.
- `work/boundary_refinement_v6/`: development tuning, comparison configurations, and evaluation evidence for seeded boundary refinement.
- `work/thin_rescue_v7/`: shape-aware thin-cleft rescue search, selected configuration, and benchmark evidence.
- `work/model_history/`: preserved version-1 results and comparison evidence.
- `work/models/`: downloaded AxonDeepSeg model.
- `work/crop_ads_validation/`: direct-crop AxonDeepSeg rejection evidence.
- `work/automatic_multi_image/`: final whole-image automatic validation evidence.
- `work/benchmark_three_mode_overlays/`: per-fiber crops, arrow-marked source fields, annotations, and both mode overlays.
- `work/separate_mode_results/`: cropped-mode and whole-image-front-end metrics packaged separately.
- `README.md`: concise commands and technical instructions.

## 14. Practical recommendation

For a dependable demonstration, use `fiber-crop` or `fiber-folder` with reviewed masks. Show `whole-image` as the ambitious automatic proposal route, clearly display its QC flags, and state that upstream fiber segmentation remains the main limitation. In every mode, inspect overlays before interpreting measurements.
