# Technical model and pipeline guide

This document describes the implementation in enough detail for a group
member to understand, reproduce, and write the Methods section of a course
paper. The code remains the authoritative implementation.

## 1. Scope and terminology

The system operates on grayscale electron-microscopy images. For each fiber it
uses three spatial regions:

- **axon**: the inner axonal compartment;
- **outer fiber**: the complete envelope enclosed by the outside of the myelin
  sheath, including the axon;
- **gross sheath**: `outer fiber AND NOT axon`.

Vacuoles are detected only inside the gross sheath. The vacuole detector is a
scale-aware, rule-based image-processing model. It is not a trained neural
network. AxonDeepSeg, which is a neural segmentation model, is used only as the
optional front end in whole-image mode to propose axon and myelin masks.

All spatial parameters are expressed in µm or µm² and converted to pixels from
the supplied `nm/pixel`. Consequently, the same biological-size rule can be
applied to images with different pixel resolutions.

## 2. The three analysis modes, two guided entry points, and launcher

### Mode 1: whole laboratory image

Input: whole TIFF/PNG and `nm/pixel`.

1. The image is resampled to the closest validated AxonDeepSeg scale: 2.36
   nm/pixel for source scales below 2 nm/pixel, otherwise 4.93 nm/pixel.
2. The expected bottom-right scale-bar region is replaced by the median image
   intensity before segmentation so it cannot become an artificial object.
3. AxonDeepSeg produces axon and myelin-class masks on the normalized image.
4. The masks are restored to the original dimensions with nearest-neighbor
   interpolation.
5. Connected axons below 0.01 µm² are rejected. The axon and myelin masks are
   joined with a one-pixel closing, and axon-labeled watershed separates
   touching fiber clusters.
6. Holes in each assigned fiber region are filled. This is important because a
   vacuole must remain inside the gross outer-fiber envelope rather than being
   interpreted as background.
7. Fibers touching the image border, scale-bar exclusion boundary, or another
   fiber's axon are rejected. Low myelin coverage, irregular axon shape, large
   axon-area outliers, and watershed-split clusters are flagged for review.
8. Each accepted fiber is cropped with a 0.25 µm margin and passed to the same
   vacuole detector used by crop modes.

Automatic measurements are only as reliable as these masks. In the controlled
43-fiber comparison, the whole-image front end recovered 25 fibers and both
automatic masks passed the predefined quality criterion for 15 fibers. This
front end, rather than the shared vacuole algorithm, is the main whole-image
limitation.

### Mode 2: one cropped fiber

Input: fiber crop, `nm/pixel`, axon mask, and outer-fiber mask.

The image and masks must have identical dimensions. The outer-fiber mask must
represent the complete filled fiber envelope, not merely a thin line or only
the dark compact-myelin pixels. Input validation rejects missing, empty, or
dimensionally inconsistent masks. No AxonDeepSeg step is used.

### Mode 3: folder of cropped fibers

Input: image folder, one common `nm/pixel`, matching axon-mask folder, and
matching outer-fiber-mask folder.

Files are matched by stem with accepted suffixes such as `_axon`, `_axon_mask`,
`_outer`, and `_outer_fiber`. After matching and validation, every fiber is
processed by the same batch and detector code as Mode 2.

### Guided preparation for Modes 2 and 3

`guided-whole-folder` is a resumable graphical front end that creates any
number of rectangular fiber crops from each whole source image, then opens
each crop for manual axon and outer-fiber labeling. `guided-crop-folder` skips
the crop-selection stage and starts from existing crops. Both use Napari,
strictly validate the two masks before completion, create an internal manifest,
and call the same frozen cropped-mode detector. They do not alter the detector
logic or its parameters. Their persistent state is an operational aid and is
not part of model training or benchmark evaluation.

### Interactive launcher

Running `python -m mvp_pipeline` without a subcommand opens a terminal wizard.
The wizard does not implement segmentation or detection. It collects and
validates the required paths and scale information, maps the user's answers to
one of the five entry points above, displays the equivalent explicit command,
checks that the selected environment components are available, and then calls
the same existing function as that explicit command. Therefore wizard and
non-wizard runs with identical inputs use identical model logic and parameters.

## 3. Common mask cleanup and quality control

Before vacuole detection:

1. Masks are converted to Boolean arrays.
2. If any axon pixel lies outside the outer-fiber mask, the outer mask is
   expanded to include it and an `axon_outside_outer_fiber` flag is recorded.
3. The gross sheath is calculated as `outer AND NOT axon`.
4. Vacuole predictions are always clipped to the gross sheath.
5. Empty axon, empty outer fiber, empty sheath, scale-bar overlap, and
   border-touching conditions are recorded. Severe flags exclude a benchmark
   row from aggregate evaluation, but the output remains available for review.

This cleanup prevents impossible output such as a vacuole inside the axon or
outside the supplied fiber boundary.

## 4. Preparing the intensity response

The detector searches for relatively bright disruptions within the generally
darker myelin sheath.

1. Only gross-sheath pixels are used to estimate intensity statistics.
2. The 1st and 99th sheath-intensity percentiles are mapped to 0 and 1, with
   values outside this range clipped. This reduces sensitivity to global image
   brightness and isolated extreme pixels.
3. Contrast-limited adaptive histogram equalization (CLAHE) is applied with
   clip limit 0.01. Its local kernel corresponds to approximately 0.25 µm.
4. A Gaussian filter with physical sigma 0.02 µm suppresses pixel-scale noise.
5. Otsu's threshold is calculated from the processed gross-sheath pixels. It
   adapts the decision to the intensity distribution of each fiber.

## 5. Strong seeds and connected weak pixels

The main candidate uses two thresholds relative to Otsu:

- high threshold: `Otsu + 0.20`;
- low threshold: `Otsu + 0.10`.

Pixels above the high threshold are strong **seed pixels**. Pixels between the
low and high thresholds are retained only when they are connected to a strong
seed. This is hysteresis thresholding. It avoids accepting every moderately
bright texture pixel while allowing a genuine vacuole to contain gradual
intensity variation.

Pixels outside the gross sheath are assigned an unusable response during this
step, so they cannot connect two otherwise separate candidates.

## 6. Standard morphology and area filtering

The raw hysteresis candidate undergoes circular closing followed by opening,
both with radius 0.02 µm:

- **closing** fills small gaps and joins nearby pixels within a candidate;
- **opening** removes thin protrusions and isolated noise.

For larger pixel radii the implementation uses distance-transform equivalents
of circular morphology for speed. Connected components smaller than 0.01 µm²
are removed from the main branch.

The 0.01 µm² value is a noise-control parameter, not the biological definition
of a vacuole. Manual annotations contain smaller valid clefts, which motivated
the separate rescue branch below.

## 7. Seeded boundary refinement

Standard morphology often identifies a vacuole but underestimates its border.
Each surviving main-branch component is therefore refined separately:

1. Search only inside the gross sheath and within 0.05 µm of the existing
   component.
2. Permit connected growth through pixels with response at least
   `Otsu - 0.05`.
3. Use binary propagation starting from the detected seed. Unconnected bright
   regions cannot appear during refinement.
4. Reject the growth and retain the original seed if the refined area exceeds
   1.5 times the seed area. This guardrail limits leakage through weak edges.

Refinement can improve the extent of an existing detection, but by design it
cannot rescue a fiber for which the main branch produced no seed.

## 8. Thin-seed rescue

Some manually labeled clefts pass the raw intensity threshold but are destroyed
by the main 0.02 µm morphology or 0.01 µm² filter. The rescue branch reuses the
raw hysteresis candidate and applies gentler 0.005 µm closing/opening.

Each resulting connected component is accepted only if all conditions hold:

| Feature | Rule | Purpose |
|---|---:|---|
| Area | at least 0.0015 µm² | Retain thin annotated clefts while rejecting tiny noise |
| Maximum thickness | at least 0.045 µm | Reject one-pixel lines and narrow texture |
| Median radial position | at most 0.55 | Prefer components nearer the axon than the outer boundary |
| Eccentricity | at most 0.95 | Reject extremely line-like structures |
| Solidity | at least 0.85 | Reject fragmented or strongly concave texture artifacts |

Thickness equals twice the maximum internal Euclidean distance to the component
boundary. Radial position is calculated for every component pixel as:

```text
distance to axon / (distance to axon + distance to outer boundary)
```

Its median is used, so 0 means the axon side of the sheath and 1 means the outer
edge. Accepted rescued components are united with the refined main prediction
and clipped once more to the gross sheath.

This is more selective than globally lowering the main area threshold. In the
test comparison, lowering area alone to 0.0015 µm² marked 5/7 compact fibers,
whereas the shape-aware rescue retained the former 1/7 compact false-positive
count.

## 9. Final outputs and measurements

For each fiber the program writes a binary vacuole mask and an overlay:

- translucent blue: axon;
- red line: outer-fiber boundary;
- yellow: predicted vacuole and its boundary.

Let `A` denote physical area in µm². The pixel area is
`(nm_per_pixel / 1000)²`. The program reports:

- axon area;
- outer-fiber area;
- gross-sheath area = outer-fiber area minus axon area;
- vacuole area;
- intact-myelin area = gross-sheath area minus vacuole area;
- vacuole burden = vacuole area / gross-sheath area;
- conventional g-ratio = `sqrt(axon area / outer-fiber area)`;
- intact-equivalent g-ratio =
  `sqrt(axon area / (outer-fiber area - vacuole area))`.

Vacuole count and mean vacuole size are intentionally omitted. Small boundary
breaks can split one biological cleft into multiple image components, making
count and component-average size unstable. Total area and burden better match
the project question.

## 10. Benchmark and model selection

The benchmark contains 43 student-labeled fibers: 20 development and 23 test.
Splits were assigned by source image to reduce leakage between highly related
crops. Two test rows are excluded from aggregate scoring by the pre-existing
border-touching rule, leaving 21 eligible test fibers.

Parameters were selected using development annotations. The active
configuration is `work/best_model_results/detector_config.json`. The final
cropped-mode test medians are:

- Dice: 0.752;
- IoU: 0.603;
- precision: 0.930;
- recall: 0.714;
- absolute vacuole-area error: 0.00628 µm²;
- vacuole-area percentage error: 20.29%.

Complete positive-fiber misses fell from five in boundary-refinement v6 to one
in thin-rescue v7, while compact-fiber false positives remained 1/7. Because
the group has visually inspected the test set during development, these values
are descriptive evidence for a course proof of concept rather than an
independent external validation.

Overlap metrics use pixel-level true positives, false positives, and false
negatives:

- Dice = `2TP / (2TP + FP + FN)`;
- IoU = `TP / (TP + FP + FN)`;
- precision = `TP / (TP + FP)`;
- recall = `TP / (TP + FN)`.

If both prediction and truth are empty, overlap metrics are defined as 1. Area
percentage error is undefined for a compact reference fiber with a nonempty
false prediction and is excluded from the median percentage calculation.

## 11. Reproducibility and preserved evidence

- `work/best_model_results/`: active configuration, masks, overlays, metrics,
  and summary.
- `work/model_history/`: previous official model snapshots and their overlays.
- `work/min_area_analysis_v5/`: area-filter analysis and candidate evidence.
- `work/boundary_refinement_v6/`: refinement tuning and failure-stage
  diagnostic.
- `work/thin_rescue_v7/`: selected rescue rule and development grid; the full
  active output is stored only once in `work/best_model_results/`.
- `work/separate_mode_results/`: crop-mode and whole-image-mode evaluations
  reported separately.

To verify a copied workspace:

```powershell
conda activate myelin-mvp
python -m mvp_pipeline doctor
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q
```

Disabling unrelated third-party Pytest plugin auto-loading keeps the test run
fast; it does not alter the pipeline.

## 12. Limitations to state in the paper

- The reference masks are student annotations, not expert biological ground
  truth.
- The benchmark is small and some test examples were inspected during model
  development.
- Bright compact-myelin texture can resemble a vacuole.
- Very thin clefts may be only partially recovered.
- Whole-image accuracy is limited by automatic axon/myelin segmentation and
  fiber extraction.
- The method is a research/course proof of concept and must not be presented as
  diagnostic, clinical, or fully biologically validated software.
