# Guided crop-and-mask workflow

This is the easiest route for a new user who has images but has not yet made
fiber crops or masks. The program opens Napari, gives instructions inside the
window, saves progress after every action, and runs the frozen vacuole detector
when annotation is complete.

AxonDeepSeg is the automatic segmentation library used by `whole-image` mode.
It is not an annotation program with a graphical window. The guided workflows
therefore open **Napari**, the same annotation interface used to build our
benchmark. The downstream vacuole detector and measurements are identical to
the other cropped-fiber modes.

## Before starting

From the project root:

```powershell
conda activate myelin-mvp
python -m mvp_pipeline doctor
```

For an assisted start, run `python -m mvp_pipeline` and choose either manual
preparation from whole images, existing crops without masks, or resume. The
wizard collects the same arguments used by the explicit commands below and
shows that command for confirmation before opening Napari.

For these workflows, both `crop_modes_ready` and `guided_workflow_ready` should
be `true`. If the latter is false, recreate the environment from
`environment.yml` or install `.[full]`.

Use a new, empty output folder for each experiment. Do not put the output
folder inside the input folder.

## A. Start with whole laboratory images

Put all whole TIFF/PNG images directly inside one input folder. If every image
has the same scale:

```powershell
python -m mvp_pipeline guided-whole-folder `
  --input "path\to\incoming_whole_images" `
  --nm-per-pixel 5.523 `
  --output "path\to\guided_run"
```

The program opens the images one after another. In the yellow `fiber_crops`
shapes layer, draw one rectangle around every complete fiber to analyze. There
is no software crop limit. Keep the entire outer myelin boundary inside each
rectangle.

Use the buttons in the right-hand panel:

- **Finish image and continue** saves all rectangles, creates the crops, and
  opens the next image.
- **Skip this whole image** records the skip and opens the next image.
- **Save draft and pause** saves the rectangles without completing the image.
- Closing the window has the same safe-pause behavior; it never silently marks
  the image complete.

The whole-image input folder is read-only from the workflow's point of view.
Finished and skipped source images remain exactly where the user placed them;
they are neither moved nor copied into the guided output. The saved state and
`results/input_status.csv` distinguish processed, skipped, and still-pending
inputs, so leaving the files in place does not cause them to be offered twice
when a run is resumed.

After the final whole image, the program immediately starts the mask stage.
For each crop, fill the two label layers:

- `outer_fiber`: the complete filled gross fiber envelope inside the outer
  myelin boundary, including the axon and internal spaces;
- `axon`: the complete axoplasm inside the innermost boundary.

Do **not** draw a vacuole mask. The program checks that both masks are nonempty,
that the axon lies inside the outer-fiber mask, and that a sheath remains. Then
it automatically runs vacuole recognition after the final crop.

## B. Start with an existing folder of fiber crops

This skips whole-image crop selection and begins with the mask windows:

```powershell
python -m mvp_pipeline guided-crop-folder `
  --images "path\to\fiber_crops" `
  --nm-per-pixel 5.523 `
  --output "path\to\guided_run"
```

The original crop files remain in their input folder and are referenced in
place; they are not copied into the output. Keep the input folder available if
you pause and resume the workflow. After the last two-mask annotation, the
frozen vacuole detector starts automatically.

## Mixed physical scales

Instead of one `--nm-per-pixel` value, supply a CSV with one row per input file:

```csv
filename,scale_nm_per_px
NeoView_1.tif,5.523
NeoView_2.tif,1.0908
```

Then use `--scales-csv "path\to\scales.csv"`. The filename, including its
extension, must match exactly apart from letter case. Never supply both scale
options. A copyable example is available at
`examples/guided_scales_template.csv`.

## Pause and resume

Progress is stored in `workflow_state.json`. To resume at any point, run the
same command with the same input and output paths. The state includes completed
and skipped sources, crop rectangles, draft masks, completed masks, and detector
status. Input images remain unchanged in their original folder throughout the
run.

If the detector itself was interrupted after annotations were completed,
rerunning the command starts it again. A completed run returns its saved
summary without opening the annotation windows again.

## Output layout

```text
guided_run/
  workflow_state.json             resumable progress
  workflow_summary.json           final guided-run summary
  input_manifest.csv              internal per-fiber input table
  crops/
    images/                       crops created from whole images only
    axon_masks/                   user-created axon masks
    outer_fiber_masks/            user-created outer-fiber masks
  results/
    input_status.csv              processed/skipped/pending input status
    masks/                        predicted vacuole masks
    overlays/                     blue axon, red boundary, yellow vacuole
    metrics.csv                   one measurement row per fiber
    summary.json                  detector settings and run counts
```

Always inspect every overlay before using the measurements. The workflow
removes repetitive file handling; it does not remove the scientific need to
review boundaries and predictions.

## Common mistakes

- Drawing only a thin myelin ring in `outer_fiber`. It must be a filled gross
  envelope.
- Resizing a crop without updating nm/pixel.
- Cropping through the outer fiber boundary.
- Starting a different experiment in an output folder that already contains a
  `workflow_state.json`.
- Moving the input or output folder between pause and resume.
