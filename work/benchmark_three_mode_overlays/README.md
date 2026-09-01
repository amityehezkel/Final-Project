# Benchmark overlay comparison

Each fiber folder contains:

- the original benchmark **fiber crop** used to draw the masks and score the detector;
- the corresponding **whole laboratory image with an arrow pointing to that fiber**, added for source traceability and anatomical context;
- `01_annotation_overlay.png`: the manual benchmark reference;
- `02_cropped_mode_overlay.png`: the frozen vacuole detector using the supplied manual axon and outer-fiber masks;
- `03_whole_image_mode_overlay.png`: the same frozen detector using masks generated automatically by the AxonDeepSeg front end;
- `00_three_way_comparison.png`: the three views arranged side by side.

The crop and whole-image files retain their source filenames, so their names are not uniform. The crop is the smaller field centered on one fiber. The contextual whole image is normally 1872 × 1872 pixels and contains the manually added arrow. In `extra_p1202_neoview12_01`, both source files originally had the name `NeoView_12.tif`; the crop is therefore preserved as `fiber_crop.tif` to prevent it from being overwritten by the arrow-marked whole image.

The arrow is for human orientation only. The arrow-marked image was not used for model inference, mask creation, tuning, or scoring, and the arrow must not be interpreted as an automatic detection. It connects the benchmark crop to its location in the source field.

For a controlled one-to-one comparison, the whole-image automatic-mask route was run on each benchmark field of view and the central automatic fiber was selected. This tests the automatic-mask stage used by whole-image mode while keeping the exact image fixed. It is not a rematch of every crop to a fresh run on its original full laboratory TIFF. A plain whole-mode image means AxonDeepSeg did not recover a central fiber; these failures are listed as `complete_miss` in `comparison_status.csv`.

Colors are blue for axon, red for the outer-fiber boundary, and yellow for predicted or annotated vacuole.
