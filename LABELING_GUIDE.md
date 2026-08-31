# Student-Consensus Labeling Guide

## Layers

Annotate three binary layers for exactly one complete fiber per crop:

1. `axon`: the complete axoplasm inside the innermost myelin boundary.
2. `outer_fiber`: everything enclosed by the gross outer myelin boundary, including axon, compact myelin, and suspected vacuoles.
3. `vacuole`: candidate pathological empty spaces within the gross myelin sheath.

The axon mask must be fully contained in the outer-fiber mask. Vacuole pixels must be inside `outer_fiber` and outside `axon`.

## Operational vacuole rule

Label a candidate vacuole only when it is:

- inside the gross myelin envelope;
- brighter than compact myelin;
- bounded by visible membrane/myelin structure;
- visibly wider than ordinary spacing between adjacent myelin lamellae.

Do not label:

- axoplasm or extracellular background;
- mitochondria, vesicles, nuclei, or other organelles;
- ordinary thin interlamellar spacing;
- section tears, folds, staining defects, or compression artifacts;
- the embedded scale bar or text;
- regions whose boundary cannot be distinguished at the displayed resolution.

If uncertain, leave the region unlabeled and record it in the annotation notes. Consistency is more valuable than aggressively labeling every possible anomaly.

## Independence and consensus

- Annotators A and B work independently and must not see each other's masks.
- Compute agreement before consensus review.
- The third student reviews both references and draws a new consensus mask.
- Do not show algorithm predictions during annotation or consensus resolution.
- These masks are student consensus, not expert ground truth.

