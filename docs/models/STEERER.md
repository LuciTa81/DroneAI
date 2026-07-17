# Model brief: steerer-official-ucf-qnrf

- Role: UCF-QNRF crowd-counting and localization baseline
- Family: hybrid density and point localization
- Backbone: HRNet-W48
- Native output: multi-resolution density and localization predictions
- Inference long-side cap: 3072 pixels
- Density normalization factor: 100

## Major blocks

- HRNet-W48 multi-resolution backbone
- multi-resolution counting heads
- selective inheritance/upsample module
- Gaussian density/local-maximum point decoder

## Feature scales

- x1
- x4
- x8

## Original training losses

- multi-resolution MSE
- selective inheritance routing loss
- uncertainty weighting where configured

## Output semantics

```text
estimated_count = sum(highest_resolution_density / 100)
localization_points = official local maxima merged from x1/x4/x8
zone_count = integral of count-preserving density in the calibrated image zone
paper localization F1 != harness F1@16px
```

## Paper/repository-reported results

The following UCF-QNRF values are paper/repository-reported results, not measurements produced by this evaluation harness:

- MAE: 77.8
- RMSE: 138.0
- F1: 75.6
- Precision: 79.7
- Recall: 72.0

Future harness results must be recorded separately. The harness localization label is `harness F1@16px in original-image coordinates`; it must not be substituted for the paper localization F1.

## Failure modes

- tiny heads
- extreme perspective
- occlusion
- domain shift
- padding artifacts
- threshold-sensitive localization
