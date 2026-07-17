# STEERER UCF-QNRF frozen validation result

The accepted run is `steerer-ucf-qnrf-val-smoke-7e6bebb`. It evaluates the
official STEERER UCF-QNRF checkpoint on the same deterministic 36-image
validation subset previously used for DM-Count. It does not access the official
test split and performs no training or fine-tuning.

## Model contract

- Architecture: HRNet-W48 backbone with multi-resolution density heads and
  selective inheritance across scales.
- Input: one RGB image, aspect-preserving resize with long side at most 3072,
  bottom/right padding to a multiple of 32, ImageNet normalization.
- Native output: x1/x4/x8 density maps and merged point candidates.
- Count: sum of the valid-region highest-resolution density after the official
  density factor of 100.
- Localization: local maxima from the three scales, merged with the official
  cross-scale rule and mapped back to original-image coordinates.

## Frozen data contract

- Dataset: `ucf-qnrf-kaggle-apache` (project policy accepts the Kaggle Apache-2.0
  label while retaining uploader/non-owner provenance risk).
- Source partition: original Train images only, 1,201 total.
- Frozen split: DM-Count official lists, 1,081 train and 120 validation.
- Evaluated subset: 36 validation images, 12 each from low/medium/high density.
- Seed: `20260715`.
- Train list SHA-256: `e8a926b35e55f9e7bf38498ec722d9a7f051169f03dd994024a9e5c757694cb0`.
- Validation list SHA-256: `835f660ed3373118f4d76a36ebf013ef28d90fcbcc835ddb5c180a2b26524126`.

## Measured result

| Metric | STEERER | DM-Count same 36 images |
|---|---:|---:|
| MAE | 71.59 | 154.29 |
| RMSE | 97.97 | 229.81 |
| Signed bias | +43.92 | +3.03 |
| MAPE reference | 6.20% | 12.99% |
| GAME-L1 mean | 79.15 | 168.81 |
| Median latency | 130.70 ms | 69.53 ms |
| Batch-1 throughput | 7.65 FPS | 14.38 FPS |
| Peak VRAM | 3,408.52 MB | 8,980.26 MB |

Additional STEERER means:

- Density zone MAE: 19.77
- Density PSNR/SSIM: 24.04 / 0.7467
- Localization precision/recall/F1 at 16 px: 0.8603 / 0.7515 / 0.7965
- Localization mean distance: 5.69 px
- Point zone MAE: 40.27

The model overcounted by 43.92 people on average. Signed bias rose from +6.12 in
the low band to +92.12 in the high band, which remains an important CCTV
calibration issue even though aggregate error is much lower than DM-Count.

## Gate and evidence

- Technical harness: `100/100`, `PASS_RESEARCH_ONLY`.
- Rights: code MIT; dataset label accepted for this comparison; official
  checkpoint commercial/derived-weight terms remain unverified.
- Evaluation scope: research comparison only, not production approval.
- Coverage: 36/36 successful, no unrecorded samples.
- Review bundle: 3,264,916 bytes, 22 internal references re-hashed successfully.
- Panels: 12 total—best, strong, and typical examples for every density band,
  plus severe undercount, severe overcount, and worst spatial-quality examples.

See the accepted run directory for the score, predictions, selection manifest,
environment, artifact hashes, and selected panels.
