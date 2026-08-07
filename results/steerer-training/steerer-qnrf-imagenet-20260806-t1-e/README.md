# STEERER UCF-QNRF T1/T5 evidence

This directory preserves the small, Git-safe evidence from the first and
five-epoch STEERER training stages. The approximately 740 MB checkpoints remain
on the home5090 SSD and are referenced by SHA-256 rather than copied into Git.

## Experiment boundary

- Initialization: ImageNet-pretrained HRNet-W48 backbone only; no STEERER model checkpoint
- Source population: 1,201 images from the approved UCF-QNRF Train package
- Deterministic split: 961 train / 240 validation, seed 3035
- Official UCF-QNRF Test: sealed and not accessed
- Architecture: STEERER with an HRNet-W48 ImageNet backbone and scale-aware
  multi-resolution density regression
- Input/output: RGB image to density map; estimated count is the density-map sum
- Stages: T1 at epoch 1, then an immutable-checkpoint resume through epoch 5
- Optimizer steps: 120 at T1 plus 480 resumed steps to T5
- Optimizer: AdamW, learning rate 1e-4, weight decay 1e-4
- Schedule: 10-epoch warm-up and cosine schedule with the original 800-epoch horizon
- Augmentation: 768x768 crop, 0.5-2.0 scale range, horizontal flip
- Batch: physical 8, accumulation 1
- Precision: AMP accepted by the FP32 comparison gate
- Runtime: RTX 5090, PyTorch 2.9.0a0+nv25.10, CUDA 13.0
- Observed elapsed time: 99.377 seconds for T1 and 202.437 seconds for the
  T1-to-T5 continuation (301.814 seconds combined)

## Result

| Metric | T1 | T5 | T1 to T5 change |
|---|---:|---:|---:|
| MAE | 1,574.708 | 389.995 | -75.23% |
| RMSE | 1,930.137 | 1,029.695 | -46.65% |
| Signed bias | +1,505.957 | -366.288 | overcount to undercount |
| Reference MAPE | 461.687% | 36.127% | -425.560 pp |
| GAME L1 | 1,602.514 | 429.191 | -73.22% |
| Quadrant zone MAE | 400.629 | 107.298 | -73.22% |
| Localization F1 @ 16 px | 8.076% | 52.963% | +44.887 pp |
| Localization precision @ 16 px | 4.214% | 44.898% | +40.684 pp |
| Localization recall @ 16 px | 96.842% | 64.559% | -32.283 pp |
| Median latency | 213.377 ms/image | 217.220 ms/image | +1.80% |
| Throughput derived from median | 4.687 FPS | 4.604 FPS | -1.77% |
| Peak allocated VRAM | 5,349.181 MB | 5,344.884 MB | -4.297 MB |

The T1 model strongly overpredicted almost every target, producing high recall
but unusably low precision. T5 sharply reduced that overprediction and produced
a substantially more balanced localization result. It now undercounts on
average, however, and remains far from the fully trained published baseline.

Harness score is 100/100 with status `PASS_COMMERCIAL_CANDIDATE`. This score
means the evidence, rights scope, split isolation, and artifact hashes passed;
it is not an accuracy score and is not production approval.

The authoritative T5 score bundle retains a legacy description saying
`required T1 validation metrics`; this was a display-only defect. The scorer
validated all 17 T5 metrics, and the source now uses the actual stage name with
a regression test. The immutable score JSON was not altered after generation.

T5 is the best result observed in this training lineage so far, but it is an
early milestone rather than a final model. See `paper-comparison.md` for the
paper/repository gap and comparability limits. T50 requires a separate approval.
