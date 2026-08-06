# STEERER UCF-QNRF T1 evidence

This directory preserves the small, Git-safe evidence from the first complete
STEERER training stage. The 740 MB checkpoints remain on the home5090 SSD and
are referenced by SHA-256 rather than copied into Git.

## Experiment boundary

- Initialization: ImageNet-pretrained HRNet-W48 backbone only; no STEERER model checkpoint
- Source population: 1,201 images from the approved UCF-QNRF Train package
- Deterministic split: 961 train / 240 validation, seed 3035
- Official UCF-QNRF Test: sealed and not accessed
- Stage: one epoch, 120 optimizer steps
- Optimizer: AdamW, learning rate 1e-4, weight decay 1e-4
- Batch: physical 8, accumulation 1
- Precision: AMP accepted by the FP32 comparison gate
- Runtime: RTX 5090, PyTorch 2.9.0a0+nv25.10, CUDA 13.0
- End-to-end elapsed time: 99.377 seconds

## Result

| Metric | T1 value |
|---|---:|
| MAE | 1,574.708 |
| RMSE | 1,930.137 |
| Signed bias | +1,505.957 |
| Reference MAPE | 461.687% |
| GAME L1 | 1,602.514 |
| Quadrant zone MAE | 400.629 |
| Localization F1 @ 16 px | 8.076% |
| Localization precision @ 16 px | 4.214% |
| Localization recall @ 16 px | 96.842% |
| Median latency | 213.377 ms/image |
| Throughput derived from median | 4.687 FPS |
| Peak allocated VRAM | 5,349.181 MB |

Harness score is 100/100 with status `PASS_COMMERCIAL_CANDIDATE`. This score
means the evidence, rights scope, split isolation, and artifact hashes passed;
it is not an accuracy score and is not production approval.

The one-epoch model strongly overcounts and is not a usable final model. See
`paper-comparison.md` for the paper/repository gap and comparability limits.

