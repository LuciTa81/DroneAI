# STEERER UCF-QNRF training result at the user-approved epoch-400 ceiling

## Outcome

The run completed the full fixed validation at epoch 400, persisted
`milestone-400.pth`, updated the checkpoint manifest, and then stopped without
starting an authoritative checkpoint beyond epoch 400. The best count result
within the approved window is the epoch-350 checkpoint:

- MAE: **120.409**
- RMSE: **429.212**
- localization F1 at 16 px: **0.7455**
- reference MAPE: **13.39%**
- median latency: **215.40 ms/image** (4.64 FPS)
- peak VRAM: **5.22 GiB** (5,349 MiB)

The lowest-RMSE checkpoint is epoch 175, with MAE/RMSE
**129.808/379.434**. It is retained as a separate Pareto candidate rather than
combining metrics from two different checkpoints.

## Training protocol

- Data: 961 UCF-QNRF Train images for training and a fixed, disjoint 240-image
  Train-derived validation holdout.
- Official Test: sealed and not accessed.
- Initialization: ImageNet-pretrained HRNet-W48 backbone only; no STEERER model
  checkpoint was loaded.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`.
- Schedule: 10-epoch warmup and cosine schedule with the pinned 800-epoch
  horizon; execution was stopped at the user-approved epoch-400 ceiling.
- Batch: physical/effective batch 8, seed 3035.
- Precision: FP32 recovery after the earlier AMP path produced a non-finite
  gradient norm. No non-finite value recurred in FP32.
- FP32 recovery segment time: 20,047.8 seconds (about 5 h 34 min), from the
  verified epoch-5 checkpoint through the epoch-400 validation boundary.

## Independently re-evaluated checkpoints

| Checkpoint | Epoch | MAE | RMSE | Interpretation |
|---|---:|---:|---:|---|
| `best-mae.pth` | 350 | **120.409** | 429.212 | Primary count candidate |
| `best-rmse.pth` | 175 | 129.808 | **379.434** | Lower large-error sensitivity |
| `milestone-400.pth` | 400 | 122.830 | 460.392 | User-ceiling recovery point |
| `last.pth` | 400 | 122.830 | 460.392 | Mutable last pointer frozen at shutdown |

Each checkpoint was loaded into a fresh model instance. All 240 validation
images were processed again, the restored hash matched the manifest, and the
report MAE/RMSE matched the stored boundary metrics exactly. The independently
loaded `last.pth` and `milestone-400.pth` produced identical model metrics.

## Diagnostic comparison with published results

The STEERER paper reports UCF-QNRF MAE/RMSE **74.3/128.3**, while the pinned
official repository reports **77.8/138.0**. Against the primary epoch-350
checkpoint, the numerical gaps are:

| Reference | MAE gap | RMSE gap |
|---|---:|---:|
| Paper 74.3 / 128.3 | +46.109 (+62.1%) | +300.912 (+234.5%) |
| Repository 77.8 / 138.0 | +42.609 (+54.8%) | +291.212 (+211.0%) |

These are **diagnostic gaps, not an apples-to-apples leaderboard claim**. The
published figures use the official UCF-QNRF Test after the authors' full
training protocol. This run uses a Train-derived validation holdout, withholds
20% of the official training images, and was stopped at epoch 400 rather than
completing the 800-epoch pinned schedule. The result therefore proves that the
commercially safer training lane works and is reproducible, but it does not yet
reproduce the paper's final Test accuracy.

## Harness score at the user ceiling

Applying the existing evidence-harness weights to the authorized epoch-400
boundary gives **100/100 for research evidence integrity**: provenance 25/25,
split and Test isolation 20/20, finite/checkpoint stability 20/20, complete
validation metrics 20/20, and artifact hash verification 15/15. This score
means the result is traceable and reproducible; it does not mean the published
accuracy was reproduced.

The formal `T800` scorer is intentionally not run because it fails closed
without `milestone-800.pth`. Its status is `not applicable` after the user
reduced the authorized ceiling to epoch 400, rather than a fabricated T800
pass.

## Decision

For count-oriented validation use, retain epoch 350 as the primary checkpoint.
Retain epoch 175 when avoiding very large per-image errors is more important.
The epoch-400 checkpoint is a resume/recovery artifact, not the selected best
model. No official Test evaluation or product approval is asserted by this
report.
