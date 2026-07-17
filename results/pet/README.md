# PET evaluation evidence

PET is being evaluated as a point-based research comparison only.

- Rights gate: `PASS_RESEARCH_ONLY`, 84/100
- Allowed: catalog, research reproduction, research asset download, research checkpoint evaluation
- Blocked: commercial fine-tuning, derived commercial weights, festival deployment
- Dataset lane: frozen UCF-QNRF validation subset; official test remains sealed
- Preflight: `PASS` on RTX 5090 with current NGC Torch/CUDA
- Checkpoint: strict load, 20,909,385 parameters, no missing/unexpected keys
- One-sample gate: `PASS` on frozen medium-density validation image `img_0067`
- One-sample result: GT 1453, predicted 1762, localization F1 0.6407,
  latency 231.28 ms, peak VRAM 3444.80 MB
- Current next gate: review and advance to the next eligible model

The accepted rights artifact is
`pet-rights-99bfc15/rights-decision.json` with SHA-256
`40d5f7c9af3417567ae55d8f1e634c144a68a23b44314057313455cf27c438d7`.

## Frozen 36-image benchmark

Accepted run: `pet-ucf-qnrf-val-smoke-9b0fe87`.

- Input: one RGB image, UCF-QNRF long side capped at 1536, ImageNet normalization
- Native output: thresholded point coordinates/confidences and quadtree split map
- Count: number of point queries retained above probability 0.5
- Zone output: number of predicted points inside each calibrated image zone
- Density: no native density map; any heatmap must be labelled as a derived point visualization
- Training/fine-tuning: none
- Split: the same frozen 36 validation images as DM-Count and STEERER; official test remains sealed

| Metric | PET | STEERER | DM-Count |
|---|---:|---:|---:|
| MAE | 76.44 | 71.59 | 154.29 |
| RMSE | 129.80 | 97.97 | 229.81 |
| Signed bias | +15.67 | +43.92 | +3.03 |
| MAPE reference | 6.05% | 6.20% | 12.99% |
| Median latency | 57.99 ms | 130.70 ms | 69.53 ms |
| Batch-1 throughput | 17.24 FPS | 7.65 FPS | 14.38 FPS |
| Peak VRAM | 4,291.91 MB | 3,408.52 MB | 8,980.26 MB |

PET-specific spatial means at the harness 16-pixel radius:

- Localization precision/recall/F1: 0.5097 / 0.5161 / 0.5121
- Localization mean distance: 8.44 px
- Point-zone MAE: 66.32
- Density-band signed bias: low +5.0, medium +34.5, high +7.5

The technical harness passed `100/100` with 36/36 successful samples. This is
not an accuracy percentage. The review bundle is 2,613,293 bytes, contains 12
selected panels, and all 22 internal references were re-hashed successfully.

PET's count error is close to STEERER and much lower than DM-Count on this
subset, while PET is fastest. STEERER remains substantially stronger for the
measured localization and zone metrics. PET is explicitly research-only, so
this result cannot authorize commercial fine-tuning or deployment.
