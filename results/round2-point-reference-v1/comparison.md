# Round 2 reference comparison

- Scope: `PASS_RESEARCH_ONLY`
- Comparison: `research_reference_only`
- Ranking eligible: `false`
- Shared sample manifest: `37adb8e058611de79d6f97c3181106f6606f5d1bc1a9b611720e4a699395cc04`

## ucf-qnrf-kaggle-apache

Rights: `PASS_COMMERCIAL_CANDIDATE`; samples/model: `334`

| Model | MAE | RMSE | Bias | MAPE ref. (%) | Median ms | FPS | Peak VRAM MB | Spatial | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| steerer | 86.67 | 152.57 | 17.19 | 11.61 | 141.15 | 7.08 | 4014.92 | game_l1=95.994 | PASS_RESEARCH_ONLY |
| pet | 82.28 | 152.91 | -10.71 | 11.46 | 57.49 | 17.40 | 5132.44 | localization_f1=0.448 | PASS_RESEARCH_ONLY |
| apgcc | 125.86 | 228.05 | -62.40 | 19.73 | 67.75 | 14.76 | 6229.94 | localization_f1=0.650 | PASS_RESEARCH_ONLY |

## jhu-crowd-plus-v2

Rights: `PASS_RESEARCH_ONLY`; samples/model: `500`

| Model | MAE | RMSE | Bias | MAPE ref. (%) | Median ms | FPS | Peak VRAM MB | Spatial | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| steerer | 94.99 | 243.78 | 67.76 | 40.70 | 21.98 | 45.49 | 3366.30 | game_l1=95.017 | PASS_RESEARCH_ONLY |
| pet | 89.04 | 229.25 | 48.66 | 44.18 | 31.66 | 31.59 | 5133.14 | localization_f1=0.282 | PASS_RESEARCH_ONLY |
| apgcc | 89.46 | 259.29 | 25.18 | 46.49 | 9.83 | 101.74 | 5614.81 | localization_f1=0.640 | PASS_RESEARCH_ONLY |

## up-count-v1

Rights: `PASS_RESEARCH_ONLY`; samples/model: `166`

| Model | MAE | RMSE | Bias | MAPE ref. (%) | Median ms | FPS | Peak VRAM MB | Spatial | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| steerer | 17.72 | 43.76 | -17.29 | 58.40 | 119.74 | 8.35 | 2585.27 | game_l1=19.487 | PASS_RESEARCH_ONLY |
| pet | 19.22 | 46.82 | -17.81 | 68.53 | 55.01 | 18.18 | 3451.92 | localization_f1=0.047 | PASS_RESEARCH_ONLY |
| apgcc | 69.65 | 207.54 | 53.36 | 1214.41 | 57.90 | 17.27 | 3963.69 | localization_f1=0.258 | PASS_RESEARCH_ONLY |

## Limitations

- Metrics are reported per dataset and are not pooled into a commercial score.
- JHU-CROWD++ and UP-COUNT evidence is non-commercial research reference only.
- Frozen checkpoints use different training domains; this comparison does not authorize production selection.
- STEERER checkpoint terms remain research-only pending independent clearance.
- PET code and official checkpoint are academic-use only.
- APGCC code is MIT, but the published checkpoint has no separate commercial terms.
