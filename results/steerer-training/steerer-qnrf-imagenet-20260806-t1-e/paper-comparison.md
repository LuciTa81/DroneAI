# STEERER paper comparison

## Primary references

- ICCV 2023 paper: <https://arxiv.org/abs/2308.10468>
- Authors' implementation: <https://github.com/taohan10200/STEERER>

The paper reports UCF-QNRF MAE/RMSE-style MSE of 74.3/128.3 and localization
F1/precision/recall of 75.5/78.6/72.7%. The authors' repository reports a
separate reproduction of 77.8/138.0 and 75.6/79.7/72.0%.

## Diagnostic gap after one epoch

| Reference | Metric | Reference | T1 | Absolute gap | Relative error gap |
|---|---|---:|---:|---:|---:|
| Paper | MAE | 74.3 | 1,574.708 | +1,500.408 | +2,019.39% |
| Paper | RMSE | 128.3 | 1,930.137 | +1,801.837 | +1,404.39% |
| Authors' repo | MAE | 77.8 | 1,574.708 | +1,496.908 | +1,924.05% |
| Authors' repo | RMSE | 138.0 | 1,930.137 | +1,792.137 | +1,298.65% |

The localization figures are not an apples-to-apples comparison. This T1 run
uses a fixed 16-pixel radius in original-image coordinates and records
F1/precision/recall of 8.076/4.214/96.842%. The paper follows the IIM/NWPU
localization protocol. The values can be used as diagnostic evidence, but the
percentage-point gaps must not be presented as a leaderboard reproduction.

Likewise, T1 is evaluated on a deterministic 240-image validation subset made
from the official Train package. The paper and authors' repository report the
official benchmark test protocol after full training. This one-epoch result is
therefore a pipeline and learning-signal checkpoint, not a paper reproduction.

The paper text describes Adam with a warm-up to 1e-5, while the pinned authors'
`configs/QNRF_final.py` uses AdamW with base learning rate 1e-4. This run follows
the pinned repository configuration and records that discrepancy explicitly.

