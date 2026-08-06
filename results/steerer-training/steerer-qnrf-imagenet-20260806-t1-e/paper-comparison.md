# STEERER paper comparison

## Primary references

- ICCV 2023 paper: <https://arxiv.org/abs/2308.10468>
- Authors' implementation: <https://github.com/taohan10200/STEERER>

The paper reports UCF-QNRF MAE/RMSE-style MSE of 74.3/128.3 and localization
F1/precision/recall of 75.5/78.6/72.7%. The authors' repository reports a
separate reproduction of 77.8/138.0 and 75.6/79.7/72.0%.

## Diagnostic gap at T1 and T5

| Reference | Metric | Reference | T1 | T5 | T5 absolute gap | T5 relative gap |
|---|---|---:|---:|---:|---:|---:|
| Paper | MAE | 74.3 | 1,574.708 | 389.995 | +315.695 | +424.89% |
| Paper | RMSE | 128.3 | 1,930.137 | 1,029.695 | +901.395 | +702.57% |
| Authors' repo | MAE | 77.8 | 1,574.708 | 389.995 | +312.195 | +401.28% |
| Authors' repo | RMSE | 138.0 | 1,930.137 | 1,029.695 | +891.695 | +646.16% |

T1 to T5 reduced MAE by 1,184.713 (-75.23%) and RMSE by 900.442
(-46.65%). This is clear learning progress, but T5 MAE is still about 5.25 times
the paper value and T5 RMSE is about 8.03 times the paper value.

The localization figures are not an apples-to-apples comparison. This run uses
a fixed 16-pixel radius in original-image coordinates. T5 records
F1/precision/recall of 52.963/44.898/64.559%, versus the paper's
75.5/78.6/72.7%. The raw differences are -22.537/-33.702/-8.141 percentage
points, but the paper follows the IIM/NWPU localization protocol. These values
are diagnostic evidence and must not be presented as a leaderboard reproduction.

Likewise, T1 and T5 are evaluated on a deterministic 240-image validation subset
made from the official Train package. The official Test remained sealed. The
paper and authors' repository report the official benchmark Test protocol after
full training. These stages are learning milestones, not a paper reproduction
or a leaderboard-equivalent comparison.

The paper text describes Adam with a warm-up to 1e-5, while the pinned authors'
`configs/QNRF_final.py` uses AdamW with base learning rate 1e-4. This run follows
the pinned repository configuration and records that discrepancy explicitly.

## Decision

T5 proves that the training pipeline, resume lineage, and learning signal work.
It does not yet establish production accuracy. The next evidence gate is T50,
which must resume from the hash-identified T5 checkpoint and must not access the
official Test. Only after that gate should a full-training budget or final Test
evaluation be approved.
