# Stage 3 — DM-Count official reproduction

## Decision

Stage 3 uses the official DM-Count implementation pinned at commit
`cc5f2132e0d1328909f31b6d665b8e0b15c30467` and starts with ShanghaiTech
Part A. The paper reports MAE **59.7** and RMSE **95.7** on its 182-image test
set. Passing requires the three-seed median of both metrics to be within 5%.

ShanghaiTech Part A is used here only for private research reproduction. The
Kaggle copy linked by the official DM-Count repository is public and about
174 MB, but its license field is `Unknown`. This stage must not imply permission
for commercial training or deployment.

## Two lanes

The upstream trainer maps ShanghaiTech `test_data` to its validation loader and
uses those metrics to select the best checkpoint. Repeating that behavior can
help reproduce the reported number, but it is not an acceptable test-isolated
experiment for DroneAI.

1. **Faithful lane:** preserve the upstream behavior, label every checkpoint as
   test-selected, and measure the paper gap.
2. **Clean lane:** freeze a deterministic validation partition from the 300
   training images, select checkpoints only on that partition, then evaluate the
   182 test images once per seed.

The faithful lane answers "can the official artifact be reproduced?" The clean
lane answers "can we trust our own model-selection process?" Both are required.

## 100-point gate

| Group | Points | Blocking evidence |
|---|---:|---|
| Paper gap | 20 | Three-seed median MAE and RMSE within 5% |
| Seed stability | 20 | Seeds 2026, 2027 and 2028 complete |
| Evaluation integrity | 35 | Correct split, 182 predictions per run, clean test isolation |
| Traceability | 20 | Pinned commit, license, configs, predictions and checkpoint hashes |
| Runtime | 5 | GPU and per-image inference time recorded |

Pass threshold: **85/100**, with every blocker passing. Dataset-rights evidence
is separately visible as a five-point review item and does not convert this
public research benchmark into a production asset.

## Review artifacts

Stage 3 writes the following under `MyDrive/DroneAI/runs/stage-3/`:

1. `score.md` and `score.json`;
2. `evidence.snapshot.json`;
3. per-seed prediction files;
4. environment and runtime snapshots;
5. checkpoint SHA-256 records.
