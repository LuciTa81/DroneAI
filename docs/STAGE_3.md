# Stage 3 — DM-Count official reproduction

## Decision

Stage 3 uses the official DM-Count implementation pinned at commit
`cc5f2132e0d1328909f31b6d665b8e0b15c30467` and starts with ShanghaiTech
Part A. The paper reports MAE **59.7** and RMSE **95.7** on its 182-image test
set. Passing requires the three-seed median of both metrics to be within 5%.

ShanghaiTech Part A is used here only for private research reproduction. The
Kaggle copy linked by the official DM-Count repository is public. The archive
downloaded on 2026-07-13 was 348,994,252 bytes, but its license field is
`Unknown`. This stage must not imply permission for commercial training or
deployment.

## Official-checkpoint preflight (2026-07-13)

The upstream ShanghaiTech Part A checkpoint was evaluated on all 182 test
images in a Colab T4 runtime using the unmodified upstream evaluation contract:
`Crowd_sh(test_data, 512, 8, method="val")`, batch size 1, and the sum of the
predicted density map.

| Metric | Observed | Paper | Relative gap | 5% check |
|---|---:|---:|---:|---|
| MAE | 60.0325 | 59.7 | +0.5569% | PASS |
| RMSE | 95.9966 | 95.7 | +0.3100% | PASS |

The 182-image pass took 15.53 seconds on a Tesla T4 with PyTorch 2.11.0+cu128.
This proves that the pinned code, dataset and official weights reproduce the
paper result within tolerance. It does **not** complete Stage 3: the three-seed
faithful training lane and the test-isolated clean lane remain required.

Traceability:

- checkpoint SHA-256: `810ea89b2e3de766a2611230d272fc9c1ca1dc17f636ee2ca2aee51ba0f30f9f`
- Part A content-manifest SHA-256 (964 image/MAT files): `9f638d75f2e04a1bec007512e694c7abc0840fdcce5de55687a718a0a6b0d52b`
- archive SHA-256: `af5dc0978d676abc9f559540c24a903d18ab449206b970f0cc90f90e733146e3`
- recorded result: `results/stage3/official_checkpoint_preflight.json`
- reproducible Colab notebook: `notebooks/04_stage3_dm_count_official_checkpoint.ipynb`

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

## Faithful pilot policy

Before committing a suitable GPU server to the full 1000-epoch, three-seed
experiment, a single-seed pilot may stop after at least 100 epochs. The pilot
must evaluate both the latest test-selected `best_model_*.pth` and the last
resumable checkpoint on all 182 test images. It records predictions, checkpoint
hashes, the environment, and a separate 100-point pilot score.

Passing the pilot review does **not** pass Stage 3. The full 1000-epoch faithful
lane, three seeds, and the test-isolated clean lane remain required. The pilot
must be labelled `PILOT_COMPLETE_FULL_1000_EPOCH_DEFERRED` so it cannot be
mistaken for the paper reproduction or a production result.

### Seed 2026 pilot result (2026-07-14)

The Colab T4 run was stopped after epoch 140 and evaluated on all 182 test
images. The separate pilot harness passed 100/100. The test-selected checkpoint
and the last resumable checkpoint were evaluated independently:

| Checkpoint | MAE | RMSE | Images | Meaning |
|---|---:|---:|---:|---|
| `best_model_6.pth` | 71.2820 | 113.2125 | 182 | Best model selected using `test_data` |
| `last_ckpt.tar` | 81.3608 | 129.0051 | 182 | Model state at the epoch-140 stop |

The full result, predictions and score are stored under
`MyDrive/DroneAI/runs/stage-3/faithful/seed-2026/evaluation/pilot-epoch-140`.
The Git snapshot is `results/stage3/faithful_seed_2026_pilot.json`. This result
does not pass the paper-gap gate; the 1000-epoch run is deferred until a
suitable graphics server is available.

## Stage 3B clean-protocol smoke

Before any clean-lane long run, Stage 3B freezes a deterministic split inside
the 300 official training images: 240 for training and 60 for validation. The
split ranks image paths by SHA-256 with seed 2026 and persists the complete
manifest and split hash before training begins.

The official DM-Count loss, optimizer and training loop are preserved. Only the
ShanghaiTech dataset binding is replaced so both training and validation load
from `train_data`; `test_data` is neither loaded nor evaluated. A one-epoch
smoke must produce a resumable checkpoint, a validation-selected model, finite
validation metrics, environment metadata and a 100-point review bundle.

Stage 3B passes at 90/100 only if all split, isolation and training blockers
pass. Its validation accuracy is not a model-quality claim. The official 182
test images remain sealed until the final clean-lane evaluation for each seed.

Reproducible Colab notebook: `notebooks/05_stage3b_clean_smoke.ipynb`.

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
