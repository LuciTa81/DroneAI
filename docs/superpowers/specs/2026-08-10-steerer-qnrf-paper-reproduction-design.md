# STEERER UCF-QNRF Paper Reproduction Design

## Goal

Reproduce the official STEERER-HRNet-W48 UCF-QNRF protocol without changing or
weakening the existing development lane. The reproduction lane trains on all
1,201 official UCF-QNRF Train images, evaluates on all 334 official Test images,
and records that Test is used for checkpoint selection exactly as in the pinned
upstream implementation.

## Context and decision

The existing lane is a development experiment: 961 Train images are used for
training, 240 Train images are held out for validation, and official Test is
sealed. Its results remain valid and immutable.

Three implementation approaches were considered:

1. Add a protocol flag to the existing runner. This minimizes new code but risks
   weakening the sealed-Test invariants and making evidence from the two lanes
   easy to confuse.
2. Invoke the upstream scripts directly. This is closest to the repository but
   omits DroneAI dataset hashes, rights scope, checkpoint lineage, restart
   evidence, and result manifests.
3. Add an isolated paper-reproduction lane that reuses small, already-tested
   conversion and runtime primitives. This adds a narrow interface while keeping
   the development lane closed to Test.

Approach 3 is selected.

## Protocol

- Model: STEERER with `MocHRBackbone`, HRNet-W48, and `withMOE`.
- Initialization: the pinned ImageNet HRNet-W48 backbone only. The official full
  STEERER checkpoint is evaluation evidence and is never used to initialize the
  new training run. The immutable ImageNet source file and SHA-256 remain
  unchanged; the initialized backbone parameters are fine-tuned with the rest of
  STEERER because freezing them would depart from the official protocol.
- Data: official UCF-QNRF Train 1,201 and Test 334, with no internal validation
  split in this lane.
- Upstream source: commit
  `5b1854dbc2d280f2326d67c65515d8baf9083810` and `configs/QNRF_final.py`.
- Training settings: crop 768x768, scale 0.5-2.0, horizontal flip, density factor
  100, AdamW, learning rate 1e-4, weight decay 1e-4, 10 warm-up epochs, cosine
  schedule to epoch 800, effective batch 8, and Test long side 3072.
- Evaluation cadence: the pinned upstream `val_span` behavior. The official Test
  set is used during training to select best-MAE and best-RMSE checkpoints. Every
  report must label these numbers `official Test, test-selected`; they are not an
  untouched final estimate.
- Metrics: MAE and RMSE are primary paper-comparison metrics. Signed bias, MAPE
  reference, latency, FPS, peak VRAM, localization, and zone metrics may be
  computed as secondary DroneAI diagnostics but must not be presented as paper
  metrics.

## Isolated storage and identities

- Processed dataset:
  `/workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-paper-reproduction-v1`
- Checkpoints:
  `/workspace/data/checkpoints/steerer-paper-reproduction/<run-id>`
- Results:
  `/workspace/data/results/steerer-paper-reproduction/<run-id>`
- Official checkpoint validation result:
  `/workspace/data/results/steerer-paper-reproduction/official-checkpoint-test334-v1`

No existing B-lane directory is modified. Preparation refuses a non-empty
destination. Git contains only code, configuration, manifests, summaries, and
curated figures under 25 MiB; raw data and weights remain on the SSD.

## Gates and data flow

1. **B0 — freeze the development-lane checkpoint selection:** preregister epoch
   625 `best-mae.pth` as primary, epoch 175 `best-rmse.pth` as secondary, and
   epoch 800 `milestone-800.pth` as final-reference using their existing
   validation-selected roles and SHA-256 values. Test results cannot change these
   roles or authorize more B-lane tuning.
2. **B1 — one-time B final Test:** evaluate all three frozen B checkpoints on all
   334 official Test images, publish the same count metrics for each, and then
   mark the B lane closed. These results are `official Test, validation-selected`
   and are the untouched final estimate for the already-completed B development
   process.
3. **G0 — source and data audit:** verify project/upstream commits, raw Train/Test
   counts, annotation pairing, ImageNet backbone SHA-256, official checkpoint
   SHA-256, container digest, GPU, and storage.
4. **G1 — evaluator reproduction:** use the same frozen Test adapter to evaluate
   the pinned official checkpoint on all 334 images. Proceed only if the
   run is complete, finite, and close to the repository claim of 77.8 MAE and
   138.0 RMSE. Initial compatibility bounds are absolute difference at most 5.0
   MAE and 10.0 RMSE. A miss blocks training and starts a preprocessing/evaluator
   audit; the bound is never widened after seeing results.
5. **G2 — Train/Test preparation:** convert all 1,201 Train and 334 Test samples,
   write `train.txt` and `test.txt`, verify disjoint official partitions, and save
   content-addressed input/output inventories.
6. **G3 — synthetic and one-update smoke:** verify imports, CUDA matmul, one
   optimizer update, finite loss, checkpoint round-trip, and manifest reload.
7. **G4 — one-epoch smoke:** train one full epoch, run the approved official Test
   evaluation cadence, and verify all 334 observations and restart evidence.
8. **G5 — 800-epoch reproduction:** launch a fresh FP32 run from the pinned
   ImageNet backbone in the existing `crowd` tmux session. Persist atomic status,
   checkpoint/RNG/optimizer/scheduler state, and resume only from a manifested
   completed boundary.
9. **G6 — final comparison:** reload best-MAE, best-RMSE, and epoch-800
   checkpoints; compare with paper 74.3/128.3, official checkpoint 77.8/138.0,
   and the separate B-lane results. Publish the exact protocol and Test-selection
   caveat with every table.

## Failure handling

- Missing, duplicate, overlapping, or hash-mismatched samples stop the gate.
- Non-finite loss, density, count, or metric stops training before checkpoint
  promotion.
- The official checkpoint G1 miss blocks the new 800-epoch run because it means
  the local evaluator is not yet comparable to the claimed result.
- Interruption does not trigger automatic deletion or restart. Resume requires a
  matching checkpoint manifest and an inactive advisory lock.
- Existing B results, official checkpoint, raw UCF-QNRF, and unrelated GPU jobs
  are never removed or overwritten.

## Evidence and scope

Every result records project commit, upstream commit, container image digest,
Python/PyTorch/CUDA/GPU, seed, raw and processed inventory hashes, split-list
hashes, initialization artifact hash, checkpoint hash, command, timing, and
rights scope.

The lane is `PASS_COMMERCIAL_CANDIDATE`, not `PRODUCTION_APPROVED`. Code is MIT;
the project has internally accepted the Kaggle Apache-2.0 listing for this
experiment, while dataset provenance risk remains documented. The official
STEERER checkpoint remains research/evaluation-only because its weight license is
not explicit. Newly trained weights do not inherit that checkpoint, but they are
not approved for deployment until dataset, derived-weight, and field-validation
reviews pass.

## Acceptance criteria

- Existing development-lane tests and sealed-Test guards continue to pass.
- New tests prove the A lane requires exactly Train 1,201 and Test 334, rejects
  generic split substitution, uses isolated roots, and labels Test-selected
  results.
- New tests prove the B final-Test manifest freezes all three checkpoint roles
  and hashes before Test metrics exist, and forbids post-Test continuation.
- G1 evaluates exactly 334 unique Test samples and meets the frozen compatibility
  bounds before any long training starts.
- G2-G4 produce complete, hash-verifiable evidence before G5 begins.
- No official Test result is described as validation or as an untouched final
  generalization estimate.

## Non-goals

- No hyperparameter tuning on official Test.
- No use of the official STEERER full checkpoint as training initialization.
- No modification, deletion, or reinterpretation of the existing 961/240 lane.
- No production deployment decision from this reproduction alone.
