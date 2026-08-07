# STEERER T800 Long-Run Training Design

## Purpose

Extend the existing authoritative STEERER UCF-QNRF training lane from the T5
checkpoint through epoch 800 while preserving reproducibility, restart safety,
validation-based best-model selection, and the sealed official Test boundary.
The result must support a defensible comparison with the ICCV 2023 paper and
the authors' repository without claiming leaderboard equivalence.

## Approved operating policy

- Resume the existing run `steerer-qnrf-imagenet-20260806-t1-e` from the
  immutable epoch-5 milestone whose SHA-256 is
  `2ce6800bcbbdaace881e5f2ecc9f481616a67b9b96254232b7e8e5285ceeafc4`.
- Preserve the pinned 800-epoch cosine scheduler timeline and its 10-epoch
  warm-up. A resume must restore optimizer, scheduler, AMP scaler, RNG, and
  best-metric state rather than starting a new schedule.
- Run the complete 240-image Train-derived validation split every 25 epochs.
- Atomically overwrite `last.pth` at every validation boundary so a machine
  restart loses at most 25 epochs of work.
- Preserve `best-mae.pth` and `best-rmse.pth` whenever their corresponding
  validation metric improves.
- Preserve immutable `milestone-100.pth` through `milestone-800.pth` at each
  100-epoch boundary. T800 also produces `milestone-800.pth`.
- Keep the official UCF-QNRF Test partition sealed for the entire T800 run.
- Keep checkpoints and full logs on the home5090 SSD. Commit only small JSON,
  Markdown, hash manifests, and a curated set of visual outputs to Git.

## Architecture

### Stage contract

Add `T800` as a first-class approved stage alongside T0, T1, T5, and T50.
The CLI must require both `--stage T800` and `--approved-stage T800`, as well as
an in-run checkpoint verified by the authoritative checkpoint manifest. The
profile declares epoch 800 as the stop ceiling. Any unsupported stage name,
missing approval, missing resume checkpoint, wrong predecessor, or scheduler
horizon other than 800 fails before model construction or GPU work.

T800 may resume from a verified T5, T50, or T800 progress checkpoint in the
same run lineage. It must never resume from T0/T1 or from an externally supplied
checkpoint. Resuming a partially completed T800 run continues from the last
completed 25-epoch validation boundary.

### Training and validation loop

The runner derives validation boundaries from the current completed epoch and
the fixed interval 25: `25, 50, ..., 800`, skipping boundaries already passed.
For each boundary it trains only to that ceiling, captures the training-boundary
RNG state, evaluates all 240 validation images, computes the existing 17 metrics,
and finalizes evidence before proceeding.

Validation must not perturb the following training segment. After validation,
the runner restores the captured training-boundary RNG state into the checkpoint.
A resumed run must therefore produce the same subsequent sampling and learning
rate sequence as an uninterrupted run.

### Evidence and checkpoints

Each validation boundary writes an immutable metric file named
`metrics.epoch-NNN.json` and a content-addressed environment manifest. The
checkpoint manifest records epoch, SHA-256, byte count, environment hash,
metric reason, and parent checkpoint SHA-256.

Checkpoint writes remain atomic. The boundary checkpoint becomes the parent of
the next boundary, producing an auditable chain from milestone 005 through 800.
`last.pth` is replaceable recovery state; best and milestone files are preserved
artifacts. A write or reload/hash failure stops the run and leaves the previous
verified `last.pth` usable.

The final `metrics.t800.json` is an immutable copy of the epoch-800 observation,
while the T800 summary also identifies the epoch and hash of the best-MAE and
best-RMSE checkpoints. The authoritative stage score verifies every referenced
artifact from disk and reports `PASS_COMMERCIAL_CANDIDATE` only when all hard
blockers pass.

### Best-model decision

MAE is the primary selection metric because the paper and repository rank
UCF-QNRF counting performance primarily by MAE. RMSE is the secondary model
selection metric. The report also preserves MAPE, signed bias, GAME L1,
quadrant zone MAE, localization precision/recall/F1, latency, throughput, and
peak VRAM for operational interpretation.

The final report evaluates the selected best-MAE checkpoint again on the same
fixed 240-image validation split. Best-RMSE is also reloaded and verified; if it
is a different checkpoint, its recorded RMSE and full metric bundle are shown
separately. No hyperparameter or checkpoint choice may use the sealed Test.

## Runtime and recovery

The long run executes inside `crowd-jupyter` under the existing `tmux crowd`
workflow on `home5090-pop`. Stdout and stderr are written to a run-specific SSD
log. A lightweight status JSON is updated atomically after every epoch and
validation boundary with epoch, global step, most recent loss, validation state,
checkpoint hash, elapsed time, and estimated remaining time.

The launcher holds a non-blocking Linux advisory lock for the lifetime of the
STEERER process and refuses to start if another live process owns it. The lock
is released by the kernel on normal exit, process death, container stop, or host
reboot; persistent metadata in the lock file is informational and cannot create
a stale-lock dead end. After host reboot, the operator restarts the same command
with the verified `last.pth`; the runner recomputes its hash and lineage before
allocating training work.

Expected runtime is approximately 8 hours from epoch 5 to 800 based on the T1
and T5 measurements, with roughly 30-40 minutes attributable to periodic full
validation. Runtime is an estimate rather than an acceptance criterion.

## Failure handling

- CUDA OOM: stop, preserve the last verified boundary checkpoint and captured
  diagnostic evidence; do not silently change batch size or accumulation.
- Non-finite loss, gradient, density, or metric: stop without promotion.
- Dataset, split, upstream, backbone, config, container, or checkpoint hash
  mismatch: stop before training.
- Official Test path access: hard failure.
- Interrupted epoch between boundaries: discard its incomplete state and resume
  from the preceding verified boundary.
- Validation or artifact-finalization failure: stop before entering the next
  training segment.

## Verification strategy

Implementation follows test-first development. Tests must first fail for the
missing T800 behavior and then pass after the minimum implementation.

Required automated coverage:

1. CLI accepts T800 only with matching approval and a resume checkpoint.
2. Profile and stage parsers map T800 to exactly epoch 800.
3. Resume policy accepts verified T5/T50/T800 boundary checkpoints and rejects
   T0/T1, foreign, unmanifested, and lineage-mismatched files.
4. Boundary generation is exactly every 25 epochs through 800 and skips already
   completed boundaries.
5. Validation covers exactly 240 samples at every boundary.
6. RNG, optimizer, scheduler, scaler, and global-step state round-trip across a
   boundary and resume.
7. Atomic recovery keeps the prior valid checkpoint after injected failures.
8. Best-MAE, best-RMSE, last, and 100-epoch milestone policies retain only the
   intended artifacts with correct hashes and parent lineage.
9. T800 scoring requires finite complete metrics, sealed Test evidence, and
   disk-recomputed hashes.
10. Existing T0/T1/T5/T50 behavior remains unchanged.

Before launching the real run, the focused training suite, a synthetic T800
boundary simulation, `git diff --check`, remote repository alignment, CUDA
matmul smoke, dataset/split hashes, free disk, and GPU-idle checks must pass.

## Completion criteria

T800 is complete only when all of the following are true:

- epoch 800 is reached on the uninterrupted scheduler lineage from T5;
- every scheduled validation boundary has a complete 240-sample metric bundle;
- the final, best-MAE, best-RMSE, and milestone-800 checkpoints reload and
  hash-verify;
- the official Test remains sealed;
- the harness records 100/100 `PASS_COMMERCIAL_CANDIDATE` with no blockers;
- the best-MAE checkpoint receives a fresh full-validation confirmation;
- the final report gives T5-to-best improvement and absolute/relative gaps to
  the paper's 74.3/128.3 and repository's 77.8/138.0 figures, with split and
  protocol limitations stated explicitly;
- small evidence is committed and pushed, while large weights remain on SSD.

This completion does not grant `PRODUCTION_APPROVED` and does not authorize use
of the official Test. Those remain separate decisions.
