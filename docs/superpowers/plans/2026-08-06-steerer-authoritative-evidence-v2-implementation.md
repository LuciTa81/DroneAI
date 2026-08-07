# STEERER Authoritative Evidence v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a checkpoint-bound, independently scoreable T0 and T1 STEERER UCF-QNRF training result on home5090, then compare T1 to the paper and official-repository baselines.

**Architecture:** The pinned training runtime returns complete measured observations. A new pure evidence module converts those observations into the exact Task 7 metric schema and builds immutable schema-v2 environment bytes. The runner writes `metrics.json`, then the content-addressed environment, then the checkpoint containing that environment hash; the existing authority gate recomputes every hash before scoring.

**Tech Stack:** Python 3.12, PyTorch, NumPy, existing DroneAI evaluation metrics, pinned STEERER at `5b1854dbc2d280f2326d67c65515d8baf9083810`, pytest, RTX 5090 in `crowd-jupyter`.

## Global Constraints

- UCF-QNRF official Test remains sealed; only the prepared official-Train-derived 961/240 split is permitted.
- Initialization is the verified ImageNet HRNet-W48 backbone only; no STEERER model checkpoint may be loaded.
- Existing run `steerer-qnrf-imagenet-20260806-a` is immutable and remains unscored.
- Fresh run IDs are `steerer-qnrf-imagenet-20260806-t0-b` and `steerer-qnrf-imagenet-20260806-t1-b`.
- Artifact order is metrics, immutable schema-v2 environment, then checkpoint bound to the environment SHA-256.
- T0 must score at least 70 without a blocker before T1; T1 must score at least 75 without a blocker.
- Do not start T5 or T50 in this execution cycle.
- Checkpoints and raw outputs stay under `/workspace/data`; Git receives only small verified evidence.

---

### Task 1: Deterministic Stage Metric Aggregation

**Files:**
- Create: `src/droneai/steerer_training_evidence.py`
- Create: `tests/test_steerer_training_evidence.py`

**Interfaces:**
- Consumes: `UpdateObservation`, `EpochObservation`, and per-sample `ValidationSampleObservation` values supplied by the runner.
- Produces: `build_t0_metrics(update: UpdateObservation) -> dict[str, float]`, `build_t1_metrics(epoch: EpochObservation, validation: ValidationObservation) -> dict[str, float]`, and `strict_metrics_payload(run_id: str, stage: str, metrics: Mapping[str, float]) -> dict[str, object]`.

- [ ] **Step 1: Write failing T0 schema test**

```python
def test_t0_metrics_use_actual_optimizer_update_loss() -> None:
    update = UpdateObservation(1.25, (10.0,), 0.5, 1e-4)
    assert build_t0_metrics(update) == {"train_loss": 1.25}
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `python -m pytest tests/test_steerer_training_evidence.py::test_t0_metrics_use_actual_optimizer_update_loss -q`

Expected: import failure because `steerer_training_evidence` does not exist.

- [ ] **Step 3: Implement T0 aggregation and strict finite validation**

`build_t0_metrics` accepts only a finite measured `UpdateObservation.loss`. `strict_metrics_payload` requires the exact stage key set from `steerer_training_gate`, rejects NaN/Infinity, and emits schema version 1 with run/stage lineage.

- [ ] **Step 4: Write failing T1 formula tests**

Use two hand-calculated sample observations so expected MAE, RMSE, signed bias, percent MAPE, GAME L1, quadrant MAE, localization precision/recall/F1, median/p95 latency, FPS, peak VRAM, density/count difference, validation loss, mean train loss, and first-versus-last 10% window improvement are literals.

- [ ] **Step 5: Run the T1 tests and confirm RED**

Run: `python -m pytest tests/test_steerer_training_evidence.py -q`

Expected: failure because T1 aggregation is incomplete.

- [ ] **Step 6: Implement minimal T1 aggregation**

Aggregate exactly the 17 names in `REQUIRED_T1_METRICS`. Require 240 unique sample IDs in production through the supplied observation count, one-to-one localization totals, and finite values. Define `fps = 1000 / median_latency_ms`; reject non-positive median latency.

- [ ] **Step 7: Verify and commit**

Run: `python -m pytest tests/test_steerer_training_evidence.py tests/test_steerer_training_gate.py -q`

Commit: `git add src/droneai/steerer_training_evidence.py tests/test_steerer_training_evidence.py && git commit -m "Add STEERER training metric evidence"`

---

### Task 2: Complete Real Validation Observations

**Files:**
- Modify: `src/droneai/steerer_training_runner.py`
- Modify: `tests/test_steerer_training_runner.py`

**Interfaces:**
- Produces: `ValidationSampleObservation(sample_id, target_count, predicted_count, game_l1, quadrant_zone_mae, localization_tp, localization_fp, localization_fn, latency_ms, peak_vram_mb, density_sum_count_difference)` and an expanded `ValidationObservation(samples, loss, density_values, mae, rmse)`.
- Consumes: prepared `jsons/<sample_id>.json`, loader resize metadata, highest-resolution prediction/target density, and `extract_steerer_points`.

- [ ] **Step 1: Write failing runner contract tests**

Add tests proving duplicate/missing validation sample observations fail, a T1 result retains epoch losses, and the fake engine exposes 240 complete sample observations rather than two illustrative counts.

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_steerer_training_runner.py -q`

Expected: dataclass/signature failures for the new observation contract.

- [ ] **Step 3: Implement validation metadata preservation**

Keep the loader names and resize ratios alongside tensors. Resolve ground-truth points from the approved processed JSON only. Reject paths outside `processed_root/jsons`, duplicate IDs, mismatched human counts, and any point outside original image bounds.

- [ ] **Step 4: Implement measured per-sample metrics**

Time only synchronized batch-1 inference. Convert highest-resolution predicted/ground-truth densities to NumPy, calculate four equal quadrants and GAME L1, extract STEERER points at the pinned 16-pixel original-coordinate radius, accumulate one-to-one TP/FP/FN, and record peak allocated CUDA memory. Count is the highest-resolution density sum.

- [ ] **Step 5: Preserve training-window observations**

Retain every optimizer-update loss from epoch 1. The first and last windows each contain `max(1, ceil(update_count * 0.1))` updates; do not use AMP probe losses.

- [ ] **Step 6: Verify and commit**

Run: `python -m pytest tests/test_steerer_training_runner.py tests/test_steerer_adapter.py -q`

Commit: `git add src/droneai/steerer_training_runner.py tests/test_steerer_training_runner.py && git commit -m "Collect complete STEERER validation observations"`

---

### Task 3: Authoritative Finalization and CLI Runtime Identity

**Files:**
- Modify: `src/droneai/steerer_training_evidence.py`
- Modify: `src/droneai/steerer_training_runner.py`
- Modify: `scripts/run_steerer_ucf_training.py`
- Modify: `tests/test_steerer_training_evidence.py`
- Modify: `tests/test_steerer_training_runner.py`
- Modify: `tests/test_steerer_training_runbook.py`
- Modify: `docs/runbooks/steerer-ucf-qnrf-training-home5090.md`

**Interfaces:**
- Produces: `finalize_stage_evidence(...) -> FinalizedStageEvidence` containing metrics path/hash, content-addressed environment path/hash, checkpoint path/hash, and `StageResult.metrics_path/environment_path/checkpoint_path`.
- Consumes: actual Docker image digest, project/profile/upstream/dataset/backbone identities, batch plan, measured metrics, and checkpoint state.

- [ ] **Step 1: Write failing artifact-order and hash-binding tests**

Tests must observe that the checkpoint save is never called before `metrics.json` and `environment.<sha>.json` exist, that the environment embeds the recomputed metrics SHA-256, and that the checkpoint embeds the same environment SHA-256. A simulated environment-write failure must leave no new authoritative checkpoint.

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_steerer_training_evidence.py tests/test_steerer_training_runner.py -q`

Expected: failure because the current runner writes schema-v1 environment before training.

- [ ] **Step 3: Build exact schema-v2 environment bytes**

Populate the exact sections enforced by `_environment_sections`: project, upstream, dataset, initialization, training, runtime, batch, and metrics. Obtain Git, upstream, dataset, split, sealed-marker, backbone, profile, and config facts from disk; obtain Python/Torch/CUDA/GPU from the active runtime; accept only a `sha256:` plus 64-hex container image ID passed by CLI.

- [ ] **Step 4: Refactor stage finalization order**

For T0, derive metrics from the actual update. For T1, derive metrics from the complete epoch and validation observations. Atomically write immutable metrics, then immutable environment, then call `_persist_and_reload_checkpoint` with the environment hash. Existing resume behavior remains unchanged for future stages.

- [ ] **Step 5: Require runtime digest in the CLI**

Add required `--container-image-digest`, validate its exact syntax before model construction, pass it to `run_training_stage`, and include artifact paths/hashes in the completion JSON. No Test, trust override, or checkpoint-origin option is added.

- [ ] **Step 6: Update exact runbook commands**

Use `docker inspect --format '{{.Image}}' crowd-jupyter` on the host and pass the returned image ID to fresh T0/T1 commands. Document distinct run IDs and the T0 score stop before T1.

- [ ] **Step 7: Verify existing authority gate and checkpoint suites**

Run: `python -m pytest tests/test_steerer_training_evidence.py tests/test_steerer_training_runner.py tests/test_steerer_training_checkpoint.py tests/test_steerer_training_gate.py tests/test_steerer_training_runbook.py -q`

Commit: `git add src/droneai/steerer_training_evidence.py src/droneai/steerer_training_runner.py scripts/run_steerer_ucf_training.py tests docs/runbooks/steerer-ucf-qnrf-training-home5090.md && git commit -m "Bind STEERER checkpoints to measured evidence"`

---

### Task 4: Local Verification, Remote T0/T1, and Paper Comparison

**Files:**
- Add after verified remote execution: small JSON/Markdown evidence under `results/steerer-training/`
- Modify only for defects demonstrated by a failing test: Task 1–3 files.

**Interfaces:**
- Consumes: T0/T1 authoritative artifacts, existing score CLI, official STEERER paper/repository baselines.
- Produces: scored T0/T1 result bundle and a comparison reporting absolute and relative deltas.

- [ ] **Step 1: Run local focused and full verification**

Run: `python -m pytest tests/test_steerer_training_evidence.py tests/test_steerer_training_runner.py tests/test_steerer_training_checkpoint.py tests/test_steerer_training_gate.py tests/test_steerer_training_runbook.py -q`

Run: `python -m pytest -q`

Run: `git diff --check`

- [ ] **Step 2: Push and fast-forward the remote worktree safely**

Push `agent/steerer-ucf-training-implementation`, then use `git pull --ff-only` in `/workspace/.worktrees/steerer-ucf-training`. Confirm both worktrees are clean and at the same commit.

- [ ] **Step 3: Run fresh T0 and score it**

Use `steerer-qnrf-imagenet-20260806-t0-b`. Require one optimizer update, finite `train_loss`, checkpoint round-trip, authoritative verification, no blocker, and score at least 70. Preserve any failure without overwriting it.

- [ ] **Step 4: Run fresh T1 only after T0 passes**

Use `steerer-qnrf-imagenet-20260806-t1-b`. Run exactly epoch 1 from ImageNet-backbone initialization and all 240 validation samples. Do not access Test.

- [ ] **Step 5: Score T1 and copy only small evidence**

Require the exact 17 metrics, score at least 75, and no blocker. Keep `.pth`, raw density maps, and full logs on SSD. Copy only score, metrics, environment, lineage, README, and small summaries to Git after recomputing hashes.

- [ ] **Step 6: Compare against published baselines**

Report our MAE/RMSE against both the ICCV paper result and the official repository reproduction. For each lower-is-better metric calculate `ours - baseline` and `(ours - baseline) / baseline * 100`. Report localization separately because its protocol may differ; never imply direct comparability without matching split and radius.

- [ ] **Step 7: Final verification and stop**

Run the full test suite, `git diff --check`, re-run authoritative verification from SSD, and verify copied hashes. Report elapsed time, batch/AMP choice, all metrics, checkpoint hash, rights scope, paper/repository deltas, and limitations. Do not start T5.

