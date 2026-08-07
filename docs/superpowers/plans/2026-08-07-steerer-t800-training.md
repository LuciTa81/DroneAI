# STEERER T800 Long-Run Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume the authoritative STEERER UCF-QNRF run from epoch 5 through epoch 800 with 25-epoch validation, restart-safe checkpointing, best-model selection, and a verified paper-gap report.

**Architecture:** Add T800 to the existing stage trust boundary, isolate long-run boundary/status/lock mechanics in a focused module, and make the runner finalize a hash-bound validation checkpoint every 25 epochs. The run retains one 800-epoch optimizer/scheduler/RNG lineage, keeps the official Test sealed, selects best MAE/RMSE checkpoints on the fixed 240-image validation split, and stores large artifacts only on the home5090 SSD.

**Tech Stack:** Python 3.12/3.13, PyTorch 2.9 nv25.10, pytest, JSON/SHA-256 evidence, Docker `crowd-jupyter`, RTX 5090, tmux, Git/GitHub.

## Global Constraints

- Resume run ID `steerer-qnrf-imagenet-20260806-t1-e` from SHA-256 `2ce6800bcbbdaace881e5f2ecc9f481616a67b9b96254232b7e8e5285ceeafc4`.
- Keep the scheduler horizon exactly 800 epochs and restore optimizer, scheduler, scaler, RNG, global step, and best metrics.
- Validate all 240 fixed Train-derived validation samples every 25 epochs.
- Preserve milestones at epochs 100, 200, 300, 400, 500, 600, 700, and 800.
- Keep the official UCF-QNRF Test sealed and reject any Test path access.
- Do not silently change physical batch 8, accumulation 1, AMP decision, optimizer, loss, augmentation, split, or seed 3035.
- Keep checkpoints/logs on `/workspace/data`; commit only small evidence and curated outputs.
- Every production behavior change follows red-green TDD.

---

### Task 1: Extend the authoritative stage vocabulary to T800

**Files:**
- Modify: `configs/training/steerer_ucf_qnrf_imagenet.home5090.json`
- Modify: `src/droneai/steerer_training_profile.py`
- Modify: `src/droneai/steerer_training_checkpoint.py`
- Modify: `src/droneai/steerer_training_evidence.py`
- Modify: `src/droneai/steerer_training_gate.py`
- Modify: `src/droneai/steerer_training_upstream.py`
- Modify: `scripts/run_steerer_ucf_training.py`
- Modify: `scripts/score_steerer_ucf_training.py`
- Test: `tests/test_steerer_training_profile.py`
- Test: `tests/test_steerer_training_checkpoint.py`
- Test: `tests/test_steerer_training_gate.py`
- Test: `tests/test_steerer_training_runner.py`
- Test: `tests/test_steerer_training_upstream.py`

**Interfaces:**
- Consumes: existing T0/T1/T5/T50 stage validation and immutable evidence APIs.
- Produces: `T800` as an accepted stage mapped to epoch 800 with score threshold 90 and explicit approval/resume requirements.

- [ ] **Step 1: Write failing profile and CLI tests**

Add assertions equivalent to:

```python
def test_profile_declares_t800_epoch_ceiling() -> None:
    profile = load_training_profile(PROFILE_PATH)
    assert profile.stage_epochs == {
        "T0": 0, "T1": 1, "T5": 5, "T50": 50, "T800": 800,
    }


def test_cli_requires_matching_t800_approval_and_resume(tmp_path: Path) -> None:
    args = run_cli._parser().parse_args(
        _required_cli_args(tmp_path)
        + ["--stage", "T800", "--approved-stage", "T800", "--resume", "last.pth"]
    )
    run_cli._validate_stage_ceiling(args)
    args.approved_stage = "T50"
    with pytest.raises(PermissionError, match="T800.*approved-stage"):
        run_cli._validate_stage_ceiling(args)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
py -3.13 -m pytest tests/test_steerer_training_profile.py tests/test_steerer_training_runner.py -k "t800" -q
```

Expected: FAIL because `T800` is not accepted and the profile lacks epoch 800.

- [ ] **Step 3: Write failing checkpoint and score-gate tests**

Add T800 fixtures and assertions equivalent to:

```python
def test_t800_checkpoint_payload_and_score_are_authoritative(tmp_path: Path) -> None:
    fixture = _authoritative_fixture(tmp_path, stage="T800")
    evidence = verify_authoritative_training_evidence(
        fixture.inputs, _profile_override=fixture.profile, torch_module=_FakeTorch()
    )
    report = score_training_stage(evidence, stage="T800")
    assert report.threshold == 90
    assert report.score == 100
    assert report.status == "PASS_COMMERCIAL_CANDIDATE"
```

- [ ] **Step 4: Run the checkpoint/gate tests and verify RED**

Run:

```powershell
py -3.13 -m pytest tests/test_steerer_training_checkpoint.py tests/test_steerer_training_gate.py -k "t800" -q
```

Expected: FAIL because stage validation rejects T800.

- [ ] **Step 5: Implement the minimum T800 vocabulary**

Make the exact stage maps include T800:

```python
_STAGES = frozenset({"T0", "T1", "T5", "T50", "T800"})
_STAGE_EPOCHS = {"T0": 0, "T1": 1, "T5": 5, "T50": 50, "T800": 800}
STAGE_THRESHOLDS = {"T0": 70, "T1": 75, "T5": 80, "T50": 85, "T800": 90}
```

Extend both CLI `choices` tuples and require `--approved-stage T800` plus a
resume checkpoint. Update the profile JSON and exact profile validator with
`"T800": 800`. Keep all pre-existing stage behavior unchanged.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```powershell
py -3.13 -m pytest tests/test_steerer_training_profile.py tests/test_steerer_training_checkpoint.py tests/test_steerer_training_gate.py tests/test_steerer_training_runner.py -k "t800 or stage" -q
```

Expected: PASS.

- [ ] **Step 7: Commit the stage contract**

```powershell
git add configs/training/steerer_ucf_qnrf_imagenet.home5090.json src/droneai/steerer_training_profile.py src/droneai/steerer_training_checkpoint.py src/droneai/steerer_training_gate.py scripts/run_steerer_ucf_training.py scripts/score_steerer_ucf_training.py tests/test_steerer_training_profile.py tests/test_steerer_training_checkpoint.py tests/test_steerer_training_gate.py tests/test_steerer_training_runner.py
git commit -m "Add authoritative STEERER T800 stage"
```

---

### Task 2: Add deterministic long-run boundaries, status, and locking

**Files:**
- Create: `src/droneai/steerer_training_longrun.py`
- Create: `tests/test_steerer_training_longrun.py`

**Interfaces:**
- Consumes: completed epoch, target epoch, run result directory, run ID.
- Produces: `validation_boundaries(completed_epoch: int, target_epoch: int = 800) -> tuple[int, ...]`, `RunLock`, and `write_longrun_status(path: Path, payload: Mapping[str, object]) -> None`.

- [ ] **Step 1: Write failing boundary tests**

```python
def test_validation_boundaries_resume_after_completed_epoch() -> None:
    assert validation_boundaries(5)[:3] == (25, 50, 75)
    assert validation_boundaries(50)[:2] == (75, 100)
    assert validation_boundaries(799) == (800,)
    assert validation_boundaries(800) == ()


@pytest.mark.parametrize("epoch", [-1, 801])
def test_validation_boundaries_reject_invalid_epoch(epoch: int) -> None:
    with pytest.raises(ValueError):
        validation_boundaries(epoch)
```

- [ ] **Step 2: Run the boundary tests and verify RED**

```powershell
py -3.13 -m pytest tests/test_steerer_training_longrun.py -q
```

Expected: collection FAIL because the module does not exist.

- [ ] **Step 3: Implement boundary generation**

```python
VALIDATION_INTERVAL = 25
MILESTONE_INTERVAL = 100
TARGET_EPOCH = 800


def validation_boundaries(completed_epoch: int, target_epoch: int = TARGET_EPOCH) -> tuple[int, ...]:
    if isinstance(completed_epoch, bool) or not isinstance(completed_epoch, int):
        raise TypeError("completed_epoch must be an integer")
    if not 0 <= completed_epoch <= target_epoch == TARGET_EPOCH:
        raise ValueError("T800 epoch range must stay within 0..800")
    return tuple(
        epoch for epoch in range(VALIDATION_INTERVAL, target_epoch + 1, VALIDATION_INTERVAL)
        if epoch > completed_epoch
    )
```

- [ ] **Step 4: Write failing lock and status tests**

Tests must prove a second live advisory lock is rejected, process death releases
the kernel lock even when metadata remains, status JSON is strict and atomic,
and an injected replace failure leaves the previous status readable. Exercise
the real `fcntl.flock` behavior in the Linux container; skip only those POSIX
integration cases on Windows while still running boundary/status tests there.

```python
def test_run_lock_rejects_duplicate_owner(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    with RunLock(path, run_id="run-3035"):
        with pytest.raises(FileExistsError, match="already active"):
            RunLock(path, run_id="run-3035").acquire()


def test_status_write_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    write_longrun_status(path, {"run_id": "run-3035", "epoch": 25, "state": "validated"})
    assert json.loads(path.read_text(encoding="utf-8"))["epoch"] == 25
    assert not list(tmp_path.glob(".status.json.*"))
```

- [ ] **Step 5: Implement `RunLock` and atomic status writes**

Open the stable lock path with mode `0o600`, then use
`fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)` for process-lifetime ownership.
After locking, truncate and store strict JSON containing run ID, PID, hostname,
and creation time. Release with `LOCK_UN` and close the owning descriptor; the
kernel also releases it after crashes or reboot. On non-POSIX systems, raise a
clear runtime error rather than claiming long-run locking is active. Write
status to a same-directory temporary file, `fsync`, then `os.replace`.

- [ ] **Step 6: Run long-run tests and verify GREEN**

```powershell
py -3.13 -m pytest tests/test_steerer_training_longrun.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the long-run primitives**

```powershell
git add src/droneai/steerer_training_longrun.py tests/test_steerer_training_longrun.py
git commit -m "Add STEERER long-run control primitives"
```

---

### Task 3: Extend checkpoint policy for 100-epoch T800 milestones

**Files:**
- Modify: `src/droneai/steerer_training_checkpoint.py`
- Modify: `tests/test_steerer_training_checkpoint.py`

**Interfaces:**
- Consumes: validated checkpoint state and `preserve_milestone: bool | None`.
- Produces: atomic `last`, conditional bests, and caller-authorized T800 milestone files without changing legacy stage defaults.

- [ ] **Step 1: Write failing periodic milestone tests**

```python
def test_t800_policy_preserves_only_authorized_100_epoch_milestones(tmp_path: Path) -> None:
    at_100 = save_checkpoint_with_policy(
        tmp_path, _state(epoch=100, stage="T800"), current_mae=90.0,
        current_rmse=150.0, preserve_milestone=True, torch_module=_FakeTorch(),
    )
    assert "milestone-100" in at_100.artifacts
    at_125 = save_checkpoint_with_policy(
        tmp_path, _state(epoch=125, stage="T800"), current_mae=89.0,
        current_rmse=149.0, preserve_milestone=False, torch_module=_FakeTorch(),
    )
    assert "milestone-125" not in at_125.artifacts
```

Also assert `preserve_milestone=True` is rejected for non-100 T800 epochs and
legacy T1/T5/T50 milestone behavior stays unchanged when the argument is `None`.

- [ ] **Step 2: Run the tests and verify RED**

```powershell
py -3.13 -m pytest tests/test_steerer_training_checkpoint.py -k "milestone" -q
```

Expected: FAIL because the argument and T800 policy do not exist.

- [ ] **Step 3: Implement the explicit milestone policy**

Add the keyword-only parameter:

```python
preserve_milestone: bool | None = None
```

Resolve it as follows:

```python
if preserve_milestone is None:
    should_preserve = _MILESTONE_EPOCHS.get(stage) == validated["epoch"]
else:
    should_preserve = preserve_milestone
    if stage != "T800" or int(validated["epoch"]) % 100 != 0:
        raise ValueError("explicit T800 milestones must be 100-epoch boundaries")
```

Use `should_preserve` for the existing atomic milestone write and manifest entry.

- [ ] **Step 4: Run checkpoint tests and verify GREEN**

```powershell
py -3.13 -m pytest tests/test_steerer_training_checkpoint.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the checkpoint policy**

```powershell
git add src/droneai/steerer_training_checkpoint.py tests/test_steerer_training_checkpoint.py
git commit -m "Preserve STEERER T800 milestones"
```

---

### Task 4: Execute T800 as 25-epoch authoritative validation segments

**Files:**
- Modify: `src/droneai/steerer_training_runner.py`
- Modify: `tests/test_steerer_training_runner.py`
- Modify: `tests/test_steerer_training_evidence.py`
- Modify: `src/droneai/steerer_training_evidence.py` only if the existing strict payload cannot express the boundary epoch without weakening its schema.

**Interfaces:**
- Consumes: `validation_boundaries`, `write_longrun_status`, T5/T50/T800 resume checkpoints, and the existing training engine.
- Produces: immutable `metrics.epoch-NNN.json`, environment manifests, chained checkpoint boundaries, final `metrics.t800.json`, and a `StageResult` for epoch 800.

- [ ] **Step 1: Write failing resume-policy tests**

Tests must prove T800 accepts epoch-5 T5, epoch-50 T50, and partial T800 boundary
states; rejects T0/T1, non-boundary T800 states, and stage/epoch mismatches.

```python
@pytest.mark.parametrize(
    ("resume_stage", "epoch"),
    [("T5", 5), ("T50", 50), ("T800", 25), ("T800", 775)],
)
def test_t800_accepts_only_verified_predecessor_boundaries(
    tmp_path: Path, resume_stage: str, epoch: int
) -> None:
    checkpoint = _write_resume_checkpoint(tmp_path, stage=resume_stage, epoch=epoch)
    result = run_training_stage(
        profile, stage="T800", run_id="run-3035", resume=checkpoint,
        engine=_boundary_engine(completed_epoch=epoch),
    )
    assert result.completed_epoch == 800
```

- [ ] **Step 2: Run resume tests and verify RED**

```powershell
py -3.13 -m pytest tests/test_steerer_training_runner.py -k "t800 and resume" -q
```

Expected: FAIL because T800 predecessor validation is absent.

- [ ] **Step 3: Implement T800 predecessor validation**

Extend the resume rule without loosening T5/T50:

```python
valid_t800_resume = (
    (resume_stage == "T5" and epoch == 5)
    or (resume_stage == "T50" and epoch == 50)
    or (resume_stage == "T800" and 25 <= epoch < 800 and epoch % 25 == 0)
)
```

Reject every other T800 predecessor before restoring model state.

- [ ] **Step 4: Write failing boundary-loop and RNG-continuity tests**

Use a fake engine with inexpensive epoch observations and 240 validation samples.
Assert validation occurs exactly at every expected boundary, the runner does not
cross a boundary before finalization, parent SHA-256 advances after each
checkpoint, and validation RNG use does not alter the next segment's sample
tokens.

```python
def test_t800_validates_every_25_epochs_and_chains_checkpoints(tmp_path: Path) -> None:
    engine = FakeTrainingEngine(validation_count=240, completed_epoch=5)
    result = run_training_stage(
        profile, stage="T800", run_id="run-3035", resume=_t5_checkpoint(tmp_path),
        engine=engine,
    )
    assert engine.validation_epochs == list(range(25, 801, 25))
    assert result.completed_epoch == 800
    assert result.validation_samples == 240
    assert engine.checkpoint_parent_epochs == [5, *range(25, 800, 25)]
```

- [ ] **Step 5: Run the boundary-loop test and verify RED**

```powershell
py -3.13 -m pytest tests/test_steerer_training_runner.py -k "t800 and (boundary or rng or chain)" -q
```

Expected: FAIL because the runner currently validates only once at stage end.

- [ ] **Step 6: Extract one boundary finalizer**

Add `_finalize_validation_boundary(engine, *, run_id, stage,
epoch_observations, boundary_epoch, container_image_digest, physical_batch,
accumulation_steps, parent_checkpoint_sha256, preserve_milestone) ->
BoundaryResult`. Define `BoundaryResult` as a frozen dataclass containing
`metrics_path`, `metrics_sha256`, `environment_path`, `environment_sha256`,
`checkpoint_path`, `checkpoint_sha256`, `validation`, and `optimizer_steps`.

The function validates all 240 samples, combines only the epoch observations
since the preceding boundary, writes `metrics.epoch-NNN.json`, writes the
content-addressed environment, saves and reloads the checkpoint, restores the
captured training-boundary RNG into that checkpoint, and returns the new parent
hash as `checkpoint_sha256`. Every write uses the existing strict payload and
atomic artifact functions; no alternate self-attested evidence path is added.

- [ ] **Step 7: Implement the T800 segmented loop**

For T800 only, iterate `validation_boundaries(engine.completed_epoch)`. Train
from the prior boundary through the next ceiling, finalize that boundary, clear
the in-memory epoch-observation window, atomically update `status.json`, then
continue. Use `preserve_milestone=(boundary_epoch % 100 == 0)`. Existing T1/T5/
T50 control flow must remain unchanged.

- [ ] **Step 8: Write failing injected-failure recovery tests**

Inject failures during validation, metric write, checkpoint serialization, and
manifest replacement. Each test must prove the next segment never starts and
the previous verified `last.pth` plus manifest still hash-verify.

- [ ] **Step 9: Implement failure-safe boundary promotion**

Order operations as validation → immutable metrics → environment → atomic
checkpoint/manifest → status. Do not mutate the parent hash or enter the next
segment until all reload/hash checks pass.

- [ ] **Step 10: Run runner/evidence tests and verify GREEN**

```powershell
py -3.13 -m pytest tests/test_steerer_training_runner.py tests/test_steerer_training_evidence.py tests/test_steerer_training_checkpoint.py tests/test_steerer_training_longrun.py -q
```

Expected: PASS.

- [ ] **Step 11: Commit segmented T800 execution**

```powershell
git add src/droneai/steerer_training_runner.py src/droneai/steerer_training_evidence.py tests/test_steerer_training_runner.py tests/test_steerer_training_evidence.py
git commit -m "Run STEERER T800 with validated boundaries"
```

---

### Task 5: Add T800 scoring, runbook, and operator evidence

**Files:**
- Modify: `src/droneai/steerer_training_gate.py`
- Modify: `scripts/score_steerer_ucf_training.py`
- Modify: `docs/runbooks/steerer-ucf-qnrf-training-home5090.md`
- Modify: `tests/test_steerer_training_gate.py`
- Modify: `tests/test_steerer_training_runbook.py`

**Interfaces:**
- Consumes: epoch-800 final evidence, checkpoint manifest, best checkpoint identities, and sealed-Test evidence.
- Produces: T800 100-point score bundle and exact launch/resume/status/result-pull commands.

- [ ] **Step 1: Write failing T800 score-bundle tests**

Require score stage `T800`, threshold 90, 17 finite metrics, epoch 800, scheduler
horizon 800, official Test sealed, a valid milestone-800, and hash-verifiable
best-MAE/best-RMSE entries.

```python
def test_t800_score_requires_final_and_best_checkpoint_lineage(tmp_path: Path) -> None:
    fixture = _authoritative_fixture(tmp_path, stage="T800")
    evidence = verify_authoritative_training_evidence(
        fixture.inputs, _profile_override=fixture.profile, torch_module=_FakeTorch()
    )
    report = score_training_stage(evidence, stage="T800")
    assert report.score == 100
    assert report.threshold == 90
    assert report.failed_blockers == ()
```

- [ ] **Step 2: Run scoring tests and verify RED**

```powershell
py -3.13 -m pytest tests/test_steerer_training_gate.py -k "t800" -q
```

Expected: FAIL until T800 artifact requirements are enforced.

- [ ] **Step 3: Implement T800 hard blockers**

Reuse the 17-metric schema. Add blockers for epoch 800, scheduler horizon 800,
sealed Test, milestone-800, best-MAE and best-RMSE disk hashes, and the complete
parent chain. Do not turn these blockers into averaged optional points.

- [ ] **Step 4: Write the exact runbook commands**

Document preflight, launch, status, resume, score, evidence copy, and checkpoint
hash checks. The launch command must include:

```text
--stage T800 --approved-stage T800
--resume /workspace/data/checkpoints/steerer-ucf-training/steerer-qnrf-imagenet-20260806-t1-e/milestone-005.pth
```

The runbook must explicitly say that after a restart the `--resume` path changes
to the hash-verified `last.pth`, and that official Test access remains forbidden.

- [ ] **Step 5: Run score/runbook tests and verify GREEN**

```powershell
py -3.13 -m pytest tests/test_steerer_training_gate.py tests/test_steerer_training_runbook.py -q
```

Expected: PASS.

- [ ] **Step 6: Run the complete focused suite**

```powershell
py -3.13 -m pytest tests/test_steerer_adapter.py tests/test_steerer_training_profile.py tests/test_steerer_training_split.py tests/test_steerer_training_data.py tests/test_steerer_training_checkpoint.py tests/test_steerer_training_evidence.py tests/test_steerer_training_gate.py tests/test_steerer_training_longrun.py tests/test_steerer_training_runner.py tests/test_steerer_training_runbook.py -q
git diff --check
```

Expected: all tests PASS and `git diff --check` exits 0.

- [ ] **Step 7: Commit T800 scoring and operations**

```powershell
git add src/droneai/steerer_training_gate.py scripts/score_steerer_ucf_training.py docs/runbooks/steerer-ucf-qnrf-training-home5090.md tests/test_steerer_training_gate.py tests/test_steerer_training_runbook.py
git commit -m "Document and score STEERER T800 runs"
```

---

### Task 6: Synchronize and preflight the home5090 run

**Files:**
- Verify only: local worktree, GitHub branch/PR, home5090 worktree, SSD artifacts.

**Interfaces:**
- Consumes: tested Git commit and immutable T5 checkpoint.
- Produces: a clean, hash-aligned remote worktree and a recorded no-training preflight.

- [ ] **Step 1: Push the tested branch**

```powershell
git push -u origin agent/steerer-ucf-training-implementation
```

- [ ] **Step 2: Fast-forward the remote worktree without resetting**

```powershell
ssh home5090-pop "cd /home/lucita/crowd-counting-lab && git fetch origin agent/steerer-ucf-training-implementation"
ssh home5090-pop "docker exec crowd-jupyter git -C /workspace/.worktrees/steerer-ucf-training merge --ff-only origin/agent/steerer-ucf-training-implementation"
```

- [ ] **Step 3: Verify runtime, storage, code, data, and checkpoint hashes**

Check exact commit alignment, clean trees, CUDA matmul, GPU idle state, at least
50 GB free SSD, upstream commit `5b1854dbc2d280f2326d67c65515d8baf9083810`,
dataset/split hashes from the T5 environment, backbone SHA-256
`0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8`,
and milestone-005 SHA-256
`2ce6800bcbbdaace881e5f2ecc9f481616a67b9b96254232b7e8e5285ceeafc4`.

- [ ] **Step 4: Run the synthetic T800 boundary simulation in the container**

```powershell
ssh home5090-pop "docker exec crowd-jupyter bash -lc 'cd /workspace/.worktrees/steerer-ucf-training && /workspace/.venvs/steerer/bin/python -m pytest tests/test_steerer_training_longrun.py tests/test_steerer_training_runner.py -k t800 -q'"
```

Expected: PASS without accessing the real official Test or starting real training.

- [ ] **Step 5: Record preflight evidence and commit only the small JSON/Markdown**

Write the environment, hashes, free-space result, CUDA smoke, and exact launch
identity under `results/steerer-training/steerer-qnrf-imagenet-20260806-t1-e/t800-preflight/`, verify their hashes, commit, and push.

---

### Task 7: Launch and monitor the epoch-800 training run

**Files:**
- Large runtime outputs: `/workspace/data/checkpoints/steerer-ucf-training/steerer-qnrf-imagenet-20260806-t1-e/`
- Large/runtime logs: `/workspace/data/results/steerer-ucf-training/steerer-qnrf-imagenet-20260806-t1-e/`

**Interfaces:**
- Consumes: verified T800 code/profile and milestone-005 checkpoint.
- Produces: boundary evidence through epoch 800 and recoverable status/logs.

- [ ] **Step 1: Launch inside `tmux crowd` with a run-specific log**

Use the exact command from the approved runbook with the pinned config, processed
root, upstream, backbone/hash, container digest, run ID, `--stage T800`,
`--approved-stage T800`, and the milestone-005 resume path. Redirect stdout and
stderr to `run-result-t800.jsonl` on the SSD.

- [ ] **Step 2: Confirm the process crossed startup gates**

Verify one active process, the run lock owner, CUDA allocation, restored epoch 5,
restored global step 600, scheduler horizon 800, and no Test path in open files
or logs.

- [ ] **Step 3: Monitor boundary evidence without busy polling**

At status updates, inspect `status.json`, process state, GPU utilization,
temperature, free disk, last log lines, and the newest manifest entry. Do not
alter hyperparameters in response to normal metric fluctuation.

- [ ] **Step 4: Resume only from verified `last.pth` after interruption**

If interrupted, verify `last.pth` against the manifest and repeat the identical
launch command with that path. Never resume from an incomplete epoch or copy.

- [ ] **Step 5: Verify epoch-800 completion**

Require completed epoch 800, all 32 boundary metric files, milestone-800,
best-MAE/best-RMSE, final `last.pth`, finite loss, and zero failed checkpoint
round trips.

---

### Task 8: Score the best model and publish the paper-gap report

**Files:**
- Create: `results/steerer-training/steerer-qnrf-imagenet-20260806-t1-e/t800/README.md`
- Create: `results/steerer-training/steerer-qnrf-imagenet-20260806-t1-e/t800/paper-comparison.md`
- Create: `results/steerer-training/steerer-qnrf-imagenet-20260806-t1-e/t800/artifact-sha256.txt`
- Copy: small final/best metric, environment, manifest, status, and score-bundle files.

**Interfaces:**
- Consumes: epoch-800 boundary evidence and the selected best-MAE/best-RMSE checkpoints.
- Produces: a Git-safe authoritative score bundle and a report comparing T5, T800 final, T800 best, paper, and repository figures.

- [ ] **Step 1: Reload and revalidate best-MAE and best-RMSE checkpoints**

Run the complete 240-image validation adapter for each distinct checkpoint.
Record all 17 metrics, latency, VRAM, checkpoint SHA-256, selected epoch, and
validation split SHA-256. Refuse any Test path.

- [ ] **Step 2: Generate and verify the T800 score bundle**

Run `scripts/score_steerer_ucf_training.py --stage T800` with the exact final
checkpoint manifest, environment, metrics, and output directory. Require score
100/100, `PASS_COMMERCIAL_CANDIDATE`, and no failed blockers.

- [ ] **Step 3: Calculate comparison deltas**

Report absolute and relative changes from T5 MAE/RMSE 389.995/1029.695, paper
74.3/128.3, and repository 77.8/138.0. Report localization differences only as
diagnostic because the fixed 16-pixel protocol is not paper-equivalent.

- [ ] **Step 4: Copy only small evidence and recompute every Git hash**

Keep all `.pth` files on SSD. Copy JSON, Markdown, manifest, and curated PNG
evidence; generate `artifact-sha256.txt`; independently recompute every listed
hash locally.

- [ ] **Step 5: Run final regression and evidence verification**

```powershell
py -3.13 -m pytest tests/test_steerer_adapter.py tests/test_steerer_training_profile.py tests/test_steerer_training_split.py tests/test_steerer_training_data.py tests/test_steerer_training_checkpoint.py tests/test_steerer_training_evidence.py tests/test_steerer_training_gate.py tests/test_steerer_training_longrun.py tests/test_steerer_training_runner.py tests/test_steerer_training_runbook.py -q
git diff --check
```

Also verify local/remote commits, PR head, checkpoint hashes, score JSON, official
Test sealing, and a clean worktree.

- [ ] **Step 6: Commit and push the authoritative T800 report**

```powershell
git add results/steerer-training/steerer-qnrf-imagenet-20260806-t1-e/t800
git commit -m "Record authoritative STEERER T800 evidence"
git push
```

- [ ] **Step 7: Report the decision**

Lead with the best observed checkpoint and whether it materially closes the paper
gap. State architecture/input/output/count method, split and rights scope,
training method/time, all operational metrics, evidence hashes/locations,
harness score, comparability limits, and the next approval gate. Do not label the
result production-approved or paper-reproduced.
