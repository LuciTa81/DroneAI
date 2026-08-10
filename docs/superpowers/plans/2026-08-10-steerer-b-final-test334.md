# STEERER B-Lane Final Test334 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the three validation-selected STEERER B-lane checkpoints once on all 334 official UCF-QNRF Test images and permanently close the B lane to further tuning.

**Architecture:** Add a dedicated, fail-closed Test evaluation profile and runner rather than adding a Test switch to the existing sealed development runner. Reuse the existing `STEERERAdapter`, common evaluation runner, and UCF-QNRF sample contract, while identifying project-trained checkpoints separately from the unlicensed official research checkpoint.

**Tech Stack:** Python 3.12, pytest, NumPy/SciPy/Pillow, PyTorch/CUDA in `crowd-jupyter`, existing DroneAI evaluation harness.

## Global Constraints

- The official UCF-QNRF Test split contains exactly 334 image/annotation pairs and is accessed only under the user's 2026-08-10 authorization.
- Freeze checkpoint roles before Test metrics exist: epoch 625 `best-mae.pth` primary, epoch 175 `best-rmse.pth` secondary, epoch 800 `milestone-800.pth` final-reference.
- Expected checkpoint SHA-256 values are respectively `e41df787f546a5195e078486ec6d1bd62ffcb2e3e77010436f74b65045667ad2`, `ab1f65573c7f18f115e711998f4f4fb252e816bc5b9ae3564d5b379f22f981e6`, and `d13316a818ddaa2ed0da680718b234bfd21377934c7ef3b33fa074746c93c8f0`.
- Test results cannot change checkpoint roles, resume B training, or authorize Test-driven tuning.
- Keep all raw Test data, checkpoints, predictions, and complete outputs under `/workspace/data`; Git receives only code, configuration, summaries, hashes, and curated evidence under 25 MiB.
- Do not modify the immutable ImageNet backbone file or existing B result/checkpoint directories.

---

### Task 1: Freeze the B final-Test contract

**Files:**
- Create: `configs/evaluation/steerer_b_ucf_qnrf_final_test.json`
- Create: `src/droneai/steerer_final_test.py`
- Create: `tests/test_steerer_final_test.py`

**Interfaces:**
- Consumes: the three checkpoint identities above and the existing training profile.
- Produces: `load_b_final_test_profile(path) -> BFinalTestProfile` and `build_b_final_test_protocol(profile, *, checkpoint, rights_path) -> EvaluationProtocol`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_profile_freezes_three_validation_selected_checkpoints():
    profile = load_b_final_test_profile(PROFILE)
    assert [(c.role, c.epoch, c.sha256) for c in profile.checkpoints] == [
        ("primary-best-val-mae", 625, "e41df787f546a5195e078486ec6d1bd62ffcb2e3e77010436f74b65045667ad2"),
        ("secondary-best-val-rmse", 175, "ab1f65573c7f18f115e711998f4f4fb252e816bc5b9ae3564d5b379f22f981e6"),
        ("final-reference", 800, "d13316a818ddaa2ed0da680718b234bfd21377934c7ef3b33fa074746c93c8f0"),
    ]

def test_profile_forbids_post_test_training():
    profile = load_b_final_test_profile(PROFILE)
    assert profile.test_access_approved is True
    assert profile.post_test_training_allowed is False
```

- [ ] **Step 2: Verify RED**

Run: `.venv/Scripts/python -m pytest tests/test_steerer_final_test.py -q`

Expected: collection failure because `droneai.steerer_final_test` does not exist.

- [ ] **Step 3: Implement the strict dataclasses, JSON loader, and protocol builder**

The loader accepts only schema version 1, `official_test334`, exactly 334 samples, the pinned STEERER upstream commit, the three ordered checkpoint identities, `checkpoint_origin=project_training`, `test_access_approved=true`, and `post_test_training_allowed=false`. The protocol sets `split_role="test"`, `sealed_test_access_approved=True`, `checkpoint_training_split_status="VERIFIED_DISJOINT"`, and the existing comparison scope `held_out_performance`.

- [ ] **Step 4: Verify GREEN and mutation coverage**

Run: `.venv/Scripts/python -m pytest tests/test_steerer_final_test.py -q`

Expected: all tests pass; mutations to sample count, role order, hash, Test approval, or post-Test training flag are rejected.

- [ ] **Step 5: Commit**

```bash
git add configs/evaluation/steerer_b_ucf_qnrf_final_test.json src/droneai/steerer_final_test.py tests/test_steerer_final_test.py
git commit -m "feat: freeze STEERER B final Test contract"
```

### Task 2: Add the explicit official-Test index and project-weight model brief

**Files:**
- Modify: `src/droneai/ucf_qnrf.py`
- Modify: `src/droneai/steerer_adapter.py`
- Modify: `tests/test_ucf_qnrf.py`
- Modify: `tests/test_steerer_adapter.py`

**Interfaces:**
- Produces: `index_ucf_qnrf_test(test_root) -> tuple[UCFQNRFRecord, ...]` with IDs prefixed `test_` and `STEERERAdapter.brief()` rights/protocol text appropriate to `checkpoint_origin="project_training"`.
- Consumes: existing point parsing, bounds normalization, hashing, adapter checkpoint unwrapping, and inference behavior.

- [ ] **Step 1: Write failing behavior tests**

```python
def test_test_index_prefixes_partition_identity_and_requires_pairs(tmp_path):
    records = index_ucf_qnrf_test(make_qnrf_partition(tmp_path, count=2))
    assert [row.sample_id for row in records] == ["test_img_0001", "test_img_0002"]

def test_project_checkpoint_brief_does_not_claim_official_weight_rights(adapter):
    brief = adapter(checkpoint_origin="project_training").brief()
    assert brief.rights_status == "PASS_COMMERCIAL_CANDIDATE"
    assert "project-trained" in brief.official_protocol
```

- [ ] **Step 2: Verify RED**

Run: `.venv/Scripts/python -m pytest tests/test_ucf_qnrf.py tests/test_steerer_adapter.py -q`

Expected: the Test index is missing and the adapter still reports official research-checkpoint rights.

- [ ] **Step 3: Implement only the explicit Test index and conditional brief metadata**

Extract a private partition indexer so the existing `index_ucf_qnrf_train` behavior remains unchanged. The new Test function prefixes sample IDs but retains source filenames, hashes, dimensions, points, counts, and density bands. Do not add Test arguments to any B training CLI.

- [ ] **Step 4: Verify GREEN and the sealed-runner regression tests**

Run: `.venv/Scripts/python -m pytest tests/test_ucf_qnrf.py tests/test_steerer_adapter.py tests/test_steerer_training_split.py tests/test_steerer_training_runner.py -q`

Expected: all tests pass, including assertions that the development runner exposes no Test switch.

- [ ] **Step 5: Commit**

```bash
git add src/droneai/ucf_qnrf.py src/droneai/steerer_adapter.py tests/test_ucf_qnrf.py tests/test_steerer_adapter.py
git commit -m "feat: support explicit STEERER project-weight Test samples"
```

### Task 3: Implement the one-time three-checkpoint runner

**Files:**
- Create: `scripts/run_steerer_b_final_test.py`
- Modify: `src/droneai/steerer_final_test.py`
- Modify: `tests/test_steerer_final_test.py`
- Create: `docs/runbooks/steerer-b-final-test334-home5090.md`

**Interfaces:**
- Produces: a CLI accepting only `--config`, `--test-root`, `--upstream-dir`, `--checkpoint-root`, `--checkpoint-manifest`, `--training-profile`, `--output-dir`, and `--device cuda`.
- Produces: per-role standard evaluation bundles plus immutable `checkpoint-selection.json`, `comparison.json`, and `b-lane-closed.json`.

- [ ] **Step 1: Write failing CLI and closure tests**

```python
def test_cli_has_test_root_but_no_train_resume_or_checkpoint_override():
    options = parser_option_strings()
    assert "--test-root" in options
    assert {"--train-root", "--resume", "--checkpoint", "--checkpoint-sha256"}.isdisjoint(options)

def test_complete_run_writes_closed_marker_and_refuses_rerun(tmp_path):
    run_with_fake_adapter(tmp_path)
    assert json.loads((tmp_path / "b-lane-closed.json").read_text())["post_test_training_allowed"] is False
    with pytest.raises(FileExistsError):
        run_with_fake_adapter(tmp_path)
```

- [ ] **Step 2: Verify RED**

Run: `.venv/Scripts/python -m pytest tests/test_steerer_final_test.py -q`

Expected: CLI and aggregate result functions are absent.

- [ ] **Step 3: Implement the fail-closed runner**

Index exactly 334 Test samples once, write a source inventory, verify all checkpoint files against the frozen configuration and existing checkpoint manifest before any inference, then evaluate roles in frozen order with `checkpoint_origin="project_training"`. Atomically write the comparison and closure markers only after three complete evaluation bundles exist. Preserve partial output on interruption and refuse silent reuse.

- [ ] **Step 4: Verify GREEN, focused suite, and full suite**

Run: `.venv/Scripts/python -m pytest tests/test_steerer_final_test.py tests/test_ucf_qnrf.py tests/test_steerer_adapter.py tests/test_steerer_training_split.py tests/test_steerer_training_runner.py -q`

Run: `.venv/Scripts/python -m pytest -q`

Expected: all tests pass. The full suite may exceed two minutes, so allow a five-minute command timeout without treating duration as failure.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_steerer_b_final_test.py src/droneai/steerer_final_test.py tests/test_steerer_final_test.py docs/runbooks/steerer-b-final-test334-home5090.md
git commit -m "feat: run one-time STEERER B final Test334"
```

### Task 4: Publish, run on home5090, and report the frozen result

**Files:**
- Create after execution: `results/steerer-training/steerer-b-final-test334-20260810/README.md`
- Create after execution: `results/steerer-training/steerer-b-final-test334-20260810/metrics-summary.json`

**Interfaces:**
- Consumes: the verified branch, `crowd-jupyter`, Test root, upstream checkout, checkpoint root, checkpoint manifest, and training profile.
- Produces: three 334-sample Test results and a Git-sized summary without weights or raw predictions.

- [ ] **Step 1: Push the feature branch and create/update a matching remote worktree safely**

Run locally: `git push -u origin agent/steerer-qnrf-test334-a`

On the host, fetch the branch; inside the container create a separate linked worktree if absent. Never reset `/workspace` or `/workspace/.worktrees/steerer-ucf-training`.

- [ ] **Step 2: Run remote preflight**

Verify exact branch/upstream commits, Test 334 pairs, all three checkpoint hashes, checkpoint-manifest membership, at least 20 GiB free, CUDA matmul, idle GPU, and a new empty output path.

- [ ] **Step 3: Execute the three frozen evaluations sequentially**

Run the new CLI in `crowd-jupyter` with output root `/workspace/data/results/steerer-b-final-test334/steerer-b-final-test334-20260810`. Do not start A training in the same command or session.

- [ ] **Step 4: Verify evidence before claiming completion**

Require 334 unique prediction rows and finite MAE/RMSE/bias/MAPE for each role, matching checkpoint and Test inventory hashes, three passing score bundles, one comparison file, and `b-lane-closed.json` with `post_test_training_allowed=false`.

- [ ] **Step 5: Commit only the compact result summary**

Copy the aggregate metrics, evidence hashes, and concise interpretation into the two Git-sized result files, run `git diff --check`, and commit with `docs: record STEERER B final Test334 result`. Do not copy `.pth`, complete `predictions.csv`, raw density arrays, or all panels into Git.
