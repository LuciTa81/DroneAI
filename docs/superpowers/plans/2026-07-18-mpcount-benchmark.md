# MPCount Frozen Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the accepted deterministic MPCount checkpoint on exactly 36 frozen UCF-QNRF validation images and accept a hash-verified common evidence bundle.

**Architecture:** Add a thin MPCount CLI that validates the existing rights, split, checkpoint, and config contracts, constructs `MPCountAdapter` once, and delegates all aggregation, scoring, curation, and artifact writing to `run_evaluation`. Keep the complete run on the home5090 SSD and commit only the small evidence bundle and curated panels.

**Tech Stack:** Python 3.12, PyTorch 2.9 NGC CUDA runtime, pytest, existing DroneAI evaluation harness, Git, home5090 Docker.

## Global Constraints

- Use exactly 36 samples from `dm-count-upstream-qnrf-val-smoke-36-v1` with seed `20260715`.
- The split role is validation; never accept or read a test root.
- Do not train, fine-tune, download, resume, or change checkpoint weights.
- Use MPCount commit `6eb06772bcf7dfb771c43a14d67146fce767f103` and split commit `cc5f2132e0d1328909f31b6d665b8e0b15c30467`.
- Use checkpoint SHA-256 `4a0ba849ace0ef2bc87364ec8551de4a3e9dca74fad27764c9b948ebf7928eb0` and patch size `3584`.
- Rights remain `PASS_COMMERCIAL_CANDIDATE`; this run is research comparison only and not production approval.
- Never overwrite a non-empty output directory.

---

### Task 1: Add the frozen MPCount benchmark runner

**Files:**
- Create: `scripts/run_mpcount_ucf_qnrf_smoke.py`
- Create: `tests/test_run_mpcount_ucf_qnrf_smoke.py`

**Interfaces:**
- Consumes: `load_mpcount_smoke_config`, `validate_mpcount_rights_decision`, `validate_mpcount_split_upstream`, `prepare_smoke_samples`, `write_split_source_manifest`, `build_mpcount_protocol`, `MPCountAdapter`, and `run_evaluation`.
- Produces: `main(argv: Sequence[str] | None = None) -> int`, returning `0` for a successful common gate and `2` for a completed but blocked gate.

- [ ] **Step 1: Write the failing CLI and delegation tests**

Create tests that assert the parser contains only `config`, `train_root`, `upstream_dir`, `split_upstream_dir`, `train_list`, `validation_list`, `checkpoint`, `checkpoint_sha256`, `rights_decision`, `rights_manifest`, `output_dir`, `device`, and `patch_size`; rejects training/test/download flags; rejects a non-empty output before validation; passes patch size to `MPCountAdapter`; requires exactly 36 prepared samples; and delegates one atomic call to `run_evaluation`.

- [ ] **Step 2: Run the test and verify RED**

Run: `../../.venv/Scripts/python.exe -m pytest tests/test_run_mpcount_ucf_qnrf_smoke.py -q`

Expected: collection fails because `scripts.run_mpcount_ucf_qnrf_smoke` does not exist.

- [ ] **Step 3: Implement the minimal runner**

Follow the established STEERER runner boundary, use the MPCount validators and adapter, assert `len(prepared.samples) == config["expected_samples"] == 36`, build the protocol, change to the repository root, and call `run_evaluation` once with the split manifest as provenance.

- [ ] **Step 4: Run focused GREEN tests**

Run: `../../.venv/Scripts/python.exe -m pytest tests/test_run_mpcount_ucf_qnrf_smoke.py tests/test_mpcount_adapter.py tests/test_mpcount_smoke.py tests/test_evaluation_runner.py -q`

Expected: all selected tests pass with no failure.

- [ ] **Step 5: Commit the runner**

```bash
git add scripts/run_mpcount_ucf_qnrf_smoke.py tests/test_run_mpcount_ucf_qnrf_smoke.py
git commit -m "Add MPCount frozen benchmark runner"
```

### Task 2: Execute and accept the 36-sample evidence

**Files:**
- Modify: `configs/evaluation/model_queue.json`
- Create: `results/mpcount/<run-id>/` small evidence bundle copied from SSD

**Interfaces:**
- Consumes: runner from Task 1 and the already accepted rights, split, environment, checkpoint, and one-sample evidence.
- Produces: common `score.json`, `metrics.json`, `predictions.csv`, `selection_manifest.json`, `evidence-manifest.json`, summaries, and 12 panels; queue benchmark evidence entries with exact SHA-256.

- [ ] **Step 1: Push and fast-forward home5090**

Require clean Windows and Pop!_OS worktrees, push the branch, and use only `git pull --ff-only` on home5090.

- [ ] **Step 2: Run exactly 36 validation samples**

Inside `crowd-jupyter`, run `/workspace/.venvs/mpcount/bin/python scripts/run_mpcount_ucf_qnrf_smoke.py` with the frozen config, UCF-QNRF Train root, pinned MPCount and DM-Count directories/lists, accepted rights files, checkpoint and SHA-256, CUDA, patch size `3584`, and a new `/workspace/data/results/mpcount/<run-id>` output.

- [ ] **Step 3: Verify the complete bundle**

Check process exit code, 36 CSV records, no failure state, 12 selected panels, score and metrics content, file sizes, and every artifact SHA-256. Do not call one-sample values aggregate accuracy.

- [ ] **Step 4: Curate and accept evidence**

Copy only files under the Git size policy, add the exact `score.json` and `metrics.json` SHA-256 entries to the MPCount queue, and confirm the SSD-backed workflow status advances to `review_and_advance` or the next declared state without changing the active model prematurely.

- [ ] **Step 5: Run complete verification and commit**

Run: `../../.venv/Scripts/python.exe -m pytest -q`

Expected: zero failures. Then commit the queue and curated evidence, push, fast-forward home5090, and re-run the SSD-backed status command.
