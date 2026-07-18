# MPCount One-Sample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one frozen UCF-QNRF validation image through the pinned deterministic ShanghaiTech-A MPCount checkpoint on home5090 and capture fail-closed compatibility evidence without training or test-split access.

**Architecture:** A thin `MPCountAdapter` imports the clean pinned upstream repository in an isolated namespace, recreates the official 16-pixel padding, `[-1, 1]` normalization, patch inference, and `density.sum()/1000` count contract, then returns a standard `NativePrediction`. A separate one-sample runner reuses the accepted DM-Count validation split and Stage 3C rights evidence, renders one panel, and writes an immutable result JSON to SSD.

**Tech Stack:** Python 3.12, PyTorch 2.9 NGC runtime, CUDA/RTX 5090, NumPy, Pillow, existing DroneAI evaluation contracts and pytest.

## Global Constraints

- Use model upstream commit `6eb06772bcf7dfb771c43a14d67146fce767f103` and checkpoint SHA-256 `4a0ba849ace0ef2bc87364ec8551de4a3e9dca74fad27764c9b948ebf7928eb0`.
- Use dataset `ucf-qnrf-kaggle-apache`, split `dm-count-upstream-qnrf-val-smoke-36-v1`, validation only, seed `20260715`.
- Keep official UCF-QNRF Test sealed; expose no test-root argument.
- Perform no training or fine-tuning.
- Keep raw density arrays, checkpoints, and full results under `/workspace/data`; commit only code, configuration, documentation, hashes, and a curated panel under 25 MiB.
- Treat one-sample numbers as compatibility evidence, not benchmark accuracy.

---

### Task 1: Freeze the MPCount validation and rights contract

**Files:**
- Create: `configs/evaluation/mpcount_ucf_qnrf_smoke.json`
- Create: `src/droneai/mpcount_smoke.py`
- Create: `tests/test_mpcount_smoke.py`

**Interfaces:**
- Consumes: `load_smoke_config`, `prepare_smoke_samples`, and `write_split_source_manifest` from `droneai.dm_count_smoke`; Stage 3C rights files for candidate `mpcount-official-sta-to-ucf-qnrf-apache`.
- Produces: `load_mpcount_smoke_config`, `validate_mpcount_rights_decision`, `validate_mpcount_split_upstream`, `write_split_source_manifest`, and `build_mpcount_protocol`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_mpcount_config_preserves_frozen_validation_contract():
    config = load_mpcount_smoke_config(CONFIG)
    assert config["expected_samples"] == 36
    assert config["split_role"] == "validation"
    assert config["checkpoint_training_split_status"] == "VERIFIED_DISJOINT"
    assert config["sealed_test_access_approved"] is False

def test_mpcount_rights_require_commercial_candidate_for_evaluation():
    decision = validate_mpcount_rights_decision(RIGHTS, manifest_path=MANIFEST)
    assert decision["status"] == "PASS_COMMERCIAL_CANDIDATE"
    assert "frozen_checkpoint_evaluation" in decision["allowed_actions"]
```

- [ ] **Step 2: Verify RED**

Run: `../../.venv/Scripts/python.exe -m pytest tests/test_mpcount_smoke.py -q`

Expected: FAIL because `droneai.mpcount_smoke` and its config do not exist.

- [ ] **Step 3: Implement the frozen wrapper**

Create the config by copying the accepted DM-Count split identities and replacing only:

```json
{
  "run_id": "mpcount-ucf-qnrf-validation-smoke",
  "protocol_id": "mpcount-official-sta-to-qnrf-val-smoke-v1",
  "model_id": "mpcount-official-sta-to-ucf-qnrf",
  "upstream_commit": "6eb06772bcf7dfb771c43a14d67146fce767f103",
  "candidate_id": "mpcount-official-sta-to-ucf-qnrf-apache",
  "required_action": "frozen_checkpoint_evaluation",
  "checkpoint_training_split_status": "VERIFIED_DISJOINT",
  "comparison_scope": "compatibility_smoke"
}
```

Implement strict whole-object config comparison, candidate/manifest semantic binding, `PASS_COMMERCIAL_CANDIDATE`, required action validation, clean pinned model/split repositories, and the density `game_l1` protocol.

- [ ] **Step 4: Verify GREEN**

Run: `../../.venv/Scripts/python.exe -m pytest tests/test_mpcount_smoke.py -q`

Expected: all MPCount smoke tests PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/evaluation/mpcount_ucf_qnrf_smoke.json src/droneai/mpcount_smoke.py tests/test_mpcount_smoke.py
git commit -m "Add frozen MPCount validation contract"
```

### Task 2: Implement the official MPCount density adapter

**Files:**
- Create: `src/droneai/mpcount_adapter.py`
- Create: `tests/test_mpcount_adapter.py`

**Interfaces:**
- Consumes: `EvaluationSample`, `NativePrediction`, `ModelBrief`, a clean pinned MPCount repository, and a SHA-verified checkpoint.
- Produces: `MPCountAdapter.predict(sample, retain_native=False) -> NativePrediction` and `MPCountAdapter.brief() -> ModelBrief`.

- [ ] **Step 1: Write failing adapter tests**

```python
def test_adapter_preserves_official_padding_normalization_and_count(adapter):
    backend = FakeBackend(density=np.full((32, 48), 1000.0, dtype=np.float32))
    prediction = adapter(backend).predict(sample, retain_native=True)
    assert prediction.output_type == "density"
    assert prediction.predicted_count == pytest.approx(32 * 48)
    assert prediction.native_density is not None
    assert prediction.metadata["log_para"] == 1000.0

def test_adapter_fails_closed_on_nonfinite_or_negative_density(adapter, bad_density):
    prediction = adapter(FakeBackend(density=bad_density)).predict(sample)
    assert prediction.failure_state is not None
```

- [ ] **Step 2: Verify RED**

Run: `../../.venv/Scripts/python.exe -m pytest tests/test_mpcount_adapter.py -q`

Expected: FAIL because `droneai.mpcount_adapter` does not exist.

- [ ] **Step 3: Implement the adapter and CUDA backend**

Implement these exact boundaries:

```python
class MPCountBackend(Protocol):
    parameter_count: int
    missing_keys: Sequence[str]
    unexpected_keys: Sequence[str]
    def infer(self, normalized_bchw: np.ndarray, patch_size: int) -> tuple[np.ndarray, float, float]:
        raise NotImplementedError

class MPCountAdapter(ModelAdapter):
    def predict(self, sample: EvaluationSample, *, retain_native: bool = False) -> NativePrediction:
        raise NotImplementedError
    def brief(self) -> ModelBrief:
        raise NotImplementedError
```

The backend must instantiate `DGModel_final(pretrained=False, deterministic=True)`, load the state dictionary with `weights_only=True`, reject missing/unexpected keys, use CUDA inference mode, reset/synchronize peak-memory counters, and stitch official non-overlapping patches. The adapter must pad symmetrically to a multiple of 16, normalize RGB tensors with mean/std `0.5`, crop the predicted density back to the original image region, reject nonfinite or negative values, and calculate `predicted_count = density.sum() / 1000.0`.

- [ ] **Step 4: Verify GREEN**

Run: `../../.venv/Scripts/python.exe -m pytest tests/test_mpcount_adapter.py -q`

Expected: all adapter tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/droneai/mpcount_adapter.py tests/test_mpcount_adapter.py
git commit -m "Add official MPCount density adapter"
```

### Task 3: Add and execute the one-sample gate

**Files:**
- Create: `scripts/run_mpcount_one_sample.py`
- Create: `tests/test_run_mpcount_one_sample.py`
- Create: `docs/models/MPCOUNT.md`
- Modify after verified execution: `configs/evaluation/model_queue.json`

**Interfaces:**
- Consumes: Task 1 frozen split/rights functions and Task 2 `MPCountAdapter`.
- Produces: `result.json`, `split-source-manifest.json`, `environment-summary.json`, `model-brief.md`, and `one-sample-panel.png` under a new SSD result directory.

- [ ] **Step 1: Write the failing runner contract test**

```python
def test_mpcount_one_sample_runner_exposes_no_test_path():
    help_text = subprocess.run(
        [sys.executable, "scripts/run_mpcount_one_sample.py", "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "--sample-id" in help_text
    assert "--checkpoint-sha256" in help_text
    assert "--test-root" not in help_text
```

- [ ] **Step 2: Verify RED**

Run: `../../.venv/Scripts/python.exe -m pytest tests/test_run_mpcount_one_sample.py -q`

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement runner and model brief**

The runner must select the first sorted medium-density sample unless `--sample-id` names a member of the frozen 36, validate rights and both upstream commits before constructing the adapter, call `evaluate_sample`, render one review panel, hash every evidence file, and return nonzero when `failure_state` is set. Document native density, count summation, no native localization, official training losses, the ShanghaiTech-A to UCF domain-transfer boundary, and the Torch 2.0.1 to 2.9 compatibility boundary.

- [ ] **Step 4: Verify GREEN and the relevant suite**

Run:

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_mpcount_adapter.py tests/test_mpcount_smoke.py tests/test_run_mpcount_one_sample.py tests/test_model_workflow.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit, push, and fast-forward home5090**

```bash
git add scripts/run_mpcount_one_sample.py tests/test_run_mpcount_one_sample.py docs/models/MPCOUNT.md
git commit -m "Add MPCount one-sample gate"
git push origin HEAD
ssh home5090-pop 'cd /home/lucita/crowd-counting-lab && git pull --ff-only'
```

- [ ] **Step 6: Execute one frozen validation sample**

Run the runner inside `crowd-jupyter` with `/workspace/.venvs/mpcount/bin/python`, the accepted Stage 3C decision/manifest, `/workspace/upstreams/MPCount`, `/workspace/upstreams/DM-Count`, the frozen train/validation lists, the SHA-verified deterministic checkpoint, and a new `/workspace/data/results/mpcount/mpcount-one-sample-<commit>` directory.

Expected: exit `0`, `status=PASS`, finite nonnegative density, `predicted_count == cropped_density_sum / 1000`, finite latency, finite peak VRAM, and one panel.

- [ ] **Step 7: Accept evidence and verify next action**

Hash `result.json`, add it as MPCount `one_sample` evidence in `configs/evaluation/model_queue.json`, run workflow tests, commit/push/pull, then run the compact status command.

Expected: `completed_gates=["rights","preflight","one_sample"]`, `next_action="benchmark"`, and `approval_required=true`.

- [ ] **Step 8: Run the full suite and report**

Run: `../../.venv/Scripts/python.exe -m pytest -q`

Expected: zero failures. Report one-sample values strictly as compatibility evidence and stop at the benchmark approval gate.
