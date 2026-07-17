# STEERER UCF-QNRF Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rights-scoped adapter for the official STEERER UCF-QNRF checkpoint and evaluate the same frozen 36-image validation sample set used by DM-Count, producing count, density, localization, runtime, provenance, and selected-panel evidence without training or test-split access.

**Architecture:** Keep the official STEERER repository pinned and clean under `/workspace/upstreams/STEERER`, keep the accepted UCF-QNRF archive, checkpoint, and large outputs under `/workspace/data`, and add only adapter/configuration/test code plus a small review bundle to DroneAI. Reuse the existing UCF-QNRF index, official 1,081/120 train-validation split, common evaluation runner, scoring gate, and curation logic; isolate STEERER-specific preprocessing, multi-resolution density decoding, point extraction, dependencies, and rights scope.

**Tech Stack:** Python 3.12, PyTorch 2.9/CUDA 13 from `nvcr.io/nvidia/pytorch:25.10-py3`, NumPy, Pillow, OpenCV-headless, SciPy, MMCV 1.x, timm, einops, yacs, fvcore, pytest, RTX 5090.

## Global Constraints

- Use branch `agent/steerer-ucf-qnrf-adapter` in the isolated worktree `.worktrees/steerer-ucf-qnrf`; never edit the DM-Count worktree in this plan.
- Pin official STEERER upstream commit `5b1854dbc2d280f2326d67c65515d8baf9083810`; reject modified, untracked, or mismatched upstream trees.
- Use the accepted Kaggle UCF-QNRF dataset record `configs/datasets/ucf_qnrf.kaggle_apache.json`; do not download or use the STEERER-hosted dataset copy.
- Reuse exactly the frozen validation split and 36 selected sample IDs defined by `configs/evaluation/dm_count_ucf_qnrf_smoke.json`; do not enumerate or access the official test split.
- The official checkpoint's commercial terms are not explicit. Record it as an ambiguous commercial candidate, allow only an internally approved research checkpoint comparison, and block commercial training, derived-weight use, weight reuse, and deployment.
- Do not train or fine-tune STEERER. Fine-tuning remains the final step for the selected winner only.
- Use official inference preprocessing: RGB ImageNet normalization, preserve original size when the long side is at most 3072 pixels, otherwise resize the long side to 3072, then zero-pad height and width to multiples of 32.
- Use batch size 1. Synchronize CUDA before and after timed inference, reset peak allocated memory per sample, and record median latency, FPS, and peak VRAM.
- Preserve native outputs: highest-resolution density divided by `den_factor=100`, plus points extracted from x1/x4/x8 density maps using STEERER's threshold `0.15` and official cross-scale point merge.
- Count is the sum of highest-resolution density mass. Point count is a separate localization diagnostic and must not replace density count.
- Map density mass and localization points back to original-image coordinates before common metrics, zone aggregation, and rendering.
- Label harness localization as `F1@16px` and not as the paper's box-size-aware F1; preserve the paper's reported 75.6/79.7/72.0 only as external reference metadata.
- Store raw checkpoint, dependency environment, all native density arrays, and full run outputs on the 5090 SSD. Commit only small JSON/Markdown evidence and curated best/representative/worst panels.
- Never overwrite a non-empty output directory, delete personal files, or write outside `/mnt/crowd-data/CrowdCounting` on the Pop host.
- Stop after the 36-image result and 100-point harness score for user review.

---

## File Structure

- `src/droneai/stage3c.py`: add a distinct research-checkpoint action without weakening commercial frozen-evaluation rules.
- `configs/candidates/steerer_ucf_qnrf.candidate.json`: five-component rights inventory for code, accepted dataset, ambiguous official checkpoint, absent derived weights, and blocked deployment.
- `configs/models/steerer.official.json`: architecture, inputs, outputs, losses, official paper/repository metrics, and runtime risks.
- `src/droneai/steerer_adapter.py`: clean-upstream checks, official preprocessing, model construction, checkpoint validation, CUDA inference, density mass conversion, and multi-scale point extraction.
- `tests/test_steerer_adapter.py`: model-independent fake-backend tests and pinned configuration assertions.
- `src/droneai/steerer_smoke.py`: model-specific configuration and rights validation layered on the existing frozen UCF-QNRF sample builder.
- `tests/test_steerer_smoke.py`: frozen-sample identity and research-scope tests.
- `configs/evaluation/steerer_ucf_qnrf_smoke.json`: protocol identity, frozen split hashes, thresholds, and evidence requirements.
- `scripts/run_steerer_ucf_qnrf_smoke.py`: command-line entry point with no test-root or training arguments.
- `tests/test_run_steerer_ucf_qnrf_smoke.py`: parser, output safety, and fake-adapter end-to-end tests.
- `requirements/models/steerer-home5090.txt`: isolated model dependency overlay that does not replace NGC Torch/CUDA.
- `docs/models/STEERER.md`: human-readable architecture, output interpretation, metric caveats, and exact reproduction command.

---

### Task 1: Add the Ambiguous-Weight Research Evaluation Scope

**Files:**
- Modify: `src/droneai/stage3c.py`
- Modify: `tests/test_stage3c.py`
- Create: `configs/candidates/steerer_ucf_qnrf.candidate.json`

**Interfaces:**
- Consumes: `classify_rights(manifest: dict[str, Any]) -> RightsDecision`
- Produces: allowed action strings `research_asset_download` and `research_checkpoint_evaluation`; commercial actions `asset_download` and `frozen_checkpoint_evaluation` remain unchanged and stricter.

- [ ] **Step 1: Write the failing scope test**

```python
def test_public_checkpoint_with_internal_research_approval_stays_noncommercial() -> None:
    manifest = _manifest()
    weight = next(c for c in manifest["components"] if c["component_type"] == "pretrained_weights")
    weight["rights"].update({
        "status": "unverified",
        "license_id": None,
        "evidence_url": "https://github.com/taohan10200/STEERER#reproduce-counting-and-localization-performance",
        "commercial_use": None,
        "research_use": True,
        "basis": "internal_approval",
    })
    decision = classify_rights(manifest)
    assert decision.status == "PASS_COMMERCIAL_CANDIDATE"
    assert {"research_asset_download", "research_checkpoint_evaluation"}.issubset(decision.allowed_actions)
    assert "frozen_checkpoint_evaluation" not in decision.allowed_actions
    assert not {"commercial_training", "weight_reuse", "deployment"}.intersection(decision.allowed_actions)
```

- [ ] **Step 2: Run the focused test and confirm the missing-action failure**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_stage3c.py::test_public_checkpoint_with_internal_research_approval_stays_noncommercial`

Expected: FAIL because `research_checkpoint_evaluation` is not in `allowed_actions`.

- [ ] **Step 3: Add the narrowly scoped classifier rule**

```python
research_checkpoint_ok = bool(
    complete_ok
    and research_verified.get("code")
    and research_verified.get("dataset")
    and components["pretrained_weights"]["rights"].get("research_use") is True
    and components["pretrained_weights"]["rights"].get("basis") == "internal_approval"
    and components["pretrained_weights"]["rights"].get("evidence_url")
)
```

Return `research_checkpoint_ok` from `_inspect_manifest`, then in `classify_rights` append `("research_asset_download", "research_checkpoint_evaluation")` only when `status == "PASS_COMMERCIAL_CANDIDATE"` and `inspection["research_checkpoint_ok"]` is true. Do not add `asset_download`, `frozen_checkpoint_evaluation`, or any production action through this rule.

- [ ] **Step 4: Add the STEERER manifest**

Record these exact rights decisions:

```json
{
  "candidate_id": "steerer-official-ucf-qnrf-research-comparison",
  "code": {"status": "verified", "license_id": "MIT", "commercial_use": true, "research_use": true},
  "dataset": {"status": "verified", "license_id": "Apache-2.0", "commercial_use": true, "research_use": true},
  "pretrained_weights": {"status": "unverified", "license_id": null, "commercial_use": null, "research_use": true, "basis": "internal_approval"},
  "derived_weights": {"status": "pending", "commercial_use": null, "research_use": null},
  "deployment": {"status": "pending", "commercial_use": null, "research_use": null}
}
```

The actual manifest must use the existing five-component schema and stable source/evidence URLs; the compact JSON above is the assertion target, not a replacement schema.

- [ ] **Step 5: Run Stage 3C tests and generate a temporary decision**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_stage3c.py`

Run: `.venv/Scripts/python.exe scripts/run_stage3c.py --manifest configs/candidates/steerer_ucf_qnrf.candidate.json --output-dir "$env:TEMP\steerer-stage3c-$([guid]::NewGuid())"`

Expected: tests PASS; decision is `PASS_COMMERCIAL_CANDIDATE`; research asset download and research checkpoint evaluation are allowed; commercial asset download, commercial frozen evaluation, and deployment are absent.

- [ ] **Step 6: Commit the rights boundary**

```bash
git add src/droneai/stage3c.py tests/test_stage3c.py configs/candidates/steerer_ucf_qnrf.candidate.json
git commit -m "feat: add scoped research checkpoint evaluation"
```

---

### Task 2: Freeze the STEERER Model Contract

**Files:**
- Create: `configs/models/steerer.official.json`
- Create: `docs/models/STEERER.md`
- Modify: `tests/test_model_brief.py`

**Interfaces:**
- Consumes: `ModelBrief`
- Produces: model contract `steerer-official-ucf-qnrf` and paper-versus-measured metric labels.

- [ ] **Step 1: Write the failing contract test**

```python
def test_steerer_contract_separates_paper_and_harness_localization() -> None:
    config = json.loads((REPO_ROOT / "configs/models/steerer.official.json").read_text())
    assert config["architecture"]["backbone"] == "HRNet-W48"
    assert config["architecture"]["output_type"] == "hybrid"
    assert config["inference"]["long_side_cap"] == 3072
    assert config["inference"]["den_factor"] == 100
    assert config["reported_results"]["ucf_qnrf"] == {
        "mae": 77.8, "rmse": 138.0, "f1": 75.6, "precision": 79.7, "recall": 72.0
    }
    assert config["measured_localization_metric"] == "harness F1@16px in original-image coordinates"
```

- [ ] **Step 2: Run the contract test and confirm the missing-file failure**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_model_brief.py::test_steerer_contract_separates_paper_and_harness_localization`

Expected: FAIL because the STEERER model contract does not exist.

- [ ] **Step 3: Add the model JSON and model brief**

Record the exact blocks `HRNet-W48 multi-resolution backbone`, `multi-resolution counting heads`, `selective inheritance/upsample module`, and `Gaussian density/local-maximum point decoder`; feature scales `x1`, `x4`, and `x8`; original training losses `multi-resolution MSE`, `selective inheritance routing loss`, and `uncertainty weighting where configured`; and failure modes `tiny heads`, `extreme perspective`, `occlusion`, `domain shift`, `padding artifacts`, and `threshold-sensitive localization`.

- [ ] **Step 4: Document the output semantics**

Include this exact distinction in `docs/models/STEERER.md`:

```text
estimated_count = sum(highest_resolution_density / 100)
localization_points = official local maxima merged from x1/x4/x8
zone_count = integral of count-preserving density in the calibrated image zone
paper localization F1 != harness F1@16px
```

- [ ] **Step 5: Run and commit the contract tests**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_model_brief.py`

```bash
git add configs/models/steerer.official.json docs/models/STEERER.md tests/test_model_brief.py
git commit -m "docs: freeze STEERER model contract"
```

---

### Task 3: Implement the Official STEERER Adapter with TDD

**Files:**
- Create: `src/droneai/steerer_adapter.py`
- Create: `tests/test_steerer_adapter.py`

**Interfaces:**
- Produces: `calculate_steerer_size(width: int, height: int, long_side_cap: int = 3072) -> tuple[int, int, float]`
- Produces: `extract_steerer_points(densities: Sequence[np.ndarray], gaussian_maximum: float, resize_ratio: float, original_size: tuple[int, int], threshold: float = 0.15) -> tuple[Point, ...]`
- Produces: `class STEERERAdapter(ModelAdapter)` with `brief()` and `predict()`.

- [ ] **Step 1: Write preprocessing and coordinate tests**

```python
def test_size_caps_long_side_and_pads_to_32() -> None:
    assert calculate_steerer_size(4000, 2000) == (3072, 1536, pytest.approx(0.768))
    assert calculate_steerer_size(1000, 769) == (1024, 800, 1.0)

def test_rgb_imagenet_normalization_and_zero_padding(tmp_path: Path) -> None:
    adapter, backend = fixture_adapter(tmp_path, density=np.ones((4, 4), np.float32))
    adapter.predict(red_rgb_sample(tmp_path, width=33, height=31), retain_native=True)
    assert backend.observed.shape == (3, 32, 64)
    assert backend.observed[0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229)
    assert backend.observed[1, 0, 0] == pytest.approx((0.0 - 0.456) / 0.224)
```

- [ ] **Step 2: Write integrity and output tests**

```python
def test_adapter_rejects_dirty_upstream_and_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="working tree must be clean"):
        build_dirty_adapter(tmp_path)
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        build_hash_mismatch_adapter(tmp_path)

def test_hybrid_prediction_keeps_density_count_separate_from_point_count(tmp_path: Path) -> None:
    adapter = fixture_adapter_with_multiscale_outputs(tmp_path)
    prediction = adapter.predict(sample_with_four_zones(tmp_path), retain_native=True)
    assert prediction.output_type == "hybrid"
    assert prediction.predicted_count == pytest.approx(float(prediction.density.sum()))
    assert prediction.metadata["localization_point_count"] == len(prediction.points)
    assert prediction.coordinate_space == "original_pixels"
```

- [ ] **Step 3: Run the tests and confirm missing symbols**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_steerer_adapter.py`

Expected: collection FAIL because `droneai.steerer_adapter` does not exist.

- [ ] **Step 4: Implement model-independent helpers**

Implement long-side capping, multiple-of-32 bottom/right padding, ImageNet normalization, mass-preserving density resize through the existing density helper, and original-coordinate clipping. The returned `resize_ratio` must describe only geometric resize, not padding.

- [ ] **Step 5: Implement the Torch backend without editing upstream**

Use `mmcv.Config.fromfile(upstream / "configs/QNRF_final.py")`, construct `Baseline_Counter(config.network, config.dataset.den_factor, config.train.route_size, device)`, load the checkpoint with `strict=False`, record missing/unexpected key lists, move to CUDA, and call `model(normalized_batch, labels=None)`. Divide output indices `0`, `-2`, and `-1` by `100`. Pass cloned tensors to the official `local_maximum_points` implementation and reproduce the official x1/x4/x8 distance-based merge.

- [ ] **Step 6: Enforce runtime evidence**

Use this timing order:

```python
torch.cuda.reset_peak_memory_stats(device)
torch.cuda.synchronize(device)
started = time.perf_counter()
with torch.inference_mode():
    outputs = model(batch, labels=None)
torch.cuda.synchronize(device)
latency_ms = (time.perf_counter() - started) * 1000.0
peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
```

- [ ] **Step 7: Convert backend failures to explicit failed predictions**

Return `NativePrediction(output_type="hybrid", predicted_count=None, density=None, points=(), failure_state="<ExceptionType>: <message>")`; never substitute zeros or partial outputs.

- [ ] **Step 8: Run tests and commit**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_steerer_adapter.py`

```bash
git add src/droneai/steerer_adapter.py tests/test_steerer_adapter.py
git commit -m "feat: add official STEERER adapter"
```

---

### Task 4: Reuse the Frozen UCF-QNRF Validation Smoke

**Files:**
- Create: `src/droneai/steerer_smoke.py`
- Create: `tests/test_steerer_smoke.py`
- Create: `configs/evaluation/steerer_ucf_qnrf_smoke.json`

**Interfaces:**
- Consumes: `prepare_smoke_samples`, `write_split_source_manifest`, and split-path checks from `droneai.dm_count_smoke`.
- Produces: `load_steerer_smoke_config`, `validate_steerer_rights_decision`, and `build_steerer_protocol`.

- [ ] **Step 1: Write frozen-identity tests**

```python
def test_steerer_uses_the_exact_dm_count_validation_sample_identity() -> None:
    dm = load_smoke_config(REPO_ROOT / "configs/evaluation/dm_count_ucf_qnrf_smoke.json")
    st = load_steerer_smoke_config(REPO_ROOT / "configs/evaluation/steerer_ucf_qnrf_smoke.json")
    for key in ("dataset_id", "split_id", "expected_samples", "seed", "train_list_sha256", "validation_list_sha256"):
        assert st[key] == dm[key]
    assert st["split_role"] == "validation"
    assert st["sealed_test_access_approved"] is False

def test_research_decision_cannot_be_misreported_as_commercial() -> None:
    decision = validate_steerer_rights_decision(decision_path, manifest_path=manifest_path)
    assert decision["status"] == "PASS_COMMERCIAL_CANDIDATE"
    assert {"research_asset_download", "research_checkpoint_evaluation"}.issubset(decision["allowed_actions"])
    assert "frozen_checkpoint_evaluation" not in decision["allowed_actions"]
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_steerer_smoke.py`

Expected: FAIL because the STEERER smoke module and config do not exist.

- [ ] **Step 3: Add the protocol config**

Copy the frozen split identity and count/runtime targets from DM-Count, change only `run_id`, `protocol_id`, `model_id`, `candidate_id`, `required_action`, and set `localization_radius` to `16.0`. Use `required_action: research_checkpoint_evaluation` and preserve `require_clean_git: true`.

- [ ] **Step 4: Implement a thin STEERER wrapper**

Reuse the existing indexing and sample preparation rather than copying it. Reject any decision that lacks the exact research action or contains `deployment`; include `evaluation_scope: research_comparison_only` in the split/provenance manifest.

- [ ] **Step 5: Run and commit**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_steerer_smoke.py tests/test_dm_count_smoke.py tests/test_ucf_qnrf.py`

```bash
git add src/droneai/steerer_smoke.py tests/test_steerer_smoke.py configs/evaluation/steerer_ucf_qnrf_smoke.json
git commit -m "feat: freeze STEERER UCF-QNRF validation smoke"
```

---

### Task 5: Add the Safe CLI and Isolated Dependency Overlay

**Files:**
- Create: `scripts/run_steerer_ucf_qnrf_smoke.py`
- Create: `tests/test_run_steerer_ucf_qnrf_smoke.py`
- Create: `requirements/models/steerer-home5090.txt`

**Interfaces:**
- Consumes: `STEERERAdapter`, `load_steerer_smoke_config`, `prepare_smoke_samples`, `build_steerer_protocol`, and `run_evaluation`.
- Produces: a CLI that returns `0` only for a successful common report, `2` for a completed failing gate, and raises on integrity/rights/safety violations.

- [ ] **Step 1: Write parser and safety tests**

```python
def test_parser_exposes_no_test_or_training_arguments() -> None:
    destinations = {action.dest for action in _parser()._actions}
    assert "test_root" not in destinations
    assert not {"train", "epochs", "optimizer", "resume"}.intersection(destinations)

def test_nonempty_output_directory_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "keep.txt").write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="new or empty"):
        main(valid_args(output_dir=output))
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_run_steerer_ucf_qnrf_smoke.py`

Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement the CLI by following the DM-Count runner pattern**

Required arguments are `--config`, `--train-root`, `--upstream-dir`, `--train-list`, `--validation-list`, `--checkpoint`, `--checkpoint-sha256`, `--rights-decision`, `--rights-manifest`, `--output-dir`, and `--device`. Add no URL, download, test-root, train, or fine-tune argument.

- [ ] **Step 4: Add the dependency overlay**

Pin these model-only dependencies exactly, without pinning or installing `torch`, `torchvision`, `triton`, CUDA, or NVIDIA packages:

```text
mmcv==1.7.2
opencv-python-headless==4.12.0.88
timm==1.0.19
einops==0.8.1
dict_recursive_update==1.0.1
yacs==0.1.8
tensorboardX==2.6.2.2
fvcore==0.1.5.post20221221
```

- [ ] **Step 5: Run fake end-to-end and commit**

Run: `.venv/Scripts/python.exe -m pytest -q tests/test_run_steerer_ucf_qnrf_smoke.py`

```bash
git add scripts/run_steerer_ucf_qnrf_smoke.py tests/test_run_steerer_ucf_qnrf_smoke.py requirements/models/steerer-home5090.txt
git commit -m "feat: add safe STEERER smoke runner"
```

---

### Task 6: Verify Locally and Synchronize to home5090

**Files:**
- Modify only if verification exposes a tested defect in files from Tasks 1-5.

**Interfaces:**
- Produces: clean commit, full local test evidence, and a fast-forwarded remote working copy.

- [ ] **Step 1: Run the full Windows test suite with the known-good harness environment**

Run: `..\common-evaluation-harness-mvp\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS; the prior baseline was 232 tests before STEERER additions.

- [ ] **Step 2: Run repository integrity checks**

Run: `git diff --check`

Run: `git status --short`

Run: `git ls-files | ForEach-Object { if ((Get-Item $_).Length -gt 5MB) { $_ } }`

Expected: no whitespace errors, only intentional plan/implementation state, and no tracked file larger than 5 MB.

- [ ] **Step 3: Push without rewriting history**

Run: `git push -u origin agent/steerer-ucf-qnrf-adapter`

Expected: branch is published; no force push.

- [ ] **Step 4: Fast-forward the Pop host repository**

```bash
ssh home5090-pop "git -C /home/lucita/crowd-counting-lab status --short && git -C /home/lucita/crowd-counting-lab fetch origin && git -C /home/lucita/crowd-counting-lab checkout agent/steerer-ucf-qnrf-adapter && git -C /home/lucita/crowd-counting-lab pull --ff-only"
```

Expected: the status check is clean before checkout; checkout and pull are fast-forward only.

---

### Task 7: Probe Compatibility, Hash the Official Checkpoint, and Run One Sample

**Files:**
- Create on SSD only: `/workspace/data/checkpoints/steerer/<checkpoint-file>`
- Create on SSD only: `/workspace/data/results/steerer/<new-run-id>/`
- Record small evidence later under: `reports/stage-4/steerer-ucf-qnrf-<run-id>/`

**Interfaces:**
- Produces: environment lock, checkpoint SHA-256, upstream commit proof, one-sample CUDA result, and a pass/fail compatibility decision.

- [ ] **Step 1: Create a model-isolated environment without replacing base Torch**

```bash
ssh home5090-pop "docker exec crowd-jupyter bash -lc 'python -m venv --system-site-packages /workspace/.venvs/steerer && /workspace/.venvs/steerer/bin/python -m pip install -r /workspace/requirements/models/steerer-home5090.txt'"
```

- [ ] **Step 2: Run import and CUDA probes**

```bash
ssh home5090-pop "docker exec crowd-jupyter bash -lc '/workspace/.venvs/steerer/bin/python -c \"import torch,mmcv,cv2,timm; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), mmcv.__version__)\"'"
```

Expected: RTX 5090 is named, CUDA is available, and base Torch remains `2.9.0a0+...nv25.10`.

- [ ] **Step 3: Generate the Stage 3C decision before checkpoint access**

Run `scripts/run_stage3c.py` in the STEERER venv and verify the exact `research_asset_download` and `research_checkpoint_evaluation` actions. Stop immediately if either action is absent.

- [ ] **Step 4: Download to a new SSD filename and hash before loading**

Use the official README checkpoint URL only after the decision in Step 3. Refuse an existing destination. Record URL, byte count, download timestamp, and SHA-256 in `artifact-manifest.json`; do not copy the checkpoint into Git.

- [ ] **Step 5: Inspect checkpoint structure without model execution**

Run `torch.load(path, map_location="cpu", weights_only=False)` in the isolated venv, record the top-level type, state-dict key count, and first/last sorted key names, and compare against `Baseline_Counter.state_dict()` using `strict=False`. Treat a missing/unexpected-key pattern outside the official script's expected behavior as a blocker.

- [ ] **Step 6: Run one frozen validation sample**

Use the first ID from the frozen 36-sample manifest. Verify finite density, nonnegative mass, points within original image bounds, count-density agreement, CUDA synchronization, peak VRAM, and one composite panel.

- [ ] **Step 7: Stop on compatibility failure**

If Python 3.12/MMCV/PyTorch 2.9 compatibility fails, preserve logs and implement a separate Blackwell-compatible image or a tested adapter shim in a new plan. Do not patch the upstream tree or silently downgrade CUDA/Torch.

---

### Task 8: Run the Same 36 Samples, Curate Evidence, and Score

**Files:**
- Create on SSD: full run directory under `/workspace/data/results/steerer/`
- Create in Git: `reports/stage-4/steerer-ucf-qnrf-<run-id>/` containing only small evidence and curated panels.

**Interfaces:**
- Produces: common report, per-sample records, count/density/localization/runtime metrics, model brief, rights decision, provenance hashes, curated panels, and 100-point stage score.

- [ ] **Step 1: Run the 36-image command in the persistent tmux/container session**

Use a new output directory and the exact frozen split/checkpoint hashes. Do not pass an official test path.

- [ ] **Step 2: Verify complete accounting**

Assert `expected_samples == recorded_samples == 36`, `explicit_failures == 0`, and all selected sample IDs match the DM-Count manifest in the same order.

- [ ] **Step 3: Verify reported metrics**

The report must contain MAE, RMSE, signed bias, MAPE reference, GAME-L1, density-zone MAE, localization precision/recall/F1@16px, localization mean distance, median latency, batch-1 FPS, and peak VRAM. Paper values must remain under `official_reported_metrics`, never under measured values.

- [ ] **Step 4: Curate panels**

Keep one best, one upper-middle, one representative median, one lower-middle, and one worst sample by absolute count error, with deterministic tie-breaking by sample ID. Each panel shows original image, predicted density heatmap, GT points, predicted points, four zones, GT count, predicted density count, point count, absolute error, latency, and VRAM.

- [ ] **Step 5: Recompute hashes and the 100-point gate**

Re-hash every committed artifact, rerun provenance validation, ensure the rights banner says `research_comparison_only`, run the common scoring gate, and generate `score.json` plus `score.md`.

- [ ] **Step 6: Run final verification**

Run the full test suite, `git diff --check`, tracked-file size check, upstream-clean check, checkpoint hash check, and selected-panel count check.

- [ ] **Step 7: Commit and push only the review bundle**

```bash
git add reports/stage-4/steerer-ucf-qnrf-* docs/models/STEERER.md
git commit -m "results: add STEERER UCF-QNRF validation smoke"
git push
```

- [ ] **Step 8: Stop for user approval**

Report the model structure, input/output behavior, exact measured values, paper-reference values, run time, score, rights classification, failure modes, and the next recommended model. Do not start a full validation run, official test run, training, fine-tuning, APGCC, or another model.

---

## Self-Review Result

- Spec coverage: architecture explanation, same UCF-QNRF samples, count/density/localization, zones, runtime, hashes, rights separation, selected panels, scoring, and stop point are each assigned to a task.
- Placeholder scan: no deferred implementation markers or unspecified error-handling steps remain.
- Type consistency: STEERER returns `NativePrediction(output_type="hybrid")`; density count remains the scalar prediction; localization points use original-image coordinates; the protocol uses the existing `EvaluationProtocol` and runner.
- Deliberate non-goals: official UCF-QNRF test split, paper-metric reproduction, training, fine-tuning, CCTV calibration, cross-camera merge, and production approval.
