# STEERER UCF-QNRF ImageNet Initialization Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pinned STEERER code and a provenance-verified ImageNet HRNet-W48 backbone are used to train a new STEERER weight from UCF-QNRF Train only, while the official STEERER QNRF checkpoint and UCF-QNRF Test remain blocked.

**Architecture:** Add an isolated `steerer-ucf-training` lane beside the existing comparison harness. Project-owned modules create a deterministic 961/240 split, transform raw UCF-QNRF assets, validate rights and hashes, synthesize a pinned upstream config, run T0/T1 training, save atomic resumable checkpoints, and score evidence through the existing `score_stage` primitive. All large artifacts remain under `/workspace/data`; Git receives code, configuration, manifests, score summaries, and curated panels only.

**Tech Stack:** Python 3.12, PyTorch 2.9 nv25.10, CUDA on RTX 5090, official STEERER at commit `5b1854dbc2d280f2326d67c65515d8baf9083810`, MMCV 1.7.2, NumPy, SciPy, Pillow/OpenCV headless, pytest, Git/GitHub.

## Global Constraints

- Work on `agent/steerer-ucf-training-design` or a clean worktree created from it. Do not modify the active Round 2 queue or its CSRNet state.
- Never read UCF-QNRF `Test/` in preparation, T0, T1, T5, or T50. The training CLI accepts only `--train-root`; it has no dataset-root or test-root argument.
- Never load an official STEERER QNRF checkpoint. The only permitted initial model weight is the exact HRNet-W48 ImageNet backbone artifact declared in the training profile.
- Treat Kaggle UCF-QNRF Apache labeling and the ImageNet-derived backbone as internally approved commercial-candidate inputs, not `PRODUCTION_APPROVED` assets.
- Pin and verify STEERER Git commit, clean status, code LICENSE hash, dataset archive/inventory hashes, backbone URL and SHA-256, split hashes, config hash, and every resume checkpoint hash.
- Write processed data, checkpoints, logs, full predictions, and raw density arrays only below `/workspace/data`. No individual Git artifact may exceed 25 MiB.
- Do not overwrite a non-empty processed-data or result directory. Checkpoints use temp-file plus `os.replace` atomic writes.
- Stop after T1 and report for user approval. T5, T50, UCF-QNRF Test, 1,201-image retraining, and full-length training remain separate approval gates.
- Every feature or bug fix starts with a failing test. Run the narrow test, implement the smallest behavior, rerun it, then run the related suite before committing.

## File and Responsibility Map

| File | Responsibility |
|---|---|
| `configs/training/steerer_ucf_qnrf_imagenet.home5090.json` | Immutable training policy and storage/runtime defaults |
| `src/droneai/steerer_training_profile.py` | Typed profile loading, rights scope, path and forbidden-weight validation |
| `src/droneai/steerer_training_split.py` | Deterministic density-stratified 961/240 split and manifest |
| `src/droneai/steerer_training_data.py` | Raw JPG/MAT to STEERER image/JSON conversion and integrity manifest |
| `src/droneai/steerer_training_upstream.py` | Pinned upstream audit, official-config synthesis, backbone-only load audit |
| `src/droneai/steerer_training_checkpoint.py` | Atomic checkpoint, hash manifest, RNG capture/restore, resume verification |
| `src/droneai/steerer_training_runner.py` | T0/T1 engine, effective-batch handling, finite-value and leakage blockers |
| `src/droneai/steerer_training_gate.py` | 100-point T0/T1/T5/T50 gate using `droneai.scoring` |
| `scripts/prepare_steerer_ucf_training.py` | Safe preprocessing CLI |
| `scripts/run_steerer_ucf_training.py` | Stage-bounded CUDA train/resume CLI |
| `scripts/score_steerer_ucf_training.py` | Evidence validation and Markdown/JSON score CLI |
| `tests/test_steerer_training_*.py` | Unit/contract tests for each module and CLI |
| `docs/runbooks/steerer-ucf-qnrf-training-home5090.md` | Exact SSH, prepare, T0, T1, status, resume, result-pull commands |

---

### Task 1: Freeze the Training Profile and Rights Boundary

**Files:**
- Create: `configs/training/steerer_ucf_qnrf_imagenet.home5090.json`
- Create: `src/droneai/steerer_training_profile.py`
- Test: `tests/test_steerer_training_profile.py`

- [ ] **Step 1: Write failing profile contract tests**

```python
def test_profile_freezes_approved_training_contract() -> None:
    profile = load_training_profile(PROFILE)
    assert profile.model_upstream_commit == "5b1854dbc2d280f2326d67c65515d8baf9083810"
    assert profile.dataset_population == 1201
    assert (profile.train_count, profile.validation_count) == (961, 240)
    assert profile.seed == 3035
    assert profile.initialization == "imagenet_backbone_only"
    assert profile.sealed_test_access is False
    assert profile.success_scope == "PASS_COMMERCIAL_CANDIDATE"

def test_profile_rejects_official_steerer_checkpoint() -> None:
    payload = json.loads(PROFILE.read_text())
    payload["initialization"]["model_checkpoint"] = "/weights/QNRF_mae_78.4.pth"
    with pytest.raises(PermissionError, match="official STEERER checkpoint"):
        validate_training_profile(payload)
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run: `python -m pytest tests/test_steerer_training_profile.py -q`

Expected: FAIL with `ModuleNotFoundError: droneai.steerer_training_profile`.

- [ ] **Step 3: Add the exact JSON profile**

The profile must contain these values, with no unresolved value:

```json
{
  "schema_version": 1,
  "run_family": "steerer-ucf-qnrf-imagenet-home5090",
  "runtime_backend": "home5090_docker",
  "model_upstream": {"url": "https://github.com/taohan10200/STEERER.git", "commit": "5b1854dbc2d280f2326d67c65515d8baf9083810", "license_sha256": "5c3649a9ac14d2839d2580710c10bdbc9c70cb6a79c07c06a3858952223b6733"},
  "dataset_id": "ucf-qnrf-kaggle-apache",
  "dataset_population": 1201,
  "split": {"seed": 3035, "train": 961, "validation": 240, "test_role": "sealed"},
  "initialization": {"mode": "imagenet_backbone_only", "model_checkpoint": null},
  "training": {"crop": [768, 768], "scale_range": [0.5, 2.0], "flip": true, "density_factor": 100, "optimizer": "AdamW", "learning_rate": 0.0001, "weight_decay": 0.0001, "warmup_epochs": 10, "scheduler": "cosine", "schedule_horizon_epochs": 800, "effective_batch": 8, "validation_long_side": 3072},
  "stage_epochs": {"T0": 0, "T1": 1, "T5": 5, "T50": 50},
  "storage": {"processed_root": "/workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1", "checkpoint_root": "/workspace/data/checkpoints/steerer-ucf-training", "result_root": "/workspace/data/results/steerer-ucf-training"},
  "rights": {"success_scope": "PASS_COMMERCIAL_CANDIDATE", "production_approved": false, "sealed_test_access": false}
}
```

- [ ] **Step 4: Implement strict typed loading and blockers**

```python
@dataclass(frozen=True)
class SteererTrainingProfile:
    model_upstream_commit: str
    dataset_population: int
    train_count: int
    validation_count: int
    seed: int
    initialization: str
    sealed_test_access: bool
    success_scope: str
    processed_root: Path
    checkpoint_root: Path
    result_root: Path

def validate_initialization(*, backbone: ArtifactReference, model_checkpoint: Path | None) -> None:
    if model_checkpoint is not None:
        raise PermissionError("official STEERER checkpoint loading is forbidden")
    if not backbone.source_url.startswith("https://") or not is_sha256(backbone.sha256):
        raise PermissionError("ImageNet backbone source URL and SHA-256 are required")
```

Also reject unknown keys, non-absolute home5090 paths, any storage path outside `/workspace/data`, non-1201 population, non-961/240 split, and success scopes other than `PASS_COMMERCIAL_CANDIDATE`.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/test_steerer_training_profile.py tests/test_scoring.py -q`

Expected: PASS.

Commit: `git add configs/training/steerer_ucf_qnrf_imagenet.home5090.json src/droneai/steerer_training_profile.py tests/test_steerer_training_profile.py && git commit -m "Add STEERER training policy gate"`

---

### Task 2: Build a Deterministic 961/240 Split Without Test Access

**Files:**
- Create: `src/droneai/steerer_training_split.py`
- Create: `tests/test_steerer_training_split.py`
- Modify: `scripts/prepare_steerer_ucf_training.py` (create CLI skeleton)

- [ ] **Step 1: Write failing split tests using synthetic `UCFQNRFRecord` rows**

```python
def test_split_is_deterministic_disjoint_complete_and_stratified() -> None:
    first = build_training_split(records, seed=3035, validation_count=240)
    second = build_training_split(tuple(reversed(records)), seed=3035, validation_count=240)
    assert first.train_ids == second.train_ids
    assert first.validation_ids == second.validation_ids
    assert len(first.train_ids) == 961
    assert len(first.validation_ids) == 240
    assert set(first.train_ids).isdisjoint(first.validation_ids)
    assert set(first.train_ids) | set(first.validation_ids) == {r.sample_id for r in records}

def test_split_api_cannot_accept_a_test_root() -> None:
    assert "test_root" not in inspect.signature(build_training_split).parameters
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_steerer_training_split.py -q`

Expected: FAIL because the split module does not exist.

- [ ] **Step 3: Implement band-proportional deterministic allocation**

Use `UCFQNRFRecord.density_band` and the existing SHA-256 stable ordering. Compute each validation quota as `floor(240 * band_size / 1201)`, then give remaining slots to the largest fractional remainders with band name as the tie-breaker. Choose validation rows by `_stable_key(sample_id, 3035, "steerer-val-<band>")`; all remaining rows are train.

```python
@dataclass(frozen=True)
class SteererTrainingSplit:
    seed: int
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    band_counts: dict[str, dict[str, int]]
    population_sha256: str
    train_sha256: str
    validation_sha256: str

def build_training_split(
    records: Iterable[UCFQNRFRecord], *, seed: int, validation_count: int
) -> SteererTrainingSplit: ...
```

The population hash is SHA-256 over newline-delimited `sample_id,image_sha256,annotation_sha256,count,density_band` rows sorted by sample ID. Split hashes are SHA-256 over newline-delimited IDs in their written order.

- [ ] **Step 4: Implement atomic split artifact writing**

`write_training_split(output_dir, split)` creates a new empty directory containing `train.txt`, `val.txt`, `test-sealed.json`, and `split-manifest.json`. `test-sealed.json` contains only `{"role":"sealed","sample_count":334,"accessed":false}` and no Test IDs or paths.

- [ ] **Step 5: Add CLI argument-surface test**

```python
def test_prepare_cli_exposes_train_root_but_no_test_or_dataset_root() -> None:
    help_text = subprocess.run(
        [sys.executable, "scripts/prepare_steerer_ucf_training.py", "--help"],
        text=True, capture_output=True, check=True,
    ).stdout
    assert "--train-root" in help_text
    assert "--test-root" not in help_text
    assert "--dataset-root" not in help_text
```

- [ ] **Step 6: Verify and commit**

Run: `python -m pytest tests/test_steerer_training_split.py tests/test_ucf_qnrf.py -q`

Expected: PASS.

Commit: `git add src/droneai/steerer_training_split.py scripts/prepare_steerer_ucf_training.py tests/test_steerer_training_split.py && git commit -m "Add sealed deterministic UCF training split"`

---

### Task 3: Convert Raw UCF-QNRF Assets Into Deterministic STEERER Inputs

**Files:**
- Create: `src/droneai/steerer_training_data.py`
- Create: `tests/test_steerer_training_data.py`
- Modify: `scripts/prepare_steerer_ucf_training.py`

- [ ] **Step 1: Write failing point/count/geometry tests**

```python
def test_convert_sample_preserves_count_and_scales_geometry(tmp_path: Path) -> None:
    result = convert_training_sample(record, output_root=tmp_path)
    payload = json.loads(result.json_path.read_text())
    assert payload["human_num"] == record.count == len(payload["points"])
    assert payload["source_sha256"] == record.image_sha256
    assert all(0 <= x < result.width and 0 <= y < result.height for x, y in payload["points"])
    assert all(0 <= x1 <= x2 < result.width and 0 <= y1 <= y2 < result.height for x1, y1, x2, y2 in payload["boxes"])

def test_preparation_refuses_non_empty_output(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("owned")
    with pytest.raises(FileExistsError, match="new or empty"):
        prepare_training_dataset(records, split, output_root=tmp_path)
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_steerer_training_data.py -q`

Expected: FAIL because the conversion module does not exist.

- [ ] **Step 3: Implement the documented resize and annotation algorithm**

Preserve the raw image bytes and image size for training assets; inspection of the pinned QNRF loader confirms that STEERER performs stochastic 768×768 crop/scale at load time. Normalize only invalid edge coordinates using the already indexed record. JSON contains `points`, `human_num`, `boxes`, source dimensions/hashes, and scale `[1.0, 1.0]`.

For each point, derive an audit box from the mean Euclidean distance to up to three nearest other points. Use half-size `clip(mean_distance * 0.25, 2.0, 64.0)` pixels; a single point uses 8 pixels. Boxes are clipped to image bounds. These boxes are provenance/visualization metadata; the pinned QNRF loader consumes `points` only.

```python
def point_boxes(points: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 4), dtype=np.float32)
    distances = scipy.spatial.distance.cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    k = min(3, max(len(points) - 1, 1))
    local = np.partition(distances, kth=k - 1, axis=1)[:, :k].mean(axis=1)
    if len(points) == 1:
        local[:] = 32.0
    half = np.clip(local * 0.25, 2.0, 64.0)
    return clipped_xyxy(points, half, width=width, height=height)
```

- [ ] **Step 4: Add atomic dataset preparation and integrity manifest**

Write images to `images/<sample_id>.jpg`, annotations to `jsons/<sample_id>.json`, split lists to the root, and manifests to `manifests/`. Copy source image bytes exactly with a streamed copy so the output image SHA-256 equals the input image SHA-256; do not recompress JPEGs. Write each file to a sibling `.tmp`, fsync, then `os.replace`.

The final manifest must show 1,201 converted samples, 961 train, 240 validation, 0 count mismatches, 0 out-of-bounds points/boxes, input/output inventory SHA-256, and `source_partition: official_train_only`.

- [ ] **Step 5: Wire the prepare CLI**

```powershell
python scripts/prepare_steerer_ucf_training.py `
  --config configs/training/steerer_ucf_qnrf_imagenet.home5090.json `
  --train-root /workspace/data/datasets/ucf-qnrf-kaggle-apache/raw/Train `
  --output-root /workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1
```

The CLI prints one JSON object with status, sample counts, manifest path, and manifest SHA-256; it exits 2 on a gate violation.

- [ ] **Step 6: Verify and commit**

Run: `python -m pytest tests/test_steerer_training_data.py tests/test_steerer_training_split.py tests/test_ucf_qnrf.py -q`

Expected: PASS.

Commit: `git add src/droneai/steerer_training_data.py scripts/prepare_steerer_ucf_training.py tests/test_steerer_training_data.py && git commit -m "Add deterministic STEERER training preparation"`

---

### Task 4: Audit the Pinned Upstream and Enforce Backbone-Only Initialization

**Files:**
- Create: `src/droneai/steerer_training_upstream.py`
- Create: `tests/test_steerer_training_upstream.py`

- [ ] **Step 1: Write failing audit and config tests**

```python
def test_audit_rejects_dirty_or_wrong_upstream(fake_git: FakeGit) -> None:
    with pytest.raises(ValueError, match="pinned|clean"):
        audit_upstream(fake_git.path, expected_commit=PINNED, expected_license_sha256=LICENSE_HASH)

def test_initialize_model_loads_backbone_only(fake_backbone: Path) -> None:
    audit = initialize_steerer_model(model, fake_backbone, expected_sha256=sha256_file(fake_backbone))
    assert audit.loaded_scope == "backbone_only"
    assert audit.model_checkpoint_loaded is False
    assert audit.random_head_parameter_count > 0
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_steerer_training_upstream.py -q`

Expected: FAIL because the upstream training module does not exist.

- [ ] **Step 3: Implement upstream audit and generated MMCV config**

```python
@dataclass(frozen=True)
class UpstreamAudit:
    commit: str
    clean: bool
    license_sha256: str
    config_sha256: str

def synthesize_official_config(
    *, upstream_dir: Path, processed_root: Path, backbone_path: Path,
    stage: Literal["T0", "T1", "T5", "T50"], physical_batch: int,
    accumulation_steps: int,
) -> dict[str, object]: ...
```

Load `configs/QNRF_final.py` from the pinned checkout, then assert its architecture, crop, density factor, optimizer, LR, weight decay, warmup, scheduler, scale, seed, and `train.end_epoch=800` equal the approved profile before overriding only dataset root, `train_set=train.txt`, `test_set=val.txt`, device list `(0,)`, workers, batch, and output root. Keep the scheduler horizon at 800 epochs for every stage; T1/T5/T50 are project runner stop ceilings, not scheduler reconfiguration. Reject a `test_set` value containing `test` and any `resume_path` outside the current approved run directory.

- [ ] **Step 4: Implement a model-load audit**

Reset all RNGs to seed 3035 and create a reference `Baseline_Counter` with `pretrained_backbone=None`. Reset the same RNGs again and create the training model with only `pretrained_backbone=<verified ImageNet path>`. Compare parameter digests between the two models: at least one backbone parameter must change, while every counting/FSIA/MSIL head parameter must remain byte-identical to the seeded random reference. The audit JSON records loaded backbone keys, unmatched backbone keys, randomly initialized non-backbone keys, and `model_checkpoint_loaded=false`. Abort if the upstream constructor reads any other weight or if any head key differs.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/test_steerer_training_upstream.py tests/test_steerer_smoke.py -q`

Expected: PASS.

Commit: `git add src/droneai/steerer_training_upstream.py tests/test_steerer_training_upstream.py && git commit -m "Gate STEERER backbone-only initialization"`

---

### Task 5: Add Atomic Checkpoints and Exact Resume Semantics

**Files:**
- Create: `src/droneai/steerer_training_checkpoint.py`
- Create: `tests/test_steerer_training_checkpoint.py`

- [ ] **Step 1: Write failing round-trip and corruption tests**

```python
def test_checkpoint_round_trip_restores_optimizer_scheduler_and_rng(tmp_path: Path) -> None:
    saved = save_training_checkpoint(tmp_path / "last.pth", state)
    restored = load_training_checkpoint(saved.path, expected_sha256=saved.sha256)
    assert restored["epoch"] == 1
    assert restored["optimizer"] == state["optimizer"]
    assert restored["scheduler"] == state["scheduler"]
    assert restored["rng"]["seed"] == 3035

def test_resume_rejects_hash_mismatch_and_wrong_lineage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256|lineage"):
        verify_resume(checkpoint, expected_sha256="0" * 64, expected_run_id="another-run")
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_steerer_training_checkpoint.py -q`

Expected: FAIL because the checkpoint module does not exist.

- [ ] **Step 3: Implement complete state capture and atomic save**

```python
def capture_rng_state(seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }

def atomic_torch_save(payload: dict[str, object], path: Path) -> ArtifactReference:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return artifact_reference(path)
```

Checkpoint payload keys are exactly: schema version, run ID, stage, epoch, global step, model, optimizer, scheduler, scaler, best MAE, best RMSE, config SHA-256, split SHA-256s, dataset inventory SHA-256, backbone SHA-256, upstream commit, RNG state, and environment manifest SHA-256.

- [ ] **Step 4: Add manifest and milestone policy**

Always update `last.pth`; additionally write `best-mae.pth`, `best-rmse.pth`, and stage milestone `milestone-001.pth`, `milestone-005.pth`, or `milestone-050.pth`. Manifest updates are atomic and contain filename, SHA-256, byte count, epoch, metric reason, and parent checkpoint SHA-256.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/test_steerer_training_checkpoint.py tests/test_evaluation_artifacts.py -q`

Expected: PASS.

Commit: `git add src/droneai/steerer_training_checkpoint.py tests/test_steerer_training_checkpoint.py && git commit -m "Add resumable STEERER checkpoint lineage"`

---

### Task 6: Implement the Stage-Bounded T0/T1 Training Runner

**Files:**
- Create: `src/droneai/steerer_training_runner.py`
- Create: `scripts/run_steerer_ucf_training.py`
- Create: `tests/test_steerer_training_runner.py`
- Modify: `src/droneai/steerer_adapter.py`
- Test: `tests/test_steerer_adapter.py`

- [ ] **Step 1: Write failing stage, finite-value, and checkpoint-load tests**

```python
def test_t0_runs_one_update_and_checkpoint_reload(fake_engine: FakeTrainingEngine) -> None:
    result = run_training_stage(profile, stage="T0", engine=fake_engine)
    assert result.optimizer_steps == 1
    assert result.finite_loss is True
    assert result.checkpoint_round_trip is True
    assert result.validation_samples == 0

def test_t1_runs_exactly_one_epoch_and_full_validation(fake_engine: FakeTrainingEngine) -> None:
    result = run_training_stage(profile, stage="T1", engine=fake_engine)
    assert result.completed_epoch == 1
    assert result.validation_samples == 240

def test_t1_to_t5_resume_matches_uninterrupted_lr_sequence(fake_engine: FakeTrainingEngine) -> None:
    staged = run_epochs(fake_engine, stops=(1, 5), schedule_horizon=800)
    uninterrupted = run_epochs(fake_engine, stops=(5,), schedule_horizon=800)
    assert staged.learning_rates == uninterrupted.learning_rates

def test_runner_stops_on_non_finite_loss(fake_engine: FakeTrainingEngine) -> None:
    fake_engine.loss = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite"):
        run_training_stage(profile, stage="T0", engine=fake_engine)
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_steerer_training_runner.py -q`

Expected: FAIL because the runner module does not exist.

- [ ] **Step 3: Implement the pinned upstream engine**

Use project control flow and pinned upstream model, dataset, optimizer, scheduler, loss, and validation primitives. Do not call upstream `tools/train_cc.py` directly because it does not save scheduler/scaler/RNG and does not write atomically.

```python
@dataclass(frozen=True)
class StageResult:
    stage: str
    completed_epoch: int
    optimizer_steps: int
    validation_samples: int
    finite_loss: bool
    checkpoint_round_trip: bool
    physical_batch: int
    accumulation_steps: int
    elapsed_seconds: float

def run_training_stage(
    profile: SteererTrainingProfile,
    *, stage: Literal["T0", "T1", "T5", "T50"],
    run_id: str, resume: Path | None, device: str = "cuda:0",
) -> StageResult: ...
```

Set Python, NumPy, Torch CPU/CUDA seeds to 3035. Use AMP only if the one-batch numerical comparison against FP32 produces finite loss and relative count difference below `1e-3`; record the result. Check every loss, density tensor, gradient norm, and validation count for NaN/Inf.

- [ ] **Step 4: Implement effective batch fallback explicitly**

First probe physical batch 8. On CUDA OOM only, empty cache and retry physical batch 4 with accumulation 2. No other batch change is allowed. Record the selected values and OOM evidence in `environment.json`; do not silently continue after other CUDA errors.

- [ ] **Step 5: Extend the STEERER adapter for project checkpoints**

The adapter may unwrap either upstream `state_dict` or project `model` only when `checkpoint_origin="project_training"` is explicit. Keep the current frozen-checkpoint path unchanged. Reject a payload containing both keys or neither key. Add tests proving old research-smoke checkpoints still load with the same behavior.

- [ ] **Step 6: Add the CLI and hard stage ceiling**

```bash
/workspace/.venvs/steerer/bin/python scripts/run_steerer_ucf_training.py \
  --config configs/training/steerer_ucf_qnrf_imagenet.home5090.json \
  --stage T0 \
  --run-id steerer-qnrf-imagenet-20260806-a \
  --processed-root /workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1 \
  --upstream-dir /workspace/upstreams/STEERER \
  --backbone /workspace/data/checkpoints/backbones/hrnetv2_w48_imagenet_pretrained.pth \
  --backbone-sha256 <64-character-approved-hash>
```

The CLI accepts only `T0` or `T1` without `--resume`. T5/T50 require a verified resume checkpoint and an explicit `--approved-stage` value matching the requested stage. No CLI option can enable Test.

- [ ] **Step 7: Verify and commit**

Run: `python -m pytest tests/test_steerer_training_runner.py tests/test_steerer_adapter.py tests/test_steerer_smoke.py -q`

Expected: PASS.

Commit: `git add src/droneai/steerer_training_runner.py src/droneai/steerer_adapter.py scripts/run_steerer_ucf_training.py tests/test_steerer_training_runner.py tests/test_steerer_adapter.py && git commit -m "Add gated STEERER T0 T1 training runner"`

---

### Task 7: Produce Validation Evidence and the 100-Point Gate

**Files:**
- Create: `src/droneai/steerer_training_gate.py`
- Create: `scripts/score_steerer_ucf_training.py`
- Create: `tests/test_steerer_training_gate.py`

- [ ] **Step 1: Write failing scoring and blocker tests**

```python
def test_t1_score_uses_approved_weights_and_threshold() -> None:
    report = score_training_stage(evidence, stage="T1")
    assert report.threshold == 75
    assert sum(check.weight for check in report.checks) == 100
    assert weights_by_category(report) == {
        "provenance": 25, "split": 20, "stability": 20,
        "validation": 20, "artifacts": 15,
    }

def test_leakage_or_official_checkpoint_is_a_hard_blocker() -> None:
    evidence["split"]["overlap"] = 1
    evidence["initialization"]["model_checkpoint_loaded"] = True
    report = score_training_stage(evidence, stage="T1")
    assert report.status == "BLOCKED"
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_steerer_training_gate.py -q`

Expected: FAIL because the gate module does not exist.

- [ ] **Step 3: Implement evidence schema and scoring**

Use `CheckResult` and `score_stage`. Stage thresholds are T0 70, T1 75, T5 80, T50 85; success status is always `PASS_COMMERCIAL_CANDIDATE`. The five category totals are exactly 25/20/20/20/15.

Hard blockers are split overlap or incompleteness, official Test access, any NaN/Inf, source/split/backbone/checkpoint hash mismatch, official model checkpoint load, dirty/wrong upstream, and resume round-trip failure.

T1 validation evidence includes MAE, RMSE, signed bias, reference MAPE, GAME L1, quadrant zone MAE, localization precision/recall/F1 at 16 px, median/p95 latency, FPS, peak VRAM, density-sum/count difference, train loss, validation loss, and improvement from the first to last training window. A metric may be poor without blocking T1, but it must be finite and present.

- [ ] **Step 4: Write immutable result artifacts**

`score_steerer_ucf_training.py` verifies all referenced artifacts from disk before writing `score.json`, `score.md`, `metrics.json`, `environment.json`, `lineage.json`, and `README.md` to a new empty stage result directory. Hashes in `lineage.json` include Git commit, upstream commit, container image digest, Python/Torch/CUDA/GPU, dataset inventory, split files, backbone, input checkpoint, output checkpoint, config, and rights decision.

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest tests/test_steerer_training_gate.py tests/test_scoring.py tests/test_evaluation_runner.py -q`

Expected: PASS.

Commit: `git add src/droneai/steerer_training_gate.py scripts/score_steerer_ucf_training.py tests/test_steerer_training_gate.py && git commit -m "Score STEERER training stages with hard blockers"`

---

### Task 8: Document Exact Home5090 Operations and Approval Stops

**Files:**
- Create: `docs/runbooks/steerer-ucf-qnrf-training-home5090.md`
- Modify: `README.md`
- Test: `tests/test_steerer_training_runbook.py`

- [ ] **Step 1: Write a failing runbook-command test**

```python
def test_runbook_contains_all_operational_commands_and_approval_stops() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    for token in ("home5090-pop", "T0", "T1", "T5", "T50", "resume", "score.json", "rsync"):
        assert token in text
    assert "Do not run T5 before user approval" in text
    assert "UCF-QNRF Test remains sealed" in text
```

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest tests/test_steerer_training_runbook.py -q`

Expected: FAIL because the runbook does not exist.

- [ ] **Step 3: Write the end-to-end commands**

Document:

1. `ssh home5090-pop` connectivity and `docker ps`/`nvidia-smi` checks.
2. `git fetch origin` and a clean isolated worktree for the training branch.
3. STEERER environment verification with `scripts/setup_steerer_home5090.py --verify-only`.
4. Backbone source URL capture, download outside Git, and SHA-256 calculation.
5. Dataset inventory/hash audit and preparation command.
6. T0, score, T1, score, log/status, and interrupt commands.
7. PC reboot recovery using verified `last.pth` and manifest hash.
8. Explicitly disabled T5/T50 commands shown only under approval sections.
9. Result pull with `rsync` for small JSON/Markdown/panels only; checkpoints remain on SSD.

- [ ] **Step 4: Verify links and commit**

Run: `python -m pytest tests/test_steerer_training_runbook.py -q`

Expected: PASS.

Run: `python -c "from pathlib import Path; assert Path('docs/runbooks/steerer-ucf-qnrf-training-home5090.md').is_file()"`

Commit: `git add docs/runbooks/steerer-ucf-qnrf-training-home5090.md README.md tests/test_steerer_training_runbook.py && git commit -m "Document STEERER home5090 training operations"`

---

### Task 9: Full Verification, Remote T0/T1 Execution, and Handoff

**Files:**
- Modify only if verification reveals an in-scope defect: files introduced in Tasks 1–8
- Create on SSD, not Git: `/workspace/data/results/steerer-ucf-training/<run-id>/...`
- Add to Git after review: `results/steerer-training/<run-id>/T0/` and `results/steerer-training/<run-id>/T1/` small summaries only

- [ ] **Step 1: Run local focused and full tests**

Run:

```powershell
python -m pytest tests/test_steerer_training_profile.py tests/test_steerer_training_split.py tests/test_steerer_training_data.py tests/test_steerer_training_upstream.py tests/test_steerer_training_checkpoint.py tests/test_steerer_training_runner.py tests/test_steerer_training_gate.py tests/test_steerer_training_runbook.py -q
python -m pytest -q
```

Expected: both PASS with zero failed tests.

- [ ] **Step 2: Scan for unresolved placeholders and accidental large files**

Run:

```powershell
rg -n "TODO|TBD|FIXME|<64-character-approved-hash>" configs/training src/droneai/steerer_training* scripts/*steerer*training* docs/runbooks/steerer-ucf-qnrf-training-home5090.md tests/test_steerer_training*
git status --short
git diff --check
```

Expected: no unresolved placeholders in implementation/config/runbook, no whitespace errors, and only intentional files. The angle-bracket hash in this plan is explanatory and is not copied into executable files.

- [ ] **Step 3: Review spec coverage and data-type consistency**

Check every approved design section against at least one module and test: rights, split, preprocessing, hyperparameters, stage ceilings, checkpoint/resume, metrics, scoring, Git boundary, failure handling, approval stops. Verify IDs and hashes remain strings, counts/epochs are integers, metrics are finite floats, and paths are serialized strings.

- [ ] **Step 4: Push the implementation branch and create an isolated remote worktree**

Run after the local tree is clean:

```powershell
git push -u origin agent/steerer-ucf-training-design
ssh home5090-pop "docker exec crowd-jupyter git -C /workspace fetch origin agent/steerer-ucf-training-design"
ssh home5090-pop "docker exec crowd-jupyter git -C /workspace worktree add /workspace/.worktrees/steerer-ucf-training origin/agent/steerer-ucf-training-design"
```

Expected: no forced reset, no modification to `/workspace` active branch, and a clean isolated training worktree.

- [ ] **Step 5: Run remote preflight and preparation**

Verify CUDA matmul, container/image digest, venv, pinned upstream, backbone URL/hash, UCF archive/inventory, free disk, and split/preprocessing integrity. Stop without training if the HRNet backbone file or exact URL/hash is still unavailable.

- [ ] **Step 6: Run and score T0**

Run the exact T0 command from the runbook in the existing `tmux crowd` context or a non-interactive runner. Require one optimizer step, finite outputs, atomic checkpoint write/read, and a non-blocked score before T1.

- [ ] **Step 7: Run and score T1**

Run exactly one epoch, then all 240 validation samples. Do not touch official Test. Require a complete metrics/environment/lineage/score bundle and threshold at least 75 with no blocker.

- [ ] **Step 8: Copy only small evidence into Git and verify hashes**

Copy T0/T1 `score.json`, `score.md`, `metrics.json`, `environment.json`, `lineage.json`, `README.md`, and curated panels below 25 MiB into `results/steerer-training/<run-id>/`. Recompute each copied SHA-256; do not copy `.pth`, optimizer state, raw density, full predictions, or event logs.

- [ ] **Step 9: Final verification and commit**

Run:

```powershell
python -m pytest -q
git diff --check
git status --short
```

Expected: all tests pass and only intentional T0/T1 evidence is uncommitted.

Commit: `git add results/steerer-training docs && git commit -m "Record STEERER UCF T0 T1 training evidence"`

- [ ] **Step 10: Stop at the user approval gate**

Report T0/T1 score, loss behavior, validation MAE/RMSE/bias/MAPE, GAME/zone/localization metrics, latency/FPS/VRAM, checkpoint SHA-256, elapsed time, rights scope, deviations such as batch 4×accumulation 2, and known limitations. Ask for explicit approval before T5. Do not start T5 in the same execution cycle.
