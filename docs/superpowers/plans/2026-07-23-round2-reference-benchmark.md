# Round 2 Reference Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a hash-frozen, research-only 1,000-image reference benchmark for STEERER, DM-Count, and MPCount across UCF-QNRF Test, JHU-CROWD++ Validation, and UP-COUNT validation/test frames without training or fine-tuning.

**Architecture:** Add dataset-specific read-only indexers that normalize official point labels into the existing `EvaluationSample` contract, then freeze one shared 1,000-sample manifest before inference. Extend the common evaluation runner with an append-only, identity-bound progress ledger and expose one dataset-neutral Round 2 CLI that constructs the accepted existing adapters. Each dataset produces its own metric bundle; a final comparison package refuses to pool mixed-rights metrics into a commercial score.

**Tech Stack:** Python 3.12, pytest, dataclasses, Pillow, NumPy/SciPy, existing DroneAI evaluation contracts and model adapters, Git, Docker `crowd-jupyter`, RTX 5090 CUDA environments.

## Global Constraints

- Use `home5090-pop`, the existing `crowd-jupyter` container, `/workspace`, and `/workspace/data`; do not create or replace the container.
- Use the exact accepted Round 1 upstream commits, checkpoint SHA-256 values, and model environments.
- Process exactly 334 UCF-QNRF official Test images, 500 JHU-CROWD++ official Validation images, and 166 UP-COUNT official validation/test frames per model.
- The combined benchmark is `PASS_RESEARCH_ONLY`; UCF-QNRF remains separately addressable as `PASS_COMMERCIAL_CANDIDATE` under the recorded internal decision.
- No optimizer, backward pass, gradient update, adaptation, learned calibration, checkpoint write, training, or fine-tuning is allowed.
- JHU-CROWD++ and UP-COUNT source/derived images stay on SSD unless a separate sharing-rights decision permits copying.
- Git receives only source, configs, compact manifests, aggregate metrics, scores, and cleared curated panels; total review bundle stays below 25 MiB.
- Stop after every one-sample or full-model gate and report the status contract before advancing.

---

### Task 1: Freeze Round 2 configuration and dataset rights

**Files:**
- Create: `configs/datasets/jhu_crowd_plus.research.json`
- Create: `configs/evaluation/round2_reference_benchmark.json`
- Create: `src/droneai/round2_config.py`
- Create: `tests/test_round2_config.py`

**Interfaces:**
- Consumes: existing dataset manifests and accepted model IDs.
- Produces: `Round2Config`, `DatasetLane`, `ModelLane`, and `load_round2_config(path: str | Path) -> Round2Config`.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_round2_config_freezes_exact_matrix() -> None:
    config = load_round2_config(CONFIG)
    assert [model.model_id for model in config.models] == ["steerer", "dm-count", "mpcount"]
    assert [(lane.dataset_id, lane.partition, lane.samples) for lane in config.datasets] == [
        ("ucf-qnrf-kaggle-apache", "test", 334),
        ("jhu-crowd-plus-v2", "val", 500),
        ("up-count-v1", "val_test", 166),
    ]
    assert config.total_samples_per_model == 1000
    assert config.combined_rights_scope == "PASS_RESEARCH_ONLY"
    assert config.fine_tuning is False

def test_round2_config_rejects_training_action(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text())
    payload["action"] = "fine_tuning"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frozen_internal_reference_evaluation"):
        load_round2_config(path)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_round2_config.py -q`

Expected: collection fails because `droneai.round2_config` does not exist.

- [ ] **Step 3: Implement immutable typed configuration validation**

```python
@dataclass(frozen=True)
class DatasetLane:
    dataset_id: str
    partition: str
    samples: int
    rights_scope: str
    manifest_path: Path

@dataclass(frozen=True)
class ModelLane:
    model_id: str
    family: str
    rights_scope: str

@dataclass(frozen=True)
class Round2Config:
    round_id: str
    action: str
    models: tuple[ModelLane, ...]
    datasets: tuple[DatasetLane, ...]
    total_samples_per_model: int
    combined_rights_scope: str
    fine_tuning: bool

def load_round2_config(path: str | Path) -> Round2Config:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["action"] != "frozen_internal_reference_evaluation":
        raise ValueError("action must be frozen_internal_reference_evaluation")
    models = tuple(ModelLane(**row) for row in payload["models"])
    datasets = tuple(
        DatasetLane(manifest_path=Path(row["manifest_path"]), **{k: v for k, v in row.items() if k != "manifest_path"})
        for row in payload["datasets"]
    )
    if [row.model_id for row in models] != ["steerer", "dm-count", "mpcount"]:
        raise ValueError("model order differs from the approved Round 2 shortlist")
    if sum(row.samples for row in datasets) != 1000:
        raise ValueError("Round 2 requires exactly 1,000 samples per model")
    if payload["combined_rights_scope"] != "PASS_RESEARCH_ONLY" or payload["fine_tuning"] is not False:
        raise ValueError("Round 2 must remain research-only with fine_tuning=false")
    return Round2Config(models=models, datasets=datasets, **{k: payload[k] for k in (
        "round_id", "action", "total_samples_per_model", "combined_rights_scope", "fine_tuning"
    )})
```

The JHU manifest records official URL `https://www.crowd-counting.com/`, download ID `1FkdvHyAom1B2aVj6_jZpZPW01sQNiI7n`, terms URL `https://www.crowd-counting.com/terms-of-use.html`, `commercial_use=false`, and allowed action `noncommercial_research`.

- [ ] **Step 4: Run focused tests and commit**

Run: `python -m pytest tests/test_round2_config.py tests/test_stage1.py -q`

Expected: all tests pass.

```bash
git add configs/datasets/jhu_crowd_plus.research.json configs/evaluation/round2_reference_benchmark.json src/droneai/round2_config.py tests/test_round2_config.py
git commit -m "Freeze Round 2 reference benchmark scope"
```

### Task 2: Normalize UCF Test, JHU Validation, and UP-COUNT selection

**Files:**
- Modify: `src/droneai/ucf_qnrf.py`
- Create: `src/droneai/jhu_crowd.py`
- Modify: `src/droneai/up_count.py`
- Create: `src/droneai/round2_samples.py`
- Modify: `tests/test_ucf_qnrf.py`
- Create: `tests/test_jhu_crowd.py`
- Modify: `tests/test_prepare_up_count.py`
- Create: `tests/test_round2_samples.py`

**Interfaces:**
- Produces `index_ucf_qnrf_partition(...)`, `index_jhu_validation(...)`, `select_up_count_reference(...)`, and `load_round2_samples(...) -> tuple[EvaluationSample, ...]`.

- [ ] **Step 1: Write failing tests for official partition isolation**

```python
def test_index_ucf_test_requires_exactly_334_when_frozen(tmp_path: Path) -> None:
    _write_ucf_fixture(tmp_path, count=2)
    with pytest.raises(ValueError, match="expected 334"):
        index_ucf_qnrf_partition(tmp_path, partition="test", expected_samples=334)

def test_index_jhu_validation_reads_official_points_and_labels(tmp_path: Path) -> None:
    _write_jhu_sample(tmp_path, "0001", points=[(10, 11), (20, 21)], scene="stadium", weather=0)
    rows = index_jhu_validation(tmp_path / "val", expected_samples=1)
    assert rows[0].count == 2
    assert rows[0].condition_tags == {"scene": "stadium", "weather": "none", "distractor": "false", "density_band": "low"}

def test_up_count_selection_is_deterministic_sequence_balanced(tmp_path: Path) -> None:
    rows = _up_count_inventory_fixture(tmp_path, sequences=3, frames=90)
    first = select_up_count_reference(rows, sample_count=12, namespace="round2-upcount-v1", minimum_frame_gap=30)
    second = select_up_count_reference(tuple(reversed(rows)), sample_count=12, namespace="round2-upcount-v1", minimum_frame_gap=30)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert {row["split"] for row in first} <= {"val", "test"}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_ucf_qnrf.py tests/test_jhu_crowd.py tests/test_prepare_up_count.py tests/test_round2_samples.py -q`

Expected: failures identify the three missing public interfaces.

- [ ] **Step 3: Implement the dataset readers**

JHU `gt/{sample_id}.txt` rows are parsed as six integers:

```python
center_x, center_y, width, height, occlusion, blur = map(int, line.split())
point = (float(center_x), float(center_y))
```

JHU `image_labels.txt` rows are parsed as `img_index,count,scene,weather,distractor`; the declared count must match GT rows and every point must fall inside the image. UCF uses existing `_annotation_path`, `read_ucf_qnrf_points`, `_normalized_points`, and `UCFQNRFRecord`. UP-COUNT selection uses largest-remainder per-sequence quotas and SHA-256 order of `round2-upcount-v1:{sequence_id}:{frame_id}` with a 30-frame gap, recording `gap_relaxed=true` only for shortfall fill.

`load_round2_samples` converts each normalized record to the existing `EvaluationSample` with dataset-specific `split_id`, point density downsample 8, quadrant zones, and explicit `source_partition` condition tags.

- [ ] **Step 4: Run focused tests and commit**

Run: `python -m pytest tests/test_ucf_qnrf.py tests/test_jhu_crowd.py tests/test_prepare_up_count.py tests/test_round2_samples.py -q`

Expected: all focused tests pass.

```bash
git add src/droneai/ucf_qnrf.py src/droneai/jhu_crowd.py src/droneai/up_count.py src/droneai/round2_samples.py tests/test_ucf_qnrf.py tests/test_jhu_crowd.py tests/test_prepare_up_count.py tests/test_round2_samples.py
git commit -m "Normalize Round 2 dataset partitions"
```

### Task 3: Build and verify the shared 1,000-sample manifest

**Files:**
- Create: `src/droneai/round2_manifest.py`
- Create: `scripts/build_round2_reference_manifest.py`
- Create: `tests/test_round2_manifest.py`
- Create: `tests/test_build_round2_reference_manifest.py`

**Interfaces:**
- Produces `build_round2_manifest(config, dataset_roots) -> dict[str, object]`, `verify_round2_manifest(payload, dataset_roots) -> tuple[EvaluationSample, ...]`, and a CLI with `--verify-only`.

- [ ] **Step 1: Write failing integrity tests**

```python
def test_manifest_contains_exact_counts_and_aggregate_hash(fixture_roots: dict[str, Path]) -> None:
    payload = build_round2_manifest(_fixture_config(), fixture_roots)
    assert payload["counts"] == {"ucf-qnrf-kaggle-apache": 2, "jhu-crowd-plus-v2": 2, "up-count-v1": 2}
    assert len(payload["sample_manifest_sha256"]) == 64
    assert len({row["global_sample_id"] for row in payload["samples"]}) == 6

def test_verify_manifest_rejects_changed_annotation(frozen_manifest: Path, fixture_roots: dict[str, Path]) -> None:
    target = next(fixture_roots["jhu"].rglob("*.txt"))
    target.write_text(target.read_text() + "1 1 1 1 1 0\n")
    with pytest.raises(ValueError, match="annotation SHA-256"):
        verify_round2_manifest(json.loads(frozen_manifest.read_text()), fixture_roots)
```

- [ ] **Step 2: Verify RED, implement atomic manifest writing, then verify GREEN**

Run RED: `python -m pytest tests/test_round2_manifest.py tests/test_build_round2_reference_manifest.py -q`

The manifest canonicalizes JSON with `sort_keys=True`, separators `(',', ':')`, UTF-8, hashes every source and annotation, prefixes IDs with dataset ID, and writes through `{output}.tmp` followed by `Path.replace` only after self-verification.

Run GREEN: `python -m pytest tests/test_round2_manifest.py tests/test_build_round2_reference_manifest.py -q`

- [ ] **Step 3: Commit**

```bash
git add src/droneai/round2_manifest.py scripts/build_round2_reference_manifest.py tests/test_round2_manifest.py tests/test_build_round2_reference_manifest.py
git commit -m "Freeze shared Round 2 sample manifest"
```

### Task 4: Add identity-bound resumable evaluation progress

**Files:**
- Create: `src/droneai/evaluation_progress.py`
- Modify: `src/droneai/evaluation_runner.py`
- Create: `tests/test_evaluation_progress.py`
- Modify: `tests/test_evaluation_runner.py`

**Interfaces:**
- Produces `ProgressLedger.open(path, identity_sha256, ordered_sample_ids)`, `completed_records()`, and `append(record, native_metadata, fingerprint)`.
- Extends `run_evaluation(..., resume: bool = False)` without changing default behavior.

- [ ] **Step 1: Write failing ledger tests**

```python
def test_progress_resume_reuses_immutable_prefix(tmp_path: Path) -> None:
    ledger = ProgressLedger.open(tmp_path / "progress.jsonl", "a" * 64, ("a", "b"))
    ledger.append(_record("a"), {"shape": "1x1"}, "b" * 64)
    resumed = ProgressLedger.open(tmp_path / "progress.jsonl", "a" * 64, ("a", "b"))
    assert [row.sample_id for row in resumed.completed_records()] == ["a"]

def test_progress_rejects_changed_run_identity(tmp_path: Path) -> None:
    ProgressLedger.open(tmp_path / "progress.jsonl", "a" * 64, ("a",))
    with pytest.raises(ValueError, match="identity"):
        ProgressLedger.open(tmp_path / "progress.jsonl", "b" * 64, ("a",))
```

- [ ] **Step 2: Verify RED and implement append-only fsync ledger**

The first JSONL row is a header containing schema version, run identity, and ordered sample-ID hash. Each later row stores the complete `ScalarEvaluation`, native metadata scalars, and native-output fingerprint. `append` opens in binary append mode, writes one canonical JSON line, flushes, and calls `os.fsync`. Existing rows are never rewritten.

Run: `python -m pytest tests/test_evaluation_progress.py -q`

Expected: PASS.

- [ ] **Step 3: Integrate resume into the common runner**

`run_evaluation` computes the progress identity from protocol JSON, checkpoint SHA-256, rights SHA-256, provenance references, and the ordered sample manifest. With `resume=True`, it loads a verified prefix, skips those samples, and appends after each successful `adapter.predict`. Final reports are still written only once all expected rows exist. Existing non-resume callers retain their prior reserved-output behavior.

Run: `python -m pytest tests/test_evaluation_runner.py tests/test_evaluation_progress.py -q`

Expected: all existing tests plus resume tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/droneai/evaluation_progress.py src/droneai/evaluation_runner.py tests/test_evaluation_progress.py tests/test_evaluation_runner.py
git commit -m "Resume hash-bound evaluation runs"
```

### Task 5: Add the dataset-neutral Round 2 model runner

**Files:**
- Create: `src/droneai/round2_runner.py`
- Create: `scripts/run_round2_reference.py`
- Create: `tests/test_round2_runner.py`
- Create: `tests/test_run_round2_reference.py`

**Interfaces:**
- Produces `build_round2_adapter(model_id, ...) -> ModelAdapter`, `build_round2_protocol(model_id, dataset_lane, ...) -> EvaluationProtocol`, and CLI modes `--dry-run`, `--one-sample`, `--resume`, and full dataset execution.

- [ ] **Step 1: Write failing CLI and factory tests**

```python
def test_dry_run_lists_nine_dataset_runs_without_inference(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(CONFIG), "--manifest", str(MANIFEST), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["runs"]) == 9
    assert payload["fine_tuning"] is False

def test_mixed_rights_protocol_cannot_claim_ranking() -> None:
    protocol = build_round2_protocol("steerer", _jhu_lane(), _rights_fixture())
    assert protocol.comparison_scope == "research_reference_only"
    assert protocol.checkpoint_training_split_status == "UNKNOWN"
    assert protocol.sealed_test_access_approved is False
```

- [ ] **Step 2: Verify RED and implement the minimal model factory**

The factory constructs only existing `STEERERAdapter`, `DMCountAdapter`, or `MPCountAdapter`, verifies the same Round 1 upstream/checkpoint hashes, and rejects any unknown model. No model-specific dataset sampling code is allowed.

- [ ] **Step 3: Implement one-sample and full-run modes**

Each model is loaded once per dataset lane. `--one-sample` selects the first manifest sample and writes a separate compatibility result. Full mode calls `run_evaluation(..., resume=True)`. UCF Test sets `sealed_test_access_approved=true` only because this implementation plan is the recorded current user authorization; JHU uses split role `validation`; UP-COUNT uses split role `validation` with condition tag `official_partition` preserving val/test origin.

Run: `python -m pytest tests/test_round2_runner.py tests/test_run_round2_reference.py tests/test_evaluation_runner.py -q`

Expected: all tests pass and dry-run performs no adapter construction.

- [ ] **Step 4: Commit**

```bash
git add src/droneai/round2_runner.py scripts/run_round2_reference.py tests/test_round2_runner.py tests/test_run_round2_reference.py
git commit -m "Run frozen Round 2 model matrix"
```

### Task 6: Build dataset-wise comparison and document operations

**Files:**
- Create: `src/droneai/round2_comparison.py`
- Create: `scripts/build_round2_comparison.py`
- Create: `tests/test_round2_comparison.py`
- Modify: `docs/HOME5090_RUNBOOK.md`

**Interfaces:**
- Produces a hash-verified comparison JSON/Markdown with per-dataset metrics, per-model architecture/output notes, rights columns, and no pooled commercial score.

- [ ] **Step 1: Write failing comparison safety tests**

```python
def test_comparison_refuses_missing_or_mixed_manifest_hashes(tmp_path: Path) -> None:
    runs = _nine_run_fixture(tmp_path)
    _change_manifest_hash(runs[-1])
    with pytest.raises(ValueError, match="shared manifest"):
        build_round2_comparison(runs)

def test_comparison_is_research_only_and_dataset_wise(tmp_path: Path) -> None:
    payload = build_round2_comparison(_nine_run_fixture(tmp_path))
    assert payload["rights_scope"] == "PASS_RESEARCH_ONLY"
    assert "pooled_commercial_score" not in payload
    assert set(payload["datasets"]) == {"ucf-qnrf-kaggle-apache", "jhu-crowd-plus-v2", "up-count-v1"}
```

- [ ] **Step 2: Verify RED, implement comparison, and verify GREEN**

Run: `python -m pytest tests/test_round2_comparison.py -q`

The comparison re-hashes every metrics, score, prediction, selection, environment, rights, and evidence file; requires 9/9 accepted runs; writes atomically; and reports MAE, RMSE, bias, MAPE reference, spatial metrics, latency, FPS, VRAM, and limitations separately per dataset.

- [ ] **Step 3: Document exact home5090 commands and commit**

Run: `python -m pytest -q && git diff --check`

Expected: 487 existing tests plus all new tests pass, two existing skips remain, and diff check is clean.

```bash
git add src/droneai/round2_comparison.py scripts/build_round2_comparison.py tests/test_round2_comparison.py docs/HOME5090_RUNBOOK.md
git commit -m "Compare Round 2 reference runs"
```

### Task 7: Push, acquire datasets, freeze evidence, and execute gates

**Files:**
- Runtime only: `/workspace/data/datasets/*`, `/workspace/data/results/round2-reference-v1/*`
- Commit later: compact verified result manifests and summaries only.

**Interfaces:**
- Consumes all implementation tasks and the current user authorization.
- Produces three Stage 1 decisions, one shared manifest, nine one-sample results, nine full dataset results, and the final comparison.

- [ ] **Step 1: Verify and publish implementation branch**

```powershell
python -m pytest -q
git diff --check
git status --short
git push -u origin agent/round2-reference-benchmark
```

Expected: all tests pass and only intentional files are committed.

- [ ] **Step 2: Safely synchronize home5090**

```bash
cd /home/lucita/crowd-counting-lab
git status --short
git fetch origin
git switch --track -c agent/round2-reference-benchmark origin/agent/round2-reference-benchmark
git pull --ff-only
docker exec crowd-jupyter bash -lc 'cd /workspace && git rev-parse HEAD && git status --short'
```

Stop if either host or container reports uncommitted files.

- [ ] **Step 3: Acquire into versioned staging directories and run Stage 1**

Use only the recorded official UCF/Kaggle source, JHU official Google Drive ID, and UP-COUNT Zenodo record. Downloads land under `.partial` directories and are atomically renamed only after archive/file hashes, expected partitions, and file counts verify. Run `scripts/prepare_up_count.py`, the JHU normalizer, and Stage 1 for all datasets. Expected statuses: UCF `PASS_COMMERCIAL_CANDIDATE`; JHU and UP-COUNT `PASS_RESEARCH_ONLY`.

- [ ] **Step 4: Freeze and verify the exact 1,000-sample manifest**

```bash
/workspace/.venvs/harness/bin/python scripts/build_round2_reference_manifest.py \
  --config configs/evaluation/round2_reference_benchmark.json \
  --ucf-root /workspace/data/datasets/ucf-qnrf-kaggle-apache \
  --jhu-root /workspace/data/datasets/jhu-crowd-plus-v2 \
  --up-count-root /workspace/data/datasets/up-count-v1 \
  --output /workspace/data/results/round2-reference-v1/sample-manifest.json
```

Expected: exactly 334/500/166 rows, 1,000 unique global IDs, and a printed SHA-256.

- [ ] **Step 5: Execute the STEERER gates**

Run dry-run, then three one-sample lanes, verify every result/hash/panel, and report. Only after that report passes, run the three full STEERER lanes with `--resume` in tmux `crowd`. Verify exactly 1,000 accepted rows and report metrics/score/rights before advancing.

- [ ] **Step 6: Execute DM-Count and MPCount gates**

Repeat the same one-sample then full-run sequence for DM-Count, report and verify after each model, and never start the next model before the prior model's 1,000-row evidence is accepted.

- [ ] **Step 7: Build comparison and pull compact evidence**

```bash
/workspace/.venvs/harness/bin/python scripts/build_round2_comparison.py \
  --config configs/evaluation/round2_reference_benchmark.json \
  --results-root /workspace/data/results/round2-reference-v1 \
  --output-dir /workspace/data/results/round2-reference-comparison-v1
```

Verify all evidence hashes and the 25 MiB Git budget before copying only compact summaries/manifests/cleared panels to Windows and Git. Keep datasets, full predictions, restricted panels, and native outputs on SSD.

## Final verification

Before declaring completion:

```powershell
python -m pytest -q
git diff --check
git status --short --branch
ssh home5090-pop "docker exec crowd-jupyter bash -lc 'cd /workspace && git status --short --branch && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'"
```

Confirm 3 models × 1,000 identical sample identities, no training artifacts, per-dataset metrics only, `PASS_RESEARCH_ONLY` combined scope, and separate UCF commercial-candidate evidence.
