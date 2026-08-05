# Round 2 Point-Model Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. This plan is
> intentionally executable by one agent; do not require parallel reviewers or
> subagents.

**Goal:** Replace the unstarted Round 2 DM-Count and MPCount lanes with PET and
APGCC while preserving the completed STEERER results and every historical
artifact.

**Architecture:** Keep the legacy three-model config readable and add a second,
explicit ordered shortlist (`steerer`, `pet`, `apgcc`). Reuse the existing
frozen dataset rows by their sample-manifest SHA-256, but generate the new
shortlist's config identity and manifest at a distinct path. Extend the common
runner and comparison code with PET/APGCC adapters, model-specific rights, and
heterogeneous spatial metrics. Run PET fully before APGCC on home5090, with a
three-dataset one-sample gate before each model's full 334/500/166 evaluation.

**Tech Stack:** Python 3.12, pytest, dataclasses, JSON, NumPy/Pillow, existing
DroneAI evaluation harness, isolated PET and APGCC virtual environments, Docker
`crowd-jupyter`, CUDA on RTX 5090, Git/GitHub.

## Non-negotiable constraints

- Work only in branch `agent/round2-reference-benchmark`.
- Preserve the legacy configs, manifest, Round 1 results, and completed Round 2
  STEERER directories byte-for-byte.
- Never use `git reset --hard`, force-push, or recursive deletion.
- Never train, fine-tune, calibrate, backpropagate, or write a derived weight.
- Keep `/workspace/data` as the only remote data/results root.
- Keep PET `PASS_RESEARCH_ONLY` in all lanes.
- Allow APGCC UCF only as `PASS_COMMERCIAL_CANDIDATE`; JHU and UP-COUNT narrow
  it to `PASS_RESEARCH_ONLY`.
- Treat the combined comparison as `PASS_RESEARCH_ONLY`.
- Do not report a point-rendered visualization as a native density map.
- Advance from one-sample to full evaluation only after the gate artifacts,
  hashes, counts, coordinates, rights, and panel are valid.
- Preserve failed attempts and stop the next lane rather than overwriting.

---

### Task 1: Add the point shortlist without breaking the legacy shortlist

**Files:**

- Modify: `src/droneai/round2_config.py`
- Create: `configs/evaluation/round2_point_reference_benchmark.json`
- Modify: `tests/test_round2_config.py`
- Modify: `tests/test_round2_manifest.py`

**Interfaces:**

- Preserve `load_round2_config(path) -> Round2Config`.
- Export ordered constants for the legacy and point shortlists.
- Reject mixed, reordered, missing, duplicated, and extra model entries.

- [ ] **Step 1: Write failing config tests**

Add assertions equivalent to:

```python
def test_round2_point_config_freezes_exact_matrix() -> None:
    config = load_round2_config(POINT_CONFIG)
    assert tuple(model.model_id for model in config.models) == (
        "steerer",
        "pet",
        "apgcc",
    )
    assert tuple(lane.samples for lane in config.datasets) == (334, 500, 166)
    assert config.total_samples_per_model == 1000
    assert config.training is False
    assert config.fine_tuning is False


@pytest.mark.parametrize(
    "models",
    [
        ["steerer", "pet", "mpcount"],
        ["pet", "steerer", "apgcc"],
        ["steerer", "pet"],
        ["steerer", "pet", "apgcc", "dm-count"],
    ],
)
def test_round2_config_rejects_unapproved_model_matrix(
    tmp_path: Path,
    models: list[str],
) -> None:
    payload = json.loads(POINT_CONFIG.read_text(encoding="utf-8"))
    by_id = {item["model_id"]: item for item in payload["models"]}
    payload["models"] = [
        by_id.get(model_id, {"model_id": model_id, "family": "test"})
        for model_id in models
    ]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="approved ordered model shortlist"):
        load_round2_config(path)
```

Add a manifest identity test that builds both configs from the same fixture
dataset roots and asserts:

```python
assert point.config_identity_sha256 != legacy.config_identity_sha256
assert point.sample_manifest_sha256 == legacy.sample_manifest_sha256
```

- [ ] **Step 2: Run RED**

Run:

```powershell
python -m pytest tests/test_round2_config.py tests/test_round2_manifest.py -q
```

Expected: point config/shortlist support is missing.

- [ ] **Step 3: Implement the two explicit ordered matrices**

Use one source of truth:

```python
LEGACY_MODEL_IDS = ("steerer", "dm-count", "mpcount")
POINT_MODEL_IDS = ("steerer", "pet", "apgcc")
APPROVED_MODEL_MATRICES = (LEGACY_MODEL_IDS, POINT_MODEL_IDS)
SUPPORTED_MODEL_IDS = tuple(dict.fromkeys(LEGACY_MODEL_IDS + POINT_MODEL_IDS))
```

Validate the exact tuple against `APPROVED_MODEL_MATRICES`. Keep any legacy
public constant needed by existing callers as a compatibility alias rather
than silently changing its meaning.

- [ ] **Step 4: Create the point benchmark config**

Create `configs/evaluation/round2_point_reference_benchmark.json` with:

```json
{
  "round_id": "round-2-reference-benchmark-v1",
  "action": "frozen_internal_reference_evaluation",
  "training": false,
  "fine_tuning": false,
  "models": [
    {
      "model_id": "steerer",
      "family": "density_and_points",
      "rights_scope": "PASS_COMMERCIAL_CANDIDATE"
    },
    {
      "model_id": "pet",
      "family": "points",
      "rights_scope": "PASS_RESEARCH_ONLY"
    },
    {
      "model_id": "apgcc",
      "family": "points",
      "rights_scope": "PASS_COMMERCIAL_CANDIDATE"
    }
  ]
}
```

Copy the dataset, metric, review-panel, and safety fields from the approved
legacy config exactly. Do not hand-edit sample IDs.

- [ ] **Step 5: Run GREEN and commit**

Run:

```powershell
python -m pytest tests/test_round2_config.py tests/test_round2_manifest.py -q
git diff --check
git add src/droneai/round2_config.py tests/test_round2_config.py tests/test_round2_manifest.py configs/evaluation/round2_point_reference_benchmark.json docs/superpowers/plans/2026-07-23-round2-point-model-replacement.md
git commit -m "Add Round 2 point-model shortlist"
```

---

### Task 2: Add an exact home5090 runtime profile

**Files:**

- Modify: `src/droneai/round2_runner.py`
- Modify: `scripts/run_round2_reference.py`
- Create: `configs/evaluation/round2_point_reference_home5090.json`
- Modify: `tests/test_round2_runner.py`
- Modify: `tests/test_run_round2_reference.py`

**Interfaces:**

- Preserve the legacy runtime config.
- Require runtime models to match the selected benchmark config in exact order.
- Verify upstream commit, checkpoint SHA-256, environment, and rights path.

- [ ] **Step 1: Write failing runtime and CLI tests**

Cover:

```python
def test_point_runtime_config_freezes_exact_model_order() -> None:
    config = load_round2_runtime_config(POINT_RUNTIME)
    assert tuple(config) == ("steerer", "pet", "apgcc")


def test_runtime_config_rejects_mixed_shortlist(tmp_path: Path) -> None:
    payload = json.loads(POINT_RUNTIME.read_text(encoding="utf-8"))
    payload["models"]["mpcount"] = payload["models"].pop("apgcc")
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="approved ordered model shortlist"):
        load_round2_runtime_config(path)


def test_cli_rejects_runtime_config_mismatch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_round2_reference.py",
            "--config",
            str(POINT_CONFIG),
            "--runtime-config",
            str(LEGACY_RUNTIME),
            "--dry-run",
        ],
    )
    with pytest.raises(ValueError, match="runtime models must exactly match"):
        main()
```

Also assert parser choices include PET/APGCC and preserve DM-Count/MPCount for
legacy verification.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_round2_runner.py tests/test_run_round2_reference.py -q
```

- [ ] **Step 3: Create the exact runtime profile**

Create `configs/evaluation/round2_point_reference_home5090.json` with these
identities:

| Model | Upstream | Commit | Environment | Checkpoint SHA-256 |
|---|---|---|---|---|
| STEERER | `/workspace/upstreams/STEERER` | `5b1854dbc2d280f2326d67c65515d8baf9083810` | `/workspace/.venvs/steerer` | `85f71763595527e419257d459288e4a784766f41687ff0b859dcff65a151bbec` |
| PET | `/workspace/upstreams/PET` | `5b4dd7da8b11568a3305a88bb7c99a7fc831a998` | `/workspace/.venvs/pet` | `58324f86782051591075a923522f33d245ce97f2b50cc43f618313d884b966ad` |
| APGCC | `/workspace/upstreams/APGCC` | `e3e997bf592a70fd34233a432a974200d8c0c847` | `/workspace/.venvs/apgcc` | `cd9aa0f65882c81bb753a4ea8c821e07573b2ae4a31fffd5595fa0a43925cf90` |

PET must carry `long_side_cap: 1536`. Point to the already accepted model
configs, rights decisions, and checkpoints; do not duplicate weights.

- [ ] **Step 4: Implement exact config/runtime matching**

In the CLI, validate before any dataset or model import:

```python
expected = tuple(model.model_id for model in benchmark.models)
actual = tuple(runtime.models)
if actual != expected:
    raise ValueError(
        f"runtime models must exactly match benchmark models: "
        f"expected={expected!r}, actual={actual!r}"
    )
```

- [ ] **Step 5: Run GREEN and commit**

```powershell
python -m pytest tests/test_round2_runner.py tests/test_run_round2_reference.py -q
git diff --check
git add src/droneai/round2_runner.py scripts/run_round2_reference.py tests/test_round2_runner.py tests/test_run_round2_reference.py configs/evaluation/round2_point_reference_home5090.json
git commit -m "Add point-model home5090 runtime"
```

---

### Task 3: Integrate PET and APGCC adapters and rights

**Files:**

- Modify: `src/droneai/round2_runner.py`
- Modify: `tests/test_round2_runner.py`
- Modify: `tests/test_pet_adapter.py`
- Modify: `tests/test_apgcc_adapter.py`

**Interfaces:**

- `load_and_validate_model_config(runtime: RuntimeModel) -> dict[str, object]`
- `build_round2_adapter(model_id: str, *, runtime: RuntimeModel,
  adapter_types: Mapping[str, type[Any]] | None = None) -> ModelAdapter`
- `_checkpoint_claim(model_id, dataset_id)`
- `write_scoped_rights_decision(path: str | Path, *,
  runtime: RuntimeModel, lane: DatasetLane) -> Path`

- [ ] **Step 1: Write failing model-integration tests**

Cover all of:

1. PET loads through `load_pet_smoke_config` and
   `validate_pet_rights_decision`.
2. APGCC loads through `load_apgcc_smoke_config` and
   `validate_apgcc_rights_decision`.
3. PET construction receives `long_side_cap=1536`.
4. PET/APGCC construction uses their existing isolated adapter factories.
5. PET has a conservative checkpoint split claim and is always
   `PASS_RESEARCH_ONLY`.
6. APGCC checkpoint is declared trained on ShanghaiTech-A and disjoint from
   UCF/JHU/UP target rows.
7. APGCC UCF stays `PASS_COMMERCIAL_CANDIDATE`, while JHU/UP narrow to
   `PASS_RESEARCH_ONLY`.
8. PET/APGCC protocols select `localization_f1` as their native spatial metric
   and never claim density PSNR/SSIM.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_round2_runner.py tests/test_pet_adapter.py tests/test_apgcc_adapter.py -q
```

- [ ] **Step 3: Implement the smallest adapter mapping**

Use lazy imports inside the existing factory:

```python
from droneai.apgcc_adapter import APGCCAdapter
from droneai.dm_count_adapter import DMCountAdapter
from droneai.mpcount_adapter import MPCountAdapter
from droneai.pet_adapter import PETAdapter
from droneai.steerer_adapter import STEERERAdapter

adapter_types = {
    "steerer": STEERERAdapter,
    "dm-count": DMCountAdapter,
    "mpcount": MPCountAdapter,
    "pet": PETAdapter,
    "apgcc": APGCCAdapter,
}
if runtime.model_id == "pet":
    kwargs["long_side_cap"] = runtime.long_side_cap
return cast(ModelAdapter, adapter_types[model_id](**kwargs))
```

Reuse the accepted constructors and smoke config validators. Do not fork or
copy upstream model code.

- [ ] **Step 4: Implement model-specific rights**

Rights rules:

```python
if model_id == "pet":
    effective_scope = "PASS_RESEARCH_ONLY"
elif dataset_scope == "PASS_RESEARCH_ONLY":
    effective_scope = "PASS_RESEARCH_ONLY"
else:
    effective_scope = requested_model_scope
```

Record code, checkpoint, and dataset terms separately. `PASS_COMMERCIAL_CANDIDATE`
is not `PRODUCTION_APPROVED`.

- [ ] **Step 5: Run GREEN and commit**

```powershell
python -m pytest tests/test_round2_runner.py tests/test_pet_adapter.py tests/test_apgcc_adapter.py -q
git diff --check
git add src/droneai/round2_runner.py tests/test_round2_runner.py tests/test_pet_adapter.py tests/test_apgcc_adapter.py
git commit -m "Integrate PET and APGCC into Round 2"
```

---

### Task 4: Generalize the comparison to the approved model matrix

**Files:**

- Modify: `src/droneai/round2_comparison.py`
- Modify: `scripts/build_round2_comparison.py`
- Modify: `tests/test_round2_comparison.py`
- Modify: `tests/test_build_round2_comparison.py`

**Interfaces:**

- Preserve legacy three-model comparison support.
- Add explicit point-model order.
- Require exactly three datasets for every selected model.
- Label heterogeneous spatial metrics rather than pooling them.

- [ ] **Step 1: Write failing comparison tests**

Construct nine minimal fixture runs for `steerer`, `pet`, and `apgcc`, then
assert:

```python
assert report["model_order"] == ["steerer", "pet", "apgcc"]
assert report["combined_rights_scope"] == "PASS_RESEARCH_ONLY"
assert report["lanes"] == 9
assert {row["spatial_metric_name"] for row in report["rows"]} == {
    "density_zone_mae",
    "localization_f1",
}
assert report["models"]["pet"]["native_density_map"] is False
assert report["models"]["apgcc"]["native_density_map"] is False
```

Add failure cases for missing lanes, extra lanes, mismatched
`sample_manifest_sha256`, and a PET/APGCC row that claims density PSNR/SSIM.
Retain an explicit legacy-matrix test.

- [ ] **Step 2: Run RED**

```powershell
python -m pytest tests/test_round2_comparison.py tests/test_build_round2_comparison.py -q
```

- [ ] **Step 3: Implement config-driven approved order**

Add `LEGACY_MODEL_ORDER` and `POINT_MODEL_ORDER`, accept only either tuple, and
pass the selected config order from `scripts/build_round2_comparison.py`.

Add notes:

- PET: point/non-point scores, accepted point coordinates, confidences, and
  quadtree split representation; no native density map.
- APGCC: person logits, point coordinates, confidence, and offsets; no native
  density map.
- A point-rendered heatmap is `derived point visualization`.

- [ ] **Step 4: Run GREEN and commit**

```powershell
python -m pytest tests/test_round2_comparison.py tests/test_build_round2_comparison.py -q
git diff --check
git add src/droneai/round2_comparison.py scripts/build_round2_comparison.py tests/test_round2_comparison.py tests/test_build_round2_comparison.py
git commit -m "Support point-model Round 2 comparisons"
```

---

### Task 5: Verify locally, publish, and synchronize home5090 safely

**Files:**

- No new product files unless a test reveals a defect.

- [ ] **Step 1: Run focused tests**

```powershell
python -m pytest tests/test_round2_config.py tests/test_round2_manifest.py tests/test_round2_runner.py tests/test_run_round2_reference.py tests/test_round2_comparison.py tests/test_build_round2_comparison.py tests/test_pet_adapter.py tests/test_apgcc_adapter.py -q
```

- [ ] **Step 2: Run the complete suite**

```powershell
python -m pytest -q
git diff --check
git status --short --branch
```

Expected: at least the previous `530 passed, 2 skipped`, plus the new tests.

- [ ] **Step 3: Push the tested branch**

```powershell
git push origin agent/round2-reference-benchmark
```

- [ ] **Step 4: Verify remote state before pulling**

From Windows:

```powershell
ssh home5090-pop "cd /home/lucita/crowd-counting-lab && git status --short --branch && git branch --show-current && git rev-parse HEAD"
```

Stop if the remote worktree is dirty or on an unexpected branch.

- [ ] **Step 5: Fast-forward only**

```powershell
ssh home5090-pop "cd /home/lucita/crowd-counting-lab && git pull --ff-only origin agent/round2-reference-benchmark"
```

- [ ] **Step 6: Verify container and CUDA without modifying environments**

```powershell
ssh home5090-pop "docker exec crowd-jupyter bash -lc 'cd /workspace && git status --short --branch && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'"
ssh home5090-pop "docker exec crowd-jupyter bash -lc '/workspace/.venvs/pet/bin/python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"'"
ssh home5090-pop "docker exec crowd-jupyter bash -lc '/workspace/.venvs/apgcc/bin/python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"'"
```

---

### Task 6: Generate and verify the new frozen manifest

**Files/artifacts:**

- Create only on SSD:
  `/workspace/data/results/round2-point-reference-v1/sample-manifest.json`
- Preserve:
  `/workspace/data/results/round2-reference-v1/sample-manifest.json`

- [ ] **Step 1: Run CLI dry-run**

```bash
cd /workspace
/workspace/.venvs/harness/bin/python scripts/run_round2_reference.py \
  --config configs/evaluation/round2_point_reference_benchmark.json \
  --runtime-config configs/evaluation/round2_point_reference_home5090.json \
  --manifest-path /workspace/data/results/round2-point-reference-v1/sample-manifest.json \
  --results-root /workspace/data/results/round2-reference-v1 \
  --dry-run
```

Expected: exactly nine planned lanes in `steerer, pet, apgcc` order.

- [ ] **Step 2: Build the manifest at the new path**

Use the existing manifest-generation command exposed by the CLI; do not point
it at the legacy path.

- [ ] **Step 3: Verify identities**

Use a read-only Python check:

```python
assert new["sample_manifest_sha256"] == (
    "37adb8e058611de79d6f97c3181106f6606f5d1bc1a9b611720e4a699395cc04"
)
assert new["sample_manifest_sha256"] == old["sample_manifest_sha256"]
assert new["config_identity_sha256"] != old["config_identity_sha256"]
assert len(new["samples"]) == 1000
```

Also recompute the manifest artifact SHA-256 and record it in the run log.

- [ ] **Step 4: Verify the three existing STEERER lane manifests**

Require every STEERER result to reference the exact frozen
`sample_manifest_sha256`. If any differs, stop and do not include it in the
new comparison.

---

### Task 7: Gate and run PET sequentially

**Artifacts:**

- Runtime: `/workspace/.venvs/pet/bin/python`
- New result lanes:
  `/workspace/data/results/round2-reference-v1/pet/{dataset_id}`
- Failed attempts:
  `/workspace/data/results/round2-reference-v1/pet/attempts/{UTC timestamp}-{dataset_id}`

- [ ] **Step 1: Verify immutable PET identity**

Check:

```bash
cd /workspace/upstreams/PET
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "5b4dd7da8b11568a3305a88bb7c99a7fc831a998"
sha256sum /workspace/data/checkpoints/pet/UCF_QNRF.pth
```

Require checkpoint SHA:

```text
58324f86782051591075a923522f33d245ce97f2b50cc43f618313d884b966ad
```

Recompute and validate the PET rights-decision artifact. Confirm CUDA import
and one batch-1 matrix operation in the PET environment.

- [ ] **Step 2: Run three one-sample lanes**

Use distinct, timestamped `attempts` output directories. Invoke the Round 2 CLI
once per dataset with `--model pet --max-samples 1` and the new config/runtime
and manifest paths.

Require for each sample:

- finite count and latency;
- non-negative, original-image in-bounds points;
- `len(points) == len(confidences) == predicted_count`;
- thresholded point count exactly reproduces the reported count;
- valid 5 × 5 zone aggregation whose sum equals the count;
- recorded peak VRAM;
- one valid panel with `derived point visualization` labelling where relevant;
- technical score and rights gate both pass at the permitted scope.

Visually inspect all three panels before proceeding.

- [ ] **Step 3: Run PET UCF 334**

Only after the UCF one-sample gate:

```bash
/workspace/.venvs/pet/bin/python scripts/run_round2_reference.py \
  --config configs/evaluation/round2_point_reference_benchmark.json \
  --runtime-config configs/evaluation/round2_point_reference_home5090.json \
  --manifest-path /workspace/data/results/round2-point-reference-v1/sample-manifest.json \
  --results-root /workspace/data/results/round2-reference-v1 \
  --model pet --dataset ucf-qnrf-kaggle-apache
```

Verify exactly 334 accepted samples, zero implicit failure, 12 unique curated
panels, all evidence hashes, and `PASS_RESEARCH_ONLY`.

- [ ] **Step 4: Run PET JHU 500**

Repeat with `jhu-crowd-plus-v2`; verify exactly 500 and research-only.

- [ ] **Step 5: Run PET UP-COUNT 166**

Repeat with `up-count-v1`; verify exactly 166 and research-only.

- [ ] **Step 6: Stop-and-report PET gate**

Report per dataset: MAE, RMSE, bias, MAPE reference, localization
precision/recall/F1, mean matched distance, point-zone MAE, median/P95 latency,
FPS, peak VRAM, elapsed time, failed samples, panel count, technical score, and
rights scope. Do not start APGCC unless all three PET lanes are complete and
hash-verified.

---

### Task 8: Gate and run APGCC sequentially

**Artifacts:**

- Runtime: `/workspace/.venvs/apgcc/bin/python`
- New result lanes:
  `/workspace/data/results/round2-reference-v1/apgcc/{dataset_id}`
- Failed attempts:
  `/workspace/data/results/round2-reference-v1/apgcc/attempts/{UTC timestamp}-{dataset_id}`

- [ ] **Step 1: Verify immutable APGCC identity**

Require:

```text
upstream commit:
e3e997bf592a70fd34233a432a974200d8c0c847

checkpoint SHA-256:
cd9aa0f65882c81bb753a4ea8c821e07573b2ae4a31fffd5595fa0a43925cf90
```

Verify clean upstream, rights-decision hash, isolated CUDA import, and batch-1
matrix operation.

- [ ] **Step 2: Run three one-sample lanes**

Run `--model apgcc --max-samples 1` separately for UCF, JHU, and UP-COUNT into
timestamped attempts. Apply the same point/count/zone/latency/VRAM/panel
checks as PET.

Expected scopes:

- UCF: no broader than `PASS_COMMERCIAL_CANDIDATE`;
- JHU: `PASS_RESEARCH_ONLY`;
- UP-COUNT: `PASS_RESEARCH_ONLY`;
- none: `PRODUCTION_APPROVED`.

- [ ] **Step 3: Run full lanes in order**

Run and verify:

1. UCF-QNRF Test: 334;
2. JHU-CROWD++ Validation: 500;
3. UP-COUNT val/test selection: 166.

Stop on the first failure. Preserve the failed attempt and diagnose before
resuming.

- [ ] **Step 4: Stop-and-report APGCC gate**

Report the same full metric and provenance contract as PET, including the
checkpoint-rights ambiguity.

---

### Task 9: Build the final 3 × 3 evidence package

**Artifacts:**

- Create:
  `/workspace/data/results/round2-point-reference-v1/comparison/`
- Git receives only the compact, rights-cleared subset approved by the existing
  export policy.

- [ ] **Step 1: Build comparison**

```bash
cd /workspace
/workspace/.venvs/harness/bin/python scripts/build_round2_comparison.py \
  --config configs/evaluation/round2_point_reference_benchmark.json \
  --results-root /workspace/data/results/round2-reference-v1 \
  --output-dir /workspace/data/results/round2-point-reference-v1/comparison
```

- [ ] **Step 2: Verify the comparison contract**

Require:

- nine lanes exactly;
- model order `steerer, pet, apgcc`;
- sample counts 334/500/166 for every model;
- one frozen sample-manifest SHA across all lanes;
- `training=false`, `fine_tuning=false`;
- all evidence manifest hashes verify;
- 12 unique panels per lane;
- spatial metric names retained per row;
- no PET/APGCC native density accuracy claim;
- combined scope `PASS_RESEARCH_ONLY`.

- [ ] **Step 3: Score the completed stage**

Publish separate scores:

1. technical reproducibility;
2. dataset integrity;
3. artifact/evidence integrity;
4. runtime completeness;
5. rights/commercial readiness.

Do not average a research-only rights blocker into a misleading general PASS.
Use the scope labels:

- `PASS_RESEARCH_ONLY`;
- `PASS_COMMERCIAL_CANDIDATE`;
- `PRODUCTION_APPROVED` only when all code/data/weight/deployment rights are
  explicitly verified (not expected in this plan).

- [ ] **Step 4: Verify branch state and publish compact code/results**

Run the full local suite again before the final commit. Use the existing export
allowlist and size limit; do not copy dataset images, unrestricted checkpoints,
or JHU/UP-derived panels into GitHub without an explicit sharing decision.

## Completion criteria

This plan is complete only when:

- both legacy and point configs pass all tests;
- the remote repository and home5090 are on the same tested commit;
- the new manifest has a different config identity and exact frozen sample-row
  SHA-256;
- completed STEERER evidence remains untouched;
- PET and APGCC each pass three one-sample gates and three full lanes;
- exactly nine verified lanes enter the final comparison;
- every result records input, output, count derivation, spatial metrics,
  train/validation/test partition, zero-training statement, runtime, elapsed
  time, environment, commit, checkpoint/dataset/artifact hashes, and rights;
- no long-term training or fine-tuning has started.
