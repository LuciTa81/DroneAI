# Round 1 Dual-Scenario Share Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a verified 18-page share report that shows all six Round 1 models on the same moderate-count and high-density UCF-QNRF validation scenes.

**Architecture:** Add a report-only scenario inference lane that writes new one-sample evidence without modifying accepted benchmark runs, package the twelve panels into a SHA-256-bound scenario manifest, and render two full-size pages per model from that manifest. Keep the existing 36-image comparison JSON as the only aggregate metric source and keep scenario examples explicitly sample-level.

**Tech Stack:** Python 3.12, PyTorch/CUDA model-specific venvs on home5090, ReportLab, Pillow, pypdf, pytest, JSON manifests, Git/GitHub.

## Global Constraints

- Frozen inputs are exactly `img_0775` (GT 195) and `img_0221` (GT 1,762) from `dm-count-upstream-qnrf-val-smoke-36-v1` validation.
- Run six models for both scenarios; no test path, training, or fine-tuning argument may exist.
- Do not modify accepted 36-image benchmark directories, queue state, rights decisions, or comparison metrics.
- Store native outputs and logs under `/workspace/data/results/round1-report-scenarios-v1`; commit only twelve curated panels and compact evidence under the 25 MiB Git budget.
- Use neutral Korean labels `적정 인원 공통 장면` and `고밀도 공통 장면`; do not use `강점 사례`, `best case`, or `winner` in report text.
- The report status remains `PASS_RESEARCH_ONLY`; technical and rights conclusions remain independent.
- Existing untracked `output/` and `tmp/` in the isolated worktree must be preserved.

---

## File structure

- Create `configs/reporting/round1_dual_scenarios.json`: frozen scenario IDs, GT counts, model order, and deterministic result paths.
- Create `src/droneai/report_scenario_package.py`: validate one-sample evidence and atomically copy twelve panels into a report input package.
- Create `scripts/build_round1_scenario_inputs.py`: CLI for the package builder.
- Create `scripts/run_dm_count_one_sample.py`: report-only DM-Count validation-sample runner.
- Create `scripts/run_steerer_one_sample.py`: report-only STEERER validation-sample runner.
- Create `configs/reporting/round1_dual_scenarios.home5090.json`: pinned Python/runner/argument matrix for all six model environments.
- Create `scripts/run_round1_report_scenarios.py`: fail-closed twelve-command orchestrator with dry-run support.
- Modify `src/droneai/round1_share_report.py`: scenario manifest validation and exact 18-page rendering.
- Modify `tests/test_round1_share_report.py`: 18-page, two-scenario, image, copy, and wording contracts.
- Create `tests/test_report_scenario_package.py`: scenario evidence and hash tests.
- Create `tests/test_run_dm_count_one_sample.py`: DM runner CLI and sample-selection tests.
- Create `tests/test_run_steerer_one_sample.py`: STEERER runner CLI and sample-selection tests.
- Create `tests/test_run_round1_report_scenarios.py`: exact twelve-command and fail-closed orchestration tests.
- Modify `docs/HOME5090_RUNBOOK.md`: exact report-scenario run, resume, packaging, and pull commands.

---

### Task 1: Freeze and package the two-scenario evidence contract

**Files:**
- Create: `configs/reporting/round1_dual_scenarios.json`
- Create: `src/droneai/report_scenario_package.py`
- Create: `scripts/build_round1_scenario_inputs.py`
- Create: `tests/test_report_scenario_package.py`

**Interfaces:**
- Consumes: accepted `results/round1-cctv-comparison-34ad450/comparison.json`, twelve `result.json`/`one-sample-panel.png` pairs, and the deterministic config.
- Produces: `build_report_scenario_package(comparison_path: str | Path, results_root: str | Path, config_path: str | Path, output_dir: str | Path) -> Path` and `assets-manifest.json` schema version 2.

- [ ] **Step 1: Write the failing frozen-config test**

```python
def test_dual_scenario_config_is_exact() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["scenarios"] == {
        "moderate": {"sample_id": "img_0775", "ground_truth_count": 195.0},
        "high_density": {"sample_id": "img_0221", "ground_truth_count": 1762.0},
    }
    assert tuple(payload["model_order"]) == MODEL_PAGE_ORDER
    assert payload["dataset_id"] == "ucf-qnrf-kaggle-apache"
    assert payload["split_role"] == "validation"
```

- [ ] **Step 2: Write failing package tests for completeness, identities, and hashes**

```python
def test_package_contains_six_models_for_both_scenarios(tmp_path: Path) -> None:
    comparison, results, config = make_scenario_fixture(tmp_path)
    output = build_report_scenario_package(comparison, results, config, tmp_path / "package")
    manifest = json.loads((output / "assets-manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert set(manifest["scenarios"]) == {"moderate", "high_density"}
    for scenario in manifest["scenarios"].values():
        assert set(scenario["models"]) == set(MODEL_PAGE_ORDER)
        assert all(len(row["source_sha256"]) == 64 for row in scenario["models"].values())

def test_package_rejects_changed_panel_hash(tmp_path: Path) -> None:
    comparison, results, config = make_scenario_fixture(tmp_path)
    panel = results / "round1-report-scenarios-v1/steerer/img_0775/one-sample-panel.png"
    panel.write_bytes(b"changed")
    with pytest.raises(ValueError, match="panel SHA-256 mismatch"):
        build_report_scenario_package(comparison, results, config, tmp_path / "package")
```

- [ ] **Step 3: Run RED**

Run: `../../.venv/Scripts/python.exe -m pytest tests/test_report_scenario_package.py -q`

Expected: FAIL because the config and package module do not exist.

- [ ] **Step 4: Implement the frozen config and package validator**

Use this public shape and fail before copying when any binding differs:

```python
MODEL_IDS = ("steerer", "dm-count", "pet", "mpcount", "apgcc", "csrnet")

def build_report_scenario_package(comparison_path, results_root, config_path, output_dir):
    comparison = read_object(comparison_path)
    config = read_object(config_path)
    _validate_frozen_comparison(comparison, config)
    scenarios = {}
    assets = []
    for scenario_id, scenario_spec in config["scenarios"].items():
        models = {}
        for model_id in MODEL_IDS:
            result_dir = Path(results_root) / config["result_paths"][model_id][scenario_id]
            row = _validated_result(result_dir, scenario_spec, comparison, model_id)
            models[model_id] = row
        scenarios[scenario_id] = {**scenario_spec, "models": models}
    return _atomic_copy_and_write(output_dir, comparison_path, scenarios, assets)
```

`_validated_result` must recompute the result, panel, source image, annotation,
checkpoint, split-manifest, rights-decision, and environment hashes that the
result declares; require `status == "PASS"`, `split_role == "validation"`,
`failure_state is None`, finite count/error/spatial/runtime fields, and the exact
scenario sample/GT.

- [ ] **Step 5: Implement the CLI**

```python
def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    output = build_report_scenario_package(
        args.comparison, args.results_root, args.config, args.output_dir
    )
    print(json.dumps({"status": "PASS_RESEARCH_ONLY", "output_dir": str(output)}))
    return 0
```

- [ ] **Step 6: Run GREEN and commit**

Run: `../../.venv/Scripts/python.exe -m pytest tests/test_report_scenario_package.py -q`

Expected: all tests PASS.

Commit:

```bash
git add configs/reporting/round1_dual_scenarios.json src/droneai/report_scenario_package.py scripts/build_round1_scenario_inputs.py tests/test_report_scenario_package.py
git commit -m "Add dual-scenario report asset contract"
```

---

### Task 2: Add report-only one-sample runners for DM-Count and STEERER

**Files:**
- Create: `scripts/run_dm_count_one_sample.py`
- Create: `scripts/run_steerer_one_sample.py`
- Create: `tests/test_run_dm_count_one_sample.py`
- Create: `tests/test_run_steerer_one_sample.py`

**Interfaces:**
- Consumes: the existing model adapters, `prepare_smoke_samples`, accepted rights decision/manifest, frozen split lists, and one explicit `--sample-id`.
- Produces: new directories containing `result.json`, `one-sample-panel.png`, `split-source-manifest.json`, `environment-summary.json`, and `model-brief.md` without exposing a test or training path.

- [ ] **Step 1: Write CLI and selection RED tests for both runners**

```python
@pytest.mark.parametrize("runner", [
    "scripts/run_dm_count_one_sample.py",
    "scripts/run_steerer_one_sample.py",
])
def test_report_runner_exposes_only_validation_sample_selection(runner: str) -> None:
    help_text = subprocess.run(
        [sys.executable, runner, "--help"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    assert "--sample-id" in help_text
    assert "--validation-list" in help_text
    assert "--test-root" not in help_text
    assert "--train" not in help_text
```

Add a unit test that supplies two prepared samples and asserts an unknown ID
raises `ValueError("sample-id is not in the frozen split")` before adapter
construction.

- [ ] **Step 2: Run RED**

Run: `../../.venv/Scripts/python.exe -m pytest tests/test_run_dm_count_one_sample.py tests/test_run_steerer_one_sample.py -q`

Expected: FAIL because both runner files do not exist.

- [ ] **Step 3: Implement the minimal report-only runners**

Each runner must use the same selection helper:

```python
def _select_sample(samples, sample_id: str):
    matches = [sample for sample in samples if sample.sample_id == sample_id]
    if len(matches) != 1:
        raise ValueError(f"sample-id is not in the frozen split: {sample_id}")
    return matches[0]
```

The runner sequence is exact: validate rights binding, load the 36-sample frozen
validation preparation, select the explicit sample, write the split manifest,
construct the pinned adapter, call `predict(sample, retain_native=True)`, evaluate
with `evaluate_sample`, render `one-sample-panel.png`, and write a schema-version-1
result whose hashes bind every artifact. Require a new or empty output directory.

- [ ] **Step 4: Run GREEN and relevant regression tests**

Run:

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_run_dm_count_one_sample.py tests/test_run_steerer_one_sample.py tests/test_dm_count_adapter.py tests/test_steerer_adapter.py tests/test_dm_count_smoke.py tests/test_steerer_smoke.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_dm_count_one_sample.py scripts/run_steerer_one_sample.py tests/test_run_dm_count_one_sample.py tests/test_run_steerer_one_sample.py
git commit -m "Add report-only DM-Count and STEERER runners"
```

---

### Task 3: Add a fail-closed home5090 scenario orchestrator

**Files:**
- Create: `configs/reporting/round1_dual_scenarios.home5090.json`
- Create: `scripts/run_round1_report_scenarios.py`
- Create: `tests/test_run_round1_report_scenarios.py`
- Modify: `docs/HOME5090_RUNBOOK.md`

**Interfaces:**
- Consumes: exact model venv Python paths, six runner paths, pinned common/model arguments, two scenario IDs, and a new output root.
- Produces: twelve non-overwriting one-sample result directories and a final `scenario-run-manifest.json` containing command digests and artifact hashes.

- [ ] **Step 1: Write the exact command-matrix RED test**

```python
def test_command_matrix_is_exactly_six_models_times_two_scenarios() -> None:
    commands = build_commands(CONFIG, Path("/workspace/data/results/round1-report-scenarios-v1"))
    assert len(commands) == 12
    assert {(row.model_id, row.sample_id) for row in commands} == {
        (model, sample)
        for model in MODEL_PAGE_ORDER
        for sample in ("img_0775", "img_0221")
    }
    assert all("--sample-id" in row.argv for row in commands)
    assert all("--test-root" not in row.argv for row in commands)
```

Also test that an existing nonempty output directory blocks execution and that
`--dry-run` creates no directories or subprocesses.

- [ ] **Step 2: Run RED**

Run: `../../.venv/Scripts/python.exe -m pytest tests/test_run_round1_report_scenarios.py -q`

Expected: FAIL because the orchestrator and home profile do not exist.

- [ ] **Step 3: Implement the command builder and sequential executor**

```python
@dataclass(frozen=True)
class ScenarioCommand:
    model_id: str
    scenario_id: str
    sample_id: str
    output_dir: Path
    argv: tuple[str, ...]

def execute(commands, *, dry_run: bool) -> list[dict[str, object]]:
    if dry_run:
        return [{"model_id": c.model_id, "sample_id": c.sample_id, "argv": list(c.argv)} for c in commands]
    completed = []
    for command in commands:
        subprocess.run(command.argv, cwd=ROOT, check=True)
        completed.append(validate_result(command))
    return completed
```

The home profile must pin each Python interpreter and every argument already
recorded in `docs/HOME5090_RUNBOOK.md`; do not discover or download checkpoints
during execution. Write the run manifest only after all twelve results validate.

- [ ] **Step 4: Add exact runbook commands**

```bash
cd /workspace
/workspace/.venvs/harness/bin/python scripts/run_round1_report_scenarios.py \
  --config configs/reporting/round1_dual_scenarios.home5090.json \
  --output-root /workspace/data/results/round1-report-scenarios-v1 \
  --dry-run
/workspace/.venvs/harness/bin/python scripts/run_round1_report_scenarios.py \
  --config configs/reporting/round1_dual_scenarios.home5090.json \
  --output-root /workspace/data/results/round1-report-scenarios-v1
```

- [ ] **Step 5: Run GREEN and commit**

Run: `../../.venv/Scripts/python.exe -m pytest tests/test_run_round1_report_scenarios.py -q`

Expected: all tests PASS.

Commit the four files with message `Add home5090 report scenario runner`.

---

### Task 4: Render exactly eighteen pages from the scenario manifest

**Files:**
- Modify: `src/droneai/round1_share_report.py`
- Modify: `tests/test_round1_share_report.py`

**Interfaces:**
- Consumes: scenario package schema version 2 and the unchanged 36-image comparison/content JSON.
- Produces: `build_round1_share_pdf(...) -> Path` with exactly 18 A4 portrait pages and two full-size pages per model.

- [ ] **Step 1: Replace the page-contract test with an exact 18-page RED test**

```python
def test_report_page_contract_has_two_scenarios_per_model() -> None:
    expected = [("cover",), ("summary",), ("evaluation",)]
    for model_id in MODEL_PAGE_ORDER:
        expected.extend(((model_id, "moderate"), (model_id, "high_density")))
    expected.extend((("performance",), ("rights",), ("conclusion",)))
    assert report_page_contract() == tuple(expected)
```

- [ ] **Step 2: Add RED tests for images, captions, and neutral language**

```python
def test_each_model_scenario_page_has_the_common_image_and_metrics(tmp_path: Path) -> None:
    output = build_fixture_report(tmp_path)
    reader = PdfReader(output)
    assert len(reader.pages) == 18
    pages = reader.pages[3:15]
    expected = [(m, s) for m in MODEL_PAGE_ORDER for s in ("moderate", "high_density")]
    for page, (model_id, scenario_id) in zip(pages, expected, strict=True):
        text = page.extract_text() or ""
        sample_id = "img_0775" if scenario_id == "moderate" else "img_0221"
        for token in (DISPLAY_NAMES[model_id], sample_id, "GT", "Pred", "절대오차", "오차율", "PASS_"):
            assert token in text
        assert len(page.images) >= 1
    all_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for forbidden in ("강점 사례", "best case", "winner"):
        assert forbidden not in all_text
```

- [ ] **Step 3: Run RED**

Run: `../../.venv/Scripts/python.exe -m pytest tests/test_round1_share_report.py -q`

Expected: FAIL on the old 12-page/density-points contract.

- [ ] **Step 4: Implement scenario scope validation and page rendering**

Set `PAGE_COUNT = 18`, validate schema version 2 and exact scenario IDs, and
replace `MODEL_PAGE_ASSETS` lookup with:

```python
panel = ctx["assets"]["scenarios"][scenario_id]["models"][model_id]
```

Implement `_page_model_scenario(c, ctx, *, model_id, scenario_id, number)`.
Moderate pages render the aggregate metric strip, full-width panel, sample
caption, and structure/native-output card. High-density pages render a
sample-metric strip, full-width panel, applicable spatial metric, operational
interpretation, limitation, and rights card. Keep body/caption/header minimum
font contracts unchanged. Renumber performance, rights, and conclusion sections
to 15, 16, and 17.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
../../.venv/Scripts/python.exe -m pytest tests/test_round1_share_report.py tests/test_report_scenario_package.py tests/test_pdf_report_content.py -q
```

Expected: all tests PASS.

Commit with message `Render dual-scenario model report`.

---

### Task 5: Push reviewed code, synchronize home5090, and run twelve inferences

**Files:**
- Produce on SSD: `/workspace/data/results/round1-report-scenarios-v1/**`
- Do not modify accepted benchmark directories or `configs/evaluation/model_queue.json`.

**Interfaces:**
- Consumes: Tasks 1-4 code at one clean Git commit and the existing six pinned model environments/checkpoints.
- Produces: twelve `PASS` results and `scenario-run-manifest.json` bound to that commit.

- [ ] **Step 1: Run the focused and full local suites**

```powershell
$env:PYTHONPATH='src;C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages'
& 'C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\work\DroneAI\.venv\Scripts\python.exe' -m pytest -q
```

Expected: all tests PASS with zero collection errors.

- [ ] **Step 2: Push and fast-forward home5090**

```powershell
git push -u origin agent/round1-dual-scenario-report
ssh home5090-pop "cd /home/lucita/crowd-counting-lab && git fetch origin && git checkout agent/round1-dual-scenario-report && git pull --ff-only"
```

Require clean host/container worktrees and identical Windows/host/container commits.

- [ ] **Step 3: Dry-run and execute the twelve commands**

Run the two exact orchestrator commands from Task 3. Use tmux only for the
execution command, not the dry-run. Stop immediately on the first failure and
report the model/sample/log path; do not continue with a partial manifest.

- [ ] **Step 4: Verify scenario results and unchanged workflow status**

```bash
/workspace/.venvs/harness/bin/python scripts/run_round1_report_scenarios.py \
  --config configs/reporting/round1_dual_scenarios.home5090.json \
  --output-root /workspace/data/results/round1-report-scenarios-v1 \
  --verify-only
python scripts/run_model_workflow.py status \
  --queue configs/evaluation/model_queue.json \
  --results-root /workspace/data/results --format json
```

Require 12/12 verified, active model CSRNet, `next_action=review_and_advance`,
expected samples 36, validation split, and `fine_tuning=false`.

---

### Task 6: Package, render, visually inspect, and publish the review PDF

**Files:**
- Produce locally: `tmp/pdfs/round1-dual-scenario-report/assets/`
- Produce locally: `output/pdf/DroneAI_Round1_CCTV_공통장면_모델비교_공유용_검수본.pdf`
- Optionally commit after size check: `results/round1-report-scenarios-v1/` curated panels and compact manifests only.

**Interfaces:**
- Consumes: verified SSD results and Tasks 1/4 builders.
- Produces: 18-page `PASS_RESEARCH_ONLY` PDF, 18 rendered PNG pages, contact sheets, PDF SHA-256, and a compact delivery summary.

- [ ] **Step 1: Pull only compact scenario evidence and panels**

Use a source SHA-256 manifest and `scp` only the twelve `result.json`, twelve
panels, split/environment/model briefs required by the packager. Do not copy
datasets, checkpoints, raw density arrays, or full benchmark directories.

- [ ] **Step 2: Build the scenario package and PDF**

```powershell
python scripts/build_round1_scenario_inputs.py --comparison results/round1-cctv-comparison-34ad450/comparison.json --results-root tmp/pdfs/round1-dual-scenario-report/results --config configs/reporting/round1_dual_scenarios.json --output-dir tmp/pdfs/round1-dual-scenario-report/assets
python scripts/build_round1_share_report.py --comparison results/round1-cctv-comparison-34ad450/comparison.json --assets-manifest tmp/pdfs/round1-dual-scenario-report/assets/assets-manifest.json --content configs/reporting/round1_model_analysis_ko.json --output output/pdf/DroneAI_Round1_CCTV_공통장면_모델비교_공유용_검수본.pdf --report-date 2026-07-19
```

- [ ] **Step 3: Perform machine verification**

Require exactly 18 A4 portrait pages, pages 4-15 in model/scenario order, at
least one image per scenario page, all twelve scenario panels, required GT/Pred/
error/spatial/rights text, no forbidden terms, all asset hashes, and the frozen
comparison SHA-256. Recompute the final PDF SHA-256 and byte size.

- [ ] **Step 4: Render and inspect every page**

Use the PDF skill renderer to create `page-01.png` through `page-18.png`, inspect
all twelve model/scenario pages at original detail, and inspect contact sheets
for pages 1-6, 7-12, and 13-18. Fix any clipping, ellipsis, tiny text, table
spacing, or weak image scaling and rerun focused/full tests after a code change.

- [ ] **Step 5: Final verification, commit, push, and synchronize**

Run the full suite once more. Check the curated Git addition stays below 25 MiB,
commit only reviewed compact evidence/code, push the feature branch, and update
home5090 with `git pull --ff-only`. Keep the feature worktree and PDF until the
user approves the visual result; do not merge to `main` yet.

