# Company Share Full-Panel Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-render all 3,000 accepted Round 2 model-image outputs, assemble the approved seven-folder company share tree, and deliver one hash-verified Google Drive-ready ZIP.

**Architecture:** A pure export module validates frozen prediction rows and maintains an append-only resumable panel ledger. A thin Round 2 CLI reuses the accepted adapters, manifests, samples, metrics, and renderer. A separate package module copies only approved artifacts, writes a complete SHA-256 manifest, and creates a single deterministic ZIP outside Git.

**Tech Stack:** Python 3.12, PyTorch/CUDA through the existing model environments, Pillow, standard-library CSV/JSON/ZIP/path utilities, pytest, SSH, Docker, and the existing `crowd-jupyter` container.

## Global Constraints

- Use only `home5090-pop`, `crowd-jupyter`, `/workspace`, and `/workspace/data`.
- Do not train, fine-tune, calibrate, download, replace a checkpoint, or access a new split.
- Use the accepted 3-model × 3-dataset Round 2 shared manifest for exactly 3,000 panels.
- Match every fresh count to the accepted `predictions.csv` value within `1e-6`.
- Never overwrite a conflicting panel, ledger, manifest, staging tree, or ZIP.
- Exclude datasets, checkpoints, weights, virtual environments, caches, credentials, raw density arrays, attempts, and incomplete acquisitions.
- Keep generated panels, staging files, and ZIP files outside Git.
- Preserve the `PASS_RESEARCH_ONLY` package scope and per-component rights notes.
- Use test-first development and verify every long-running result from hashes and counts.

---

### Task 1: Add the resumable full-panel export contract

**Files:**
- Create: `src/droneai/full_panel_export.py`
- Test: `tests/test_full_panel_export.py`

**Interfaces:**
- Consumes: accepted `predictions.csv` rows, `EvaluationSample`, fresh `ScalarEvaluation`, runtime identities, and panel paths.
- Produces:
  - `AcceptedPrediction(sample_id: str, predicted_count: float, output_type: str)`
  - `PanelExportIdentity(model_id, dataset_id, git_commit, upstream_commit, checkpoint_sha256, shared_manifest_sha256)`
  - `load_accepted_predictions(path: Path) -> dict[str, AcceptedPrediction]`
  - `verify_fresh_record(accepted, fresh) -> None`
  - `PanelProgressLedger(path: Path, identity: PanelExportIdentity)`
  - `panel_filename(sample_id: str) -> str`

- [ ] **Step 1: Write failing contract tests**

Add tests proving that accepted rows require unique sample IDs, finite
non-negative counts, and output types; fresh records must match sample ID,
output type, and count within `1e-6`; unsafe sample IDs cannot become paths; and
resume accepts only a matching immutable identity and an existing panel whose
SHA-256 matches the ledger.

```python
def test_verify_fresh_record_rejects_count_drift():
    accepted = AcceptedPrediction("img_1", 10.0, "density")
    fresh = scalar_record(sample_id="img_1", predicted_count=10.001)
    with pytest.raises(ValueError, match="accepted predicted count"):
        verify_fresh_record(accepted, fresh)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_full_panel_export.py -q
```

Expected: collection fails because `droneai.full_panel_export` does not exist.

- [ ] **Step 3: Implement the minimal contract**

Implement frozen dataclasses, strict CSV parsing, `math.isclose` with
`rel_tol=0.0` and `abs_tol=1e-6`, a hexadecimal sample-name suffix for safe PNG
paths, and an append-only JSONL ledger. Each completed ledger row records:

```python
{
    "schema_version": 1,
    "identity": asdict(identity),
    "sample_id": sample_id,
    "source_sha256": source_sha256,
    "accepted_predicted_count": accepted.predicted_count,
    "fresh_predicted_count": fresh.predicted_count,
    "panel_path": relative_panel_path,
    "panel_sha256": sha256_file(panel_path),
    "latency_ms": fresh.latency_ms,
}
```

The ledger loads all rows on startup, rejects duplicate/conflicting sample
records, and returns completed sample IDs only after verifying the corresponding
panel hashes.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_full_panel_export.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/droneai/full_panel_export.py tests/test_full_panel_export.py
git commit -m "Add resumable full-panel export contract"
```

### Task 2: Add the Round 2 full-panel CLI

**Files:**
- Create: `scripts/export_round2_full_panels.py`
- Test: `tests/test_export_round2_full_panels.py`
- Modify: `docs/HOME5090_RUNBOOK.md`

**Interfaces:**
- Consumes: `load_round2_config`, `load_round2_runtime_config`,
  `verify_round2_manifest`, `build_round2_adapter`,
  `scope_adapter_to_dataset`, `build_round2_protocol`,
  `evaluate_sample`, `render_review_panel`, and Task 1 contracts.
- Produces:
  - one model × dataset invocation;
  - `panels/*.png`;
  - `progress.jsonl`;
  - `export-summary.json`;
  - exit code `0` only when the lane has exactly the declared sample count.

- [ ] **Step 1: Write failing CLI tests**

Test parser and orchestration behavior with injected adapters:

- required frozen config/runtime/manifest/result-lane/output arguments;
- explicit model and dataset choices;
- no training, test-selection, download, overwrite, or checkpoint override flag;
- output must be outside the accepted result lane;
- existing matching progress resumes;
- existing conflicting files block;
- fresh count mismatch blocks before panel acceptance;
- exactly the lane sample count is required.

```python
def test_parser_has_no_training_or_download_flags():
    actions = {action.dest for action in module._parser()._actions}
    assert "train" not in actions
    assert "download" not in actions
    assert "checkpoint" not in actions
```

- [ ] **Step 2: Run the tests and verify RED**

```powershell
python -m pytest tests/test_export_round2_full_panels.py -q
```

Expected: collection fails because the script does not exist.

- [ ] **Step 3: Implement the CLI**

Follow `scripts/run_round2_reference.py` for configuration, runtime, rights, and
sample validation. For each uncompleted sample:

```python
prediction = adapter.predict(sample, retain_native=True)
fresh = evaluate_sample(
    sample,
    prediction,
    localization_radius=protocol.localization_radius,
)
verify_fresh_record(accepted[sample.sample_id], fresh)
temporary = panels_dir / f".{panel_filename(sample.sample_id)}.tmp.png"
render_review_panel(..., output_path=temporary, category="company_share_full")
temporary.replace(final_panel)
ledger.append_completed(...)
```

Write `export-summary.json` only after verifying the exact panel count and every
panel hash. Include training/fine-tuning/calibration false, rights scope,
environment identity, elapsed time, and aggregate bytes.

- [ ] **Step 4: Document the exact home5090 commands**

Add one loop-free command template per model/dataset lane and document that
model-specific Python environments are mandatory. Record remote export root:

```text
/workspace/data/exports/DroneAI-company-share-v1/full-panels/{model}/{dataset}
```

- [ ] **Step 5: Run focused tests and verify GREEN**

```powershell
python -m pytest tests/test_full_panel_export.py tests/test_export_round2_full_panels.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add scripts/export_round2_full_panels.py src/droneai/full_panel_export.py tests/test_export_round2_full_panels.py docs/HOME5090_RUNBOOK.md
git commit -m "Add Round 2 full-panel exporter"
```

### Task 3: Add the company-share package builder

**Files:**
- Create: `src/droneai/company_share_package.py`
- Create: `scripts/build_company_share_package.py`
- Test: `tests/test_company_share_package.py`
- Test: `tests/test_build_company_share_package.py`

**Interfaces:**
- Consumes: nine completed full-panel export summaries, accepted Round 1 result
  roots, compact comparisons, report PDF paths, model integration files, and
  optional STEERER field/stress outputs.
- Produces:
  - `build_company_share(spec: CompanyShareSpec) -> PackageReport`
  - the approved `Drone AI` staging tree;
  - `결과해석/패키지_검증정보/files.sha256`;
  - `package-manifest.json`;
  - one `DroneAI_회사공유용_v1.zip`.
  - one external `DroneAI_회사공유용_v1.zip.sha256` sidecar.

- [ ] **Step 1: Write failing package tests**

Use temporary directories to prove:

- exactly seven root directories are created;
- exactly 3,000 Round 2 PNGs are required;
- Round 1 DM-Count, MPCount, and CSRNet folders are labeled as 36-image smoke;
- forbidden suffixes (`.pth`, `.pt`, `.ckpt`, `.npy`, `.npz`) are rejected;
- forbidden path components (`datasets`, `checkpoints`, `.venv`, `attempts`,
  `partial`) are rejected;
- symlinks and junction-like entries are rejected;
- every staged content file other than `files.sha256` and
  `package-manifest.json` is in `files.sha256`;
- ZIP members exactly match staged files;
- UTF-8 Korean member names survive extraction;
- a pre-existing staging root or ZIP blocks rather than being overwritten.

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/test_company_share_package.py tests/test_build_company_share_package.py -q
```

Expected: collection fails because the package module and CLI do not exist.

- [ ] **Step 3: Implement safe package assembly**

Implement explicit allowlists for every copied evidence filename and model-code
file. Copy with `shutil.copy2`, reject links using `os.lstat`, write the Korean
readmes in UTF-8, hash every staged file in normalized POSIX path order, then
create the ZIP with `ZIP_DEFLATED`, UTF-8 member names, and no absolute or
parent-traversal paths.

The manifest records:

```python
{
    "schema_version": 1,
    "scope": "PASS_RESEARCH_ONLY",
    "round2_panel_count": 3000,
    "round1_models": ["dm-count", "mpcount", "csrnet"],
    "excluded_assets": ["datasets", "checkpoints", "weights", "raw_density"],
    "files": file_count,
    "uncompressed_bytes": total_bytes,
    "content_manifest_sha256": content_manifest_sha256,
}
```

Write the ZIP SHA-256 to an external `.zip.sha256` sidecar after the ZIP closes;
never place a ZIP self-hash inside the ZIP.

- [ ] **Step 4: Run focused tests and verify GREEN**

```powershell
python -m pytest tests/test_company_share_package.py tests/test_build_company_share_package.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/droneai/company_share_package.py scripts/build_company_share_package.py tests/test_company_share_package.py tests/test_build_company_share_package.py
git commit -m "Add company share package builder"
```

### Task 4: Verify code, synchronize home5090, and run a 30-panel pilot

**Files:**
- Modify only if a failing test proves a defect in Tasks 1-3.
- Produce outside Git:
  `/workspace/data/exports/DroneAI-company-share-v1/pilot`

**Interfaces:**
- Consumes: committed exporter and existing accepted evidence.
- Produces: 30 hash-verified pilot panels, measured average size, elapsed time,
  and no change to accepted result lanes.

- [ ] **Step 1: Run focused and full tests**

```powershell
python -m pytest tests/test_full_panel_export.py tests/test_export_round2_full_panels.py tests/test_company_share_package.py tests/test_build_company_share_package.py -q
python -m pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Push and fast-forward home5090**

```powershell
git push origin agent/round2-reference-benchmark
ssh home5090-pop "docker exec crowd-jupyter bash -lc 'cd /workspace && git pull --ff-only'"
```

Verify both repositories identify the same clean commit.

- [ ] **Step 3: Run ten deterministic samples per model**

Run three model-specific pilot invocations using their existing environments and
the frozen Round 2 manifest. Limit only the export count to ten per model for
pilot measurement; do not change the accepted sample manifest or result files.

- [ ] **Step 4: Verify pilot**

Require 30/30 panels, zero count mismatches, zero explicit failures, valid panel
hashes, and an average size consistent with a final package below the available
Windows and home5090 disk budgets.

### Task 5: Run all 3,000 panels and build the final ZIP

**Files:**
- Produce outside Git:
  - `/workspace/data/exports/DroneAI-company-share-v1`
  - `C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\outputs\DroneAI_회사공유용_v1`
  - `C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\outputs\DroneAI_회사공유용_v1.zip`

**Interfaces:**
- Consumes: nine frozen lanes and verified package builder.
- Produces: final staged tree, ZIP, ZIP SHA-256, and delivery report.

- [ ] **Step 1: Launch nine resumable lane exports**

Run one lane at a time in a persistent remote session so model environments do
not contaminate each other. Preserve progress on interruption and resume from
verified panels.

- [ ] **Step 2: Verify all lane summaries**

Require:

```text
STEERER: 334 + 500 + 166 = 1,000
PET:     334 + 500 + 166 = 1,000
APGCC:   334 + 500 + 166 = 1,000
Total:                       3,000
```

Every lane must report zero count mismatches and zero explicit failures.

- [ ] **Step 3: Assemble and ZIP the share tree**

Run `scripts/build_company_share_package.py` with explicit report, result,
integration-code, field-output, and full-panel paths. Do not pass dataset or
checkpoint roots.

- [ ] **Step 4: Transfer to Windows**

Copy the completed staging tree and ZIP from home5090 into the declared Windows
delivery paths without deleting any pre-existing user directory.

- [ ] **Step 5: Verify delivery**

Verify:

- ZIP SHA-256 matches the remote package manifest;
- extraction to a fresh verification directory succeeds;
- extracted file count and every SHA-256 match the staged tree;
- exactly 3,000 Round 2 full-panel PNGs are present;
- forbidden files and path components are absent;
- the final package size is recorded;
- Git worktrees on Windows and home5090 are clean and at the same commit;
- GitHub PR CI passes after the exporter commits.

- [ ] **Step 6: Deliver**

Return clickable Windows links for the folder, ZIP, package manifest, and
`00_먼저읽기.md`, plus exact bytes, SHA-256, panel count, elapsed time, test
result, rights scope, and Google Drive upload guidance.
