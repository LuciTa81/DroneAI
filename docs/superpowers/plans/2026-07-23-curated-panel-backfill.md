# Curated Review Panel Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the frozen evaluation curation selector produce 12 unique review panels whenever at least 12 successful samples exist, without replacing its existing density-stratified or failure-oriented selections.

**Architecture:** Keep the existing selector phases unchanged, then add one deterministic backfill phase over remaining successful rows. The backfill ranks rows by distance from the global median absolute error and lexical `sample_id`, so resumed runs produce the same selection on every machine.

**Tech Stack:** Python 3.12, NumPy, pytest, the existing `droneai.evaluation_curation` and Round 2 evaluation runner.

## Global Constraints

- Preserve existing selections and their order.
- Never select an explicit failure or duplicate `sample_id`.
- Append at most enough representative samples to reach 12.
- Use category `representative` and reason `global median error backfill`.
- Do not train, fine-tune, calibrate, or change the STEERER checkpoint.
- Preserve the blocked UP-COUNT result before replacing the canonical path.
- UP-COUNT remains `PASS_RESEARCH_ONLY`.

## Approved Execution Variance

The completed ledger was cryptographically bound to Git commit `07f0bde`.
Reusing it after the selector fix at `fca7c5a` would have invalidated the
recorded runtime identity, and the runner correctly rejected that reuse. The
user approved a fresh 166-sample inference on 2026-07-23. Both blocked bundles
were preserved under `attempts`, and the clean rerun was written to the
canonical result path.

---

### Task 1: Reproduce Sparse-Stratum Shortfall

**Files:**
- Modify: `tests/test_evaluation_curation.py`

**Interfaces:**
- Consumes: `select_review_samples(rows, spatial_direction) -> tuple[Selection, ...]`
- Produces: Regression assertions for deterministic representative backfill.

- [x] **Step 1: Add the sparse-density regression test**

Add a test that creates 12 low-density rows, 12 medium-density rows, and only
two high-density rows:

```python
def test_selector_backfills_sparse_density_band_to_twelve() -> None:
    rows = []
    for band, count in (("low", 12), ("medium", 12), ("high", 2)):
        rows.extend(
            _row(
                f"{band}-{index:02d}",
                band,
                float(index - 5),
                float(index),
            )
            for index in range(count)
        )

    selected = select_review_samples(rows, spatial_direction="minimize")

    assert len(selected) == 12
    assert len({item.sample_id for item in selected}) == 12
    assert selected[-1].category == "representative"
    assert selected[-1].reason == "global median error backfill"
    assert selected == select_review_samples(
        list(reversed(rows)),
        spatial_direction="minimize",
    )
```

- [x] **Step 2: Update intentional selector expectations**

Change the positive-only and negative-only tests to expect the existing severe
error selection followed by one representative selection, with no shortfall
when `expected_panels=2`. Change the unavailable-spatial test to expect 12
unique selections, one `representative` category, and no manifest shortfall.
Keep the explicit-failure test unchanged so it continues proving failed rows
are never backfilled.

- [x] **Step 3: Run the focused test and verify RED**

Run:

```powershell
python -m pytest tests/test_evaluation_curation.py -q
```

Expected: the new sparse-density test and updated backfill expectations fail
because the current selector returns only 11 sparse-band selections and does
not append representative rows.

---

### Task 2: Implement Deterministic Representative Backfill

**Files:**
- Modify: `src/droneai/evaluation_curation.py`
- Test: `tests/test_evaluation_curation.py`

**Interfaces:**
- Consumes: the existing `successful`, `used`, and `selected` collections.
- Produces: the unchanged `select_review_samples(...) -> tuple[Selection, ...]`
  interface with a deterministic final backfill phase.

- [x] **Step 1: Add the minimal backfill phase**

Immediately after failure-oriented selection and before returning, compute the
global median absolute error and append unselected rows:

```python
    if len(selected) < 12 and successful:
        global_median = float(
            np.median(
                np.asarray(
                    [float(row.absolute_error) for row in successful],
                    dtype=np.float64,
                )
            )
        )
        backfill = sorted(
            successful,
            key=lambda row: (
                abs(float(row.absolute_error) - global_median),
                row.sample_id,
            ),
        )
        for chosen in backfill:
            if len(selected) >= 12:
                break
            if chosen.sample_id in used:
                continue
            used.add(chosen.sample_id)
            selected.append(
                Selection(
                    chosen.sample_id,
                    "representative",
                    chosen.density_band,
                    "global median error backfill",
                )
            )
```

- [x] **Step 2: Run focused curation tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_evaluation_curation.py -q
```

Expected: all curation tests pass.

- [x] **Step 3: Run evaluation-runner and comparison regression tests**

Run:

```powershell
python -m pytest tests/test_evaluation_runner.py tests/test_round2_comparison.py tests/test_pdf_input_package.py -q
```

Expected: all selected tests pass.

- [x] **Step 4: Run the complete local suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass with only the repository's two known skips.

- [x] **Step 5: Commit and push the tested implementation**

```powershell
git add tests/test_evaluation_curation.py src/droneai/evaluation_curation.py docs/superpowers/plans/2026-07-23-curated-panel-backfill.md
git commit -m "Backfill sparse curated review panels"
git push origin agent/round2-reference-benchmark
```

Expected: the branch advances without a force push.

---

### Task 3: Resume and Reverify STEERER UP-COUNT

**Files:**
- Preserve on SSD: `/workspace/data/results/round2-reference-v1/steerer/attempts/up-count-v1-review-shortfall-07f0bde`
- Regenerate on SSD: `/workspace/data/results/round2-reference-v1/steerer/up-count-v1`

**Interfaces:**
- Consumes: the existing 166-row `progress.jsonl`, frozen sample manifest,
  frozen STEERER checkpoint, and corrected curation selector.
- Produces: a 12-panel, hash-verified `PASS_RESEARCH_ONLY` evaluation bundle.

- [x] **Step 1: Preserve the blocked result**

Verify the destination does not exist, then copy the current result directory:

```bash
docker exec crowd-jupyter bash -lc '
  test ! -e /workspace/data/results/round2-reference-v1/steerer/attempts/up-count-v1-review-shortfall-07f0bde &&
  cp -a \
    /workspace/data/results/round2-reference-v1/steerer/up-count-v1 \
    /workspace/data/results/round2-reference-v1/steerer/attempts/up-count-v1-review-shortfall-07f0bde
'
```

Expected: the blocked bundle remains available unchanged in `attempts`.

- [x] **Step 2: Fast-forward the home5090 repository**

```bash
cd /home/lucita/crowd-counting-lab
git pull --ff-only
```

Expected: the remote repository reaches the new implementation commit and
remains clean.

- [x] **Step 3: Run a clean evaluation under the corrected Git identity**

Run inside `crowd-jupyter`:

```bash
cd /workspace
PYTHONPATH=src /workspace/.venvs/steerer/bin/python \
  scripts/run_round2_reference.py \
  --config configs/evaluation/round2_reference_benchmark.json \
  --runtime-config configs/evaluation/round2_reference_home5090.json \
  --manifest /workspace/data/results/round2-reference-v1/sample-manifest.json \
  --ucf-root /workspace/data/datasets/ucf-qnrf-kaggle-apache \
  --jhu-root /workspace/data/datasets/jhu-crowd-plus-v2 \
  --up-count-root /workspace/data/datasets/up-count-v1 \
  --model steerer \
  --dataset up-count-v1 \
  --output-dir /workspace/data/results/round2-reference-v1/steerer/up-count-v1
```

Expected: all 166 samples are inferred under clean commit `fca7c5a`, without
training, fine-tuning, calibration, or checkpoint changes.

- [x] **Step 4: Verify invariants and artifact hashes**

Compare the corrected bundle with the preserved blocked bundle and require:

```text
predicted_count values: exactly unchanged for all 166 samples
predictions.csv differences: latency_ms only
count and spatial aggregate values: unchanged
selected_count: 12
shortfall: 0
panel_paths: 12 unique files
all evidence-manifest artifact SHA-256 values: verified
score status: PASS_RESEARCH_ONLY
score: 100
failed_blockers: empty
```

- [x] **Step 5: Record the model gate**

Report STEERER's UCF-QNRF, JHU-CROWD++, and UP-COUNT results together, including
dataset scope, MAE, RMSE, signed bias, spatial metric, median latency, peak
VRAM, evidence locations, and independent rights status. Start no DM-Count
work until all three bundles pass.
