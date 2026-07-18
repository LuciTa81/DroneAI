# Checkpoint Split Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate local split integrity from public-checkpoint training membership so compatibility smoke results cannot be presented as held-out model rankings.

**Architecture:** Add one dependency-free comparison-claim module shared by the evaluation protocol and model workflow. Preserve immutable historical result bundles; classify them through versioned configs and queue metadata, while new runs embed the classification in their manifests and summaries.

**Tech Stack:** Python 3.12, dataclasses, JSON, pytest, Git, Docker/home5090.

## Global Constraints

- Do not open or pass a UCF-QNRF test path.
- Do not train, fine-tune, or rerun an accepted 36-image benchmark.
- Do not edit accepted SSD result artifacts or their hashes.
- `UNKNOWN` and `VERIFIED_OVERLAP` may run only as `compatibility_smoke`.
- `ranking_eligible` is derived only from `held_out_performance` plus `VERIFIED_DISJOINT`.
- Keep datasets, checkpoints, and full results under `/workspace/data`.
- Update home5090 only with `git pull --ff-only` after the Windows branch is clean and pushed.

---

### Task 1: Comparison claim contract and evaluation guard

**Files:**
- Create: `src/droneai/comparison_claim.py`
- Modify: `src/droneai/evaluation_runner.py`
- Modify: `src/droneai/dm_count_smoke.py`
- Modify: `src/droneai/steerer_smoke.py`
- Modify: `src/droneai/pet_smoke.py`
- Modify: `scripts/run_evaluation_fixture.py`
- Modify: `configs/evaluation/fixture_density.json`
- Test: `tests/test_comparison_claim.py`
- Test: `tests/test_evaluation_runner.py`
- Test: `tests/test_dm_count_smoke.py`
- Test: `tests/test_steerer_smoke.py`
- Test: `tests/test_pet_smoke.py`

**Interfaces:**
- Consumes: `checkpoint_training_split_status`, `comparison_scope`, and `checkpoint_split_evidence` from evaluation configuration.
- Produces: `validate_comparison_claim(...) -> None` and `is_ranking_eligible(...) -> bool`, plus matching immutable `EvaluationProtocol` fields.

- [ ] **Step 1: Write the failing comparison-claim tests**

```python
import pytest

from droneai.comparison_claim import is_ranking_eligible, validate_comparison_claim


@pytest.mark.parametrize(
    "status",
    ["VERIFIED_DISJOINT", "VERIFIED_OVERLAP", "UNKNOWN"],
)
def test_compatibility_smoke_accepts_every_checkpoint_status(status: str) -> None:
    validate_comparison_claim(status, "compatibility_smoke", "pinned evidence")
    assert not is_ranking_eligible(status, "compatibility_smoke")


@pytest.mark.parametrize("status", ["VERIFIED_OVERLAP", "UNKNOWN"])
def test_held_out_performance_rejects_unproven_checkpoint(status: str) -> None:
    with pytest.raises(ValueError, match="held-out"):
        validate_comparison_claim(status, "held_out_performance", "pinned evidence")


def test_verified_disjoint_held_out_is_ranking_eligible() -> None:
    validate_comparison_claim(
        "VERIFIED_DISJOINT", "held_out_performance", "official Train/Test boundary"
    )
    assert is_ranking_eligible("VERIFIED_DISJOINT", "held_out_performance")
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `python -m pytest tests/test_comparison_claim.py -q`

Expected: collection fails with `ModuleNotFoundError: droneai.comparison_claim`.

- [ ] **Step 3: Implement the minimal shared contract**

```python
from __future__ import annotations

CHECKPOINT_TRAINING_SPLIT_STATUSES = {
    "VERIFIED_DISJOINT",
    "VERIFIED_OVERLAP",
    "UNKNOWN",
}
COMPARISON_SCOPES = {"compatibility_smoke", "held_out_performance"}


def validate_comparison_claim(status: str, scope: str, evidence: str) -> None:
    if status not in CHECKPOINT_TRAINING_SPLIT_STATUSES:
        raise ValueError("checkpoint_training_split_status is invalid")
    if scope not in COMPARISON_SCOPES:
        raise ValueError("comparison_scope is invalid")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("checkpoint_split_evidence must be a non-empty string")
    if scope == "held_out_performance" and status != "VERIFIED_DISJOINT":
        raise ValueError("held-out performance requires VERIFIED_DISJOINT checkpoint evidence")


def is_ranking_eligible(status: str, scope: str) -> bool:
    return scope == "held_out_performance" and status == "VERIFIED_DISJOINT"
```

Add these immutable fields to `EvaluationProtocol` and call
`validate_comparison_claim(...)` from `__post_init__`:

```python
checkpoint_training_split_status: str
comparison_scope: str
checkpoint_split_evidence: str
```

Write all three fields and derived `ranking_eligible` to `sample-manifest.json`,
`summary.md`, and the existing protocol object inside `evidence-manifest.json`.
Update each protocol builder to pass the three configuration values without
deriving checkpoint status from `split_verified`.

- [ ] **Step 4: Verify GREEN and runner integration**

Run:

```bash
python -m pytest tests/test_comparison_claim.py tests/test_evaluation_runner.py tests/test_dm_count_smoke.py tests/test_steerer_smoke.py tests/test_pet_smoke.py tests/test_run_evaluation_fixture.py -q
```

Expected: all selected tests pass, and the fixture manifest reports
`ranking_eligible: false`.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/droneai/comparison_claim.py src/droneai/evaluation_runner.py \
  src/droneai/dm_count_smoke.py src/droneai/steerer_smoke.py \
  src/droneai/pet_smoke.py scripts/run_evaluation_fixture.py \
  configs/evaluation/fixture_density.json tests/test_comparison_claim.py \
  tests/test_evaluation_runner.py tests/test_dm_count_smoke.py \
  tests/test_steerer_smoke.py tests/test_pet_smoke.py \
  tests/test_run_evaluation_fixture.py
git commit -m "fix: separate checkpoint split provenance"
```

### Task 2: Queue status and immutable historical reclassification

**Files:**
- Modify: `src/droneai/model_workflow.py`
- Modify: `scripts/run_model_workflow.py`
- Modify: `configs/evaluation/model_queue.json`
- Modify: `configs/evaluation/dm_count_ucf_qnrf_smoke.json`
- Modify: `configs/evaluation/steerer_ucf_qnrf_smoke.json`
- Modify: `configs/evaluation/pet_ucf_qnrf_smoke.json`
- Test: `tests/test_model_workflow.py`
- Test: `tests/test_run_model_workflow.py`

**Interfaces:**
- Consumes: the shared `validate_comparison_claim` and `is_ranking_eligible` functions from Task 1.
- Produces: compact workflow status fields `checkpoint_training_split_status`, `comparison_scope`, `checkpoint_split_evidence`, and `ranking_eligible` for the active model and each roadmap entry.

- [ ] **Step 1: Write failing queue/status tests**

Extend the queue fixture with:

```python
"checkpoint_training_split_status": "UNKNOWN",
"comparison_scope": "compatibility_smoke",
"checkpoint_split_evidence": "public checkpoint training membership is not published",
```

Then assert:

```python
assert status["checkpoint_training_split_status"] == "UNKNOWN"
assert status["comparison_scope"] == "compatibility_smoke"
assert status["ranking_eligible"] is False
assert status["roadmap"][1]["ranking_eligible"] is False
```

Add a test removing `checkpoint_split_evidence` and expect `ValueError` mentioning
that field. Add a test setting `comparison_scope` to `held_out_performance` while
status is `UNKNOWN` and expect `ValueError` mentioning held-out evidence.

- [ ] **Step 2: Run the workflow tests and verify RED**

Run: `python -m pytest tests/test_model_workflow.py tests/test_run_model_workflow.py -q`

Expected: assertions fail because current status omits comparison-claim fields.

- [ ] **Step 3: Implement queue validation and status reporting**

In `_validate_model`, normalize the three fields, call
`validate_comparison_claim`, and derive ranking eligibility rather than accepting
it from JSON. In `build_workflow_status`, add the four fields to each roadmap entry
and the active-model top level. Extend Markdown output with:

```text
Comparison: `compatibility_smoke`; checkpoint split `UNKNOWN`; ranking eligible `False`
```

Reclassify versioned configuration without editing SSD result bundles:

```text
DM-Count: VERIFIED_DISJOINT / compatibility_smoke
STEERER: UNKNOWN / compatibility_smoke
PET: UNKNOWN / compatibility_smoke
CSRNet: UNKNOWN / compatibility_smoke
```

Each entry receives a precise evidence sentence. Keep existing accepted artifact
paths and SHA-256 values unchanged.

- [ ] **Step 4: Verify GREEN and compact local status**

Run:

```bash
python -m pytest tests/test_model_workflow.py tests/test_run_model_workflow.py -q
python scripts/run_model_workflow.py status \
  --queue configs/evaluation/model_queue.json \
  --results-root /workspace/data/results \
  --format markdown
```

The second command is run inside home5090 only after Task 3 sync; locally, use the
test fixture because Windows does not own `/workspace/data/results`.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/droneai/model_workflow.py scripts/run_model_workflow.py \
  configs/evaluation/model_queue.json \
  configs/evaluation/dm_count_ucf_qnrf_smoke.json \
  configs/evaluation/steerer_ucf_qnrf_smoke.json \
  configs/evaluation/pet_ucf_qnrf_smoke.json \
  tests/test_model_workflow.py tests/test_run_model_workflow.py
git commit -m "fix: classify smoke comparison eligibility"
```

### Task 3: Full verification, publication, and home5090 status

**Files:**
- Modify only if verification exposes a directly related defect.
- Verify: entire repository test suite and current Git/SSD evidence.

**Interfaces:**
- Consumes: committed Task 1 and Task 2 changes.
- Produces: a clean pushed Windows branch and an identical clean home5090 commit with PET still at `review_and_advance`, now explicitly non-ranking smoke evidence.

- [ ] **Step 1: Run focused and full verification**

```bash
python -m pytest tests/test_comparison_claim.py tests/test_evaluation_runner.py \
  tests/test_model_workflow.py tests/test_run_model_workflow.py -q
python -m pytest -q
git diff --check
git status --short --branch
```

Expected: all tests pass, `git diff --check` emits nothing, and the worktree is
clean except that the branch is ahead of origin.

- [ ] **Step 2: Push safely**

```bash
git push origin agent/steerer-ucf-qnrf-adapter
```

Expected: fast-forward push succeeds without force.

- [ ] **Step 3: Update home5090 and verify compact status**

```bash
ssh home5090-pop "cd /home/lucita/crowd-counting-lab && \
  git pull --ff-only && \
  docker exec crowd-jupyter bash -lc 'cd /workspace && \
    python scripts/run_model_workflow.py status \
      --queue configs/evaluation/model_queue.json \
      --results-root /workspace/data/results \
      --format json'"
```

Expected status fields:

```json
{
  "active_model": "pet",
  "next_action": "review_and_advance",
  "comparison_scope": "compatibility_smoke",
  "checkpoint_training_split_status": "UNKNOWN",
  "ranking_eligible": false
}
```

- [ ] **Step 4: Stop at the next model approval boundary**

Report the reclassification, unchanged PET metrics and artifact hashes, technical
score, rights scope, current commit, and next candidate gate. Do not access the
official UCF-QNRF test split or begin training.
