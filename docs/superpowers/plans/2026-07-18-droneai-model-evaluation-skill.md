# DroneAI Model Evaluation Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact, project-specific Codex skill and deterministic repository status command that reuse existing DroneAI gates, then use them to run STEERER's frozen 36-image UCF-QNRF evaluation safely.

**Architecture:** Keep all executable research behavior in DroneAI and make the personal skill a thin router. A versioned model queue plus verified SSD artifact references determines the active model, completed gates, and next action; the skill reads the compact status instead of reconstructing project history from chat. Existing model runners remain authoritative for inference and metrics.

**Tech Stack:** Python 3.12, pytest, JSON profiles, PowerShell/SSH, Codex personal skills, RTX 5090 Docker runtime.

## Global Constraints

- Personal skill path is `C:\Users\user\.codex\skills\evaluating-droneai-models`.
- Default runtime is `home5090_docker`; Colab remains legacy/fallback.
- Use one agent by default and do not browse pinned evidence again.
- Never access the UCF-QNRF official test split.
- Never commit datasets, checkpoints, raw density arrays, or full run directories.
- Keep every Git review bundle below 25 MiB.
- Rights scope and technical score remain separate.
- Do not train or fine-tune during Round 1; fine-tune at most one selected winner at the end.

---

### Task 1: Add the compact model queue and status engine

**Files:**
- Create: `configs/evaluation/model_queue.json`
- Create: `src/droneai/model_workflow.py`
- Create: `tests/test_model_workflow.py`

**Interfaces:**
- Consumes: queue JSON, `/workspace/data/results`, SHA-256 evidence references.
- Produces: `load_model_queue(path: Path) -> dict[str, object]`, `build_workflow_status(queue_path: Path, results_root: Path) -> dict[str, object]`.

- [ ] **Step 1: Write failing queue and status tests**

Test a temporary results tree containing a hash-matched DM-Count score/metrics pair and STEERER one-sample result. Assert active model `steerer`, completed gate `one_sample`, next action `benchmark`, dataset `ucf-qnrf-kaggle-apache`, expected samples `36`, and an explicit approval requirement. Add failures for a changed artifact hash, unknown active model, unsafe test split, and `benchmark` without one-sample evidence.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_model_workflow.py -q`

Expected: collection fails because `droneai.model_workflow` does not exist.

- [ ] **Step 3: Implement the queue schema and minimal status engine**

The queue must record `dm-count`, `steerer`, `pet`, and `csrnet` in order. Each accepted evidence entry contains a results-root-relative path, SHA-256, artifact kind, and gate. Validate all strings, SHA-256 values, relative paths, split role, sample count, unique order, and that the active model exists. Recalculate evidence hashes before marking a gate complete.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `python -m pytest tests/test_model_workflow.py -q`

Expected: all model-workflow tests pass.

- [ ] **Step 5: Commit**

```powershell
git add configs/evaluation/model_queue.json src/droneai/model_workflow.py tests/test_model_workflow.py
git commit -m "Add compact model evaluation status"
```

### Task 2: Add the safe workflow CLI

**Files:**
- Create: `scripts/run_model_workflow.py`
- Create: `tests/test_run_model_workflow.py`
- Modify: `docs/HOME5090_RUNBOOK.md`

**Interfaces:**
- Consumes: `build_workflow_status`, queue path, results root.
- Produces: CLI subcommand `status` with compact JSON and Markdown; nonzero exit on invalid evidence.

- [ ] **Step 1: Write failing CLI tests**

Invoke the script with a temporary queue/results tree. Assert JSON contains `active_model`, `current_stage`, `next_action`, `rights_scope`, `dataset`, `validation`, and `roadmap`. Assert `--format markdown` includes the same fields without embedding historical documents. Assert invalid evidence returns exit code 2.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_run_model_workflow.py -q`

Expected: FAIL because `scripts/run_model_workflow.py` is missing.

- [ ] **Step 3: Implement `status` only**

Use `argparse` with required `--queue`, required `--results-root`, and `--format {json,markdown}`. Do not add shell execution or generic command templates. The model-specific existing runners remain separately invoked after the status gate.

- [ ] **Step 4: Run focused and adjacent tests**

Run: `python -m pytest tests/test_run_model_workflow.py tests/test_model_workflow.py -q`

Expected: PASS.

- [ ] **Step 5: Document the home5090 status command and commit**

```powershell
git add scripts/run_model_workflow.py tests/test_run_model_workflow.py docs/HOME5090_RUNBOOK.md
git commit -m "Add model workflow status command"
```

### Task 3: Freeze STEERER evaluation dependencies

**Files:**
- Modify: `requirements/models/steerer-home5090.txt`
- Modify: `scripts/setup_steerer_home5090.py`
- Modify: `tests/test_setup_steerer_home5090.py`
- Modify: `docs/models/STEERER.md`

**Interfaces:**
- Consumes: existing `setup_environment` and current verified environment.
- Produces: exact 17-package overlay including `scikit-image==0.26.0`, `ImageIO==2.37.3`, `tifffile==2026.7.14`, and `lazy_loader==0.4`.

- [ ] **Step 1: Extend tests before the allowlist**

Require the four evaluation distributions and reject missing or changed versions. Update the expected allowlist cardinality from 13 to 17.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_setup_steerer_home5090.py -q`

Expected: FAIL because the production allowlist lacks evaluation packages.

- [ ] **Step 3: Extend exact pins and metadata verification**

Add the four exact requirements to the file and `FROZEN_REQUIREMENTS`; reuse existing generic metadata checks. Update the model brief wording to 17 packages.

- [ ] **Step 4: Run focused STEERER tests**

Run: `python -m pytest tests/test_setup_steerer_home5090.py tests/test_steerer_adapter.py tests/test_steerer_smoke.py tests/test_run_steerer_ucf_qnrf_smoke.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add requirements/models/steerer-home5090.txt scripts/setup_steerer_home5090.py tests/test_setup_steerer_home5090.py docs/models/STEERER.md
git commit -m "Freeze STEERER evaluation dependencies"
```

### Task 4: Create and validate the personal Codex skill

**Files:**
- Create: `C:\Users\user\.codex\skills\evaluating-droneai-models\SKILL.md`
- Create: `C:\Users\user\.codex\skills\evaluating-droneai-models\references\status-contract.md`
- Create: `C:\Users\user\.codex\skills\evaluating-droneai-models\agents\openai.yaml`

**Interfaces:**
- Consumes: `python scripts/run_model_workflow.py status`, active model config, existing runner and Stage 3C evidence.
- Produces: an auto-discovered project-specific skill that reports status first and advances exactly one approved gate.

- [ ] **Step 1: Record RED baseline scenarios before creating the skill**

Use the current conversation and existing status request as raw baseline evidence: the agent previously reconstructed long context, reviewed multiple documents, and required repeated manual state explanation before identifying the next action. Record the observable failures in the implementation report without adding them to the skill.

- [ ] **Step 2: Initialize the skill**

Run `init_skill.py evaluating-droneai-models --path C:\Users\user\.codex\skills --resources references` with generated `display_name`, `short_description`, and `default_prompt` interface values.

- [ ] **Step 3: Write the minimal skill and reference**

Keep `SKILL.md` below 500 words. Require compact status first, one model/one gate, no subagent by default, identity-based evidence reuse, rights/action matching, no test access, and explicit stage reports. Put the exact required report fields and standard home5090 paths in `references/status-contract.md`.

- [ ] **Step 4: Validate and forward-test**

Run `quick_validate.py` on the skill directory. Then invoke the skill in a fresh minimal-context task and verify it identifies STEERER's 36-image benchmark as the next action without rescanning the repository or proposing training.

- [ ] **Step 5: Correct any discovered ambiguity and revalidate**

Only adjust wording needed by the observed forward test, then rerun validation.

### Task 5: Verify, synchronize, and run STEERER 36

**Files:**
- Modify: `configs/evaluation/model_queue.json` only after accepting new result evidence.
- Create on SSD only: `/workspace/data/results/steerer/<new-run>/...`

**Interfaces:**
- Consumes: clean committed DroneAI branch, verified STEERER venv, rights decision, dataset/split/upstream/checkpoint identities.
- Produces: complete 36-sample evaluation, score, metrics, 12 panels, and updated compact status.

- [ ] **Step 1: Run local full tests and verify the worktree**

Run: `python -m pytest -q`, `git diff --check`, `git status --short`.

Expected: all tests pass and only intentional committed changes exist.

- [ ] **Step 2: Push and fast-forward home5090**

Push the current branch. On home5090, require a clean tree and use `git pull --ff-only`; do not reset or delete anything.

- [ ] **Step 3: Verify the 17-package environment and preflight identities**

Inside `crowd-jupyter`, run setup with `--verify-only`, the focused STEERER tests, the compact status command, and SHA-256 checks for checkpoint and accepted one-sample evidence.

- [ ] **Step 4: Run the frozen 36-sample STEERER evaluation**

Use a new timestamped directory under `/workspace/data/results/steerer`. Invoke `scripts/run_steerer_ucf_qnrf_smoke.py` with the pinned STEERER upstream, separately pinned DM-Count split upstream, accepted UCF-QNRF Train root, official train/validation lists, verified checkpoint, and Stage 3C decision/manifest. Do not pass any test path.

- [ ] **Step 5: Verify outputs and update status evidence**

Require 36 prediction rows, zero unrecorded samples, 12 panels, score/metrics/evidence manifests, a bundle below 25 MiB, and all internal artifact hashes valid. Add only the accepted score/metrics references to the queue and commit.

- [ ] **Step 6: Report the roadmap and proceed to PET onboarding**

Use the installed skill's status output to report current model, dataset, metrics, score, rights scope, and next action. Start PET with architecture/official-source/checkpoint/rights discovery and a separate TDD adapter plan; do not fine-tune.
