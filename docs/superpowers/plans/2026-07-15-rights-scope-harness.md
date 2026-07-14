# Rights-scope harness implementation plan

> Execute with the Superpowers test-driven-development, systematic-debugging,
> and verification-before-completion workflows.

**Goal:** Close known harness bypasses and add an auditable Stage 3C rights gate
that keeps silent-license assets as candidates without treating them as usable
commercial assets.

**Architecture:** Extend the shared scorer with validated scoped success states.
Keep technical readiness checks in their existing stages, add a pure rights
classifier plus Stage 3C report writer, and represent each code/data/weight/
deployment right as a separately evidenced component.

**Tech stack:** Python 3.10+, dataclasses, JSON, pytest, GitHub Actions, Docker
home5090 runtime.

---

### Task 1: Lock scoped scorer behavior

**Files:**
- Modify: `src/droneai/scoring.py`
- Modify: `tests/test_scoring.py`

1. Add failing tests for accepted scoped success statuses and rejected unknown
   statuses.
2. Add a `success_status` input with a safe allowlist and a schema-v2
   `decision_scope` serialization field.
3. Run `pytest -q tests/test_scoring.py`.

### Task 2: Close Stage 1 commercial-use contradiction

**Files:**
- Modify: `src/droneai/stage1.py`
- Modify: `tests/test_stage1.py`

1. Add a failing commercial manifest test where allowed uses claim commercial
   R&D but `commercial_use=false`.
2. Add explicit rights-consistency and scoped-result behavior.
3. Verify research fixtures emit `PASS_RESEARCH_ONLY` and contradictions block.

### Task 3: Close Stage 3 and 3B integrity bypasses

**Files:**
- Modify: `src/droneai/stage3.py`
- Modify: `src/droneai/stage3b.py`
- Modify: `tests/test_stage3.py`
- Modify: `tests/test_stage3b.py`

1. Add failing adversarial tests for short/non-hex hashes and omitted
   traceability/review bundles.
2. Require canonical SHA-256 values and make required artifact bundles blockers.
3. Return `PASS_RESEARCH_ONLY` from the Stage 3/3B runners.

### Task 4: Implement Stage 3C rights gate

**Files:**
- Create: `src/droneai/stage3c.py`
- Create: `tests/test_stage3c.py`
- Create: `configs/candidates/dm_count_shanghaitech.candidate.json`

1. Write failing tests for explicit non-commercial, silent candidate, fully
   verified production, explicit prohibition, and inconsistent metadata.
2. Implement manifest validation, deterministic classification, allowed-action
   output, scored report generation, and evidence snapshot persistence.
3. Confirm candidate actions exclude asset download, training, weight reuse, and
   deployment.

### Task 5: Align configuration and documentation

**Files:**
- Modify: `configs/datasets/dronecrowd.pending.json`
- Modify: `configs/datasets/local_festival.commercial.template.json`
- Modify: `README.md`
- Modify: `docs/ROADMAP.md`
- Create: `docs/STAGE_3C.md`

1. Represent missing terms as `commercial_use=null` plus a recorded basis,
   without claiming commercial permission.
2. Document the four states, action boundaries, user review artifacts, and
   legal-review caveat.
3. Keep the existing Colab/Drive provenance unchanged.

### Task 6: Verify locally, publish, and verify home5090

1. Run focused adversarial tests and `pytest -q` locally.
2. Inspect the patch for secrets, oversized artifacts, and unrelated edits.
3. Commit and push intentionally to the existing branch/PR.
4. Confirm GitHub checks pass.
5. Safely pull on home5090 only when its worktree is clean; run the full harness
   tests and Stage 0 CUDA smoke in the existing container.
6. Record exact commit hashes and results. Do not download a dataset or start
   long training.

### Task 7: Reboot handoff

1. Report what survives a notebook/server reboot and what Docker's restart
   policy restores.
2. Give the user a minimal shutdown/restart verification sequence.
3. Stop before any model/data stage and wait for the user's reboot confirmation.
