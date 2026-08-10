# STEERER UCF-QNRF Official-Code Reproduction A Implementation Plan

> Execute one gate at a time. A failed gate blocks every later gate. Existing B-lane
> files, checkpoints, results, and sealed-Test controls remain immutable.

**Goal:** Reproduce the pinned public STEERER UCF-QNRF implementation on the full
official Train 1,201 / Test 334 protocol, beginning with an official-checkpoint
evaluator compatibility gate and ending with a fresh 800-epoch ImageNet-initialized
run only after all short gates pass.

**Authority:** Upstream commit
`5b1854dbc2d280f2326d67c65515d8baf9083810`, especially
`configs/QNRF_final.py`, is authoritative for executable settings. The paper's
74.3/128.3 result remains a comparison target, but the lane is named
`official-code reproduction` because the paper prose says Adam while the released
configuration says AdamW.

**Isolation:** Store processed data, checkpoints, and results under
`steerer-official-code-reproduction` roots. Never add a protocol switch to the
B runner and never initialize training from the official full STEERER checkpoint.

---

## Task 1: Freeze the A protocol and G1 compatibility contract

**Files:**
- Create: `src/droneai/steerer_official_reproduction.py`
- Create: `configs/evaluation/steerer_a_official_checkpoint_test334.json`
- Create: `tests/test_steerer_official_reproduction.py`

1. Write failing tests requiring Train 1,201, Test 334, explicit Test access,
   `official Test, test-selected` labeling, upstream commit pinning, official
   checkpoint SHA-256, research-only checkpoint scope, and isolated A roots.
2. Add immutable dataclasses and strict JSON validation. Reject alternate sample
   counts, generic validation labels, commercial use of the official checkpoint,
   missing checkpoint identity, and any B-lane path.
3. Add `build_g1_protocol()` using `split_role=test`, exact 334 samples, and the
   existing evaluation contract without weakening its sealed-Test guard.
4. Run `python -m pytest tests/test_steerer_official_reproduction.py -q`.

## Task 2: Implement the dedicated G1 runner

**Files:**
- Create: `scripts/run_steerer_a_g1_official_checkpoint.py`
- Modify: `tests/test_steerer_official_reproduction.py`

1. Write failing CLI tests proving the runner accepts only the frozen profile,
   official Test root, pinned upstream, checkpoint plus artifact manifest,
   rights decision, output directory, and CUDA device. It must expose no train,
   resume, epoch, optimizer, or arbitrary checkpoint-origin options.
2. Implement pre-inference verification of upstream commit, checkpoint filename,
   size, SHA-256, artifact ID, allowed scope, and 334 unique Test records.
3. Evaluate with `STEERERAdapter(checkpoint_origin="research_checkpoint")` and
   write a G1 provenance file before invoking the generic evaluation runner.
4. After evaluation, write `compatibility-gate.json`. Pass only when output is
   complete and absolute differences from 77.8/138.0 are at most 5.0 MAE and
   10.0 RMSE. Never widen these bounds after observing results.
5. Run the focused tests, then the B regression tests.

## Task 3: Run and publish G1 on home5090

**Files:**
- Result: `/workspace/data/results/steerer-official-code-reproduction/official-checkpoint-test334-v1`
- Modify: `docs/models/STEERER.md`
- Add only compact JSON/Markdown evidence under `results/steerer/` after success.

1. Push the A branch, pull it without force on home5090, verify the container and
   CUDA matmul, and verify raw Test count and the official checkpoint hash.
2. Run G1 non-interactively inside `crowd-jupyter`; retain all large figures and
   prediction CSV files on the SSD.
3. Verify `metrics.json`, `score.json`, evidence hashes, 334/334 accounting, and
   `compatibility-gate.json` with a fresh process.
4. If G1 misses its frozen tolerance, stop A and audit preprocessing/evaluator.
   If it passes, commit only compact evidence and report the G2 approval point.

## Task 4: Prepare the full official Train/Test data lane

**Files:**
- Create: `src/droneai/steerer_official_data.py`
- Create: `scripts/prepare_steerer_a_data.py`
- Create: `tests/test_steerer_official_data.py`
- Create: `configs/training/steerer_ucf_qnrf_official_code_reproduction.home5090.json`

1. Test first that preparation requires exactly 1,201 Train and 334 Test pairs,
   rejects overlap/duplicates/missing annotations, refuses a non-empty output,
   and writes content-addressed inventories plus `train.txt` and `test.txt`.
2. Convert into the isolated processed root and verify all hashes in a fresh
   process. The Test list must be named and labeled Test, never validation.

## Task 5: Add short training gates without weakening B

**Files:**
- Create: `src/droneai/steerer_official_training.py`
- Create: `scripts/run_steerer_a_training.py`
- Create: `tests/test_steerer_official_training.py`

1. Add a separate A training profile and engine. Reuse only low-level runtime
   primitives whose semantics do not encode B's 961/240 validation split.
2. Test and run A0: CUDA import/matmul, one optimizer update, finite loss,
   checkpoint round-trip, RNG/optimizer/scheduler reload, and manifest hashes.
3. Test and run A1: one full Train epoch and one complete 334-image official-Test
   evaluation using the pinned upstream cadence and explicit test-selected label.
4. Record AdamW 1e-4, betas 0.9/0.999, weight decay 1e-4, epsilon 1e-8,
   warm-up 10, cosine minimum 1e-7, crop 768, route size 256, scale 0.5-2.0,
   flip, density factor 100, sigma 4, kernel 15, effective batch 8, seed, and
   ImageNet backbone identity in every status manifest.

## Task 6: Run 800 epochs and compare

**Files:**
- Create compact result bundle under `results/steerer/`.
- Modify: `docs/models/STEERER.md`

1. Start a fresh FP32 run in tmux from the pinned ImageNet HRNet-W48 backbone.
   Save atomic checkpoints and resumable state only at completed boundaries.
2. Follow the pinned public `val_span` cadence. Because official Test selects
   best-MAE/best-RMSE, label all selected results `official Test, test-selected`.
3. Reload best-MAE, best-RMSE, and epoch-800 checkpoints and independently
   evaluate all 334 samples.
4. Compare against paper 74.3/128.3, official repository 77.8/138.0, and B epoch
   625 83.0872/143.1877. Do not call non-identical localization protocols directly
   comparable.
5. Score experimental integrity separately from model performance and leave the
   outcome `PASS_COMMERCIAL_CANDIDATE`, never `PRODUCTION_APPROVED`, until rights
   and field validation reviews pass.
