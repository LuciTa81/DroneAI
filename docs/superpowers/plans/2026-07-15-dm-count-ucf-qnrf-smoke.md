# DM-Count / UCF-QNRF Smoke Implementation Plan

> Follow the approved focused design and stop after the 36-image non-test smoke
> review bundle. Do not train, fine-tune, or access the official test split.

**Goal:** Add a rights-scoped official DM-Count adapter and produce a
reproducible UCF-QNRF validation smoke on `home5090_docker`.

**Architecture:** Keep upstream code, dataset, checkpoint, and large outputs on
the 5090 SSD. Add small configuration and adapter modules to DroneAI, reuse the
common evaluation contracts/runner, and extend Stage 3C only enough to authorize
frozen evaluation when code, dataset, and pretrained-weight evidence are all
verified.

**Runtime:** Python 3.12, PyTorch/Torchvision inherited from
`nvcr.io/nvidia/pytorch:25.10-py3`, NumPy, Pillow, SciPy, scikit-image, pytest.

## Task 1: Freeze rights evidence and action semantics

**Files:**

- Modify: `src/droneai/stage3c.py`
- Modify: `tests/test_stage3c.py`
- Create: `configs/datasets/ucf_qnrf.kaggle_apache.json`
- Create: `configs/candidates/dm_count_ucf_qnrf.candidate.json`
- Modify: `docs/STAGE_3C.md`

1. Write failing tests proving that verified code, dataset, and pretrained
   weights add `asset_download` and `frozen_checkpoint_evaluation` while
   pending derived weights/deployment keep commercial training and deployment
   absent.
2. Implement the smallest component-aware action calculation.
3. Record Apache-2.0 dataset evidence, MIT code/checkpoint evidence, the
   uploader-ownership risk note, and pending downstream components.
4. Run `python -m pytest -q tests/test_stage3c.py`.

## Task 2: Add a test-isolated UCF-QNRF index and split builder

**Files:**

- Create: `src/droneai/ucf_qnrf.py`
- Create: `tests/test_ucf_qnrf.py`

1. Build a miniature filesystem fixture matching the accepted archive layout.
2. Write failing tests for annotation parsing, point/count agreement, source
   hashes, the complete/disjoint pinned upstream train and validation lists,
   36-sample stratified selection, and refusal to enumerate any test path.
3. Implement layout discovery through explicit train/annotation roots only.
4. Add count-preserving max-size resize and transformed point coordinates.
5. Run `python -m pytest -q tests/test_ucf_qnrf.py`.

## Task 3: Add the official DM-Count density adapter

**Files:**

- Create: `src/droneai/dm_count_adapter.py`
- Create: `tests/test_dm_count_adapter.py`
- Modify: `configs/models/dm_count.official.json`

1. Write fake-Torch/upstream tests for pinned-commit verification, checkpoint
   SHA-256 verification, no-network construction, ImageNet normalization,
   density mass/count agreement, zone integration, explicit failure behavior,
   and model brief fields.
2. Implement dynamic import from an explicit upstream directory without editing
   upstream files.
3. Load the official state dict on an explicit device and prevent pretrained
   backbone download.
4. Synchronize CUDA around latency measurement and record peak allocated VRAM.
5. Run `python -m pytest -q tests/test_dm_count_adapter.py`.

## Task 4: Connect DM-Count to the common runner

**Files:**

- Create: `configs/evaluation/dm_count_ucf_qnrf_smoke.json`
- Create: `scripts/run_dm_count_ucf_qnrf_smoke.py`
- Create: `tests/test_run_dm_count_ucf_qnrf_smoke.py`

1. Write failing CLI/config tests for explicit upstream, dataset, checkpoint,
   output and rights paths; fixed seed; 36 samples; validation role; and sealed
   test approval false.
2. Construct common `EvaluationSample` objects with point annotations and four
   zones, run the adapter through `run_evaluation`, and persist preprocessing
   evidence.
3. Refuse non-empty output directories and any dataset root that overlaps the
   configured test root.
4. Run the focused CLI tests and a CPU fake-adapter end-to-end test.

## Task 5: Verify the complete local suite

1. Run `python -m pytest -q`.
2. Run the existing CPU fixture into a new temporary directory.
3. Confirm `git diff --check`, no large files, and no dataset/checkpoint paths
   are staged.
4. Commit the reviewed implementation on
   `agent/dm-count-ucf-qnrf-adapter`.

## Task 6: Prepare and verify home5090 without long execution

1. Push the branch and fast-forward/checkout it under
   `/home/lucita/crowd-counting-lab` without reset or deletion.
2. Clone the official DM-Count repository at the pinned commit under
   `/workspace/upstreams/DM-Count` and record its commit.
3. Install only the small adapter overlay in an isolated environment; do not
   replace the NGC image's Torch/CUDA stack.
4. Run the full unit suite and CUDA synthetic fixture.
5. Run model construction with no dataset access.

## Task 7: Acquire verified artifacts and run the non-test smoke

1. Download the accepted Kaggle UCF-QNRF archive and official repository
   checkpoint to new paths under `/workspace/data`; never overwrite an existing
   artifact.
2. Record archive/checkpoint byte size and SHA-256 before extraction/use.
3. Build and freeze the pinned 1,081/120 upstream train/validation split and
   36-image validation smoke manifest while proving zero test-path access.
4. Run one sample first; inspect finite density mass, count, latency, VRAM, and
   a panel.
5. Run the 36-image smoke, then copy only the small review bundle into Git.
6. Re-run focused integrity checks and the 100-point gate.
7. Stop and report results for user approval. Do not run the full validation,
   official test, training, fine-tuning, or the next model.
