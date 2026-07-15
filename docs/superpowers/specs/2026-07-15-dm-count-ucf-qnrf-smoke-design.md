# DM-Count / UCF-QNRF Frozen-Checkpoint Smoke Design

**Date:** 2026-07-15
**Status:** Approved direction
**Parent design:** `2026-07-15-multi-model-evaluation-harness-design.md`

## Objective

Add the first real-model adapter to the common evaluation harness and run one
short, test-isolated DM-Count smoke on the RTX 5090. This checkpoint validates
the model structure, official weight compatibility, dataset adapter, native
density output, count derivation, zone aggregation, runtime evidence, review
panels, and rights-scoped gate. It does not train or fine-tune the model and it
does not open the official UCF-QNRF test split.

## Rights decision and preserved risk note

The project owner explicitly accepts the Apache-2.0 label displayed by the
Kaggle UCF-QNRF mirror as permission for project use. The dataset component is
therefore recorded as verified for research and commercial R&D evaluation.
The official UCF page remains the provenance source; the Kaggle page is the
recorded license-evidence source.

The manifest must also preserve that the Kaggle uploader says they are not the
dataset owner. This is a visible provenance risk note, not an automated blocker
under the owner's approved policy. The harness provides an engineering record,
not legal advice.

DM-Count code is pinned to upstream commit
`cc5f2132e0d1328909f31b6d665b8e0b15c30467` and MIT. The UCF-QNRF checkpoint
tracked in that repository is recorded separately from the code, with the MIT
repository license as evidence and an explicit note that this scope
interpretation covers the tracked checkpoint.

Stage 3C remains component and action scoped:

- verified code + dataset + pretrained weights authorizes asset download and
  frozen-checkpoint evaluation;
- missing derived-weight or deployment approval keeps the overall bundle at
  `PASS_COMMERCIAL_CANDIDATE`;
- commercial training, derived-weight use, weight reuse in a product, and
  deployment remain blocked until their own components are verified;
- a technical score never upgrades the rights decision.

## Model and adapter boundary

The official upstream repository remains outside Git at
`/workspace/upstreams/DM-Count`. The adapter verifies the pinned upstream
commit and a clean upstream working tree before importing reviewed code paths.
Large checkpoints stay under
`/workspace/data/checkpoints/dm-count`; datasets stay under
`/workspace/data/datasets/ucf-qnrf`; native outputs stay under
`/workspace/data/results/dm-count`.

The adapter exposes the existing common `ModelAdapter` contract:

- input: RGB image, original dimensions, source hash, frozen split identity,
  ground-truth point annotations, condition tags, and four quadrant zones;
- preprocessing: ImageNet normalization and the official QNRF maximum-size
  policy, with every resize transform recorded;
- network: VGG-19 feature extractor, four pooling stages, bilinear 2x
  upsampling, 512-256-128 regression head, and non-negative 1-channel density;
- output: a density map at approximately one eighth of the processed input
  resolution;
- count: sum of predicted density mass;
- zone count: count-preserving density integration over transformed zone
  bounds;
- runtime: synchronized CUDA latency and peak allocated VRAM;
- failure: an explicit failure record, never a silent zero count.

Model construction must not download ImageNet weights. The official QNRF
checkpoint is the only initialization artifact for inference. Compatibility
code must live in the DroneAI adapter or isolated model environment; the
upstream checkout is not edited.

## Dataset and split isolation

The official UCF-QNRF identities remain 1,201 train images and 334 test images.
The archive and every selected source image are SHA-256 recorded.

The first run uses only the validation subset fixed by the pinned official
DM-Count repository. That repository assigns 1,081 source-train images to
training and 120 to validation:

1. parse the training images and dot annotations without touching the test
   directory;
2. verify the complete, disjoint `preprocess/qnrf_train.txt` and
   `preprocess/qnrf_val.txt` lists at the pinned upstream commit, including
   their frozen SHA-256 values;
3. select a deterministic 36-image smoke subset from the official 120-image
   validation list, stratified by
   frozen density bands when the available data permits;
4. write the complete split manifest before inference;
5. assert that no test path was enumerated or opened.

Using the upstream split prevents the official checkpoint from being evaluated
on images that it saw during training. Thirty-six samples provide enough rows
for low/medium/high-density evidence and
allow the existing deterministic curation logic to select exactly 12 review
panels. The later full validation and sealed 334-image test runs require a new
protocol freeze and explicit approval.

## Metrics and visual evidence

The smoke reports, without claiming paper reproduction:

- count MAE, RMSE, signed bias, normalized error, and coverage;
- density-mass consistency;
- quadrant zone/GAME-style absolute error from point annotations;
- bias by frozen density band;
- per-image latency, throughput, peak VRAM, input/output resolution, and
  preprocessing transform;
- upstream, checkpoint, dataset, split, environment, config, and result hashes.

The Git review bundle contains the model brief, rights decision, frozen config,
split/sample manifest, predictions, summary, 100-point score, selection
manifest, and exactly 12 four-column panels. Large native density arrays,
checkpoint, dataset, and full logs remain on the 5090 SSD. The Git bundle must
remain below 25 MiB.

The 12 panels are selected deterministically from best, representative, high
density, and worst/failure cases. Each density-model panel shows source image,
ground-truth spatial evidence, predicted density heatmap, and error/zone
summary.

## Execution and review gates

The sequence is fixed:

1. unit tests and CPU fake-adapter integration;
2. home5090 CUDA model-construction smoke with no dataset access;
3. checkpoint hash verification and one approved sample inference;
4. 36-image non-test smoke and evidence bundle;
5. user review of structure, code paths, metrics, panels, rights status, and
   100-point score;
6. only after approval, freeze thresholds and run full validation or sealed
   test evaluation.

No training, fine-tuning, official test access, or next-model adapter begins in
this checkpoint.

## Acceptance criteria

- Existing tests remain green.
- Rights classification permits only the verified frozen-evaluation action and
  keeps training/deployment pending.
- The persisted rights decision is bound to the expected candidate manifest
  semantic hash and component IDs.
- Dataset indexing proves zero test access and verifies the pinned 1,081/120
  official train/validation split.
- The adapter returns finite non-negative count-preserving density output or an
  explicit failure.
- CUDA smoke records the RTX 5090 environment and hashes.
- The 36-image run produces complete metrics, a scoped 100-point result, and
  exactly 12 panels under the Git budget.
- Re-running the frozen selection reproduces the same sample IDs and hashes.
