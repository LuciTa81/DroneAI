# Round 2 reference benchmark design

## Goal

Evaluate the three commercial-path technical candidates—STEERER, DM-Count, and
MPCount—on the same 1,000 labeled images per model. This round is a frozen-model
reference comparison only. It does not train, fine-tune, adapt, calibrate model
weights, or authorize production use.

The round answers four practical questions:

1. How stable are count and spatial outputs outside the 36-image Round 1 pilot?
2. How do the models behave across dense still imagery, varied fixed-camera
   scenes, and UAV imagery?
3. What runtime, VRAM, and failure patterns should the fixed-CCTV prototype plan
   around?
4. Which evidence is research-only and which evidence may remain in the
   commercial-candidate lane?

## Frozen scope

### Models

| Order | Model | Family | Native output | Count method | Round 2 rights role |
|---|---|---|---|---|---|
| 1 | STEERER | hybrid density and points | density hierarchy and point cues | integrate final density | commercial-path candidate, checkpoint terms still pending |
| 2 | DM-Count | density regression | density map | density-map sum | commercial-path candidate |
| 3 | MPCount | density regression | density/count output | model density/count aggregation | commercial-path candidate |

All three use the exact accepted Round 1 code commit, upstream commit,
checkpoint SHA-256, model environment, and preprocessing unless a preflight
proves that a dataset adapter requires a documented input-only change. No
optimizer, backward pass, gradient update, checkpoint save, or learned
calibration is permitted.

### Dataset matrix

Each model processes exactly the same 1,000 sample identities:

| Dataset partition | Samples | Technical purpose | Rights treatment |
|---|---:|---|---|
| UCF-QNRF official Test | 334 | primary dense-crowd count and spatial evaluation | `PASS_COMMERCIAL_CANDIDATE` under the project owner's accepted Kaggle Apache-2.0 listing, with provenance caveat retained |
| JHU-CROWD++ official Validation | 500 | varied illumination, weather, distractors, and unconstrained fixed/static scenes | `PASS_RESEARCH_ONLY`; official terms prohibit commercial use |
| UP-COUNT official validation/test sequences | 166 | high-resolution UAV and temporal/domain-shift stress test | `PASS_RESEARCH_ONLY`; CC BY-NC 4.0 |
| **Total per model** | **1,000** | frozen reference benchmark | the combined round is `PASS_RESEARCH_ONLY` |

The existing 36-image UCF-QNRF validation pilot remains historical evidence and
is not part of the new 1,000 images.

## Split and sampling contract

The split manifest is immutable once inference starts.

- UCF-QNRF uses every one of the 334 official Test images, sorted by normalized
  official sample ID. No Train image is admitted.
- JHU-CROWD++ uses every one of the 500 official Validation images, sorted by
  normalized official sample ID. No Train or Test image is admitted.
- UP-COUNT draws 166 frames only from the official validation and test sequence
  lists. Candidate frames are grouped by sequence. Quotas are assigned
  proportionally by eligible frame count using largest-remainder allocation,
  with at least one frame per eligible sequence while capacity allows. Within a
  sequence, frames are ordered by SHA-256 of
  `round2-upcount-v1:{sequence_id}:{frame_id}` and accepted only when they are at
  least 30 source frames from an already selected frame in that sequence. If the
  gap rule leaves a shortfall, the remaining candidates are accepted in the same
  hash order and the relaxation is recorded per frame.

Before inference, the harness writes and hashes:

- dataset source URL and captured license text/hash;
- official partition file SHA-256;
- every sample ID, source-image SHA-256, label SHA-256, sequence ID, and frame ID;
- selection algorithm/version and the fixed selection namespace;
- dataset-level Merkle-style aggregate hash and full manifest SHA-256.

The same manifest SHA-256 must be consumed by all three model runs. A missing,
changed, duplicate, out-of-partition, or hash-mismatched sample blocks the run.

## Rights and evidence separation

Not training on a restricted dataset does not change its license. The harness
therefore enforces rights at both dataset and report level.

- JHU-CROWD++ and UP-COUNT evidence is internal, non-production technical
  research only. It cannot authorize training, fine-tuning, learned calibration,
  derived commercial weights, deployment, or `PRODUCTION_APPROVED` status.
- UCF-QNRF metrics remain separately addressable and may retain
  `PASS_COMMERCIAL_CANDIDATE`; they are never promoted to production approval
  without independent provenance and deployment review.
- Cross-dataset totals may be shown descriptively, but the harness does not
  calculate a single commercial score from mixed-rights data.
- The combined 1,000-image run and any overall technical comparison are labeled
  `PASS_RESEARCH_ONLY`.
- Dataset terms, source-image rights, code rights, checkpoint rights, and result
  rights remain separate fields in every manifest and report.

Restricted source images, copied annotations, and image-derived review panels
stay under `/workspace/data` by default. Git receives only code, configs, compact
manifests, aggregate metrics, scores, and a small set of panels whose sharing
rights have been separately cleared. A company Drive package for JHU-CROWD++ or
UP-COUNT contains aggregate metrics and license notices by default, not source or
derived images. Broader image sharing requires written permission or a separate
rights decision.

## Evaluation outputs

For every model × dataset run, retain on the SSD:

- one row per sample with GT count, predicted count, absolute error, signed
  error, percentage-error reference value, latency, and output validity checks;
- density/GAME/zone metrics when a valid density-like output exists;
- localization precision, recall, F1, and distance only when the native model
  output and ground truth support the metric without converting a density map
  into invented points;
- model-native output metadata, finite/nonnegative checks, count-sum checks, and
  input/output dimensions;
- median and percentile latency, FPS, and peak VRAM measured after warm-up;
- a compressed per-sample visualization for technical review, retained on SSD
  and governed by the dataset's sharing scope;
- exactly 12 hash-bound curated panels per model × dataset for local review:
  best, strong, representative, high-density, and failure cases without sample
  duplication where possible;
- metrics, score, selection manifest, environment manifest, rights decision,
  evidence manifest, and human-readable summary.

MAE, RMSE, signed bias, and MAPE reference are reported separately for each
dataset. MAPE is never presented as the primary crowd-counting metric. No metric
from the three datasets is silently pooled.

## Execution gates

The order is fixed and stops on the first failed gate:

1. **Round 1 closeout:** re-verify accepted evidence and preserve the existing
   `review_and_advance` comparison as historical provenance.
2. **Dataset rights:** capture official terms and record the declared action
   `frozen_internal_reference_evaluation` for all three datasets.
3. **Dataset acquisition:** download only to dedicated versioned directories
   under `/workspace/data/datasets`; verify file counts, partitions, and hashes.
4. **Manifest freeze:** build the exact shared 1,000-sample manifest before any
   model sees a new image.
5. **Adapter preflight:** verify the existing isolated model environments, CUDA,
   checkpoint identities, and read-only dataset adapters without inference.
6. **One-sample matrix:** run one declared sample from each dataset through
   STEERER, inspect native output and metrics, then repeat for DM-Count and
   MPCount.
7. **Full model run:** after the one-sample evidence passes, execute all 1,000
   samples for one model, verify artifacts and score it, and report before moving
   to the next model.
8. **Comparison:** create dataset-wise technical tables and an explicitly
   research-only combined interpretation. Do not fine-tune or select production
   weights in this round.

Long runs use the existing `crowd` tmux session or a resumable non-interactive
runner. Resume may append only missing sample IDs after rechecking the Git,
checkpoint, dataset, split, config, and environment hashes. Existing completed
rows are immutable.

## Failure handling

- A rights mismatch blocks only the affected dataset lane; it never silently
  falls back to another mirror or license.
- A download interruption resumes into the same versioned staging directory and
  is promoted atomically only after complete hash verification.
- OOM retries may use only a predeclared input tiling or scaling policy shared by
  all three models and recorded in the manifest. A model-specific silent resize
  is forbidden.
- Invalid native output, NaN/Inf, negative density outside tolerance, count-sum
  mismatch, missing labels, or duplicate IDs fails the affected sample and the
  model run stops for review.
- A reboot resumes from the last hash-verified sample; it does not recompute or
  overwrite accepted rows.

## Acceptance criteria

Round 2 is complete only when:

- all three models consume the identical 1,000-sample manifest;
- every model has 334 UCF-QNRF, 500 JHU-CROWD++, and 166 UP-COUNT accepted rows;
- train, validation, and test roles are explicit and no training occurs;
- each result is bound to Git, upstream, checkpoint, environment, dataset,
  partition, and sample-manifest SHA-256 values;
- metrics, runtime, VRAM, native-output interpretation, limitations, and rights
  scope are reported after each model;
- the combined comparison is labeled `PASS_RESEARCH_ONLY` and cannot emit a
  production approval;
- large datasets and full results remain on SSD, while Git additions stay below
  25 MiB and contain only cleared compact evidence;
- no fine-tuning, calibration training, commercial derived weights, or deployment
  action has occurred.

## Explicitly out of scope

- model training or fine-tuning;
- learned camera calibration or multi-CCTV merging;
- use of JHU-CROWD++ or UP-COUNT to create commercial weights;
- declaring any model or dataset production-approved;
- uploading restricted source or derived imagery to a broadly shared Drive;
- replacing the accepted Round 1 evidence chain.
