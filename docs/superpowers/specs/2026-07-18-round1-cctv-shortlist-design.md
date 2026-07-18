# Round 1 Fixed-CCTV Technical Shortlist Design

## Status

Approved in conversation on 2026-07-18 for design documentation. Implementation
and shortlist generation require a separate reviewed plan.

## Goal

Build a deterministic, evidence-gated comparison of the six completed Round 1
crowd-counting candidates for the festival's fixed-CCTV use case. The primary
technical objective is zone-level crowd density, spatial visualization, and risk
screening. Absolute count accuracy remains important but is not the only selection
criterion.

The comparison must keep technical merit independent from commercial-rights
status. A technically strong research-only model stays visible in the technical
shortlist, while a separate deployment gate prevents that result from being
mistaken for commercial approval.

## Non-goals

- Do not run new inference, training, fine-tuning, or test-split evaluation.
- Do not call the result a scientific model ranking or an official UCF-QNRF
  leaderboard result.
- Do not infer commercial permission from a paper, code license, Kaggle label, or
  `PASS_COMMERCIAL_CANDIDATE` status.
- Do not copy datasets, checkpoints, raw arrays, or full SSD runs into Git.
- Do not advance to drone-domain evaluation or select a fine-tuning winner in this
  change.

## Inputs and comparison scope

The source of truth is `configs/evaluation/model_queue.json` plus the accepted SSD
evidence under `/workspace/data/results`. The comparison covers exactly these six
models in queue order:

1. DM-Count
2. STEERER
3. PET
4. MPCount
5. APGCC
6. CSRNet

Every model must refer to the same dataset ID, frozen validation split ID, split
role, and expected sample count. The current required identity is:

- dataset: `ucf-qnrf-kaggle-apache`
- split: `dm-count-upstream-qnrf-val-smoke-36-v1`
- role: `validation`
- samples: 36, with 12 low-, 12 medium-, and 12 high-density images
- fine-tuning: false
- comparison scope: `compatibility_smoke`

The official checkpoints were not all trained under the same protocol. Some were
trained on ShanghaiTech Part A and some have unknown training membership. The
generated result must therefore set `ranking_eligible=false` and label its top
three as `PROVISIONAL_TECHNICAL_SHORTLIST`.

## Alternatives considered

### One universal raw-metric ranking

This is simple, but it incorrectly treats density-map and point-localization
outputs as if their family-specific spatial metrics were interchangeable.

### Two-axis fixed-CCTV scorecard

This is the selected design. It produces a 100-point technical CCTV-fit score and
an independent rights/deployment status. Family adapters map density and point
evidence into explicitly bounded operational categories.

### Pareto-only comparison

This is the most conservative method, but it does not provide the requested
shortlist. The output will still include raw metrics and category subscores so a
Pareto analysis can be performed later without rerunning models.

## Evidence validation

Comparison generation fails closed unless all of the following hold:

1. The queue and all accepted evidence records pass the existing SHA-256 checks.
2. Each model has an accepted benchmark `score` and `metrics` artifact.
3. The sibling `evidence-manifest.json` verifies `predictions.csv`,
   `selection_manifest.json`, the review bundle, and all 12 curated panels.
4. Each predictions file contains exactly 36 accounted records, no unrecorded
   sample, and finite required numeric fields.
5. The canonical tuples `(sample_id, ground_truth_count, density_band,
   source_partition)` are identical across all six runs. Their sorted canonical
   JSON representation is hashed as the comparison split fingerprint.
6. The split role is `validation`; any test role or test path is rejected.
7. No model is marked `ranking_eligible=true` unless the existing comparison-claim
   validator independently permits it. This Round 1 output requires all entries to
   remain non-ranking compatibility evidence.
8. The generated Git review bundle remains below 25 MiB.

Older accepted runs do not all aggregate the same optional fields in
`metrics.json`. The comparison may deterministically recompute an aggregate only
from the already verified `predictions.csv`:

- MAPE reference: mean of `normalized_error` multiplied by 100;
- zone MAE: mean of finite `zone_mae`;
- GAME, PSNR, SSIM, and localization fields: mean of their matching finite columns.

The output records whether each value was read from `metrics.json` or derived from
verified predictions. It never invents a missing value.

## Fixed technical scoring

Let `clamp(x) = min(1, max(0, x))`. All anchors below are frozen configuration,
not observed best/worst values, so adding or removing a model cannot change another
model's score.

### 1. Zone density and spatial quality: 35 points

- Zone MAE, 20 points:
  `20 * clamp(1 - mean_zone_mae / 150)`.
- Primary family spatial metric, 10 points:
  - density primary: `10 * clamp(1 - mean_GAME_L1 / 500)`;
  - point primary: `10 * clamp(mean_localization_F1 / 0.8)`;
  - hybrid primary: use the frozen `spatial_metric_name` and its corresponding
    formula; report all secondary spatial metrics without double-counting them.
- Density-band robustness, 5 points:
  `5 * clamp(1 - max_absolute_band_bias / 400)`.

PSNR, SSIM, localization precision/recall, and mean distance remain visible raw
metrics. They are not combined across incompatible output families.

### 2. Count and high-density stability: 25 points

- MAE, 10 points: `10 * clamp(1 - MAE / 250)`.
- RMSE, 7 points: `7 * clamp(1 - RMSE / 400)`.
- MAPE reference, 4 points: `4 * clamp(1 - MAPE_percent / 40)`.
- High-density absolute signed bias, 4 points:
  `4 * clamp(1 - abs(high_band_bias) / 400)`.

### 3. Runtime and resources: 15 points

Define `piecewise(value, full, zero)` as 1 when `value <= full`, 0 when
`value >= zero`, and linear interpolation between those bounds.

- Median batch-1 latency, 9 points:
  `9 * piecewise(latency_ms, 66.67, 500)`.
- Peak VRAM, 6 points:
  `6 * piecewise(peak_vram_mb, 8192, 24576)`.

FPS is reported as a raw metric and checked for consistency with latency; it is not
scored twice.

### 4. Operator output and integration evidence: 15 points

These are evidence-backed capability points, not subjective reviewer ratings:

- 6 points: finite zone aggregation for all 36 successful samples;
- 4 points: a verified direct density map suitable for heatmap rendering;
- 3 points: verified bounded person-point output with localization metrics;
- 2 points: verified per-output confidence that can be carried into the operator
  API.

An absent required artifact is `NOT_DEMONSTRATED` and receives zero for that
capability. A metric that does not apply to the declared family is `N/A`, not a
synthetic zero. A hybrid can receive both density-map and point-output capability
points only when both outputs were actually recorded and verified.

### 5. Reproducibility and evidence quality: 10 points

- 4 points: all 36 records accounted for with zero explicit inference failures;
- 4 points: score, metrics, predictions, selection, panels, environment, and
  evidence hashes verify;
- 2 points: checkpoint training split is `VERIFIED_DISJOINT`.

`UNKNOWN` checkpoint membership receives zero for the final two points and remains
prominently flagged. This is an evidence-quality deduction, not a license
deduction.

## Shortlist and tie-breaking

The total technical score is the sum of the five category scores, rounded only for
display. The three highest unrounded totals form the
`PROVISIONAL_TECHNICAL_SHORTLIST`.

Ties are resolved in this order:

1. lower mean zone MAE;
2. lower MAE;
3. lexical model ID.

The report must repeat that checkpoint heterogeneity makes the shortlist an
operational screening result, not a publishable model ranking.

## Independent rights and deployment axis

Rights never add or subtract technical points. Each row reports:

- code status;
- dataset status;
- pretrained-weight status;
- derived-weight status;
- deployment status;
- overall recorded scope.

The deployment label is derived without inference:

- `PRODUCT_ELIGIBLE` only for `PRODUCTION_APPROVED`;
- `PENDING_DUE_DILIGENCE` for `PASS_COMMERCIAL_CANDIDATE`;
- `RESEARCH_ONLY` for `PASS_RESEARCH_ONLY`;
- `BLOCKED` when an explicit restriction blocks the declared use.

A research-only model may appear in the technical top three, but the report must
state that it cannot be deployed with the evaluated assets. Independent
implementation and retraining on authorized data remain future due-diligence
options, not permissions granted by this scorecard.

## Components and data flow

Implementation will add four bounded components:

1. A frozen JSON scorecard configuration containing model IDs, anchors, weights,
   required evidence, and tie-break rules.
2. A comparison library that validates queue evidence, derives allowed aggregates,
   applies family adapters, and returns a serializable comparison record.
3. A CLI that reads the queue and SSD results and writes an atomic small-artifact
   bundle.
4. Focused tests for validation, scoring boundaries, family handling, rights
   independence, deterministic ordering, and fail-closed behavior.

Data flow:

`queue -> accepted hashes -> run evidence manifests -> verified predictions and metrics -> family adapters -> CCTV scorecard -> technical shortlist + independent rights gate`

## Output bundle

The generated directory will be `results/round1-cctv-comparison-<commit>/` and
contain only small artifacts:

- `comparison.json`: complete machine-readable inputs, subscores, raw metrics,
  shortlist, rights axis, limitations, and hashes;
- `comparison.csv`: one row per model for spreadsheet review;
- `comparison.md`: human-readable report with links to existing representative
  success and failure panels;
- `evidence-manifest.json`: SHA-256 values for every comparison input and output;
- `scorecard-config.snapshot.json`: exact frozen scoring configuration.

Existing curated panels are linked rather than duplicated. Full model runs remain
on `/workspace/data/results`.

## Error handling

The CLI exits nonzero and writes no final bundle when it encounters a missing
artifact, hash mismatch, mixed split, sample mismatch, test role, non-finite
required value, unknown model family, score-weight mismatch, or size-budget
violation. Output is written to a temporary sibling directory and renamed only
after all checks pass.

Optional family metrics are explicitly marked `N/A`; required operational
capabilities without evidence are marked `NOT_DEMONSTRATED`. Neither case is
silently imputed.

## Test strategy

Tests will cover:

- the six-model happy path using small fixture bundles;
- changed accepted-evidence and nested evidence-manifest hashes;
- different sample IDs, ground truth, bands, or source partitions;
- accidental test split and any test path;
- exact score boundaries and piecewise anchors;
- density, point, and hybrid family adapters;
- missing optional metrics versus missing required fields;
- technical scores remaining unchanged when only rights status changes;
- rights labels never becoming `PRODUCT_ELIGIBLE` without
  `PRODUCTION_APPROVED`;
- deterministic top-three ordering and tie-breaks;
- atomic output and the 25 MiB Git budget.

After focused tests pass, the full repository suite must pass before commit. The
branch will be pushed, and home5090 will update only with `git pull --ff-only`.

## Workflow boundary

Round 1 stays at `review_and_advance` while this comparison is designed and built.
After the generated comparison is reviewed, a separate user approval selects the
next round. Fine-tuning remains disabled. The likely next round is fixed-CCTV
calibration and domain validation, followed later by drone-domain evaluation where
relevant and fine-tuning of at most one rights-eligible winner.
