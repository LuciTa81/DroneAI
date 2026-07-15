# Multi-model crowd-counting evaluation harness design

Date: 2026-07-15

## Goal

Extend the existing DroneAI repository into an end-to-end, scored evaluation
harness for crowd-counting models. Every model and dataset evaluation must
produce comparable numeric evidence, model-native spatial output, a small and
deterministically selected visual review bundle, runtime and provenance
evidence, and a rights-scope decision before the next experiment begins.

The harness must support research reproduction, drone generalization, and
event-control use without treating a leaderboard number as a production
approval. The existing Git and Colab lineage remain part of the audit history;
new execution defaults to the home RTX 5090 backend.

## Non-goals

- Do not replace or rewrite existing Colab, ShanghaiTech, DM-Count, or Stage 0
  through Stage 3C history.
- Do not store datasets, checkpoints, raw video, or every rendered prediction
  in Git.
- Do not rank models trained or evaluated under different protocols as if they
  were directly comparable.
- Do not download, train with, or reuse an ambiguous asset merely because it is
  recorded as a commercial candidate.
- Do not start long training until the model adapter, data gate, rights scope,
  minimal CUDA smoke, checkpoint, and resume policy have passed review.

## Considered artifact approaches

### 1. Commit every prediction visualization

Rejected. It gives a complete visual archive but makes Git grow with dataset
size, input resolution, model count, and reruns. Full-resolution drone outputs
would quickly dominate the source repository.

### 2. Let a reviewer manually choose attractive examples

Rejected. It is small, but it creates selection bias and cannot be reproduced
from the recorded predictions.

### 3. Compute all metrics and deterministically render a curated subset

Selected. Inference computes per-sample metrics for the complete frozen test
split without retaining every rendered panel. A deterministic selector chooses
representative sample IDs from the metrics table. Only those samples are
rendered again for Git review. The selector configuration and source table hash
make the visual bundle reproducible.

## Preserve and extend the existing repository

The current repository remains authoritative for source, configuration,
lineage, reports, and review decisions. Existing assets already provide:

- staged 100-point scoring with blockers;
- Colab and `home5090_docker` runtime profiles;
- count-preserving CSRNet pipeline checks;
- DM-Count official-checkpoint and pilot evidence;
- exact artifact and split SHA-256 verification;
- research-only, commercial-candidate, production-approved, and blocked rights
  states.

The new harness adds common model adapters, common result schemas, family-aware
metrics, deterministic visual curation, and repeated model-by-model review. It
does not create a replacement repository or discard earlier evidence.

## Evaluation sequence

The harness uses one common dataset within a comparison stage and changes the
dataset only when the validation question changes.

### Round 1: common static benchmark

Use UCF-QNRF as the common labeled still-image benchmark for official-weight
screening. The initial technical set is:

1. DM-Count, density-map baseline with an eligible official UCF-QNRF
   checkpoint;
2. STEERER, scale-aware counting and localization candidate with an eligible
   official UCF-QNRF checkpoint;
3. PET, point/localization candidate with an official UCF-QNRF checkpoint but
   research-only licensing risk that must remain visible;
4. CSRNet, legacy density baseline.

CSRNet remains in the comparison, but its original work does not provide an
official UCF-QNRF checkpoint. The official-weight screening must therefore
record `NOT_EVALUATED_NO_ELIGIBLE_WEIGHT` rather than silently substitute a
third-party weight. A separately approved cross-domain smoke or later
project-trained UCF-QNRF checkpoint is a distinct run and cannot enter the
official-weight ranking as an equivalent artifact.

P2PNet is a lineage reference unless a unique comparison need remains after
PET and APGCC. APGCC is a second-wave point model because an eligible official
UCF-QNRF checkpoint has not been established. MPCount is evaluated in the
domain-generalization lane rather than used as a same-domain leaderboard entry.

### Round 2: drone generalization

Advance only the best two or three technically useful Round 1 models to
UP-COUNT. Add Pixel Distill/Dot as a UAV-native localization candidate when its
code, weight, and permitted research actions have passed Stage 3C review.

Evaluate altitude, density, tiny-person localization, high-resolution input,
tiling or resizing policy, latency, VRAM, and zone aggregation. UP-COUNT remains
a non-commercial research benchmark and cannot approve production weights.

### Round 3: optional drone stress sets

Use additional datasets only for a defined unresolved question and only after
their permitted action is verified:

- DroneCrowd for drone video/counting if its terms are established;
- DLR-ACD for optional non-commercial extreme-density stress;
- MovingDroneCrowd for the later moving-drone temporal track.

Not every model is run on every optional dataset. A model advances only when
the previous review identifies a concrete benefit that the next dataset can
test.

### Round 4: project-owned festival evidence

Final commercial candidates are fine-tuned and evaluated only on project-owned
or separately granted festival data. The sealed test split is separated by
camera, date, zone, event, or flight session, never by adjacent frames. This
round evaluates zone risk, lighting, occlusion, camera angle, temporal trend,
field calibration, and operator usefulness.

## Model adapter contract

Each upstream repository is isolated behind one model adapter. The adapter
accepts a normalized sample and returns a model-native result without forcing
density and point methods into the same internal representation.

Required input metadata:

- `sample_id`, dataset ID, frozen split ID, and source artifact hash;
- RGB image or explicitly declared modality;
- original width and height;
- ROI and zone geometry when available;
- ground-truth count and point annotations when licensed and available;
- condition tags such as density band, altitude, lighting, camera, and flight.

Required output metadata:

- model ID, upstream commit, checkpoint ID and SHA-256;
- scalar predicted count;
- native density map or predicted points and confidence, as applicable;
- preprocessing and output-coordinate transforms;
- inference latency, peak VRAM, input resolution, and failure state;
- optional zone-level counts derived with a declared aggregation method.

An adapter failure is not converted to a zero-person prediction. It emits an
explicit error state and is counted in evaluation coverage and robustness.

## Common result schema

Every completed test run writes a per-sample table with at least:

```text
sample_id
ground_truth_count
predicted_count
signed_error
absolute_error
relative_or_normalized_error
density_band
latency_ms
peak_vram_mb
output_type
failure_state
```

Family-specific columns are appended rather than fabricated:

- density models: density PSNR, density SSIM, GAME/zone error, predicted mass;
- point models: localization precision, recall, F1, matching threshold,
  coordinate error, and zone count;
- all models: undercount/overcount bias by density and condition.

PSNR and SSIM are auxiliary density-quality metrics. Their values depend on
ground-truth kernel, sigma, resolution, normalization, and ROI policy. The run
manifest must record those settings, and results with different density-target
generation policies cannot be compared directly.

## Per-run 100-point gate

Each `model_id x dataset_id x checkpoint_id x protocol_id` run receives one
technical score out of 100:

| Group | Points | Required evidence |
|---|---:|---|
| Data and split integrity | 15 | Frozen split, sample count, hashes, no leakage |
| Evaluation coverage | 10 | Every expected sample predicts or records an explicit failure |
| Count quality | 20 | Frozen MAE/RMSE and bias targets |
| Spatial output quality | 15 | Family-appropriate density, localization, GAME, or zone targets |
| Condition robustness | 10 | Density/altitude/lighting groups and high-density undercount |
| Runtime and resources | 10 | Latency, throughput, input policy, peak VRAM, GPU identity |
| Reproducibility and provenance | 10 | Git/upstream/checkpoint/environment/config hashes |
| Review artifacts | 10 | Scorecards, predictions, curated panels, selection manifest |

The default technical pass threshold is 80/100 with every integrity blocker
passing. Paper-reproduction runs retain the stricter existing Stage 3 contract,
including the frozen paper-gap requirement. Dataset/model configurations must
freeze all numeric targets before the full test split is opened; a run with a
missing target is `REVIEW`, not a pass.

Technical score and rights status are independent. A model can pass a research
technical gate while remaining `PASS_RESEARCH_ONLY`. Commercial use requires a
technical pass and a separate Stage 3C `PRODUCTION_APPROVED` decision for code,
dataset, pretrained weights, derived weights, and deployment.

Blocking conditions include:

- corrupt, altered, or leaking data/splits;
- missing or mismatched code, checkpoint, result, or split hashes;
- incomplete expected-sample accounting;
- non-finite outputs or silent failure-to-zero behavior;
- model output transformed with an unrecorded scale or coordinate policy;
- evidence or rights contradictions.

## Deterministic visual curation

The default Git review bundle contains 12 sample panels per completed
model-dataset run:

- three best cases: one each from low-, medium-, and high-density strata;
- three strong cases: one nearest the 25th error percentile in each stratum;
- three typical cases: one nearest the median error in each stratum;
- three failure cases: severe undercount, severe overcount, and worst spatial
  quality.

Density strata are frozen from ground-truth counts in the dataset protocol.
Primary ranking uses the frozen per-sample count error; the worst spatial case
uses the family-specific spatial metric. Ties are resolved by lexical
`sample_id`. If a dataset cannot fill a bucket, the selector records the reason
and produces fewer panels without duplicating a sample.

`selection_manifest.json` records:

- selector schema and configuration;
- density thresholds and metric directions;
- selected sample IDs and selection reason;
- SHA-256 of the complete predictions table;
- deterministic tie-breaking policy.

This keeps the visual report auditable and prevents manual cherry-picking.

## Visual panel contract

Each selected sample is rendered as a four-column panel inspired by the CSRNet
supplementary figures:

1. original input, ROI, and declared preprocessing crop;
2. ground-truth density or points with ground-truth count;
3. model-native predicted density or points with predicted count;
4. operator view with zones, density/risk level, confidence when valid, and
   count/spatial errors.

The panel labels model, dataset, checkpoint, sample, input resolution,
latency, and relevant family metrics. A density derived from point labels or
point outputs is labelled `derived` and cannot be presented as a native model
output. Full-resolution source and raw output stay on the SSD; Git panels use a
fixed review resolution that preserves aspect ratio and legibility.

## Storage and Git policy

The home RTX 5090 SSD remains authoritative for large run artifacts under:

```text
/workspace/data/results/<model>/<dataset>/<run-id>/
```

It may contain checkpoints, resumable state, full logs, compressed native
outputs, full-resolution selected renders, and other large evidence. Nothing is
automatically deleted. Redundant intermediate output may be marked `prunable`
only after a completed review and still requires explicit user approval before
deletion.

Git stores a small review snapshot under:

```text
reports/<model>/<dataset>/<run-id>/
  summary.md
  metrics.json
  predictions.csv
  score.json
  score.md
  selection_manifest.json
  environment-summary.json
  rights-decision.json
  figures/selected-01.png ... selected-12.png
```

The default Git review budget is 25 MiB per model-dataset run. If the selected
panels exceed the budget, the review renderer reduces only output review
resolution or losslessly optimizes the files; it does not omit required sample
categories. Dataset archives, checkpoints, credentials, full native maps, raw
videos, and mounted data never enter Git.

Every Git snapshot contains stable paths and SHA-256 references to its SSD
sources so a selected panel can be regenerated from the same code, data,
checkpoint, and protocol.

## Execution and review flow

Each model advances through the same sequence:

1. record code, dataset, weight, derived-weight, and deployment identities;
2. run Stage 3C for the requested research action;
3. build an isolated model environment and capture its lock/freeze;
4. inspect the adapter and inference entry point before full execution;
5. run synthetic CUDA and a five-to-ten-sample compatibility smoke drawn from
   synthetic input or an approved non-test partition;
6. generate and review the first visual panels;
7. freeze thresholds, protocol, selector settings, and test manifest;
8. evaluate the complete labeled test split;
9. calculate the 100-point technical score and rights-scoped decision;
10. curate 12 panels, verify artifacts and hashes, and commit the small report;
11. obtain explicit user review before advancing to the next model or dataset.

Long-running training uses checkpoint/resume and a non-interactive runner or
tmux, but reboot recovery depends on persisted checkpoints rather than tmux.
Failed or interrupted runs remain `INCOMPLETE` and cannot generate a pass.

## Comparison rules

- Only runs sharing dataset version, split, checkpoint class, preprocessing,
  target-generation policy, and metric implementation enter a direct ranking.
- Official-weight, project-trained, cross-domain, and third-party-weight runs
  appear in separate lanes.
- Model-native output is preserved. Derived density, points, zones, and risk
  levels are labelled and versioned transformations.
- A higher count score cannot hide missing localization, high-density collapse,
  runtime failure, or a rights blocker.
- Model selection considers technical score, spatial usefulness, robustness,
  runtime cost, review quality, and rights scope. There is no single-winner
  decision from MAE alone.

## Test strategy

Implementation tests must cover:

- adapter output schema for density and point fixtures;
- count preservation and coordinate transforms;
- online metric accumulation without saving every rendered map;
- family-aware metric applicability and explicit unavailable fields;
- deterministic 12-sample selection, tie-breaking, and sparse-bucket handling;
- selection-manifest hash verification;
- four-column panel creation for density and point outputs;
- Git size-budget enforcement and large-artifact rejection;
- explicit adapter failures rather than zero-count substitution;
- per-run 100-point score, blockers, and independent Stage 3C rights state;
- compatibility with existing Stage 0 through Stage 3C tests;
- the same unit suite and synthetic CUDA smoke in `home5090_docker`.

The first real-data implementation checkpoint is one DM-Count UCF-QNRF smoke
on an approved non-test subset, not a long training run. It must produce a
complete miniature report and visual bundle for user review before the sealed
test split or the next adapter proceeds.

## Acceptance criteria

The harness design is successfully implemented when:

1. one command can evaluate an eligible model-dataset protocol and write the
   common result bundle;
2. every expected sample is represented by a prediction or explicit failure;
3. a reproducible 100-point score and scoped rights decision are produced;
4. the complete metrics table deterministically selects and renders the 12
   review panels;
5. large artifacts remain on the 5090 SSD and the Git snapshot stays within its
   declared budget;
6. every path, configuration, split, checkpoint, and small artifact needed for
   regeneration is hash-verified;
7. existing historical experiments and tests remain valid;
8. the user can review the adapter, visual bundle, score, blockers, and rights
   state before authorizing the next model.
