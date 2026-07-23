# Round 2 Point-Model Replacement Design

## Objective

Replace the two unstarted Round 2 density-model lanes, DM-Count and MPCount,
with PET and APGCC. Preserve the completed STEERER runs and all historical
Round 1 and Round 2 evidence.

The active comparison becomes:

1. STEERER
2. PET
3. APGCC

Each model is evaluated on the same frozen dataset matrix:

| Dataset | Partition | Samples |
|---|---|---:|
| UCF-QNRF Kaggle Apache | official Test | 334 |
| JHU-CROWD++ | official Validation | 500 |
| UP-COUNT | official val/test selection | 166 |

This remains a frozen internal reference evaluation with exactly 1,000 samples
per model, 3,000 total model-sample inferences, no training, no fine-tuning, and
no calibration.

## Historical Preservation

The following artifacts remain immutable:

- `configs/evaluation/round2_reference_benchmark.json`
- `configs/evaluation/round2_reference_home5090.json`
- `/workspace/data/results/round2-reference-v1/sample-manifest.json`
- `/workspace/data/results/round2-reference-v1/steerer/*`
- every Round 1 DM-Count, MPCount, PET, and APGCC result

No DM-Count or MPCount result is deleted. They remain Round 1 smoke evidence and
are not presented as members of the new Round 2 matrix.

## Configuration and Sample Identity

Add a new point-model benchmark config and runtime config instead of overwriting
the legacy files. Both the legacy and point-model shortlists remain readable by
the harness.

The point-model config keeps the existing
`round-2-reference-benchmark-v1` round ID and the exact dataset lanes so split
IDs and sample rows do not change. It receives a new config identity because
the model shortlist changes.

A new shared manifest is generated at a new SSD path. Its
`config_identity_sha256` must differ from the legacy manifest, while its
`sample_manifest_sha256` must exactly equal:

```text
37adb8e058611de79d6f97c3181106f6606f5d1bc1a9b611720e4a699395cc04
```

The final comparison may reuse the existing STEERER runs only after verifying
that every run carries this same sample-row identity. The comparison does not
require the legacy and point-model config identities to match.

## Model Runtime Identities

### PET

- Family: point counting and localization
- Upstream: `/workspace/upstreams/PET`
- Upstream commit:
  `5b4dd7da8b11568a3305a88bb7c99a7fc831a998`
- Environment: `/workspace/.venvs/pet`
- Checkpoint: `/workspace/data/checkpoints/pet/UCF_QNRF.pth`
- Checkpoint SHA-256:
  `58324f86782051591075a923522f33d245ce97f2b50cc43f618313d884b966ad`
- Input: one RGB image, ImageNet normalization, official long-side cap 1536
- Native output: point/non-point scores, point coordinates, confidences, and
  quadtree split map
- Count: number of accepted point queries above the frozen 0.5 threshold
- Native density map: unavailable

PET code and official checkpoint are explicitly restricted to academic use.
Every PET lane is `PASS_RESEARCH_ONLY`. PET cannot be selected for commercial
fine-tuning, derived weights, or deployment without separate written
permission.

### APGCC

- Family: point counting and localization
- Upstream: `/workspace/upstreams/APGCC`
- Upstream commit:
  `e3e997bf592a70fd34233a432a974200d8c0c847`
- Environment: `/workspace/.venvs/apgcc`
- Checkpoint: `/workspace/data/checkpoints/apgcc/SHHA_best.pth`
- Checkpoint SHA-256:
  `cd9aa0f65882c81bb753a4ea8c821e07573b2ae4a31fffd5595fa0a43925cf90`
- Input: one RGB image with ImageNet normalization
- Native output: person logits, point coordinates, confidences, and offsets
- Count: number of in-bounds person points above the frozen 0.5 threshold
- Native density map: unavailable

APGCC code is MIT. The published ShanghaiTech-A checkpoint has no separate
commercial terms, so it remains a candidate under the owner's silence policy,
but the checkpoint and deployment are not production-approved. UCF may retain
`PASS_COMMERCIAL_CANDIDATE`; JHU-CROWD++ and UP-COUNT always narrow the
effective lane to `PASS_RESEARCH_ONLY`.

The combined comparison is always `PASS_RESEARCH_ONLY`.

## Harness Integration

The Round 2 configuration layer accepts two explicit, ordered shortlists:

```text
legacy: steerer, dm-count, mpcount
point:  steerer, pet, apgcc
```

The new runtime config contains exactly the point shortlist. The runner must
reject mixed, missing, reordered, or extra model entries.

Add PET and APGCC support to:

- accepted model configuration and rights validation;
- runtime identity and checkpoint verification;
- adapter construction;
- checkpoint training-split claims;
- dataset-specific rights narrowing;
- command-line model choices;
- final comparison model order and model notes.

The legacy shortlist continues to parse so its historical manifest can still be
verified. New execution uses only the point shortlist.

## Metric Contract

All models retain the common count and runtime metrics:

- MAE;
- RMSE;
- signed bias;
- MAPE reference;
- latency;
- batch-1 FPS;
- peak VRAM.

Point-capable spatial metrics use original-image coordinates and one frozen
matching radius of 16 pixels:

- localization precision;
- localization recall;
- localization F1;
- localization mean matched distance;
- point-zone MAE.

STEERER also retains its native density metrics:

- GAME(1);
- density-zone MAE;
- density PSNR;
- density SSIM.

PET and APGCC must not claim native density accuracy. Any binned or
kernel-rendered point heatmap is labelled `derived point visualization` and is
excluded from density PSNR, density SSIM, and density-ranking claims.

The final comparison reports spatial metric names beside their values and does
not pool incompatible density and point metrics into a single score.

## Review Panels

Each lane produces 12 unique, hash-verified review panels selected by the
existing deterministic curation policy.

For PET and APGCC, panels contain:

- source RGB image;
- ground-truth points;
- predicted points and confidence representation;
- count summary;
- calibrated zone overlay and zone counts;
- an explicit derived-visualization label if a point heatmap is shown.

Panels must never depict a derived point heatmap as a native density map.

## Execution Gates

PET runs before APGCC. A model may not advance until the previous model's three
dataset lanes are verified.

For each model:

1. Verify clean Git and upstream commits.
2. Recompute checkpoint, rights, dataset, and shared-manifest SHA-256 values.
3. Verify the isolated environment and CUDA import.
4. Run one sample from UCF-QNRF, JHU-CROWD++, and UP-COUNT.
5. Require finite count, in-bounds points, aligned confidence count, deterministic
   count derivation, valid zone aggregation, latency, VRAM, and a valid panel.
6. Run the full lanes sequentially: 334, then 500, then 166 samples.
7. Require complete accounting, no implicit failures, 12 panels, evidence hash
   verification, technical score, and independent rights scope after every
   lane.

If a lane fails, preserve it under `attempts`, diagnose the root cause, and do
not start the next lane. Existing results are never overwritten or deleted.

## Final Comparison

Build a 3 × 3 comparison from:

- the three verified STEERER runs already on SSD;
- three new PET runs;
- three new APGCC runs.

The comparison verifies:

- exactly nine lanes;
- the frozen per-dataset sample counts;
- one identical `sample_manifest_sha256`;
- model-specific spatial metric names;
- evidence-manifest artifact hashes;
- dataset and effective rights scopes;
- `training=false` and `fine_tuning=false`.

It reports each dataset separately. It does not pool datasets into a commercial
ranking or declare a production winner.

## Testing

Implementation follows test-driven development:

1. Add failing tests for the point shortlist, runtime config, and new manifest
   identity behavior.
2. Add failing runner tests for PET and APGCC config validation, adapter
   construction, checkpoint claims, and rights narrowing.
3. Add failing comparison tests for the new model order, model notes, and
   heterogeneous spatial metrics.
4. Implement the smallest changes that pass those tests.
5. Run focused Round 2, PET, APGCC, rights, runner, and comparison tests.
6. Run the complete repository test suite.
7. Push the tested branch and update home5090 only with `git pull --ff-only`.

## Success Criteria

The replacement is complete only when:

- legacy configuration and evidence remain present and verifiable;
- the new point shortlist and runtime config are deterministic and rights-gated;
- the new manifest has a new config identity and the exact frozen sample-row
  identity above;
- PET and APGCC each pass three one-sample gates;
- all six new full lanes pass their technical and rights gates;
- the final nine-lane comparison verifies all artifact hashes;
- no training, fine-tuning, derived weights, or production deployment occurs.
