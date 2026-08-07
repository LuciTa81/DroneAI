# STEERER Training Authoritative Evidence v2 Design

## Status and decision

The home5090 T0 smoke proved that the pinned STEERER model can complete one real
optimizer update with finite values and a checkpoint round-trip. It also exposed
an integration gap: the Task 6 runner writes schema-v1 runtime evidence before
training, while the Task 7 authority gate accepts only checkpoint-bound
schema-v2 evidence containing measured stage metrics.

This design selects an integrated stage finalization pipeline. A successful
stage writes measured metrics first, immutable schema-v2 environment evidence
second, and the authoritative checkpoint last. The checkpoint records the exact
environment SHA-256. The gate is not relaxed and fabricated or inferred metrics
are forbidden.

## Considered approaches

### 1. Integrated runner finalization — selected

The training runner retains stage observations, computes the complete metric
payload, writes schema-v2 evidence, and only then saves the authoritative
checkpoint. It requires a focused runner refactor but produces one coherent
lineage and no unbound final weight.

### 2. Post-training evaluator with a provisional checkpoint

A separate process could load a provisional checkpoint, evaluate it, build the
environment, and resave a final checkpoint. This reuses more of the frozen
evaluation CLI, but creates two weight files with different metadata, doubles
state transitions, and makes it easier to mistake an unscored provisional weight
for the authoritative artifact.

### 3. Permit schema-v1 evidence

This is the least work but defeats the hardening already implemented in Task 7.
It cannot prove that the reported metrics, environment, and checkpoint belong to
the same stage execution and is rejected.

## Scope

Included:

- T0 measured `train_loss` evidence and schema-v2 checkpoint binding.
- T1 one-epoch training observations and complete 240-image validation metrics.
- Runtime identity supplied from the actual `crowd-jupyter` container image ID.
- Atomic metrics, environment, checkpoint, and manifest writes.
- Exact Task 7 authority verification and score bundle generation.
- Runbook commands using distinct T0 and T1 run IDs.

Excluded:

- T5, T50, UCF-QNRF Test, full 1,201-image retraining, deployment approval, and
  commercialization of the derived weight.
- Any change to the active Round 1 comparison queue.
- Reuse or mutation of the existing unscored T0 artifact
  `steerer-qnrf-imagenet-20260806-a`.

## Architecture and ownership

### Stage observations

`steerer_training_runner.py` remains responsible for model execution. Its
observations are expanded only enough to retain measured values required for
evidence:

- T0: actual optimizer-update loss.
- Training epoch: per-update losses and first/last training windows.
- Validation: per-sample target count, predicted count, native density maps,
  localization points, validation loss, synchronized latency, and peak VRAM.

The validation collector uses the already loaded training model and the approved
240-image validation loader. It does not reload a checkpoint and does not access
UCF-QNRF Test.

### Metric calculation

A new small module, `steerer_training_evidence.py`, owns deterministic metric
aggregation and schema-v2 evidence creation. It reuses the common evaluation
metric primitives and STEERER localization extraction where their coordinate and
density contracts match. It does not own model training or checkpoint I/O.

### Finalization order

For every stage-scoped run ID:

1. Complete the authorized training and validation ceiling.
2. Validate all observed values as finite.
3. Atomically write `metrics.json`.
4. Recompute project, upstream, profile, split, dataset, backbone, batch, and
   runtime identity and atomically write `environment.{sha256}.json` using schema
   version 2. `environment.json` may be an identical convenience copy.
5. Save the authoritative checkpoint with `environment_manifest_sha256` equal to
   that content-addressed environment file.
6. Reload and verify the checkpoint and manifest.
7. Run `verify_authoritative_training_evidence` and write the score bundle.

If steps 2–5 fail, no artifact may be reported as an authoritative checkpoint.
Temporary files are removed by the existing atomic-write boundaries.

## Run identity and storage

T0 and T1 do not resume each other and therefore use different run IDs:

- T0 example: `steerer-qnrf-imagenet-20260806-t0-b`
- T1 example: `steerer-qnrf-imagenet-20260806-t1-b`

Each run ID owns one checkpoint directory and one result directory under the
profile-pinned `/workspace/data` roots. Existing non-empty runs are never
overwritten or deleted. Large `.pth` files remain on the SSD.

## Metric definitions

T0 contains exactly:

- `train_loss`: mean loss from the one real optimizer update, not the AMP probe.

T1 contains exactly the Task 7 required metric set:

- `mae`, `rmse`, `signed_bias`: count error over all 240 validation images.
- `reference_mape`: mean absolute error divided by `max(ground_truth, 1)`, in
  percent.
- `game_l1`: mean common-harness GAME L1 using highest-resolution predicted and
  ground-truth density maps.
- `quadrant_zone_mae`: mean absolute count error over four equal image quadrants.
- `localization_precision_16px`, `localization_recall_16px`,
  `localization_f1_16px`: common-harness one-to-one point matching at 16 original
  pixels. Predicted multiscale STEERER points are mapped back using the loader's
  resize ratio; ground-truth points come from the prepared Train-only JSON.
- `median_latency_ms`, `p95_latency_ms`: synchronized batch-1 inference latency
  across the 240 images. Loss computation is excluded from the timed region.
- `fps`: `1000 / median_latency_ms`.
- `peak_vram_mb`: maximum CUDA allocated-memory observation during validation.
- `density_sum_count_difference`: mean absolute difference between the returned
  predicted count and the highest-resolution density sum; this verifies the
  declared density-sum count contract.
- `train_loss`: mean loss across epoch 1 optimizer updates.
- `validation_loss`: mean validation loss over all 240 images.
- `training_window_improvement`: mean loss of the first 10% of epoch updates
  minus the mean loss of the last 10%, using at least one update per window.

Poor but finite metric values do not block T1. Missing, non-finite, incomplete,
or self-inconsistent values are hard blockers.

## Runtime and provenance evidence

The T0/T1 CLI requires `--container-image-digest` as the `sha256:` prefix followed
by exactly 64 hexadecimal characters. The
runbook obtains it from the Pop!_OS host with `docker inspect`; a caller cannot
silently substitute a tag. Schema-v2 environment evidence records:

- clean project commit and profile SHA-256;
- pinned clean upstream origin, commit, LICENSE hash, and config hash;
- Train-only processed root, sealed-Test flag, dataset inventory, population,
  train, validation, split-manifest, and sealed-marker hashes;
- ImageNet-backbone-only mode, canonical path, byte size, and SHA-256;
- synthesized training config SHA-256;
- actual Python, Torch, CUDA, GPU, container image digest, and batch plan;
- exact metrics path and SHA-256.

Rights scope remains `PASS_COMMERCIAL_CANDIDATE`; this evidence does not grant
`PRODUCTION_APPROVED`.

## Failure handling

- No Test path is accepted by any CLI or evidence API.
- A dirty/wrong project or upstream, hash mismatch, incomplete split, non-finite
  tensor, missing metric, or wrong runtime digest aborts before authority scoring.
- CUDA OOM retains the existing single fallback from batch 8×1 to 4×2.
- A failed run is preserved for diagnosis but is not promoted, scored, copied to
  Git, or used as a resume source.
- T1 cannot start until a fresh T0 run obtains at least 70/100 with no blocker.

## Testing and rollout

Implementation uses test-driven development in this order:

1. Unit tests for exact T0/T1 metric schemas and formulas.
2. Runner tests proving the actual update loss is retained and that all 240
   validation observations are required.
3. Evidence tests proving `metrics -> schema-v2 environment -> checkpoint` order
   and hash binding.
4. CLI tests for required container digest, stage-specific run IDs, and no Test
   arguments.
5. Existing checkpoint/resume and Task 7 gate suites.
6. One final full local suite.
7. Remote fresh T0, score gate, then fresh T1 only if T0 passes.

The first remote approval stop is the T0 score. The second is the T1 score; T5 is
not started in this cycle.
