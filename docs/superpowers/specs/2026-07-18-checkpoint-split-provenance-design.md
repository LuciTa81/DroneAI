# Checkpoint Split Provenance Design

## Goal

Prevent a locally disjoint validation split from being mislabeled as held-out
checkpoint evidence when the public checkpoint's exact training membership is
unknown. Preserve the completed 36-image runs as compatibility evidence without
rewriting historical artifacts.

## Scope

This change covers evaluation metadata, workflow status, research-result labels,
and ranking eligibility. It does not open the UCF-QNRF test partition, retrain a
model, rerun an accepted benchmark, or modify existing SSD result bundles.

## Decision

Keep the existing local split checks and add a separate checkpoint-level status:

- `VERIFIED_DISJOINT`: the evaluated samples are proven absent from checkpoint
  training;
- `VERIFIED_OVERLAP`: at least one evaluated sample is proven present in
  checkpoint training;
- `UNKNOWN`: the checkpoint training membership cannot be proven from pinned
  evidence.

Each evaluation also declares one comparison scope:

- `compatibility_smoke`: installation, inference, output, runtime, visual, and
  integration evidence; all three checkpoint statuses may run;
- `held_out_performance`: accuracy evidence eligible for direct ranking; only
  `VERIFIED_DISJOINT` may run.

The current 36-image runs are classified as follows:

| Model | Checkpoint split status | Comparison scope | Ranking eligible |
|---|---|---|---|
| DM-Count | `VERIFIED_DISJOINT` | `compatibility_smoke` | no |
| STEERER | `UNKNOWN` | `compatibility_smoke` | no |
| PET | `UNKNOWN` | `compatibility_smoke` | no |

DM-Count's known split remains valid, but the entire 36-image round stays a smoke
round so that its table has one consistent claim. Existing metrics remain useful
as engineering evidence and are not presented as a final model ranking.

## Components

### Evaluation protocol

The common protocol records `checkpoint_training_split_status`,
`comparison_scope`, and a non-empty `checkpoint_split_evidence` note or artifact
reference. Validation rejects unknown enum values. A held-out run is rejected
before inference unless its checkpoint status is `VERIFIED_DISJOINT`.

### Evaluation artifacts

New runs copy the checkpoint split fields into the sample manifest, summary, and
evidence manifest. `ranking_eligible` is derived, never supplied by callers:

```text
comparison_scope == held_out_performance
and checkpoint_training_split_status == VERIFIED_DISJOINT
```

Historical result directories remain immutable. Their current classification is
carried by the versioned model queue and workflow status.

### Workflow queue and status

Every queue entry must declare the checkpoint status and comparison scope.
`status` exposes both fields plus derived `ranking_eligible`. Review can advance a
model whose compatibility smoke is complete, but downstream comparison code must
exclude entries that are not ranking eligible.

### Scoring

The existing 100-point technical score continues to measure execution and artifact
quality. It is not converted into a held-out claim. Checkpoint overlap or unknown
membership is an independent comparison-scope blocker, analogous to the separate
rights scope.

## Error handling

- Missing or invalid checkpoint split metadata blocks configuration loading.
- `VERIFIED_DISJOINT` requires a non-empty pinned evidence description.
- `held_out_performance` with `UNKNOWN` or `VERIFIED_OVERLAP` blocks before model
  inference and before output creation.
- Historical accepted evidence is hash-checked but never edited in place.

## Test strategy

1. Prove the old behavior is unsafe with a failing test: an `UNKNOWN` checkpoint
   cannot request `held_out_performance`.
2. Prove all checkpoint statuses are accepted for `compatibility_smoke` and none
   are ranking eligible.
3. Prove only `VERIFIED_DISJOINT` held-out evaluation is ranking eligible.
4. Prove workflow status exposes the fields and rejects incomplete queue entries.
5. Run focused evaluation/workflow tests, then the full repository suite.
6. Push the clean commit and update home5090 with `git pull --ff-only`; run the
   compact status command there to verify PET is reclassified without rerunning it.

## Ordered evaluation roadmap

1. Reclassify the completed 36-image DM-Count, STEERER, and PET results as
   compatibility smoke evidence.
2. Evaluate remaining candidates with official checkpoints in the same smoke lane,
   recording source-domain and checkpoint split status.
3. Freeze adapters, preprocessing, metrics, thresholds, and panel selection.
4. Request explicit approval before one-time access to the official 334-image
   UCF-QNRF test partition.
5. Compare only held-out eligible results, with same-domain, cross-domain, and UAV
   checkpoint lanes reported separately.
6. Zero-shot the strongest rights-eligible candidates on UAV and owned fixed-CCTV
   data.
7. Fine-tune at most one final rights-eligible winner on owned data.
