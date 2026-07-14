# Rights-scope harness design

Date: 2026-07-15

## Goal

Prevent a technically successful crowd-counting experiment from being mistaken
for permission to build or ship a commercial product. At the same time, retain
models and datasets whose terms are silent or incomplete as due-diligence
candidates instead of deleting them from the comparison set.

This is an engineering control, not legal advice. A missing license is not
treated as permission. GitHub's licensing guidance says that default copyright
rules apply when no license is present, and CC BY-NC 4.0 expressly limits reuse
to non-commercial purposes.

## Considered approaches

### 1. Keep generic `PASS` and add notes

Rejected. A consumer can ignore the prose and interpret `PASS` as a commercial
approval. It does not close the existing `commercial_use=false` contradiction.

### 2. Harden only the Stage 1 dataset gate

Rejected. Dataset permission alone says nothing about reference source code,
pretrained weights, derived weights, or product deployment. It would also mix
research reproduction readiness with commercial eligibility.

### 3. Add an explicit rights state machine and Stage 3C

Selected. Infrastructure stages may retain generic `PASS`. Rights-sensitive
stages emit scoped outcomes, and Stage 3C combines all product-relevant rights
components before any commercial training or deployment is authorized.

## Decision states

| State | Meaning | Permitted harness actions |
|---|---|---|
| `PASS_RESEARCH_ONLY` | Explicit terms allow the declared research use but do not allow the commercial lane. | catalog, approved research reproduction |
| `PASS_COMMERCIAL_CANDIDATE` | No explicit commercial prohibition was found, but one or more required commercial permissions are absent, silent, pending, or unverified. This is not permission. | catalog and rights due diligence; synthetic compatibility smoke only when code rights are verified |
| `PRODUCTION_APPROVED` | Every required component has verified evidence that covers the declared commercial action. | catalog, commercial training, derived-weight use, deployment |
| `BLOCKED` | Explicit prohibition, scope contradiction, missing identity/provenance, or integrity blocker. | review only |

`REVIEW` remains available for an incomplete technical score. Generic `PASS`
remains available for non-rights infrastructure stages such as Stage 0 and
Stage 2.

## Rights components

Stage 3C evaluates these components independently:

1. `code`: the actual implementation or independently implemented policy.
2. `dataset`: the images/video and annotations used for training/evaluation.
3. `pretrained_weights`: any externally obtained checkpoint.
4. `derived_weights`: the proposed rights basis for weights produced by this
   project, including restrictions inherited from inputs and agreements.
5. `deployment`: the right to use the complete artifact in the product context,
   including any separately recorded privacy or collection approval.

Each component records a stable ID, source URL, rights status, license/terms
identifier, evidence URL, `commercial_use` as `true`, `false`, or `null`, and a
basis explaining whether the terms are explicit, not stated, or pending.

## Classification rules

1. `commercial_use=false` plus explicit non-commercial terms classifies the
   bundle as `PASS_RESEARCH_ONLY` only when the declared research action is
   itself permitted.
2. `commercial_use=null` plus `not_stated` or `pending_review`, with no explicit
   prohibition, classifies the bundle as `PASS_COMMERCIAL_CANDIDATE` when its
   minimum provenance is complete.
3. `commercial_use=true` counts toward `PRODUCTION_APPROVED` only when status is
   `verified` and an evidence URL and license/terms identifier are recorded.
4. Every required component must satisfy rule 3 before the bundle can become
   `PRODUCTION_APPROVED`.
5. Explicit prohibition for the requested scope, or contradictions such as
   `commercial_use=false` with `basis=not_stated`, are blockers.
6. Candidate classification authorizes no ambiguous asset download, training,
   checkpoint reuse, or deployment. Those actions remain absent from the
   candidate `allowed_actions` list. Synthetic smoke is present only when the
   code component is independently verified and uses no ambiguous data/weight.

## Existing-stage changes

- The scoring engine accepts a validated success status and serializes schema
  version 2 with a `decision_scope` field.
- Stage 1 reconciles `intended_use`, `allowed_uses`, and `commercial_use`.
  Verified research fixtures emit `PASS_RESEARCH_ONLY`; a commercial manifest
  with `commercial_use=false` is blocked.
- Stage 3 and Stage 3B are ShanghaiTech research-reproduction lanes and emit
  `PASS_RESEARCH_ONLY`.
- SHA-256 fields must be exactly 64 hexadecimal characters. Stage 3 split hashes,
  Stage 3B split/checkpoint hashes, and required traceability/review bundles are
  blockers rather than score-padding fields.

## Initial candidate record

The DM-Count plus ShanghaiTech bundle is recorded as a commercial candidate,
not as commercial permission:

- DM-Count code: verified MIT at the pinned upstream commit.
- ShanghaiTech dataset: terms not established in the current evidence.
- Upstream pretrained weights: permission not established separately.
- Derived weights and deployment: pending the input-rights chain and project
  approval.

Research reproduction continues in its isolated lane. Product-lane training and
deployment remain unavailable.

## Verification

Adversarial tests must prove that:

- a one-character or non-hex SHA-256 cannot pass;
- missing Stage 3B traceability or review artifacts cannot pass at the threshold;
- `commercial_use=false` cannot satisfy a commercial Stage 1 declaration;
- explicit non-commercial terms become research-only;
- silent terms become candidate without gaining download/train/deploy actions;
- only verified evidence for all five components reaches production approval;
- existing Stage 0/2 generic passes remain compatible.

Run the entire local suite, the GitHub checks, and the same suite plus Stage 0
CUDA smoke in `home5090_docker`. No real dataset download or long training is
part of this change.
