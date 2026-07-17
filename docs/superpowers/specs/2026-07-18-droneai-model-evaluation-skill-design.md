# DroneAI model-evaluation skill design

## Goal

Create a project-specific personal Codex skill that advances one crowd-counting
candidate through the existing DroneAI gates without repeatedly rebuilding the
project context. The skill reduces token use and runtime while preserving the
rights, provenance, split-isolation, metric, evidence, and user-approval rules.

## Selected approach

Use a thin personal skill at
`C:\Users\user\.codex\skills\evaluating-droneai-models` and keep executable
logic, model profiles, tests, and research evidence in the DroneAI repository.
The skill contains only the workflow, stop conditions, exact files to inspect,
and commands that invoke repository-owned automation.

Do not copy the full roadmap, model papers, runbooks, or result history into the
skill. They remain canonical in Git and are loaded only when the active stage
requires them.

## Trigger and scope

Trigger the skill when the user asks to add, evaluate, compare, resume, score,
or report the next DroneAI crowd-counting model on `home5090`.

The skill may:

- report repository, runtime, model-queue, rights, and result status;
- register a model candidate and its official sources;
- run rights, source, environment, checkpoint, one-sample, and frozen benchmark
  gates in order;
- create the small Git review bundle and stage score;
- stop at the documented user-review checkpoint.

It must not authorize training, fine-tuning, production deployment, test-split
access, destructive cleanup, or an unrecorded dataset/checkpoint download.

## Architecture

### Personal skill

`SKILL.md` stays concise and routes work according to a machine-readable status
summary. It requires one active model and one stage at a time, uses no subagents
by default, and forbids broad repository or web searches when pinned evidence is
already present.

Detailed field definitions and report shape live in one optional reference file.
The skill includes no large assets.

### Repository automation

Add one deterministic command surface with these operations:

1. `status` — return compact JSON describing Git/runtime connectivity, active
   model, completed gates, next safe action, and approval requirement.
2. `preflight` — verify rights authorization, clean/pinned sources, dataset and
   split identities, environment, and checkpoint identity without inference.
3. `one-sample` — invoke the registered model runner for one frozen validation
   sample and validate its native outputs.
4. `benchmark` — invoke the registered frozen evaluation only after preflight
   and one-sample evidence match the current identities.
5. `bundle` — copy only manifest, score, predictions summary, and curated panels
   within the Git size policy.

Existing DM-Count and STEERER runners remain the implementation of model-specific
inference. The new command dispatches to them; it does not duplicate adapters or
metrics.

### Model queue and state

Store a small versioned queue/profile that records the ordered Round 1 models,
active stage, required dataset/split, runner, rights record, checkpoint policy,
and fine-tuning eligibility. Derived run state is recalculated from committed
configuration and SSD manifests instead of trusted from conversation text.

Initial Round 1 order is:

1. DM-Count — completed frozen UCF-QNRF comparison.
2. STEERER — one-sample complete; frozen 36-sample comparison next.
3. PET — next point/localization candidate after STEERER review.
4. CSRNet — legacy density baseline; record
   `NOT_EVALUATED_NO_ELIGIBLE_WEIGHT` unless an approved weight or project
   training lane is established.

APGCC remains the second-wave point candidate; MPCount remains a domain-
generalization candidate rather than a same-domain leaderboard entry.

## Workflow and stop conditions

For each model, execute:

1. Freeze model brief, official source commit, dataset/split, checkpoint policy,
   output contract, thresholds, and expected sample count.
2. Require Stage 3C authorization for the exact next action.
3. Verify source, environment, dataset, split, and checkpoint identities.
4. Run unit and synthetic/CUDA compatibility checks.
5. Run one frozen validation sample and verify density/count/point invariants.
6. Stop for user approval before the 36-sample benchmark when the model is new.
7. Run all 36 samples, compute metrics, curate panels, and score the evidence.
8. Stop for user review before advancing the queue.

If an identity changes, invalidate only the dependent evidence. If rights are
silent, retain the candidate as `PASS_COMMERCIAL_CANDIDATE` but keep evaluation
scope research-only. An explicit prohibition blocks the prohibited action.

## Credit and latency controls

- Read the compact status output first; do not reconstruct state from chat.
- Read only the active model profile, active rights record, and active runner.
- Reuse evidence when its Git/upstream/dataset/split/checkpoint/environment hashes
  match; do not repeat completed reviews.
- Use one agent by default. Request additional review only for new legal evidence,
  a failing safety gate, or a material harness change.
- Do not browse for already pinned official sources.
- Do not run the full test suite when a focused test proves the changed boundary;
  run full verification only before committing or advancing a model.
- Never commit datasets, checkpoints, raw density arrays, or full run trees.

## Verification

Develop the skill and repository command using RED-GREEN-REFACTOR.

Baseline scenarios must demonstrate that, without the skill, an agent either
rescans excessive context, skips a required gate, repeats completed work, or
produces an incomplete status report. With the skill installed, the same
scenarios must select the correct next action and stop condition from compact
repository evidence.

Automated tests must cover:

- correct active-model and next-action selection;
- refusal on dirty/mismatched source and artifact identities;
- refusal to benchmark before rights/preflight/one-sample gates;
- research-only versus commercial-candidate scope preservation;
- no test-split access and exact frozen-sample cardinality;
- bundle size and artifact allowlist;
- concise roadmap/status output containing model, dataset, validation method,
  completed result, next action, and approval point.

Validate the personal skill with the official skill validator and forward-test
it on a fresh, minimal-context invocation before using it for PET.

## Acceptance criteria

The design is complete when:

- a fresh Codex invocation can determine the current DroneAI stage from compact
  machine evidence without relying on this conversation;
- STEERER's safe next action is identified as dependency pin/reverification then
  the frozen 36-sample UCF-QNRF run;
- the skill cannot bypass rights, provenance, split, checkpoint, or approval
  gates;
- the 36-sample result and subsequent PET onboarding use the same workflow;
- every stage report includes the current roadmap, model, dataset, metrics,
  technical score, rights scope, and next approval point.
