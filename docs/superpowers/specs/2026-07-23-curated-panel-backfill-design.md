# Curated Review Panel Backfill Design

## Context

The frozen STEERER evaluation on the selected UP-COUNT val/test reference set
completed all 166 samples, but the evaluation gate was blocked because the
curation selector generated 11 review panels instead of the required 12.

The selector first chooses up to three samples from each density band and then
adds three failure-oriented samples. UP-COUNT contains only two high-density
samples in the frozen selection, so the first phase cannot fill all nine
density-stratified positions. The later failure selections add only three more
unique samples, leaving a one-panel shortfall even though 166 successful,
eligible samples exist.

## Decision

Preserve the existing density-stratified and failure-oriented selections in
their current order. If fewer than 12 unique samples have been selected, fill
the remaining positions from successful, not-yet-selected samples using a
deterministic representative ordering:

1. Compute the global median absolute count error across eligible successful
   samples.
2. Sort remaining samples by distance from that median.
3. Break ties lexically by `sample_id`.
4. Append samples until 12 unique selections are present or no eligible sample
   remains.

Each appended item uses category `representative`, retains its source density
band, and records the reason `global median error backfill`.

## Alternatives Rejected

- Lowering the required panel count would make review quality depend on a
  dataset's density distribution and weaken the frozen gate.
- Reusing a sample would satisfy the file count without adding independent
  review evidence.
- Rebalancing or replacing the existing 11 selections would unnecessarily
  change already accepted representative and failure cases.

## Data and Execution Safety

- Do not rerun the 166 first-pass model inferences.
- Reuse the completed prediction ledger and deterministic selected-sample
  reruns supported by the existing `--resume` workflow.
- Do not train, fine-tune, calibrate, or modify the STEERER checkpoint.
- Preserve the blocked result and its evidence until the corrected result is
  verified.
- UP-COUNT remains `PASS_RESEARCH_ONLY`.

## Verification

Implementation follows test-driven development:

1. Add a regression test with only two high-density samples and enough
   remaining eligible samples.
2. Confirm the current selector returns 11 and the new assertion for 12 fails.
3. Implement only the deterministic backfill.
4. Confirm the regression and existing curation tests pass.
5. Run the full relevant test suite.
6. Resume the existing UP-COUNT result, regenerate the review bundle, and
   require:
   - 12 unique selections and panel paths;
   - unchanged first-pass predictions and aggregate metrics;
   - verified artifact SHA-256 values;
   - `PASS_RESEARCH_ONLY` with no failed blocker.

The next model may not start until this gate passes.
