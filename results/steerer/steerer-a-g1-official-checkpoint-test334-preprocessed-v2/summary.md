# STEERER A G1 official-checkpoint Test334 reproduction

## Outcome

- Model: official STEERER HRNet-W48 UCF-QNRF checkpoint
- Checkpoint SHA-256: `85f71763595527e419257d459288e4a784766f41687ff0b859dcff65a151bbec`
- Data: official UCF-QNRF Test 334, transformed with pinned
  `prepare_QNRF.py` image/annotation semantics
- Result label: `official Test, test-selected`
- Status: `PASS_RESEARCH_ONLY`
- Harness evidence score: `100/100` (experiment integrity, not accuracy)
- Full artifacts: `/workspace/data/results/steerer-official-code-reproduction/official-checkpoint-test334-preprocessed-v2`

## Metrics

| Metric | Result | Repository claim | Absolute difference |
|---|---:|---:|---:|
| MAE | **77.5165** | 77.8 | 0.2835 |
| RMSE | **139.0867** | 138.0 | 1.0867 |

- Signed bias: +0.8315 people/image
- MAPE reference: 10.4468%
- Median latency: 143.02 ms/image
- Throughput: 6.99 FPS
- Peak VRAM: 4.0149 GiB
- Harness localization P/R/F1@16px: 73.05% / 67.34% / 69.42%
- Four-quadrant density zone MAE: 21.89 people/zone
- Accounting: 334 expected, 334 recorded, 334 successful, 0 failures

## Root-cause evidence

The preserved v1 raw-image run returned 86.6723 MAE / 152.5721 RMSE and
correctly blocked long training. The pinned upstream preparation performs an
offline resize/re-encode of every QNRF image and preserves boundary annotations.
The old raw path skipped that preparation and excluded 57 annotations across 45
images. The prepared v2 Test set has a total GT count of 240,127, matching the
pinned upstream localization GT inventory.

This correction reduced MAE by about 10.6% and RMSE by about 8.8%, bringing the
official checkpoint inside the frozen compatibility bound without changing the
bound after observing results.

## Evidence hashes

- Test preparation manifest: `1a56761009f13c04eac4e0731d2a841acd2e3d9afdb62b3627edcc6384f57c66`
- `metrics.json`: `6520a833bda0ff3c90d292e6ff62220dd938c917df97593f5e4fa5789c106a6c`
- `compatibility-gate.json`: `10c0ec192021a919dc96098af5fa779aa261ce43f332744505288bfca06e5649`
- `score.json`: `18c7ad3d29f5342b6cf1d11ca712074152bd510003b2bddd4f6d3610ba4886f3`
- `predictions.csv`: `ebad4991db8f10e8bec7fba2fe4cc1f6fd1e7c6f984ee9a3350f2e211885b4c5`
- `evidence-manifest.json`: `af069987d24bb1c59e2c292e3a8eb26e3f920903a253717f713773e6950de82f`

## Next gate

G2 may prepare the complete A training derivative. Long training still remains
blocked until G2 data integrity, G3 one-update, and G4 one-epoch gates pass.
