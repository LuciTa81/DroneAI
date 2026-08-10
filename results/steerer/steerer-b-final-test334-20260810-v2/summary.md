# STEERER B-lane final UCF-QNRF Test334

## Outcome

- Result label: `official Test, validation-selected`
- Training lane: official Train 1,201 images split into Train 961 / validation 240
- Final evaluation: official Test 334 images, once, after checkpoint selection was frozen
- Status: `CLOSED_AFTER_OFFICIAL_TEST`; no post-Test B-lane tuning is allowed
- Runtime: `home5090_docker`, NVIDIA GeForce RTX 5090
- Complete artifacts remain on `/workspace/data/results/steerer-b-final-test334/steerer-b-final-test334-20260810-v2`

## Results

| Frozen role | Epoch | MAE | RMSE | MAPE reference | Bias | Localization F1 | Zone MAE | Median latency | FPS | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary best validation MAE | 625 | **83.09** | **143.19** | 11.39% | -15.72 | 67.99% | 23.09 | 140.86 ms | 7.10 | 4.01 GiB |
| secondary best validation RMSE | 175 | 99.90 | 168.71 | 14.24% | -8.56 | **68.45%** | 27.72 | 138.31 ms | 7.23 | 4.01 GiB |
| final reference | 800 | 84.90 | 150.10 | 11.64% | -12.31 | 67.95% | 23.60 | 138.37 ms | 7.23 | 4.01 GiB |

`epoch 625` is the frozen primary result. It trails the STEERER repository's
reported UCF-QNRF result (77.8 / 138.0) by 5.29 MAE and 5.19 RMSE, while using
the project's leakage-resistant validation-selected B protocol.

## Integrity and scope

- Three checkpoint SHA-256 values and checkpoint-manifest membership were verified before Test access.
- Every role contains 334 prediction rows, 334 unique IDs, zero explicit failures, and finite count/latency values.
- Harness evidence score: `100/100`. This is an experiment-integrity score, not model accuracy.
- The training profile is `PASS_COMMERCIAL_CANDIDATE`, with `production_approved=false`.
- The generic evaluation gate deliberately reports `PASS_RESEARCH_ONLY`; production approval still requires CCTV field validation and a final rights review.
- The first attempt stopped before inference because PyTorch 2.6 rejected optimizer/RNG metadata under `weights_only=True`. Its partial directory was preserved. The v2 run permits full deserialization only for hash-verified project-training checkpoints.

## Evidence hashes

- `comparison.json`: `845648e5e32e3481b8081ce33b56f4df2f0fca0177bc6d98da0b8833338d5bb3`
- `b-lane-closed.json`: `e0f1419c8ca4058b4519a15e7adc5d11970250e1b7368a9eeb0423c4979ae6b5`
- `checkpoint-selection.json`: `33e76f69fb398da11c52e521a386082a65a18423c86f37fb1957081d76ff463b`
- `test-source-inventory.json`: `694a73481464945ef1c90bdf140703a1a266a21087944af93690144aebe3b8c7`

