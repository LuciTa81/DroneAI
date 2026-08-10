# STEERER B final Test334 on home5090

This run is the one-time held-out evaluation of the completed 961/240 B lane.
Checkpoint roles were selected from the Train-derived validation split before
official Test access. After this run, do not resume or tune the B lane.

## Frozen inputs

```bash
WT=/workspace/.worktrees/steerer-qnrf-test334-a
PY=/workspace/.venvs/steerer/bin/python
UPSTREAM=/workspace/upstreams/STEERER
TEST_ROOT=/workspace/data/datasets/ucf-qnrf-kaggle-apache/raw/UCF-QNRF_ECCV18/Test
CHECKPOINT_ROOT=/workspace/data/checkpoints/steerer-ucf-training/steerer-qnrf-imagenet-20260806-t1-e
CHECKPOINT_MANIFEST=$CHECKPOINT_ROOT/checkpoint-manifest.json
TRAINING_PROFILE=$WT/configs/training/steerer_ucf_qnrf_imagenet.home5090.json
PROFILE=$WT/configs/evaluation/steerer_b_ucf_qnrf_final_test.json
OUTPUT=/workspace/data/results/steerer-b-final-test334/steerer-b-final-test334-20260810
```

Expected checkpoint identities:

| Role | Epoch | File | SHA-256 |
|---|---:|---|---|
| primary-best-val-mae | 625 | `best-mae.pth` | `e41df787f546a5195e078486ec6d1bd62ffcb2e3e77010436f74b65045667ad2` |
| secondary-best-val-rmse | 175 | `best-rmse.pth` | `ab1f65573c7f18f115e711998f4f4fb252e816bc5b9ae3564d5b379f22f981e6` |
| final-reference | 800 | `milestone-800.pth` | `d13316a818ddaa2ed0da680718b234bfd21377934c7ef3b33fa074746c93c8f0` |

## Preflight

```bash
test -z "$(git -C "$WT" status --short)"
test -z "$(git -C "$UPSTREAM" status --short)"
test "$(git -C "$UPSTREAM" rev-parse HEAD)" = 5b1854dbc2d280f2326d67c65515d8baf9083810
$PY -m pytest "$WT/tests/test_steerer_final_test.py" \
  "$WT/tests/test_ucf_qnrf.py" "$WT/tests/test_steerer_adapter.py" -q
$PY -c 'import torch; assert torch.cuda.is_available(); x=torch.ones((64,64),device="cuda"); print(float((x@x).sum()))'
nvidia-smi
df -h /workspace/data
```

Require a clean, unused output path. Never delete a partial output to rerun the
one-time Test. Audit it instead.

## Execute

```bash
mkdir -p "$(dirname "$OUTPUT")"
cd "$WT"
$PY scripts/run_steerer_b_final_test.py \
  --config "$PROFILE" \
  --test-root "$TEST_ROOT" \
  --upstream-dir "$UPSTREAM" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --training-profile "$TRAINING_PROFILE" \
  --output-dir "$OUTPUT" \
  --device cuda \
  2>&1 | tee "$OUTPUT.console.log"
```

The command runs the primary, secondary, and final-reference checkpoints in that
frozen order. A performance target miss is recorded in the role score but does
not suppress the other two measurements.

## Verify

```bash
test -f "$OUTPUT/comparison.json"
test -f "$OUTPUT/b-lane-closed.json"
$PY -c 'import csv,json,sys,pathlib; root=pathlib.Path(sys.argv[1]); c=json.load(open(root/"comparison.json")); assert len(c["results"])==3; [(_ for _ in ()).throw(AssertionError()) if len(list(csv.DictReader(open(root/r/"predictions.csv"))))!=334 else None for r in ("primary-best-val-mae","secondary-best-val-rmse","final-reference")]; print(json.dumps(c,indent=2))' "$OUTPUT"
sha256sum "$OUTPUT/comparison.json" "$OUTPUT/b-lane-closed.json"
```

Keep complete predictions and panels on `/workspace/data`. Publish only the
aggregate metrics and evidence hashes. Every result label is `official Test,
validation-selected`; B is closed after this evaluation.
