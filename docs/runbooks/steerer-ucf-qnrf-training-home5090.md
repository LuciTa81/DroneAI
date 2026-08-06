# STEERER UCF-QNRF training on home5090

This runbook operates the isolated STEERER training lane on the existing
`crowd-jupyter` container. UCF-QNRF Test remains sealed. The official STEERER
QNRF checkpoint is never loaded; initialization is limited to the approved
ImageNet HRNet-W48 backbone recorded in the training profile.

## 1. Connect and verify the existing runtime

From the Windows notebook:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=10 home5090-pop "uname -s"
ssh home5090-pop "nvidia-smi; docker ps --filter name=crowd-jupyter"
ssh home5090-pop "docker inspect --format '{{.Image}}' crowd-jupyter"
```

The expected OS is `Linux`; the expected GPU is the RTX 5090. Save the last
command's exact `sha256:<64 hex>` image ID as `CONTAINER_IMAGE_DIGEST`. A tag is
not acceptable evidence. Do not create or replace the container.

## 2. Create the isolated remote worktree

Push the clean local implementation branch first. On home5090, preserve the
active `/workspace` checkout and create a linked worktree:

```powershell
git push -u origin agent/steerer-ucf-training-implementation
ssh home5090-pop "git -C /home/lucita/crowd-counting-lab fetch origin agent/steerer-ucf-training-implementation"
ssh home5090-pop "docker exec crowd-jupyter git -C /workspace worktree add /workspace/.worktrees/steerer-ucf-training origin/agent/steerer-ucf-training-implementation"
```

Fetch runs on the Pop!_OS host because its verified GitHub SSH key is not mounted
inside the container. Checkout still runs inside the container so the linked
worktree records `/workspace` paths.

If that worktree already exists, inspect it instead of deleting or resetting it.

## 3. Verify environment, upstream, backbone, and storage

Run inside the container:

```bash
WT=/workspace/.worktrees/steerer-ucf-training
PY=/workspace/.venvs/steerer/bin/python
UPSTREAM=/workspace/upstreams/STEERER
BACKBONE=/workspace/data/checkpoints/backbones/hrnetv2_w48_imagenet_pretrained.pth
BACKBONE_SHA=0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8

cd "$WT"
git status --short --branch
$PY scripts/setup_steerer_home5090.py --verify-only
git -C "$UPSTREAM" rev-parse HEAD
git -C "$UPSTREAM" status --short
sha256sum "$BACKBONE"
nvidia-smi
df -h /workspace/data
$PY -c 'import torch; assert torch.cuda.is_available(); x=torch.ones((64,64), device="cuda"); print(float((x@x).sum()))'
```

The pinned upstream commit is
`5b1854dbc2d280f2326d67c65515d8baf9083810`. The backbone provenance URL and
source URL are frozen in
`configs/training/steerer_ucf_qnrf_imagenet.home5090.json`; recompute its
SHA-256 rather than trusting its filename.

## 4. Audit and prepare UCF-QNRF Train only

Expected raw input:
`/workspace/data/datasets/ucf-qnrf-kaggle-apache/raw/UCF-QNRF_ECCV18/Train`.
The preparation CLI
has no Test or dataset-root argument.

```bash
WT=/workspace/.worktrees/steerer-ucf-training
PY=/workspace/.venvs/steerer/bin/python
RAW_TRAIN=/workspace/data/datasets/ucf-qnrf-kaggle-apache/raw/UCF-QNRF_ECCV18/Train
PROCESSED=/workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1

cd "$WT"
$PY scripts/prepare_steerer_ucf_training.py \
  --config configs/training/steerer_ucf_qnrf_imagenet.home5090.json \
  --train-root "$RAW_TRAIN" \
  --output-root "$PROCESSED"
sha256sum "$PROCESSED/train.txt" "$PROCESSED/val.txt" \
  "$PROCESSED/manifests/output-inventory.jsonl"
```

Preparation must report 1,201 official Train samples split into 961 train and
240 validation samples with zero overlap. It refuses a non-empty destination;
never delete an existing prepared dataset to make the command pass.

## 5. T0: one optimizer update

T0 is a compatibility and checkpoint round-trip test, not an accuracy result.

```bash
WT=/workspace/.worktrees/steerer-ucf-training
PY=/workspace/.venvs/steerer/bin/python
UPSTREAM=/workspace/upstreams/STEERER
PROCESSED=/workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1
BACKBONE=/workspace/data/checkpoints/backbones/hrnetv2_w48_imagenet_pretrained.pth
BACKBONE_SHA=0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8
RUN_ID=steerer-qnrf-imagenet-20260806-t0-b
CONTAINER_IMAGE_DIGEST=sha256:90dfcacf0d65fec4357a37db72624b42febf8d3d9056924cdec3499237ed2691

cd "$WT"
$PY scripts/run_steerer_ucf_training.py \
  --config configs/training/steerer_ucf_qnrf_imagenet.home5090.json \
  --stage T0 --run-id "$RUN_ID" --processed-root "$PROCESSED" \
  --upstream-dir "$UPSTREAM" --backbone "$BACKBONE" \
  --backbone-sha256 "$BACKBONE_SHA" \
  --container-image-digest "$CONTAINER_IMAGE_DIGEST" --device cuda:0 \
  | tee "/workspace/data/results/steerer-ucf-training/$RUN_ID/run-result.jsonl"
```

Require `optimizer_steps=1`, finite loss, and `checkpoint_round_trip=true`.
Review the checkpoint manifest before any next stage:

```bash
cat "/workspace/data/checkpoints/steerer-ucf-training/$RUN_ID/checkpoint-manifest.json"
cat "/workspace/data/results/steerer-ucf-training/$RUN_ID/run-result.jsonl"
```

Score only after the runner has produced authoritative T0 metrics evidence. Use
the content-addressed paths printed by the runner, not the mutable convenience
`environment.json` name:

```bash
RESULT_JSON=/workspace/data/results/steerer-ucf-training/$RUN_ID/run-result.jsonl
METRICS_PATH=$($PY -c 'import json,sys; result=json.loads(open(sys.argv[1]).read().splitlines()[-1]); print(result["metrics_path"])' "$RESULT_JSON")
ENVIRONMENT_PATH=$($PY -c 'import json,sys; result=json.loads(open(sys.argv[1]).read().splitlines()[-1]); print(result["environment_path"])' "$RESULT_JSON")
$PY scripts/score_steerer_ucf_training.py \
  --stage T0 --run-id "$RUN_ID" \
  --profile "$WT/configs/training/steerer_ucf_qnrf_imagenet.home5090.json" \
  --project-repo "$WT" --upstream-dir "$UPSTREAM" \
  --processed-root "$PROCESSED" --backbone "$BACKBONE" \
  --checkpoint "/workspace/data/checkpoints/steerer-ucf-training/$RUN_ID/last.pth" \
  --checkpoint-manifest "/workspace/data/checkpoints/steerer-ucf-training/$RUN_ID/checkpoint-manifest.json" \
  --environment "$ENVIRONMENT_PATH" --metrics "$METRICS_PATH" \
  --output-dir "/workspace/data/results/steerer-ucf-training/$RUN_ID/score-bundle"
cat "/workspace/data/results/steerer-ucf-training/$RUN_ID/score-bundle/score.json"
```

Stop if scoring reports `BLOCKED` or if required evidence is absent.

## 6. T1: exactly one epoch and 240-image validation

T1 starts a clean one-epoch run from the same ImageNet backbone; it does not
resume the T0 probe.

Run ID `steerer-qnrf-imagenet-20260806-t1-b` is reserved for the preserved
failed integration attempt that stopped before validation evidence and did not
promote a checkpoint. Do not reuse it.

Run ID `steerer-qnrf-imagenet-20260806-t1-c` is reserved for the preserved
T1-c attempt interrupted before promotion after dense localization matching
caused excessive validation memory use. Do not reuse it.

Run ID `steerer-qnrf-imagenet-20260806-t1-d` is reserved for the preserved
T1-d attempt interrupted before promotion after the remaining dense
multiscale-point merge caused excessive validation time. Do not reuse it.

```bash
RUN_ID=steerer-qnrf-imagenet-20260806-t1-e
$PY scripts/run_steerer_ucf_training.py \
  --config configs/training/steerer_ucf_qnrf_imagenet.home5090.json \
  --stage T1 --run-id "$RUN_ID" --processed-root "$PROCESSED" \
  --upstream-dir "$UPSTREAM" --backbone "$BACKBONE" \
  --backbone-sha256 "$BACKBONE_SHA" \
  --container-image-digest "$CONTAINER_IMAGE_DIGEST" --device cuda:0 \
  | tee "/workspace/data/results/steerer-ucf-training/$RUN_ID/run-result.jsonl"
```

Require epoch 1, all 240 validation samples, finite metrics, and a reloadable
checkpoint. Re-read `result["metrics_path"]` and `result["environment_path"]`
from the T1 run result and invoke the scoring command with `--stage T1`. Inspect
`score.json`, `metrics.json`, `environment.json`, and `lineage.json` before
proceeding.

## 7. Status, interruption, and reboot recovery

Use the existing `tmux crowd` session for a foreground run:

```bash
tmux attach -t crowd
```

Detach with `Ctrl-b d`. To inspect without attaching:

```bash
tmux capture-pane -pt crowd -S -120
nvidia-smi
```

Interrupt only the foreground STEERER command with `Ctrl-c`; do not kill the
container or unrelated GPU processes. After a reboot, verify the hash of
`last.pth` against `checkpoint-manifest.json`. Resume is permitted only for a
later explicitly approved stage using the verified project checkpoint and its
recorded lineage.

## 8. Approval stops

Do not run T5 before user approval. T5 requires both `--resume <verified-last.pth>`
and `--approved-stage T5`. T50 likewise requires a separate approval and
`--approved-stage T50`. These commands are intentionally not provided as
copy-paste execution blocks.

UCF-QNRF Test remains sealed for T0, T1, T5, and T50. Full 1,201-image retraining
and Test evaluation are separate authorization gates.

## 9. Pull small result evidence only

From a shell with `rsync` (for example WSL), copy JSON, Markdown, and curated PNG
files only. Keep checkpoints, optimizer state, raw density arrays, and full logs
on `/workspace/data`:

```bash
rsync -avm --include='*/' --include='*.json' --include='*.md' \
  --include='*.png' --exclude='*' \
  home5090-pop:/workspace/data/results/steerer-ucf-training/$RUN_ID/ \
  ./results/steerer-training/$RUN_ID/
```

Recompute SHA-256 for every copied artifact before committing. Do not put `.pth`
files or any individual artifact above 25 MiB in Git.
