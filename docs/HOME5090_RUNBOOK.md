# Home RTX 5090 runbook

## Scope

`home5090_docker` is the default backend for new DroneAI execution. Colab and
Google Drive remain available for historical reproduction and fallback. This
runbook stops at repository tests and a synthetic CUDA Stage 0 smoke. It does
not authorize a dataset transfer, pretrained weight use, or long training.

Fixed infrastructure:

- SSH alias: `home5090-pop`
- host source bind: `/home/lucita/crowd-counting-lab`
- container source root: `/workspace`
- host data bind: `/mnt/crowd-data/CrowdCounting`
- container data root: `/workspace/data`
- container: `crowd-jupyter`
- image: `nvcr.io/nvidia/pytorch:25.10-py3`
- verified image digest: `nvcr.io/nvidia/pytorch@sha256:42263b2424fc237b34c4fc4a91c30d603c57eed36e37d31ff6d9a4f1f801edee`

## Laptop access

Open an SSH shell:

```powershell
ssh home5090-pop
```

Open the Jupyter tunnel in a separate PowerShell window, then browse to the
tokenized local URL printed by Jupyter:

```powershell
ssh -N -L 8888:127.0.0.1:8888 home5090-pop
```

## First safe repository sync

The bind root already contains `data/` and `test/startcodex.txt`. Preserve both.
Do not run `git clean`, `git reset --hard`, or clone with a command that removes
the non-empty directory. Run the following on the Pop!_OS host after the Windows
branch has been pushed:

```bash
cd /home/lucita/crowd-counting-lab
git init
git remote add origin https://github.com/LuciTa81/DroneAI.git
git fetch origin agent/stage3-dm-count-reproduction
git checkout -b agent/stage3-dm-count-reproduction --track origin/agent/stage3-dm-count-reproduction
grep -qxF '/test/' .git/info/exclude || echo '/test/' >> .git/info/exclude
git status --short --branch
```

On later syncs, stop if `git status --porcelain` prints anything that is not an
understood generated/ignored file. Pull only by fast-forward:

```bash
cd /home/lucita/crowd-counting-lab
git status --short --branch
git pull --ff-only
```

## Harness environment and smoke

The harness venv inherits CUDA PyTorch from the NGC image. It does not install
model-specific packages into the base container:

```bash
docker exec crowd-jupyter bash -lc '
set -eu
cd /workspace
python -m venv --system-site-packages .venvs/harness
. .venvs/harness/bin/activate
python -m pip install -e ".[dev,evaluation]"
pytest -q
python scripts/run_stage0.py --profile configs/runtime/home5090_docker.json
'
```

Review these small artifacts on the SSD:

```text
/workspace/data/results/stage-0/
├── environment.json
├── stage0_config.json
├── score.json
├── score.md
└── run_manifest.json
```

`run_manifest.json` must show the exact Git commit, a clean tree, the container
tag and digest, Python/PyTorch/CUDA/GPU data, seeds, `research_only`,
`production_approved=false`, and recomputed SHA-256 entries for the other four
artifacts.

## Common evaluation fixture

After the reviewed harness branch is clean and synchronized, run a new CUDA
fixture directory. Replace `<commit>` with `git rev-parse --short HEAD` from the
same repository checkout:

```bash
docker exec crowd-jupyter bash -lc '
set -eu
cd /workspace
test ! -e /workspace/data/results/common-harness-fixture-<commit>
. .venvs/harness/bin/activate
python scripts/run_evaluation_fixture.py \
  --config configs/evaluation/fixture_density.json \
  --output-dir /workspace/data/results/common-harness-fixture-<commit> \
  --device cuda
'
```

Inspect the following paths before approving the next model:

```text
<run-dir>/score.md
<run-dir>/score.json
<run-dir>/summary.md
<run-dir>/predictions.csv
<run-dir>/sample-manifest.json
<run-dir>/native-output-metadata.json
<run-dir>/rights-decision.json
<run-dir>/selection_manifest.json
<run-dir>/environment-summary.json
<run-dir>/figures/*.png
```

This synthetic fixture proves only the common harness and real CUDA execution.
It is not a model-quality, dataset-rights, commercial-use, or production claim.

The first verified CUDA fixture used code commit `a1620a9` on 2026-07-15. The
container suite passed 200 tests, and the fixture produced
`PASS_RESEARCH_ONLY`, 100/100, 36 prediction rows, 12 panels, and a 231630-byte
review bundle. `environment-summary.json` recorded PyTorch
`2.9.0a0+145a3a7bda.nv25.10`, CUDA 13.0, and the NVIDIA GeForce RTX 5090 with a
clean Git worktree.

## DM-Count environment isolation

Do this only when the compatibility-smoke stage is approved. The venv inherits
Torch from the pinned image and installs the exact overlay separately:

```bash
cd /workspace
python -m venv --system-site-packages .venvs/dm-count
. .venvs/dm-count/bin/activate
python -m pip install -e ".[dev,evaluation]"
python -m pip install -r requirements/models/dm_count-home5090.lock.txt
python -m pip freeze > /workspace/data/results/dm-count-environment.freeze.txt
```

The upstream requirements pin Torch 1.2, torchvision 0.4, and Pillow 6.2.2.
Those versions are not installed into the RTX 5090 environment. Compatibility
must instead be demonstrated against the pinned upstream commit with a
synthetic forward/backward smoke before any data is opened. `h5py` and `pandas`
are not imported by the pinned ShanghaiTech training path and must not be added
to this venv unless a reviewed adapter requires them.

## DM-Count UCF-QNRF frozen-checkpoint smoke

This lane uses the project owner's approved policy: the Apache-2.0 label on the
recorded Kaggle mirror is accepted as dataset permission, while the uploader's
non-owner statement remains visible in the manifest. The action gate permits
asset download and frozen-checkpoint evaluation, not training or deployment.

Create a new pinned upstream checkout and copy the official checkpoint into the
SSD checkpoint area. Do not replace an existing path:

```bash
cd /workspace
test ! -e /workspace/upstreams/DM-Count
git clone https://github.com/cvlab-stonybrook/DM-Count.git /workspace/upstreams/DM-Count
git -C /workspace/upstreams/DM-Count checkout --detach cc5f2132e0d1328909f31b6d665b8e0b15c30467
test "$(git -C /workspace/upstreams/DM-Count rev-parse HEAD)" = "cc5f2132e0d1328909f31b6d665b8e0b15c30467"
mkdir -p /workspace/data/checkpoints/dm-count/cc5f2132
test ! -e /workspace/data/checkpoints/dm-count/cc5f2132/model_qnrf.pth
cp /workspace/upstreams/DM-Count/pretrained_models/model_qnrf.pth \
  /workspace/data/checkpoints/dm-count/cc5f2132/model_qnrf.pth
sha256sum /workspace/data/checkpoints/dm-count/cc5f2132/model_qnrf.pth
```

Generate the action-scoped decision, then run the dataset-free model/checkpoint
compatibility smoke. Replace `<commit>` with the current short DroneAI commit
and `<checkpoint-sha256>` with the preceding `sha256sum` value:

```bash
. /workspace/.venvs/dm-count/bin/activate
python scripts/run_stage3c.py \
  --manifest configs/candidates/dm_count_ucf_qnrf.candidate.json \
  --output-dir /workspace/data/results/stage-3c/dm-count-ucf-qnrf-<commit>
python scripts/run_dm_count_compatibility_smoke.py \
  --upstream-dir /workspace/upstreams/DM-Count \
  --checkpoint /workspace/data/checkpoints/dm-count/cc5f2132/model_qnrf.pth \
  --checkpoint-sha256 <checkpoint-sha256> \
  --output-dir /workspace/data/results/dm-count/compatibility-<commit> \
  --device cuda
```

Only after that smoke passes, download the accepted public mirror into a new
versioned directory. The `.part` file is retained for resumable transfer:

```bash
mkdir -p /workspace/data/datasets/ucf-qnrf-kaggle-apache/archive
test ! -e /workspace/data/datasets/ucf-qnrf-kaggle-apache/archive/ucf-qnrf.zip
curl -fL --retry 5 -C - \
  -o /workspace/data/datasets/ucf-qnrf-kaggle-apache/archive/ucf-qnrf.zip.part \
  https://www.kaggle.com/api/v1/datasets/download/faihajalamtopu/ucf-qnrf
mv /workspace/data/datasets/ucf-qnrf-kaggle-apache/archive/ucf-qnrf.zip.part \
  /workspace/data/datasets/ucf-qnrf-kaggle-apache/archive/ucf-qnrf.zip
sha256sum /workspace/data/datasets/ucf-qnrf-kaggle-apache/archive/ucf-qnrf.zip
mkdir -p /workspace/data/datasets/ucf-qnrf-kaggle-apache/raw
unzip -q -n /workspace/data/datasets/ucf-qnrf-kaggle-apache/archive/ucf-qnrf.zip \
  -d /workspace/data/datasets/ucf-qnrf-kaggle-apache/raw
find /workspace/data/datasets/ucf-qnrf-kaggle-apache/raw -type d -name Train -print
```

Use the printed `Train` directory exactly. The runner has no test-root argument
and verifies the pinned 1,081/120 upstream train/validation lists before it
selects the 36 validation images:

```bash
python scripts/run_dm_count_ucf_qnrf_smoke.py \
  --config configs/evaluation/dm_count_ucf_qnrf_smoke.json \
  --train-root <printed-Train-directory> \
  --upstream-dir /workspace/upstreams/DM-Count \
  --train-list /workspace/upstreams/DM-Count/preprocess/qnrf_train.txt \
  --validation-list /workspace/upstreams/DM-Count/preprocess/qnrf_val.txt \
  --checkpoint /workspace/data/checkpoints/dm-count/cc5f2132/model_qnrf.pth \
  --checkpoint-sha256 <checkpoint-sha256> \
  --rights-decision /workspace/data/results/stage-3c/dm-count-ucf-qnrf-<commit>/rights-decision.json \
  --rights-manifest configs/candidates/dm_count_ucf_qnrf.candidate.json \
  --output-dir /workspace/data/results/dm-count/ucf-qnrf-val-smoke-<commit> \
  --device cuda
```

Stop after reviewing `score.md`, `summary.md`, `predictions.csv`,
`split-source-manifest.json`, `model-brief.md`, and the 12 panels. Do not open
the official test split or start training in this checkpoint.

## Dataset transfer gate

Do not execute this section until Stage 3C records dataset rights, intended
scope, a destination ID, expected file count, and a source SHA-256 manifest.
After approval, transfer only into the dedicated data bind and retain partial
files for restart:

```powershell
rsync -a --info=progress2 --partial --checksum <approved-local-dataset-path>/ home5090-pop:/mnt/crowd-data/CrowdCounting/datasets/<versioned-dataset-id>/
```

Recompute the inventory and SHA-256 values on the server before training. A
successful copy is not itself a rights approval.

## Future train and resume reference

Do not run this command until Stage 3C and the short DM-Count compatibility
smoke are approved. The existing runner resumes automatically from
`resume/last_ckpt.tar` when the same command is issued again:

```bash
tmux new -s dmcount-stage3
docker exec -it crowd-jupyter bash
cd /workspace
. .venvs/dm-count/bin/activate
python scripts/run_stage3a_faithful.py \
  --data-dir /workspace/data/datasets/<approved-dataset-id>/ShanghaiTech/part_A \
  --run-dir /workspace/data/results/stage-3/faithful/seed-2026 \
  --work-dir /workspace/data/results/stage-3/work/faithful-seed-2026 \
  --upstream-dir /workspace/upstreams/DM-Count \
  --seed 2026 \
  --max-epoch 1000 \
  --checkpoint-every 10
```

Detach from tmux with `Ctrl-b`, then `d`; reattach with
`tmux attach -t dmcount-stage3`. Tmux survives SSH and desktop logout but not a
PC reboot. After a reboot, start a new tmux session and issue the identical
training command to resume from the persisted checkpoint. At most the epochs
since the last `--checkpoint-every` boundary are repeated.

## Pull small results to the laptop

Stage 0 contains only small JSON/Markdown evidence, so it can be pulled in full:

```powershell
scp -r home5090-pop:/mnt/crowd-data/CrowdCounting/results/stage-0 ./home5090-stage0
```

For model runs, pull only score, manifest, metrics, and logs. Keep datasets and
weights on the SSD, and commit only reviewed small summaries to GitHub.

## Reboot verification

After both the laptop and Pop!_OS server have been shut down and started again,
run this from laptop PowerShell. It is read-only and confirms the network,
services, Docker restart policy, container, repository, and CUDA visibility:

```powershell
ssh home5090-pop "hostname; systemctl is-active tailscaled ssh docker; docker inspect -f '{{.State.Status}} {{.HostConfig.RestartPolicy.Name}}' crowd-jupyter; docker exec crowd-jupyter python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'; cd /home/lucita/crowd-counting-lab; git status --short --branch; git rev-parse HEAD"
```

The expected service lines are `active`, the container line is
`running unless-stopped`, CUDA prints `True` with the RTX 5090 name, and the Git
worktree is clean on the reviewed commit. If that passes, rerun the persisted
foundation evidence without downloading data or starting a model:

```powershell
ssh home5090-pop "docker exec crowd-jupyter bash -lc 'cd /workspace && . .venvs/harness/bin/activate && pytest -q && python scripts/run_stage0.py --profile configs/runtime/home5090_docker.json'"
```

The second command must end with Stage 0 `PASS`, a 100/100 score, and fresh
hash-verified artifacts under `/workspace/data/results/stage-0/`.
