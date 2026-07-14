# DroneAI

Reproducible crowd-counting experiments for CCTV and drone safety monitoring.

## Project goals

1. Reproduce published crowd-counting baselines with pinned code and environments.
2. Measure generalization to festival CCTV and drone imagery.
3. Evaluate whether count, density, latency, and failure behavior are useful for operators.

## Runtime and storage policy

- GitHub stores code, notebooks, configuration, small reports, and dataset manifests.
- The home RTX 5090 SSD is the default for new datasets, checkpoints, and large run artifacts.
- Google Drive remains the Colab fallback and preserves prior experiment provenance.
- Raw videos, model weights, credentials, and mounted Drive contents must never be committed.

Default container paths:

```text
/workspace/
├── source and harness
└── data/
    ├── datasets/
    ├── checkpoints/
    └── results/
```

Legacy/fallback Drive root:

```text
MyDrive/DroneAI/
├── datasets/
├── checkpoints/
└── runs/
```

See `docs/HOME5090_RUNBOOK.md` for safe SSH, sync, smoke, resume, and result-pull commands.

## Stage gates

Every stage produces both `score.json` and `score.md`. A high average cannot
hide a failed integrity or safety blocker.

| Stage | Question | Gate |
|---|---|---:|
| 0 | Can the selected backend, Git and persistent storage produce a reproducible CUDA run? | 85 |
| 1 | Is the selected dataset legal, intact and leakage-free? | 85 |
| 2 | Does the CSRNet smoke pipeline preserve counts and learn a tiny subset? | 80 |
| 3 | Does each official model reproduce its published result? | 85 |
| 3C | Are code, dataset, pretrained weight, derived weight and deployment rights safe for the requested scope? | 80 |
| 4 | Does the model generalize to drone and held-out camera conditions? | 80 |
| 5 | Does it pass a controlled field trial? | 85 |
| 6 | Does it help operators in shadow mode? | 85 |

See `docs/ROADMAP.md` for the complete review sequence.

## Current experiment

Stages 0, 1 and 2 passed on Colab. Stage 3 now reproduces the official
DM-Count implementation on ShanghaiTech Part A in two lanes: a faithful
paper-number check and a test-isolated clean protocol. See `docs/STAGE_3.md`
for the scored contract.

The Stage 3 official-checkpoint preflight passed on 2026-07-13 with MAE 60.03
and RMSE 96.00 on all 182 Part A test images. Stage 3 itself remains in progress
until the three-seed training and clean-lane gates are complete.

The seed-2026 faithful-training pilot reached epoch 140 on 2026-07-14. Its
test-selected checkpoint recorded MAE 71.28 and RMSE 113.21 on all 182 images,
and the separate pilot harness passed 100/100. The full 1000-epoch run is
deferred to a suitable GPU server; this pilot is not a Stage 3 pass.

The first model sequence is:

1. CSRNet pipeline smoke test
2. DM-Count official reproduction
3. P2PNet compatibility and reproduction
4. Drone-view fine-tuning and field evaluation

TensorFlow/Keras ports are kept separate until the official PyTorch baselines are reproduced.

New execution defaults to the `home5090_docker` profile. Colab notebooks and
their recorded Drive paths remain unchanged as legacy reproduction/fallback
assets. A home5090 foundation smoke is always `research_only` and never grants
commercial or production approval.

Public datasets are research benchmarks, not automatic production-training
assets. Rights-sensitive gates never emit an unscoped `PASS`:

- `PASS_RESEARCH_ONLY`: the declared research lane is permitted, not commercial use;
- `PASS_COMMERCIAL_CANDIDATE`: no explicit prohibition is recorded but required permission is absent or unverified;
- `PRODUCTION_APPROVED`: every required code/data/weight/deployment component has verified commercial evidence;
- `BLOCKED`: an explicit prohibition, contradiction, missing identity, or integrity blocker exists.

Candidate status permits cataloging and rights due diligence. A synthetic-only
compatibility smoke is added only when the code component itself is already
verified for that action; it never opens ambiguous data or weights. Candidate
status does not permit ambiguous asset download, training,
weight reuse, or deployment. UP-COUNT is non-commercial; DroneCrowd currently
lacks explicit license terms in its official repository and remains candidate
inventory only. Production fine-tuning requires project-owned or separately
granted rights. This harness is an engineering control, not legal advice.
