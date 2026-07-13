# DroneAI

Reproducible crowd-counting experiments for CCTV and drone safety monitoring.

## Project goals

1. Reproduce published crowd-counting baselines with pinned code and environments.
2. Measure generalization to festival CCTV and drone imagery.
3. Evaluate whether count, density, latency, and failure behavior are useful for operators.

## Storage policy

- GitHub stores code, notebooks, configuration, small reports, and dataset manifests.
- Google Drive stores datasets, checkpoints, and large run artifacts.
- Raw videos, model weights, credentials, and mounted Drive contents must never be committed.

Default Drive root:

```text
MyDrive/DroneAI/
├── datasets/
├── checkpoints/
└── runs/
```

## Stage gates

Every stage produces both `score.json` and `score.md`. A high average cannot
hide a failed integrity or safety blocker.

| Stage | Question | Gate |
|---|---|---:|
| 0 | Can Colab, Git and Drive produce a reproducible run? | 85 |
| 1 | Is the selected dataset legal, intact and leakage-free? | 85 |
| 2 | Does the CSRNet smoke pipeline preserve counts and learn a tiny subset? | 80 |
| 3 | Does each official model reproduce its published result? | 85 |
| 4 | Does the model generalize to drone and held-out camera conditions? | 80 |
| 5 | Does it pass a controlled field trial? | 85 |
| 6 | Does it help operators in shadow mode? | 85 |

See `docs/ROADMAP.md` for the complete review sequence.

## Current experiment

Stage 0 passed on Colab with a Tesla T4. Stage 1 now gates dataset rights,
checksums, point annotations and split leakage before any model is trained.
See `docs/STAGE_1.md` for the dataset decision and normalized inventory contract.

The first model sequence is:

1. CSRNet pipeline smoke test
2. DM-Count official reproduction
3. P2PNet compatibility and reproduction
4. Drone-view fine-tuning and field evaluation

TensorFlow/Keras ports are kept separate until the official PyTorch baselines are reproduced.

Public datasets are research benchmarks, not automatic production-training
assets. UP-COUNT is non-commercial, DroneCrowd currently lacks explicit license
terms in its official repository, and production fine-tuning requires data with
project-owned or separately granted rights.
