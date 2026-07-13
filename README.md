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

## First experiment

Open `notebooks/00_runtime_probe.ipynb` in Google Colab. It mounts Drive, creates the storage folders, records the runtime environment, and writes a JSON report under `MyDrive/DroneAI/runs/runtime-probe/`.

The first model sequence is:

1. CSRNet pipeline smoke test
2. DM-Count official reproduction
3. P2PNet compatibility and reproduction
4. Drone-view fine-tuning and field evaluation

TensorFlow/Keras ports are kept separate until the official PyTorch baselines are reproduced.

