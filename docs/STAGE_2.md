# Stage 2 — CSRNet pipeline smoke

## Purpose

Stage 2 does not claim that CSRNet is the production winner or reproduce a
paper benchmark. It proves that the common counting pipeline is internally
correct before DM-Count and P2PNet are introduced:

1. point annotations become density maps without changing total count;
2. the model's output resolution is understood and evaluated correctly;
3. the published CSRNet layer contract runs on current PyTorch;
4. a fixed tiny synthetic set can be learned;
5. seeds, metrics, predictions and model hashes are persisted.

## Model provenance

The architecture follows the CVPR 2018 paper: a VGG-16-style frontend through
`conv4_3`, three max-pools, six dilation-2 backend convolutions and a one-channel
output at stride 8. The authors' reference repository is pinned as behavioral
evidence, but it targets Python 2.7 and PyTorch 0.4 and has no explicit license
file. DroneAI therefore contains an independent implementation of the
published layer contract and does not vendor reference source or weights.

Important compatibility details:

- the output has no extra ReLU because the published/reference architecture
  ends with a plain 1x1 convolution;
- the target density map is count-preserving at stride 8;
- Stage 2 uses random initialization so downloading VGG weights is not required;
- paper-result reproduction belongs to Stage 3, not this smoke gate.

## 100-point gate

| Group | Points | Evidence |
|---|---:|---|
| Pipeline correctness | 30 | layer contract, output shape and finite tensors |
| Count preservation | 25 | point density and downsampling sums |
| Tiny-set overfit | 20 | loss reduction and count MAE |
| Determinism | 10 | identical initial state hash under a fixed seed |
| Artifacts | 15 | config, metrics, predictions and trained-state hash |

Pass threshold: **80/100**, with all correctness and artifact blockers passing.

## Synthetic smoke data

Four 64x64 RGB tensors contain deterministic spatial signals for 1, 2, 3 and
4 points. Their targets are 8x8 density maps. This deliberately tests plumbing,
not real-world generalization. No Stage 1 test frame is used for training, so
the UP-COUNT evaluation inventory remains untouched.

## Review checkpoint

Inspect these before Stage 3:

1. `score.md` and every blocker.
2. `metrics.json`: initial/final loss, loss reduction and four predicted counts.
3. `stage2_config.json`: seed, steps, learning rate and device.
4. `configs/models/csrnet.json`: paper/reference provenance and license status.
5. `src/droneai/csrnet.py`: the complete model code.
