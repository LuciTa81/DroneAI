# Model brief: csrnet-independent-shha-to-ucf-qnrf

- Paper: CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes (CVPR 2018)
- Role: classical density-map baseline for fixed-CCTV zone screening
- Family: density regression with dilated convolutions
- Backbone: VGG-16 frontend through conv4_3
- Parameter count: 16263489
- Input: one native-resolution RGB image
- Preprocessing: official ImageNet normalization: mean 0.485/0.456/0.406 and standard deviation 0.229/0.224/0.225; no source resize
- Coordinate transform: mass-preserving alignment from native stride-8 output to the original-image /8 evaluation grid
- Native output: raw signed stride-8 density plus audited non-negative operational density
- Count derivation: sum of max(raw_density, 0); raw signed sum retained for audit
- Zone derivation: fractional density integration inside calibrated CCTV zones
- Official protocol: ShanghaiTech Part A checkpoint evaluated cross-domain on one frozen UCF-QNRF validation sample
- Rights status: PASS_COMMERCIAL_CANDIDATE
- Code rights: project-owned independent implementation for commercial R&D
- Dataset rights: Kaggle Apache-2.0 accepted by project owner; uploader provenance risk recorded
- Checkpoint rights: research comparison authorized; commercial checkpoint reuse unverified
- Deployment rights: pending; evaluation approval is not deployment approval
- Upstream commit: `ed29d895989c188cb913a9503721271c6cf1ab1f`
- Checkpoint path: `/workspace/data/checkpoints/csrnet/safe/csrnet-official-shha-mae66.4-state-dict.pth`
- Checkpoint SHA-256: `7093f29f1469fb4e3a80781fda03665848689ada59c5a13d1b49940b230755e6`

## Major blocks

- VGG-style frontend with three max-pooling stages
- six-layer dilation-2 convolutional backend
- one-channel 1x1 density output

## Feature scales

- full-resolution RGB input
- one-eighth density output

## Original losses

- Euclidean density-map regression loss

## Official reported metrics

- ShanghaiTech Part A checkpoint best MAE 65.9016 in the saved checkpoint
- CSRNet paper ShanghaiTech Part A MAE 68.2 and MSE 115.0

## Strengths

- direct heatmap and zone aggregation
- simple well-understood baseline architecture

## Failure modes

- cross-domain undercount or overcount
- negative raw density values because the published head has no ReLU
- perspective and scale shift in festival CCTV

## Runtime risks

- native 4K input increases early-layer activation memory
- 2018 checkpoint evaluated under a modern Torch/CUDA runtime

## Reviewed code

- /workspace/src/droneai/csrnet.py
- /workspace/upstreams/CSRNet-pytorch/README.md
- /workspace/upstreams/CSRNet-pytorch/model.py

Review status: **approved**
