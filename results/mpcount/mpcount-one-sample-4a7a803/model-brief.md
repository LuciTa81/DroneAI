# Model brief: mpcount-official-sta-to-ucf-qnrf

- Paper: Single Domain Generalization for Crowd Counting (CVPR 2024)
- Role: domain-generalized density candidate for CCTV zone screening
- Family: density and single-domain generalization
- Backbone: VGG16-BN encoder with memory-enhanced density decoder
- Parameter count: 33233241
- Input: one RGB image padded symmetrically to a multiple of 16
- Preprocessing: official RGB to tensor, mean/std 0.5 normalization, non-overlapping patch inference at 3584 pixels
- Coordinate transform: crop padding, divide density by 1000, preserve mass on the original-image /8 grid
- Native output: non-negative full-resolution density map
- Count derivation: sum of cropped density-map mass divided by 1000
- Zone derivation: fractional density integration inside calibrated CCTV zones
- Official protocol: deterministic ShanghaiTech-A source checkpoint evaluated on UCF-QNRF, unit size 16 and log parameter 1000
- Rights status: PASS_COMMERCIAL_CANDIDATE
- Code rights: Apache-2.0 verified at the pinned upstream commit
- Dataset rights: Kaggle Apache-2.0 accepted by project owner; uploader provenance risk recorded
- Checkpoint rights: Apache-2.0 accepted by project policy for the official deterministic checkpoint
- Deployment rights: pending; evaluation approval is not deployment approval
- Upstream commit: `6eb06772bcf7dfb771c43a14d67146fce767f103`
- Checkpoint path: `/workspace/data/checkpoints/mpcount/mpcount_sta_deterministic_6eb06772.pth`
- Checkpoint SHA-256: `4a0ba849ace0ef2bc87364ec8551de4a3e9dca74fad27764c9b948ebf7928eb0`

## Major blocks

- VGG16-BN multi-stage convolutional encoder
- multi-scale decoder and density feature fusion
- 1024-slot density memory bank
- patch-wise density classification mask
- deterministic learned upsampling and density head

## Feature scales

- one-quarter
- one-eighth
- one-sixteenth
- full density

## Original losses

- density regression loss
- memory attention consistency loss
- content error-mask loss
- patch-wise density classification loss

## Official reported metrics

- ShanghaiTech-A to UCF-QNRF MAE 112.8
- ShanghaiTech-A to UCF-QNRF RMSE 193.8

## Strengths

- native density heatmap and zone aggregation
- explicit single-source domain-generalization design

## Failure modes

- extreme perspective or occlusion
- festival/CCTV shift outside source-domain coverage
- patch boundary artifacts
- small negative numerical outputs if runtime compatibility changes

## Runtime risks

- full-resolution density activations
- official Torch 2.0.1 code running on Torch 2.9

## Reviewed code

- /workspace/upstreams/MPCount/README.md
- /workspace/upstreams/MPCount/LICENSE
- /workspace/upstreams/MPCount/inference.py
- /workspace/upstreams/MPCount/models/models.py
- /workspace/upstreams/MPCount/utils/misc.py

Review status: **approved**
