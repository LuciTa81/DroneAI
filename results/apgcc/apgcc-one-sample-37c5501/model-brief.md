# Model brief: apgcc-official-shha-to-ucf-qnrf

- Paper: APGCC: Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance
- Role: ShanghaiTech-A to UCF-QNRF point compatibility comparison
- Family: point-based APGCC
- Backbone: VGG16-BN
- Parameter count: 17750480
- Input: one native-resolution RGB validation image
- Preprocessing: native resolution with ImageNet normalization
- Coordinate transform: native APGCC (x,y) pixels; discard out-of-bounds points
- Native output: pred_logits, pred_points, and offsets
- Count derivation: number of confidence-thresholded in-bounds point predictions
- Zone derivation: count accepted points inside each calibrated image zone
- Official protocol: SHHA checkpoint, IFI decoder, confidence threshold 0.5
- Rights status: PASS_COMMERCIAL_CANDIDATE
- Code rights: official code is MIT
- Dataset rights: Kaggle Apache-2.0 accepted with provenance caveat
- Checkpoint rights: research comparison authorized; commercial checkpoint reuse remains unverified
- Deployment rights: derived weights and deployment remain pending
- Upstream commit: `e3e997bf592a70fd34233a432a974200d8c0c847`
- Checkpoint path: `/workspace/data/checkpoints/apgcc/SHHA_best.pth`
- Checkpoint SHA-256: `cd9aa0f65882c81bb753a4ea8c821e07573b2ae4a31fffd5595fa0a43925cf90`

## Major blocks

- VGG16-BN convolutional backbone
- ASPP feature aggregation
- Implicit Feature Interpolation decoder
- confidence and point-offset heads

## Feature scales

- stride-8 anchor grid
- stride-16 VGG features

## Original losses

- person/background classification loss
- L2 point regression loss
- auxiliary positive and negative point guidance losses

## Official reported metrics

- SHHA count MAE/RMSE reported by the paper
- SHHA localization precision/recall/F1 reported by the paper

## Strengths

- native point localization
- auxiliary point guidance
- IFI decoder

## Failure modes

- ShanghaiTech-to-UCF domain shift
- tiny or occluded people
- confidence-threshold sensitivity
- out-of-frame point offsets

## Runtime risks

- native high-resolution proposal count
- legacy Python 3.8 upstream assumptions

## Reviewed code

- /workspace/upstreams/APGCC/README.md
- /workspace/upstreams/APGCC/LICENSE
- /workspace/upstreams/APGCC/apgcc/configs/SHHA_test.yml
- /workspace/upstreams/APGCC/apgcc/models/APGCC.py
- /workspace/upstreams/APGCC/apgcc/models/Decoder.py
- /workspace/upstreams/APGCC/apgcc/models/Encoder.py
- /workspace/upstreams/APGCC/apgcc/models/backbones/vgg.py
- /workspace/upstreams/APGCC/apgcc/engine.py

Review status: **approved**
