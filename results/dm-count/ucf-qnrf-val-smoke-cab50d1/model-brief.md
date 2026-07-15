# Model brief: dm-count-official-ucf-qnrf

- Paper: Distribution Matching for Crowd Counting (NeurIPS 2020)
- Role: first density-map baseline for common frozen-checkpoint screening
- Family: density
- Backbone: VGG-19 convolutional features
- Parameter count: not measured
- Input: one RGB UCF-QNRF image with frozen point annotations
- Preprocessing: official QNRF short-side 512-2048 resize and ImageNet normalization
- Coordinate transform: native processed density is mass-preserving resized to the original-image /8 grid
- Native output: non-negative density map
- Count derivation: sum of density-map mass
- Zone derivation: fractional integration of density mass over four original-image quadrants
- Official protocol: official pinned QNRF validation list, batch size 1, frozen model_qnrf.pth
- Rights status: PASS_COMMERCIAL_CANDIDATE
- Code rights: MIT verified at pinned upstream commit
- Dataset rights: Apache-2.0 accepted by project owner with provenance risk recorded
- Checkpoint rights: MIT repository scope accepted by project policy; SHA-256 required
- Deployment rights: pending; frozen benchmark approval is not deployment approval
- Upstream commit: `cc5f2132e0d1328909f31b6d665b8e0b15c30467`
- Checkpoint path: `/workspace/data/checkpoints/dm-count/cc5f2132/model_qnrf.pth`
- Checkpoint SHA-256: `16ef954a2cef40c66ee664f69273559553b735d5e2c9f90e2444f3c25dd45e05`

## Major blocks

- VGG-19 E feature extractor with four max-pooling stages
- bilinear 2x feature upsampling
- 512-256-128 convolutional regression head
- 1-channel convolution and ReLU density head

## Feature scales

- processed RGB input
- approximately one-eighth density output

## Original losses

- counting L1
- optimal transport
- total variation

## Official reported metrics

- UCF-QNRF MAE 85.6
- UCF-QNRF RMSE 148.3

## Strengths

- count-preserving density output
- usable heatmap and zone aggregation

## Failure modes

- domain shift
- extreme scale variation
- high-density undercount

## Runtime risks

- large high-resolution activations
- legacy upstream API compatibility

## Reviewed code

- /workspace/upstreams/DM-Count/models.py
- /workspace/upstreams/DM-Count/test.py
- /workspace/upstreams/DM-Count/datasets/crowd.py
- /workspace/upstreams/DM-Count/preprocess_dataset.py
- /workspace/upstreams/DM-Count/preprocess/preprocess_dataset_qnrf.py

Review status: **approved**
