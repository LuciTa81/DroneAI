# Model brief: steerer-official-ucf-qnrf

- Paper: STEERER: Resolving Scale Variations for Counting and Localization via Selective Inheritance Learning
- Role: UCF-QNRF crowd-counting and localization research baseline
- Family: hybrid density and point localization
- Backbone: HRNet-W48
- Parameter count: not measured
- Input: one RGB UCF-QNRF image with frozen point annotations
- Preprocessing: official long-side 3072 cap, bottom/right zero padding to multiples of 32, and ImageNet normalization
- Coordinate transform: density integrates only the valid resized image region via overlap-weighted mass remapping; padding mass and points are discarded and recorded; remaining points are divided by the geometric resize ratio then clipped to original pixels
- Native output: multi-resolution density maps and merged localization points
- Count derivation: sum of highest-resolution density mass within the valid resized image region
- Zone derivation: fractional integration of count-preserving density over calibrated image zones
- Official protocol: pinned QNRF_final.py, den_factor=100, batch size 1, frozen research checkpoint
- Rights status: RESEARCH_CHECKPOINT_EVALUATION_ONLY
- Code rights: MIT verified at pinned upstream commit
- Dataset rights: Apache-2.0 accepted by project owner with provenance risk recorded
- Checkpoint rights: unverified; internal approval permits frozen research comparison only
- Deployment rights: pending; research comparison is not deployment approval
- Upstream commit: `5b1854dbc2d280f2326d67c65515d8baf9083810`
- Checkpoint path: `/workspace/data/checkpoints/steerer/QNRF_mae_77.8_mse_138.0_85f71763.pth`
- Checkpoint SHA-256: `85f71763595527e419257d459288e4a784766f41687ff0b859dcff65a151bbec`

## Major blocks

- HRNet-W48 multi-resolution backbone
- multi-resolution counting heads
- selective inheritance/upsample module
- Gaussian density/local-maximum point decoder

## Feature scales

- x1
- x4
- x8

## Original losses

- multi-resolution MSE
- selective inheritance routing loss
- uncertainty weighting where configured

## Official reported metrics

- UCF-QNRF MAE 77.8
- UCF-QNRF RMSE 138.0
- UCF-QNRF localization F1 75.6

## Strengths

- multi-scale density counting
- official local-maximum point localization

## Failure modes

- tiny heads
- extreme perspective
- occlusion
- domain shift
- padding artifacts
- threshold-sensitive localization

## Runtime risks

- large high-resolution CUDA activations
- legacy MMCV and upstream API compatibility

## Reviewed code

- /workspace/upstreams/STEERER/configs/QNRF_final.py
- /workspace/upstreams/STEERER/lib/models/build_counter.py
- /workspace/upstreams/STEERER/lib/utils/points_from_den.py
- /workspace/upstreams/STEERER/lib/core/cc_function.py
- /workspace/upstreams/STEERER/lib/datasets/base_dataset.py
- /workspace/upstreams/STEERER/lib/datasets/nwpu.py

Review status: **approved**
