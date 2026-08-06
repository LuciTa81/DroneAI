# T5: STEERER UCF-QNRF T5 training

- Status: **PASS_COMMERCIAL_CANDIDATE**
- Decision scope: **commercial_candidate**
- Score: **100.0/100**
- Pass threshold: **80.0**
- Generated at: `2026-08-06T19:22:41.401425+00:00`

| Result | Category | Check | Points | Blocker | Observed |
|---|---|---|---:|:---:|---|
| PASS | provenance | Pinned, clean upstream | 15.0/15.0 | Y | clean=True; commit=5b1854dbc2d280f2326d67c65515d8baf9083810 |
| PASS | provenance | ImageNet-backbone-only initialization | 10.0/10.0 | Y | False |
| PASS | split | Train and validation do not overlap | 10.0/10.0 | Y | overlap=0 |
| PASS | split | Approved split is complete | 5.0/5.0 | Y | complete=True; population=1201; train=961; validation=240 |
| PASS | split | Official Test remains sealed | 5.0/5.0 | Y | False |
| PASS | stability | All required values are finite | 12.0/12.0 | Y | finite=True; metrics_finite=True |
| PASS | stability | Checkpoint round trip succeeds | 8.0/8.0 | Y | True |
| PASS | validation | Required stage validation evidence | 20.0/20.0 |  | all 17 required T1 validation metrics recorded |
| PASS | artifacts | All referenced artifacts hash-verify from disk | 15.0/15.0 | Y | verified: /workspace/data/checkpoints/backbones/hrnetv2_w48_imagenet_pretrained.pth; sha256=0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8; verified: /workspace/upstreams/STEERER/configs/QNRF_final.py; sha256=a1bd95fa9dcb62ced8c248f0716d22f45f23080705d8445e195982f789c1314c; verified: /workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1/manifests/output-inventory.jsonl; sha256=f38af282fc3007f0f204fb94c14b45595cc383207323eb7430c3a6546f41d042; verified: /workspace/data/results/steerer-ucf-training/steerer-qnrf-imagenet-20260806-t1-e/environment.9fdaf0b7408f9d0ca861e3dbbcd6b79b331e0c2d64445dca552d923f50a923a6.json; sha256=9fdaf0b7408f9d0ca861e3dbbcd6b79b331e0c2d64445dca552d923f50a923a6; verified: /workspace/data/checkpoints/steerer-ucf-training/steerer-qnrf-imagenet-20260806-t1-e/last.pth; sha256=d62b3e01727cd7f6bc245a1b3398fed0350a55c55d0238f26b437f81cf33ec72; verified: /workspace/data/checkpoints/steerer-ucf-training/steerer-qnrf-imagenet-20260806-t1-e/last.pth; sha256=d62b3e01727cd7f6bc245a1b3398fed0350a55c55d0238f26b437f81cf33ec72; verified: /workspace/.worktrees/steerer-ucf-training/configs/training/steerer_ucf_qnrf_imagenet.home5090.json; sha256=a29680cbd71bf1e7226c121d13c91ad673632cc9d7bc6d6fbe525ddce1f40929; verified: /workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1/train.txt; sha256=22ad8d3bf53d3c582b1b2d902b25e44f86d38f448bf2d7cce7153b458b7702f4; verified: /workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1/val.txt; sha256=d80b7fe911f80f9f48f6d5128dd94d7b08741afe0b4283425273b0560e7e142c |

## Decision rule

A stage passes only when the score reaches the threshold and every blocker passes.
