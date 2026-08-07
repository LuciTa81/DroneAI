# T1: STEERER UCF-QNRF T1 training

- Status: **PASS_COMMERCIAL_CANDIDATE**
- Decision scope: **commercial_candidate**
- Score: **100.0/100**
- Pass threshold: **75.0**
- Generated at: `2026-08-06T15:53:47.750406+00:00`

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
| PASS | artifacts | All referenced artifacts hash-verify from disk | 15.0/15.0 | Y | verified: /workspace/data/checkpoints/backbones/hrnetv2_w48_imagenet_pretrained.pth; sha256=0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8; verified: /workspace/upstreams/STEERER/configs/QNRF_final.py; sha256=a1bd95fa9dcb62ced8c248f0716d22f45f23080705d8445e195982f789c1314c; verified: /workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1/manifests/output-inventory.jsonl; sha256=f38af282fc3007f0f204fb94c14b45595cc383207323eb7430c3a6546f41d042; verified: /workspace/data/results/steerer-ucf-training/steerer-qnrf-imagenet-20260806-t1-e/environment.5d796fc2ef3ac8df1955c87bfb0970247a5f4887dd14c8ce1c43fbaf34461c64.json; sha256=5d796fc2ef3ac8df1955c87bfb0970247a5f4887dd14c8ce1c43fbaf34461c64; verified: /workspace/data/checkpoints/steerer-ucf-training/steerer-qnrf-imagenet-20260806-t1-e/last.pth; sha256=f2a526ca4e95d829b489d3d5c65df8104b023a0a70140550cd027ec6d4c23155; verified: /workspace/data/checkpoints/steerer-ucf-training/steerer-qnrf-imagenet-20260806-t1-e/last.pth; sha256=f2a526ca4e95d829b489d3d5c65df8104b023a0a70140550cd027ec6d4c23155; verified: /workspace/.worktrees/steerer-ucf-training/configs/training/steerer_ucf_qnrf_imagenet.home5090.json; sha256=a29680cbd71bf1e7226c121d13c91ad673632cc9d7bc6d6fbe525ddce1f40929; verified: /workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1/train.txt; sha256=22ad8d3bf53d3c582b1b2d902b25e44f86d38f448bf2d7cce7153b458b7702f4; verified: /workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1/val.txt; sha256=d80b7fe911f80f9f48f6d5128dd94d7b08741afe0b4283425273b0560e7e142c |

## Decision rule

A stage passes only when the score reaches the threshold and every blocker passes.
