# G3-A0: STEERER official-code one-update smoke

- Status: **PASS_COMMERCIAL_CANDIDATE**
- Decision scope: **commercial_candidate**
- Score: **100.0/100**
- Pass threshold: **100.0**
- Generated at: `2026-08-10T07:07:21.522648+00:00`

| Result | Category | Check | Points | Blocker | Observed |
|---|---|---|---:|:---:|---|
| PASS | provenance | Pinned source, dataset, config, backbone, and container identities | 20.0/20.0 | Y | upstream=5b1854dbc2d280f2326d67c65515d8baf9083810 backbone=0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8 |
| PASS | runtime | CUDA import and finite matrix multiplication | 20.0/20.0 | Y | {"available": true, "device": "cuda:0", "device_index": 0, "device_name": "NVIDIA GeForce RTX 5090", "matmul_finite": true, "matmul_shape": [256, 256]} |
| PASS | training | Exactly one finite optimizer update from Train1201 | 20.0/20.0 | Y | steps=1 loss=0.16829925775527954 |
| PASS | checkpoint | Hash-verified model/optimizer/scheduler/RNG checkpoint reload | 20.0/20.0 | Y | state=True rng=True |
| PASS | rights | ImageNet-only initialization, no official model weight, no Test access | 20.0/20.0 | Y | scope=backbone_only model_checkpoint=False test_access=0 |

## Decision rule

A stage passes only when the score reaches the threshold and every blocker passes.

