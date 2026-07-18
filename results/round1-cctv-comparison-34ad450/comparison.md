# Round 1 fixed-CCTV technical comparison

**Scope: PROVISIONAL_TECHNICAL_SHORTLIST; compatibility smoke; not an official ranking.**

| Order | Model | Technical | Zone/spatial | Count | Runtime | Integration | Evidence | Rights | Deployment |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | steerer | 83.18 | 29.63 | 18.88 | 13.67 | 13.00 | 8.00 | PASS_COMMERCIAL_CANDIDATE | PENDING_DUE_DILIGENCE |
| 2 | dm-count | 74.06 | 25.94 | 13.46 | 14.65 | 10.00 | 10.00 | PASS_COMMERCIAL_CANDIDATE | PENDING_DUE_DILIGENCE |
| 3 | pet | 73.12 | 22.13 | 18.99 | 15.00 | 9.00 | 8.00 | PASS_RESEARCH_ONLY | RESEARCH_ONLY |
| 4 | mpcount | 60.64 | 21.28 | 7.74 | 11.62 | 10.00 | 10.00 | PASS_COMMERCIAL_CANDIDATE | PENDING_DUE_DILIGENCE |
| 5 | apgcc | 57.74 | 20.53 | 3.23 | 14.98 | 9.00 | 10.00 | PASS_COMMERCIAL_CANDIDATE | PENDING_DUE_DILIGENCE |
| 6 | csrnet | 44.33 | 12.22 | 2.03 | 10.09 | 10.00 | 10.00 | PASS_COMMERCIAL_CANDIDATE | PENDING_DUE_DILIGENCE |

## Raw validation metrics

| Model | MAE | RMSE | MAPE % | Bias | High bias | Zone MAE | Spatial | Latency ms | FPS | VRAM MB |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| dm-count | 154.286 | 229.806 | 12.990 | 3.032 | 4.670 | 42.166 | game_l1=168.807 | 69.533 | 14.382 | 8980.255 |
| steerer | 71.588 | 97.965 | 6.201 | 43.924 | 92.115 | 19.768 | game_l1=79.152 | 130.696 | 7.651 | 3408.521 |
| pet | 76.444 | 129.802 | 6.046 | 15.667 | 7.500 | 66.319 | localization_f1=0.512 | 57.994 | 17.243 | 4291.915 |
| mpcount | 213.624 | 333.553 | 21.015 | -17.601 | -77.744 | 59.741 | game_l1=239.270 | 139.444 | 7.171 | 13281.501 |
| apgcc | 266.194 | 431.296 | 21.516 | -93.639 | -261.667 | 73.035 | localization_f1=0.683 | 67.663 | 14.779 | 5258.686 |
| csrnet | 332.329 | 522.031 | 32.820 | -86.804 | -268.827 | 91.055 | game_l1=364.129 | 80.336 | 12.448 | 20838.060 |

## Density, localization, and operator feature evidence

| Model | GAME L1 | PSNR | SSIM | Loc precision | Loc recall | Loc F1 | Mean distance | Zone | Density | Points | Confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| dm-count | 168.807 | 23.396 | 0.700 | N/A | N/A | N/A | N/A | DEMONSTRATED | DEMONSTRATED | NOT_DEMONSTRATED | NOT_DEMONSTRATED |
| steerer | 79.152 | 24.039 | 0.747 | 0.860 | 0.751 | 0.796 | 5.691 | DEMONSTRATED | DEMONSTRATED | DEMONSTRATED | NOT_DEMONSTRATED |
| pet | N/A | N/A | N/A | 0.510 | 0.516 | 0.512 | 8.443 | DEMONSTRATED | NOT_DEMONSTRATED | DEMONSTRATED | NOT_DEMONSTRATED |
| mpcount | 239.270 | 23.584 | 0.718 | N/A | N/A | N/A | N/A | DEMONSTRATED | DEMONSTRATED | NOT_DEMONSTRATED | NOT_DEMONSTRATED |
| apgcc | N/A | N/A | N/A | 0.732 | 0.663 | 0.683 | 6.999 | DEMONSTRATED | NOT_DEMONSTRATED | DEMONSTRATED | NOT_DEMONSTRATED |
| csrnet | 364.129 | 23.311 | 0.679 | N/A | N/A | N/A | N/A | DEMONSTRATED | DEMONSTRATED | NOT_DEMONSTRATED | NOT_DEMONSTRATED |

## Independent rights components

| Model | Code | Dataset | Pretrained weight | Derived weight | Deployment | Overall |
|---|---|---|---|---|---|---|
| dm-count | REVIEWED_NO_EXPLICIT_PROHIBITION | REVIEWED_NO_EXPLICIT_PROHIBITION | REVIEWED_NO_EXPLICIT_PROHIBITION | PENDING | PENDING | PENDING_DUE_DILIGENCE |
| steerer | REVIEWED_NO_EXPLICIT_PROHIBITION | REVIEWED_NO_EXPLICIT_PROHIBITION | PENDING | PENDING | PENDING | PENDING_DUE_DILIGENCE |
| pet | RESTRICTED | REVIEWED_NO_EXPLICIT_PROHIBITION | RESTRICTED | PENDING | PENDING | RESEARCH_ONLY |
| mpcount | REVIEWED_NO_EXPLICIT_PROHIBITION | REVIEWED_NO_EXPLICIT_PROHIBITION | REVIEWED_NO_EXPLICIT_PROHIBITION | PENDING | PENDING | PENDING_DUE_DILIGENCE |
| apgcc | REVIEWED_NO_EXPLICIT_PROHIBITION | REVIEWED_NO_EXPLICIT_PROHIBITION | PENDING | PENDING | PENDING | PENDING_DUE_DILIGENCE |
| csrnet | REVIEWED_NO_EXPLICIT_PROHIBITION | REVIEWED_NO_EXPLICIT_PROHIBITION | PENDING | PENDING | PENDING | PENDING_DUE_DILIGENCE |

## Provisional technical shortlist

1. `steerer`
2. `dm-count`
3. `pet`

## Representative panels

- `dm-count`: [panel 1](../dm-count/ucf-qnrf-val-smoke-aeb92da/figures/selected-01-fd9373c130ac1077.png), [panel 2](../dm-count/ucf-qnrf-val-smoke-aeb92da/figures/selected-02-ccb64a426e7cd837.png), [panel 3](../dm-count/ucf-qnrf-val-smoke-aeb92da/figures/selected-03-8e0a4dbe072725a1.png), [panel 4](../dm-count/ucf-qnrf-val-smoke-aeb92da/figures/selected-04-8fba49295a243a3d.png), [panel 5](../dm-count/ucf-qnrf-val-smoke-aeb92da/figures/selected-05-c21f6cc286a93bc0.png), [panel 6](../dm-count/ucf-qnrf-val-smoke-aeb92da/figures/selected-06-4203bc526bce1d84.png), [panel 7](../dm-count/ucf-qnrf-val-smoke-aeb92da/figures/selected-07-b136b5ac7f290f06.png), [panel 8](../dm-count/ucf-qnrf-val-smoke-aeb92da/figures/selected-08-c145cb679c8a0d04.png), [panel 9](../dm-count/ucf-qnrf-val-smoke-aeb92da/figures/selected-09-f30a3d3953dce3da.png), [panel 10](../dm-count/ucf-qnrf-val-smoke-aeb92da/figures/selected-10-3124be1f1fb9e789.png), [panel 11](../dm-count/ucf-qnrf-val-smoke-aeb92da/figures/selected-11-320ad007a4340168.png), [panel 12](../dm-count/ucf-qnrf-val-smoke-aeb92da/figures/selected-12-897716f616762b84.png)
- `steerer`: [panel 1](../steerer/steerer-ucf-qnrf-val-smoke-7e6bebb/figures/selected-01-be864ad6ba22cb9a.png), [panel 2](../steerer/steerer-ucf-qnrf-val-smoke-7e6bebb/figures/selected-02-fd9373c130ac1077.png), [panel 3](../steerer/steerer-ucf-qnrf-val-smoke-7e6bebb/figures/selected-03-8e0a4dbe072725a1.png), [panel 4](../steerer/steerer-ucf-qnrf-val-smoke-7e6bebb/figures/selected-04-d8fafefff7c52849.png), [panel 5](../steerer/steerer-ucf-qnrf-val-smoke-7e6bebb/figures/selected-05-505273f34f445a8f.png), [panel 6](../steerer/steerer-ucf-qnrf-val-smoke-7e6bebb/figures/selected-06-9e3a7ee5803445cf.png), [panel 7](../steerer/steerer-ucf-qnrf-val-smoke-7e6bebb/figures/selected-07-c145cb679c8a0d04.png), [panel 8](../steerer/steerer-ucf-qnrf-val-smoke-7e6bebb/figures/selected-08-b136b5ac7f290f06.png), [panel 9](../steerer/steerer-ucf-qnrf-val-smoke-7e6bebb/figures/selected-09-c70e58b5e6bae0d3.png), [panel 10](../steerer/steerer-ucf-qnrf-val-smoke-7e6bebb/figures/selected-10-ca90cca47cd2592a.png), [panel 11](../steerer/steerer-ucf-qnrf-val-smoke-7e6bebb/figures/selected-11-3124be1f1fb9e789.png), [panel 12](../steerer/steerer-ucf-qnrf-val-smoke-7e6bebb/figures/selected-12-cc4e59a84ce60dd5.png)
- `pet`: [panel 1](../pet/pet-ucf-qnrf-val-smoke-9b0fe87/figures/selected-01-8e0a4dbe072725a1.png), [panel 2](../pet/pet-ucf-qnrf-val-smoke-9b0fe87/figures/selected-02-826d4141eedcca6b.png), [panel 3](../pet/pet-ucf-qnrf-val-smoke-9b0fe87/figures/selected-03-fd9373c130ac1077.png), [panel 4](../pet/pet-ucf-qnrf-val-smoke-9b0fe87/figures/selected-04-ea1387c07af13e07.png), [panel 5](../pet/pet-ucf-qnrf-val-smoke-9b0fe87/figures/selected-05-eb49d273e13423a7.png), [panel 6](../pet/pet-ucf-qnrf-val-smoke-9b0fe87/figures/selected-06-419fec40ac00261f.png), [panel 7](../pet/pet-ucf-qnrf-val-smoke-9b0fe87/figures/selected-07-c145cb679c8a0d04.png), [panel 8](../pet/pet-ucf-qnrf-val-smoke-9b0fe87/figures/selected-08-cc4e59a84ce60dd5.png), [panel 9](../pet/pet-ucf-qnrf-val-smoke-9b0fe87/figures/selected-09-897716f616762b84.png), [panel 10](../pet/pet-ucf-qnrf-val-smoke-9b0fe87/figures/selected-10-ca90cca47cd2592a.png), [panel 11](../pet/pet-ucf-qnrf-val-smoke-9b0fe87/figures/selected-11-320ad007a4340168.png), [panel 12](../pet/pet-ucf-qnrf-val-smoke-9b0fe87/figures/selected-12-d8fafefff7c52849.png)
- `mpcount`: [panel 1](../mpcount/mpcount-ucf-qnrf-val-smoke-5cc5d72/figures/selected-01-fd9373c130ac1077.png), [panel 2](../mpcount/mpcount-ucf-qnrf-val-smoke-5cc5d72/figures/selected-02-c5e3d9aa6077e652.png), [panel 3](../mpcount/mpcount-ucf-qnrf-val-smoke-5cc5d72/figures/selected-03-8e0a4dbe072725a1.png), [panel 4](../mpcount/mpcount-ucf-qnrf-val-smoke-5cc5d72/figures/selected-04-c21f6cc286a93bc0.png), [panel 5](../mpcount/mpcount-ucf-qnrf-val-smoke-5cc5d72/figures/selected-05-4203bc526bce1d84.png), [panel 6](../mpcount/mpcount-ucf-qnrf-val-smoke-5cc5d72/figures/selected-06-9e3a7ee5803445cf.png), [panel 7](../mpcount/mpcount-ucf-qnrf-val-smoke-5cc5d72/figures/selected-07-e234c4048be37fca.png), [panel 8](../mpcount/mpcount-ucf-qnrf-val-smoke-5cc5d72/figures/selected-08-b136b5ac7f290f06.png), [panel 9](../mpcount/mpcount-ucf-qnrf-val-smoke-5cc5d72/figures/selected-09-eb87a5b258d7e8d8.png), [panel 10](../mpcount/mpcount-ucf-qnrf-val-smoke-5cc5d72/figures/selected-10-0a828eb9e6e71267.png), [panel 11](../mpcount/mpcount-ucf-qnrf-val-smoke-5cc5d72/figures/selected-11-320ad007a4340168.png), [panel 12](../mpcount/mpcount-ucf-qnrf-val-smoke-5cc5d72/figures/selected-12-897716f616762b84.png)
- `apgcc`: [panel 1](../apgcc/apgcc-ucf-qnrf-val-smoke-82d9285/figures/selected-01-1963ff1aa6492929.png), [panel 2](../apgcc/apgcc-ucf-qnrf-val-smoke-82d9285/figures/selected-02-8e0a4dbe072725a1.png), [panel 3](../apgcc/apgcc-ucf-qnrf-val-smoke-82d9285/figures/selected-03-ad26bfc3e5c98b18.png), [panel 4](../apgcc/apgcc-ucf-qnrf-val-smoke-82d9285/figures/selected-04-419fec40ac00261f.png), [panel 5](../apgcc/apgcc-ucf-qnrf-val-smoke-82d9285/figures/selected-05-90e81f32d2d93e17.png), [panel 6](../apgcc/apgcc-ucf-qnrf-val-smoke-82d9285/figures/selected-06-9e3a7ee5803445cf.png), [panel 7](../apgcc/apgcc-ucf-qnrf-val-smoke-82d9285/figures/selected-07-ca90cca47cd2592a.png), [panel 8](../apgcc/apgcc-ucf-qnrf-val-smoke-82d9285/figures/selected-08-c70e58b5e6bae0d3.png), [panel 9](../apgcc/apgcc-ucf-qnrf-val-smoke-82d9285/figures/selected-09-897716f616762b84.png), [panel 10](../apgcc/apgcc-ucf-qnrf-val-smoke-82d9285/figures/selected-10-0a828eb9e6e71267.png), [panel 11](../apgcc/apgcc-ucf-qnrf-val-smoke-82d9285/figures/selected-11-320ad007a4340168.png), [panel 12](../apgcc/apgcc-ucf-qnrf-val-smoke-82d9285/figures/selected-12-551885f3fe5f750a.png)
- `csrnet`: [panel 1](../csrnet/csrnet-ucf-qnrf-val-smoke-745d5b6/figures/selected-01-fd9373c130ac1077.png), [panel 2](../csrnet/csrnet-ucf-qnrf-val-smoke-745d5b6/figures/selected-02-c5e3d9aa6077e652.png), [panel 3](../csrnet/csrnet-ucf-qnrf-val-smoke-745d5b6/figures/selected-03-0a3e30df85e751e6.png), [panel 4](../csrnet/csrnet-ucf-qnrf-val-smoke-745d5b6/figures/selected-04-419fec40ac00261f.png), [panel 5](../csrnet/csrnet-ucf-qnrf-val-smoke-745d5b6/figures/selected-05-c21f6cc286a93bc0.png), [panel 6](../csrnet/csrnet-ucf-qnrf-val-smoke-745d5b6/figures/selected-06-5f883365dbf8fabe.png), [panel 7](../csrnet/csrnet-ucf-qnrf-val-smoke-745d5b6/figures/selected-07-ca90cca47cd2592a.png), [panel 8](../csrnet/csrnet-ucf-qnrf-val-smoke-745d5b6/figures/selected-08-c145cb679c8a0d04.png), [panel 9](../csrnet/csrnet-ucf-qnrf-val-smoke-745d5b6/figures/selected-09-897716f616762b84.png), [panel 10](../csrnet/csrnet-ucf-qnrf-val-smoke-745d5b6/figures/selected-10-0a828eb9e6e71267.png), [panel 11](../csrnet/csrnet-ucf-qnrf-val-smoke-745d5b6/figures/selected-11-320ad007a4340168.png), [panel 12](../csrnet/csrnet-ucf-qnrf-val-smoke-745d5b6/figures/selected-12-cc4e59a84ce60dd5.png)

## Limitations

- Official checkpoints do not share one training protocol.
- This is a 36-image cross-domain operational screen, not an official ranking.
- Technical score does not grant code, dataset, weight, derived-weight, or deployment rights.
- Fine-tuning and test-split access are outside this comparison stage.
