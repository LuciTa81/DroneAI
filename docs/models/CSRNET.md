# Model brief: csrnet-independent-shha-to-ucf-qnrf

- Role: classical density-map baseline for fixed-CCTV zone screening
- Family: density regression with dilated convolutions
- Backbone: VGG-16 frontend through conv4_3
- Rights scope: `PASS_COMMERCIAL_CANDIDATE`
- Native output: raw signed stride-8 density plus audited non-negative operational density
- Reference repository commit: `ed29d895989c188cb913a9503721271c6cf1ab1f`

## Structure

CSRNet uses the first ten VGG-16 convolution layers with three max-pooling
stages, followed by six dilation-2 convolution layers. A final 1x1 convolution
produces a single density channel at one-eighth of the input resolution. The
DroneAI architecture is independently implemented; the pinned reference source
is inspected only for provenance and preprocessing compatibility.

## Input and preprocessing

- Input: one native-resolution RGB CCTV image
- No source resize for the official inference contract
- Convert pixels to `[0, 1]`
- Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
  `[0.229, 0.224, 0.225]`

## Output semantics

```text
raw_density = CSRNet(normalized_image)[0, 0]
operational_density = max(raw_density, 0)
estimated_count = sum(operational_density)
zone_count = fractional operational density mass inside the calibrated CCTV zone
```

The output head has no ReLU. The frozen adapter therefore stores the signed raw
density as an SSD-only compressed audit artifact and records its hash, signed
sum, negative mass, and clipped-pixel count. Common evaluation, heatmaps, and
zone integration use the explicitly declared non-negative operational density;
this is never presented as the unmodified official count. Operational density
is mass-preserving aligned to the common original-image `/8` evaluation grid.

## Original training protocol (reference only)

The official checkpoint was trained on ShanghaiTech Part A with Euclidean
density-map regression loss. The saved checkpoint records best MAE `65.9016`.
The paper reports ShanghaiTech Part A MAE `68.2` and MSE `115.0`. These are
reference values, not DroneAI UCF-QNRF measurements. Round 1 performs no
training or fine-tuning.

## Rights boundary

The independent DroneAI code is project-owned for commercial R&D. The project
owner accepts the Kaggle UCF-QNRF Apache-2.0 label while retaining uploader
provenance risk. The official ShanghaiTech-A checkpoint has no published
commercial license, so it is authorized only for research comparison. Derived
weights and deployment remain pending; this is not `PRODUCTION_APPROVED`.

## Expected failure modes

- ShanghaiTech-A to UCF-QNRF domain shift
- negative raw density from the unconstrained output head
- 4K activation memory pressure in the high-resolution frontend
- festival-CCTV perspective, lighting, and occlusion shift
