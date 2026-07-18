# Model brief: mpcount-official-sta-to-ucf-qnrf

- Role: domain-generalized density candidate for fixed-CCTV crowd monitoring
- Family: density estimation and single-domain generalization
- Backbone: VGG16-BN
- Rights scope: `PASS_COMMERCIAL_CANDIDATE`
- Native output: full-resolution density map
- Official repository commit: `6eb06772bcf7dfb771c43a14d67146fce767f103`

## Structure

MPCount encodes the RGB image with a VGG16-BN feature extractor, merges multiple
feature scales in its decoder, and refines density features through a 1024-slot
memory bank. A patch-wise density classification branch gates the regression
map. The deterministic version replaces nondeterministic interpolation with
learned upsampling before producing the full-resolution density.

## Input and preprocessing

- Input: one RGB CCTV image
- Pad symmetrically to a multiple of 16; never resize the source image
- Convert pixels to `[0, 1]`, then normalize with mean `0.5` and standard deviation `0.5`
- Use non-overlapping patches when either padded dimension reaches the configured patch size
- One-sample patch size: the official default `3584`

## Output semantics

```text
raw_density = DGModel_final(normalized_image)[0]
aligned_density = crop_padding(raw_density) / 1000
estimated_count = sum(aligned_density)
zone_count = fractional density mass inside the calibrated CCTV zone
```

The DroneAI evaluation copy is mass-preserving resized to the same original
image `/8` grid as the point-derived ground-truth density. MPCount provides no
native person coordinates, so localization precision/recall/F1 are not claimed.

## Original training protocol (reference only)

The selected deterministic checkpoint was trained on ShanghaiTech Part A and
officially evaluated on UCF-QNRF. The architecture combines density regression,
memory attention consistency, a content error mask, and patch-wise density
classification. Round 1 performs no training or fine-tuning.

Official cross-domain result reported by the repository:

- ShanghaiTech-A to UCF-QNRF MAE: 112.8
- ShanghaiTech-A to UCF-QNRF RMSE: 193.8

These are paper/repository values, not DroneAI measurements.

## Rights boundary

The pinned code is Apache-2.0. Project policy accepts Apache-2.0 for the official
deterministic checkpoint and accepts the Kaggle UCF-QNRF Apache-2.0 label while
retaining the uploader-provenance caveat. Deployment and derived-weight rights
remain pending, so this is not `PRODUCTION_APPROVED`.

## home5090 compatibility boundary

The official environment is Python 3.10.12, PyTorch 2.0.1, and torchvision
0.15.2. DroneAI keeps the RTX 5090 NGC Torch 2.9 runtime and installs only the
isolated MPCount overlay. Preflight proved imports, CUDA, strict checkpoint
deserialization, and artifact hashes. One-sample is the first actual forward
compatibility gate.

## Expected failure modes

- extreme perspective, occlusion, or tiny heads
- CCTV/festival domain shift beyond the ShanghaiTech-A source distribution
- patch-boundary artifacts
- full-resolution activation memory pressure
- behavioral changes caused by the Torch 2.0.1 to 2.9 runtime gap
