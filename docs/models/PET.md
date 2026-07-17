# Model brief: pet-official-ucf-qnrf

- Role: point-based counting and localization comparison on UCF-QNRF
- Family: point-query quadtree
- Backbone: VGG16-BN
- Rights scope: `PASS_RESEARCH_ONLY`
- Native output: point/non-point probabilities, normalized point coordinates, and quadtree split map
- Density map: unavailable natively
- Official repository commit: `5b4dd7da8b11568a3305a88bb7c99a7fc831a998`

## Structure

PET encodes VGG16-BN image features with progressive rectangular-window
attention. Sparse point queries begin at stride 8. Queries in dense regions can
split through a quadtree up to depth 2, so computation is concentrated where
crowds are dense. The decoder produces a point/non-point probability and a
normalized coordinate for every retained query, plus the quadtree split map.

## Input and preprocessing

- Input: one RGB image tensor
- UCF-QNRF inference cap: constrain the long side to 1536 pixels
- Coordinate contract: map normalized predicted points back into original-image coordinates before scoring or zone aggregation
- Point threshold: strictly greater than 0.5 for the point class

## Output semantics

```text
estimated_count = number of point queries with point_probability > 0.5
localization_points = accepted point coordinates mapped to the original image
zone_count = number of predicted points inside the calibrated image zone
density_heatmap = not a native PET output; use binned or kernel-rendered points only as a derived visualization
```

This is useful for CCTV-zone counting and overlap calibration because each
prediction has a position. A derived heatmap can support the dashboard, but it
must be labelled as a visualization rather than a native density estimate.

## Original training protocol (reference only)

- Optimizer: Adam
- Weight decay: `5e-4`
- Learning rate: `1e-5` for VGG16-BN and `1e-4` for transformer layers
- Losses: point/non-point cross entropy, smooth L1 coordinate loss, quadtree split loss
- Loss weights: point coordinate `5`, split `0.1`

No training or fine-tuning is authorized in the current stage.

## Paper-reported UCF-QNRF result

These are paper results, not DroneAI measurements:

- MAE: 79.53
- RMSE: 144.32

## Rights boundary

The official README explicitly limits the code to academic purposes. Its
repository also contains an incomplete MIT template, but that does not erase
the explicit restriction. The official UCF-QNRF checkpoint inherits the same
research-only treatment. Commercial fine-tuning and deployment are blocked
unless the authors provide separate written permission.

The Kaggle UCF-QNRF copy is treated as Apache-2.0 under the project owner's
explicit approval, while its uploader/provenance caveat remains recorded.

## home5090 compatibility boundary

The official environment is Python 3.8, PyTorch 1.12.1, and torchvision
0.13.1. Those old Torch wheels are not the home5090 base runtime. PET must use
an isolated environment overlay and first prove model construction, checkpoint
key compatibility, CUDA execution, and one validation sample under the current
RTX 5090 runtime. Dependency installation must not replace the base
Torch/CUDA packages.

From `/workspace` inside `crowd-jupyter`:

```bash
python scripts/setup_pet_home5090.py
python scripts/setup_pet_home5090.py --verify-only
```

The script creates `/workspace/.venvs/pet` with access to the NGC system
packages and installs only the frozen PET overlay with `--no-deps`. It rejects
Torch, torchvision, NumPy, or other unreviewed additions, rejects GUI OpenCV,
and verifies CUDA matmul plus the imported runtime before returning `PASS`.

The official Torch 1.12 code moves boolean query masks to CPU before indexing
CUDA predictions. Current Torch rejects that device mismatch. The DroneAI
adapter keeps those masks on the prediction device while preserving the
official `> 0.5` threshold and concatenation order; this compatibility shim is
recorded in every prediction's metadata and does not change checkpoint values.

## Expected failure modes

- tiny or heavily occluded people
- large viewpoint/domain shift
- point-threshold sensitivity
- zone-boundary coordinate errors
- memory or latency pressure on very large frames
