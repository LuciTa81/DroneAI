# Model brief: steerer-official-ucf-qnrf

- Role: UCF-QNRF crowd-counting and localization baseline
- Family: hybrid density and point localization
- Backbone: HRNet-W48
- Native output: multi-resolution density and localization predictions
- Inference long-side cap: 3072 pixels
- Density normalization factor: 100

## Official preprocessing

- Input color space: RGB
- Normalization: ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
- Offline preparation: pinned `prepare_QNRF.py` resizes every image to its
  minimum/next-16 geometry and re-encodes it as bilinear JPEG quality 95
- Annotation preparation: scale and integer-truncate points, then clip boundary
  points without dropping count mass
- Inference resize: preserve the prepared size when long side <= 3072; otherwise
  set the long side to 3072
- Inference padding: zero-pad both dimensions to multiples of 32

## Major blocks

- HRNet-W48 multi-resolution backbone
- multi-resolution counting heads
- selective inheritance/upsample module
- Gaussian density/local-maximum point decoder

## Feature scales

- x1
- x4
- x8

## Original training losses

- multi-resolution MSE
- selective inheritance routing loss
- uncertainty weighting where configured

## Output semantics

```text
estimated_count = sum(highest_resolution_density / 100)
localization_points = official local maxima merged from x1/x4/x8
zone_count = integral of count-preserving density in the calibrated image zone
paper localization F1 != harness F1@16px
```

## Paper/repository-reported results

The following UCF-QNRF values are paper/repository-reported results, not measurements produced by this evaluation harness:

- Paper MAE/RMSE: 74.3 / 128.3
- Official repository/checkpoint MAE/RMSE: 77.8 / 138.0
- F1: 75.6
- Precision: 79.7
- Recall: 72.0

Future harness results must be recorded separately. The harness localization label is `harness F1@16px in original-image coordinates`; it must not be substituted for the paper localization F1.

## A-lane G1 official-checkpoint reproduction

The first raw-image attempt produced 86.67 MAE / 152.57 RMSE and was blocked.
Root-cause tracing found that raw-image evaluation omitted the pinned offline
QNRF preparation and dropped 57 out-of-bound annotations across 45 images.

After reproducing the pinned preparation, Test334 produced:

- MAE: **77.5165**
- RMSE: **139.0867**
- Signed bias: **+0.8315**
- MAPE reference: **10.4468%**
- Median latency: **143.02 ms/image**
- Throughput: **6.99 FPS**
- Peak VRAM: **4.0149 GiB**
- Harness localization P/R/F1@16px: **73.05% / 67.34% / 69.42%**
- Four-quadrant density zone MAE: **21.89 people/zone**

The absolute difference from the repository claim is 0.2835 MAE and 1.0867
RMSE, inside the preregistered 5.0/10.0 compatibility tolerance. G1 is
`PASS_RESEARCH_ONLY`; it validates evaluator compatibility, not commercial use
of the official checkpoint.

## Failure modes

- tiny heads
- extreme perspective
- occlusion
- domain shift
- padding artifacts
- threshold-sensitive localization

## home5090 isolated environment

From `/workspace` inside `crowd-jupyter`, create or reuse the model venv and
install the pinned headless overlay with:

```bash
python scripts/setup_steerer_home5090.py
```

The command creates `/workspace/.venvs/steerer` with
`--system-site-packages` only when the path is absent. It refuses an existing
non-venv path and never deletes or recreates a venv. Before any pip mutation,
new and reused environments must declare
`include-system-site-packages = true` in `pyvenv.cfg`.

For a setup run, the script uses only the venv Python to uninstall any existing
GUI `opencv-python` distribution, install every non-MMCV pin, install
`mmcv==1.7.2 --no-deps`, and finally force-reinstall the exact headless wheel
with `--no-deps`. The last step restores shared `cv2` files that a GUI-wheel
uninstall may have removed. No venv or filesystem tree is deleted.
The overlay must match the frozen 17-package name/version allowlist exactly;
missing, changed, or additional entries fail before installation. In
particular, Torch/CUDA packages and `opencv-contrib-python` cannot be added to
this model overlay, preserving the NGC base runtime.

Re-run the complete fail-closed verification without installing anything:

```bash
python scripts/setup_steerer_home5090.py --verify-only
```

Both commands emit one JSON status object. Verification requires the GUI
`opencv-python` distribution to be absent, requires
`opencv-python-headless==4.12.0.88`, compares the installed metadata for all 13
frozen distributions with their exact versions, emits those observed versions,
imports the model/runtime modules, and runs `pip check`. Only these two exact,
evidence-backed metadata messages are allowed:

```text
mmcv 1.7.2 requires opencv-python, which is not installed.
nvidia-resiliency-ext 0.4.1+cuda13 requires pynvml, which is not installed.
```

The first is MMCV's distribution-name mismatch: the working headless wheel
provides `cv2`, but MMCV's metadata names only `opencv-python`. The second is
inherited from the pinned NGC base image. Any different or additional
`pip check` line fails the command; the script does not edit package metadata
or install a dummy GUI distribution.
