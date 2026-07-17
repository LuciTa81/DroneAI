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
- Resize: preserve native size when long side <= 3072; resize only when above 3072, setting the long side to 3072
- Padding: zero-pad both dimensions to multiples of 32

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

- MAE: 77.8
- RMSE: 138.0
- F1: 75.6
- Precision: 79.7
- Recall: 72.0

Future harness results must be recorded separately. The harness localization label is `harness F1@16px in original-image coordinates`; it must not be substituted for the paper localization F1.

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
The overlay must match the frozen 13-package name/version allowlist exactly;
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
