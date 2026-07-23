# Company Share Full-Panel Export Design

## Goal

Produce one Google Drive-ready `DroneAI_회사공유용_v1.zip` containing a
browsable `Drone AI` folder with:

- all 3,000 Round 2 model-image output panels for STEERER, PET, and APGCC;
- existing Round 1 results for DM-Count, MPCount, and CSRNet;
- the existing review PDFs and compact Round 1/Round 2 comparisons;
- model integration code, pinned-source information, environment requirements,
  run commands, rights notes, metrics, predictions, and provenance manifests.

This is a research-result sharing package. It does not grant commercial rights
or production approval.

## Approved scope

### Full-panel export

Re-run inference, without training, fine-tuning, or calibration, for the exact
accepted Round 2 shared manifest:

| Model | UCF-QNRF | JHU-CROWD++ | UP-COUNT | Total |
|---|---:|---:|---:|---:|
| STEERER | 334 | 500 | 166 | 1,000 |
| PET | 334 | 500 | 166 | 1,000 |
| APGCC | 334 | 500 | 166 | 1,000 |
| Total panels | 1,002 | 1,500 | 498 | 3,000 |

Every exported panel must be reconstructed from a fresh native model output.
The fresh predicted count must match the accepted `predictions.csv` record
within `1e-6`. A mismatch, missing sample, non-finite result, checkpoint change,
source-tree change, or manifest hash change blocks that lane.

### Existing Round 1 models

DM-Count, MPCount, and CSRNet are included with their completed 36-image
UCF-QNRF validation compatibility-smoke results and existing curated panels.
Their folders must clearly state that this is a 36-image Round 1 result and is
not comparable in evaluation scale to the 1,000-image Round 2 lanes.

### Exclusions

The package must not contain:

- raw UCF-QNRF, JHU-CROWD++, or UP-COUNT datasets;
- pretrained checkpoints or derived weights;
- virtual environments, package caches, Docker layers, or credentials;
- raw full-resolution density arrays;
- incomplete acquisition directories, retry attempts, or superseded panels;
- any claim of production or commercial approval.

## Package layout

```text
Drone AI/
├── 결과해석/
│   ├── 00_먼저읽기.md
│   ├── Round1_공통장면_모델비교_공유용.pdf
│   ├── Round1_비교표/
│   ├── Round2_비교표/
│   ├── 라이선스_및_사용범위.md
│   └── 패키지_검증정보/
├── STEERER/
│   ├── 모델코드/
│   ├── UCF-QNRF_334/
│   ├── JHU-CROWD++_500/
│   ├── UP-COUNT_166/
│   ├── 현장사진_5x5/
│   └── 고밀도_추가시험/
├── PET/
│   ├── 모델코드/
│   ├── UCF-QNRF_334/
│   ├── JHU-CROWD++_500/
│   └── UP-COUNT_166/
├── APGCC/
│   ├── 모델코드/
│   ├── UCF-QNRF_334/
│   ├── JHU-CROWD++_500/
│   └── UP-COUNT_166/
├── MPCount/
│   ├── 모델코드/
│   └── UCF-QNRF_Validation_36/
├── DMCount/
│   ├── 모델코드/
│   └── UCF-QNRF_Validation_36/
└── CSRNet/
    ├── 모델코드/
    └── UCF-QNRF_Validation_36/
```

Each Round 2 dataset folder contains:

```text
결과이미지_전체/
metrics.json
predictions.csv
score.json
score.md
summary.md
model-brief.md
sample-manifest.json
evidence-manifest.json
rights-decision.json
native-output-metadata.json
README.md
```

## Model-code contents

Each `모델코드` directory contains the DroneAI adapter, runner entry point,
model profile, requirements file, exact upstream URL and commit, checkpoint
identity without weight bytes, execution command, and license/rights note.

An official upstream source snapshot may be copied only when the recorded code
license permits redistribution. Otherwise the package contains the pinned URL,
commit, and integration code, not a source-tree copy. Checkpoint files are never
included.

## Export architecture

### Panel exporter

A new deterministic batch exporter:

1. validates the frozen Round 2 configuration and shared manifest;
2. loads the accepted dataset lane, clean pinned upstream, and SHA-verified
   checkpoint through the existing Round 2 runner boundary;
3. reads accepted counts from the lane `predictions.csv`;
4. runs `adapter.predict(sample, retain_native=True)` once per image;
5. evaluates the fresh output and verifies count/output identity;
6. renders the existing four-column review panel;
7. writes the panel atomically;
8. appends a JSONL progress record containing source SHA-256, output SHA-256,
   count, latency, and runtime identity.

The exporter accepts one model and one dataset lane per invocation. This keeps
failure and resume boundaries independent.

### Resume behavior

The job is restartable after SSH loss, application closure, or process failure.
An existing panel is reused only when its progress record, source hash, accepted
count, panel hash, checkpoint hash, upstream commit, Git commit, and shared
manifest hash all match. A conflicting file fails closed and is never silently
overwritten.

### Package builder

A separate package builder consumes only completed, hash-verified lane exports.
It copies approved compact evidence and panels into a new staging directory,
writes a file-level SHA-256 manifest, verifies the expected 3,000 full panels,
and creates one UTF-8 ZIP. The ZIP is created outside Git and is not committed.

## Execution and storage

- Runtime: existing `home5090-pop` and `crowd-jupyter` only.
- Remote export root:
  `/workspace/data/exports/DroneAI-company-share-v1`
- Windows delivery root:
  `C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\outputs\DroneAI_회사공유용_v1`
- Windows ZIP:
  `C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\outputs\DroneAI_회사공유용_v1.zip`
- Long execution runs in the existing `tmux` workflow or a non-interactive
  restartable runner.

No existing dataset, checkpoint, result, personal NTFS file, or accepted
evidence is deleted or modified.

## Verification

Before the full run:

- unit tests prove lane validation, count matching, resume behavior, atomic
  writes, package selection, and path safety;
- a 30-panel pilot (ten panels per model across declared lanes) measures actual
  output size and catches layout/runtime regressions.

Before delivery:

- exactly 3,000 Round 2 full panels are present;
- the Round 1 DM-Count, MPCount, and CSRNet folders contain their accepted
  36-image results and curated panels;
- no dataset image, checkpoint, virtual environment, cache, or raw density array
  is present;
- every package file appears in the SHA-256 manifest;
- ZIP extraction reproduces the staged file count and hashes;
- the full relevant test suite passes;
- Windows and home5090 repositories remain clean and identify the same commit.

## Expected size and user experience

The current benchmark panels average about 196 KiB. The 3,000 full panels are
therefore expected to occupy roughly 575 MiB before package overhead, with a
final ZIP target of approximately 700 MiB to 1.2 GiB.

Google Drive users download one ZIP, extract it locally, and browse by model and
dataset. The package readme explains that `predictions.csv` is authoritative for
the full numeric result and that the PNG panels are research visualizations.
