# STEERER UCF-QNRF 자체 학습 설계

작성일: 2026-08-06  
상태: 사용자 승인 접근법의 구현 전 설계  
실행 백엔드: `home5090-pop` / `crowd-jupyter` / RTX 5090  

## 1. 목표와 완료 범위

STEERER 공식 코드 구조와 ImageNet 사전학습 HRNet-W48 백본을 사용하되,
배포된 STEERER QNRF 체크포인트는 전혀 불러오지 않는다. UCF-QNRF의 공식
Train 파티션만으로 STEERER의 counting/localization 계층과 백본 전체를
end-to-end 학습하여 프로젝트가 직접 생성한 체크포인트를 만든다.

첫 구현 주기의 완료 범위는 다음과 같다.

1. 권리·입력·split·초기 weight provenance를 검증한다.
2. UCF-QNRF 원본 이미지와 MAT point annotation에서 STEERER 학습 입력을 직접 생성한다.
3. 한 batch forward/backward와 1 epoch smoke를 통과시킨다.
4. 사용자 승인 후 5 epoch smoke를 실행한다.
5. 다시 승인받은 뒤 50 epoch pilot을 실행하고 검증 결과를 점수화한다.

800 epoch 공식형 장기학습, 공식 Test 334장 평가, 현장 데이터 fine-tuning과
제품 배포는 이 설계의 자동 실행 범위가 아니다. 각각 별도 승인 지점으로 남긴다.

## 2. 채택한 학습 방식

### 채택안: ImageNet 초기화 후 전체 end-to-end 학습

- STEERER code: pinned official commit
  `5b1854dbc2d280f2326d67c65515d8baf9083810`, MIT.
- Backbone initialization: 공식 설정이 요구하는
  `hrnetv2_w48_imagenet_pretrained.pth`.
- STEERER QNRF checkpoint: 학습 입력 및 resume 입력에서 금지한다.
- Counting head, FSIA/MSIL 관련 계층: 랜덤 초기화한다.
- HRNet backbone: ImageNet weight로 초기화하고 epoch 0부터 counting head와 함께
  end-to-end 최적화한다.

백본을 계속 고정하는 head-only 학습은 UCF 군중 도메인에 충분히 적응하지 못할
가능성이 높으므로 사용하지 않는다. 백본을 먼저 고정했다가 해제하는 단계형 방식은
공식 설정과 다른 실험으로 분리하며, 첫 pilot에는 넣지 않는다.

## 3. 권리 및 provenance 경계

각 구성요소를 별도 권리 항목으로 기록한다.

| 구성요소 | 사용 | 기록할 증거 | 현재 범위 |
|---|---|---|---|
| STEERER code | 사용 | source URL, commit, LICENSE SHA-256 | MIT verified |
| UCF-QNRF | 사용 | Kaggle URL, archive SHA-256, inventory SHA-256 | owner-approved Apache label with provenance risk |
| HRNet ImageNet weight | 사용 | exact download URL, filename, SHA-256, 제공 저장소 license | internal approval; production review pending |
| 공식 STEERER QNRF weight | 미사용 | forbidden artifact hash와 load audit | excluded |
| 프로젝트 생성 weight | 생성 | parent hashes, config, seed, split hashes, run manifest | commercial candidate only |

Kaggle 업로더가 원 데이터 소유자가 아니라고 밝힌 provenance 위험과 ImageNet
파생 weight의 제품 사용 검토 필요성을 manifest에 유지한다. 프로젝트 책임자의
승인은 연구·상용 후보 개발을 허용하지만 `PRODUCTION_APPROVED`를 의미하지 않는다.

학습 시작 전 다음 조건 중 하나라도 만족하지 않으면 차단한다.

- HRNet weight의 정확한 source URL 또는 SHA-256이 없다.
- UCF archive/inventory/split hash가 승인된 값과 다르다.
- resume 또는 model-load 인자에 공식 STEERER checkpoint가 들어간다.
- upstream worktree가 pinned commit과 다르거나 dirty하다.

## 4. 데이터와 split 설계

UCF-QNRF 공식 구성은 Train 1,201장과 Test 334장이다. Test는 개발 중 봉인한다.

### 개발 split

- 입력 모집단: 공식 Train 1,201장만 사용.
- seed: `3035`.
- stratification: annotation count 기반 density band별로 수행.
- train: 961장.
- validation: 240장.
- test: 공식 Test 334장, 접근 금지.

생성된 `train.txt`, `val.txt`, `test-sealed.txt`에는 sample ID만 기록하고 각 파일의
SHA-256, sample count, 상호 교집합 0건, 모집단 합집합 일치 여부를 manifest에
저장한다. `test-sealed.txt`는 학습과 pilot 평가 runner가 받을 수 없는 별도 역할로
표시한다.

50 epoch pilot에서 epoch 선택과 hyperparameter를 고정한 뒤에만 다음 선택을
사용자에게 제시한다.

1. 961/240 모델을 그대로 공식 Test에 한 번 평가한다.
2. 선택된 epoch 수로 공식 Train 1,201장을 다시 학습한 후 Test에 한 번 평가한다.

## 5. 전처리 설계

공식 OneDrive의 processed QNRF package는 사용하지 않는다. 프로젝트 전처리기는
다음 입력만 읽는다.

- UCF-QNRF 원본 JPG.
- 대응하는 `*_ann.mat` point annotation.
- 승인된 split manifest.

출력은 SSD의 다음 전용 경로에 둔다.

```text
/workspace/data/datasets/ucf-qnrf-kaggle-apache/processed/steerer-training-v1/
  images/
  jsons/
  train.txt
  val.txt
  manifests/
```

전처리기는 공식 STEERER가 요구하는 이미지 크기 정규화, point 좌표 변환,
scale/box metadata 생성을 재현한다. upstream의 hard-coded root, 비결정적
`random.sample`, 기존 파일이면 건너뛰는 동작을 그대로 실행하지 않고 프로젝트
wrapper에서 결정성과 원자적 출력을 강제한다.

전처리 검증 조건은 다음과 같다.

- 원본 point count와 JSON `human_num`이 모든 sample에서 일치한다.
- 변환된 모든 point와 box가 이미지 경계 안에 있다.
- 원본·출력 이미지 크기와 좌표 scale factor가 기록된다.
- train/validation sample 수가 961/240이다.
- 동일 seed 재실행 시 split 및 annotation artifact SHA-256이 같다.
- 기존 non-empty output은 명시적 새 run ID 없이는 덮어쓰지 않는다.

## 6. 학습 설정

첫 pilot은 공식 QNRF 설정을 가능한 범위에서 유지한다.

| 항목 | 값 |
|---|---|
| 모델 | STEERER, HRNet-W48, multi-resolution counting heads |
| 입력 crop | 768 × 768 |
| augmentation | horizontal flip, multi-scale 0.5–2.0 |
| density factor | 100 |
| optimizer | AdamW |
| base learning rate | `1e-4` |
| weight decay | `1e-4` |
| scheduler | 10 epoch linear warmup + cosine decay |
| seed | 3035 |
| target effective batch | 8 |
| validation input cap | long side 3072 |

RTX 5090 preflight에서 batch 8이 OOM이면 물리 batch를 4로 낮추고 gradient
accumulation 2를 사용해 effective batch 8을 유지한다. 이 변경은 자동으로
숨기지 않고 manifest와 결과 보고서에 기록한다.

### 단계별 실행

1. `T0`: 한 batch forward/backward, optimizer step, checkpoint write/read.
2. `T1`: 1 epoch smoke와 전체 validation.
3. `T5`: 사용자 승인 후 epoch 5까지 resume.
4. `T50`: 사용자 승인 후 epoch 50까지 resume.
5. `TFULL`: T50 검토 후 별도 승인으로 공식형 장기학습.

각 단계는 이전 단계 checkpoint에서 이어지며 같은 run lineage를 사용한다.

## 7. checkpoint와 재시작 정책

대용량 weight와 optimizer state는 Git에 넣지 않고 SSD에 저장한다.

```text
/workspace/data/checkpoints/steerer-ucf-training/<run-id>/
  last.pth
  best-mae.pth
  best-rmse.pth
  milestone-001.pth
  milestone-005.pth
  milestone-050.pth
  manifest.json
```

`last.pth`는 임시 파일에 완전히 기록한 뒤 rename하여 원자적으로 교체한다.
resume checkpoint에는 epoch, model, optimizer, scheduler, scaler, seed와 RNG state를
포함한다. PC 재부팅 후에도 같은 manifest와 checkpoint SHA-256이 맞을 때만 resume한다.

## 8. 평가와 점수화

학습 중 평가는 validation 240장만 사용한다.

- train/validation loss.
- count MAE, RMSE, signed bias, MAPE reference.
- density GAME L1과 zone error.
- localization precision, recall, F1과 point distance.
- median/p95 latency, FPS, peak VRAM.
- density sum과 predicted count 보존 오차.

각 단계의 harness score는 다음 100점으로 구성한다.

| 영역 | 배점 |
|---|---:|
| code/data/backbone/checkpoint provenance | 25 |
| train/validation/test split 무결성 | 20 |
| 학습 안정성 및 resume 재현성 | 20 |
| validation metric와 학습 개선 추세 | 20 |
| 결과 artifact/hash/환경 manifest 완전성 | 15 |

다음 항목은 점수와 무관한 hard blocker다.

- train/validation/test leakage.
- NaN/Inf loss 또는 density.
- source/split/backbone/checkpoint SHA-256 불일치.
- 공식 STEERER QNRF checkpoint 로드.
- checkpoint resume 검증 실패.

T1은 75점 이상, T5는 80점 이상, T50은 85점 이상이어야 다음 단계 후보가 된다.
T1/T5에서는 절대 정확도보다 정상적인 loss 감소, finite output과 resume 증명을
우선한다. T50에서도 공식 논문 MAE 재현을 성공 조건으로 강제하지 않고 차이를
원인과 함께 보고한다.

## 9. 결과와 Git 경계

SSD 결과 경로:

```text
/workspace/data/results/steerer-ucf-training/<run-id>/
```

SSD에 유지할 항목:

- processed images/jsons.
- 모든 checkpoint와 optimizer state.
- raw density arrays.
- 전체 TensorBoard/event log.
- 전체 sample prediction.

Git에 반영할 항목:

- 학습 및 전처리 config.
- split ID와 manifest/hash.
- 권리 decision과 artifact lineage.
- 단계별 summary/score/metrics JSON.
- 대표 결과 panel만 25 MiB 제한 안에서 선별.
- runbook의 train/resume/result 명령.

## 10. 오류 처리와 중단 조건

- CUDA OOM: 현재 stage를 실패로 기록하고 batch/accumulation 변경안을 보고한 뒤 재실행한다.
- NaN/Inf: 즉시 중단하고 마지막 정상 checkpoint를 보존한다.
- 데이터 손상 또는 hash mismatch: 전처리나 학습을 시작하지 않는다.
- SSH 단절: container/tmux run은 유지하되 로그와 heartbeat로 상태를 확인한다.
- PC 재부팅: 프로세스 자동 복구를 가정하지 않고 검증된 `last.pth`에서 명시적으로 resume한다.
- validation 개선 정지: T50 전에는 임의 early stop하지 않고 stage 종료 시 사용자에게 추세를 보고한다.

## 11. 구현 경계와 승인 지점

구현은 기존 Round 1 비교 queue를 수정하지 않는 별도 training lane으로 만든다.
현재 active CSRNet review state와 기존 STEERER 연구 checkpoint 결과는 보존한다.

승인 지점은 다음과 같다.

1. 이 설계 문서 승인.
2. 구현·전처리·T0/T1 결과 승인.
3. T5 실행 승인.
4. T50 실행 승인.
5. 공식 Test 접근 및 최종 1,201장 재학습 방식 승인.
6. 장기학습·현장 fine-tuning·제품 weight 승인.

첫 구현 단계에서는 2번까지 준비하고, 장시간 학습을 자동으로 시작하지 않는다.
