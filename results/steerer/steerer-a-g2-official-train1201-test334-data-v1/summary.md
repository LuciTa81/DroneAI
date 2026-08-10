# STEERER A G2 — Official Train/Test data gate

- 결과: **PASS_COMMERCIAL_CANDIDATE, 100/100**
- 범위: 모델 정확도 평가가 아니라 학습 데이터와 실험 계보의 무결성 검증
- 분할: official Train **1,201**, validation **0**, official Test **334**
- 목록: `train.txt` 1,201행, `test.txt` 334행, `val.txt` 없음
- 산출물: image 1,535개, JSON annotation 1,535개, 약 3.3GB
- GT 합계: Train 1,011,515명, Test 240,127명
- 중복: Train/Test sample ID 및 원본 image SHA-256 중복 없음
- 검증: 원본 image/annotation과 생성 image/annotation의 SHA-256을 새 프로세스에서 전부 재계산
- 초기화 경계: ImageNet HRNet-W48 backbone만 허용하며 공식 STEERER 완성 checkpoint는 학습에 사용하지 않음
- 권리 범위: 사용자 승인 Kaggle Apache-2.0 표기 + MIT 코드 기준의 후보 단계이며 `PRODUCTION_APPROVED`는 아님

## Protocol disclosure

승인된 A 장기 실험은 official Train 1,201장 전체를 사용하고 validation을
따로 두지 않습니다. 공개 `prepare_QNRF.py::divide_dataset`가 Train의 약 20%를
무작위로 제외하는 동작과는 다르므로, 이 lane은 **full-Train reproduction with
pinned official-code settings**로 부릅니다. 모델 선택 결과는 반드시
`official Test, test-selected`로 표시합니다.

## Next gate

G3/A0에서 CUDA import/matmul, optimizer 1회 update, finite loss, checkpoint
저장·재로딩, RNG/optimizer/scheduler round-trip만 검증합니다. 800 epoch 학습은
G3/A0와 G4/A1이 통과하기 전에는 시작하지 않습니다.
