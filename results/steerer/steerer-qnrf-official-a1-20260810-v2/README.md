# STEERER UCF-QNRF official-code A1

- 결과: `100/100 PASS_COMMERCIAL_CANDIDATE`
- 학습: ImageNet HRNet-W48 백본에서 새로 시작, Train 1,201장 중 `drop_last`에 따라 1,200장, 150 updates, 1 epoch
- 평가: official Test 334/334, `official Test, test-selected`
- 성능: MAE `1220.4982`, RMSE `1437.5661`, 평균 편향 `+1191.2887명`
- 실행: 학습 `60.46초`, Test 중앙 latency `216.79ms/장` (`4.61 FPS`), 학습 peak VRAM `23.67 GiB`
- 체크포인트: epoch 1 / global step 150, model·optimizer·scheduler·RNG round-trip 검증

이 값은 epoch 1 진단 결과이므로 완성 모델 성능이 아니다. 공식 설정은 cuDNN
benchmark를 활성화하고 deterministic mode를 비활성화하므로 동일 seed 반복 간 수치가
완전히 일치하지 않았다. 800 epoch 장기 실행 전 이 특성을 명시적으로 수용하거나 별도의
deterministic robustness run을 추가해야 한다.

`PASS_COMMERCIAL_CANDIDATE`는 코드·데이터·초기화 범위와 실험 증거가 후보 단계의
요구를 통과했다는 뜻이며, 현장 배포 승인이나 모델 정확도 100점을 뜻하지 않는다.
