# Round 1 공유용 기술 보고서 설계

## 목적

Round 1 고정형 CCTV 군중 계수 모델 비교 결과를 대표 및 비개발 관계자가
읽을 수 있는 A4 세로형 기술 보고서로 제공한다. 개인 검토용 15페이지의
근거와 수치는 유지하되, 내부 하네스 구현 설명은 줄이고 모델, 출력물,
성능, 권리 범위, 적용 결론을 중심으로 편집한다.

## 산출물

- 형식: 한국어 PDF, A4 세로, 9페이지
- 성격: 외부 배포 전 공유용 검토본
- 디자인: 기업 기술조사 보고서 스타일
- 저장 위치: `output/pdf/DroneAI_Round1_CCTV_모델비교_공유용_검토본.pdf`
- PDF와 중간 이미지는 Git에 추가하지 않는다.

## 편집 원칙

- 큰 지표 카드와 슬라이드형 넓은 여백을 사용하지 않는다.
- 번호가 있는 장·절, 연속 본문, 표, 그림 번호와 캡션을 사용한다.
- 상단에는 짧은 보고서명, 하단에는 페이지 번호와 검증 범위를 표시한다.
- 표와 본문은 한 페이지 안에서 자연스럽게 이어지도록 배치한다.
- 기술 점수는 정확도 백분율이나 공식 모델 순위로 표현하지 않는다.
- `PASS_COMMERCIAL_CANDIDATE`를 상용 배포 승인으로 표현하지 않는다.
- PET는 연구 비교군으로 명시한다.
- 공식 pretrained checkpoint를 이용한 UCF-QNRF validation 36장
  cross-domain compatibility smoke라는 한계를 반복해서 고지한다.

## 페이지 구성

1. **표지**: 보고서명, 목적, 기준일, 검증 범위, 핵심 결론 한 문장.
2. **요약 및 핵심 결론**: STEERER와 DM-Count 우선 검증, PET 연구 비교군,
   현 결과만으로 제품 배포를 승인할 수 없다는 결론.
3. **검증 목적·데이터·조건**: UCF-QNRF validation 36장, density band,
   RTX 5090, batch 1, fine-tuning 없음, 공통·계열별 metric.
4. **모델 구조와 입출력 비교**: 여섯 모델의 계열, backbone, native output,
   count 방식, CCTV 활용 형태를 한 표로 비교.
5. **성능 비교와 해석**: technical score, MAE, RMSE, zone MAE, FPS, VRAM을
   표와 간결한 해설로 제시. 수치의 비교 범위와 한계를 각주로 표시.
6. **Density 계열 출력 사례**: DM-Count, STEERER, MPCount, CSRNet의 동일
   `img_0097` 패널을 2x2로 배치하고 모델별 출력 차이를 해설.
7. **Point/Hybrid 출력 사례**: STEERER, PET, APGCC의 동일 `img_0062`
   패널과 localization 해석을 제시.
8. **상용화 권리 및 CCTV 적용 판단**: code, dataset, pretrained weight,
   derived weight, deployment 권리를 분리한 표와 fixed-CCTV calibration,
   overlap ownership 적용 판단.
9. **최종 추천과 다음 단계**: CCTV ROI·perspective·zone calibration,
   overlap merge 검증, 현장 데이터 수집, 추가 domain 평가, 권리 적격
   최종 1개 모델 fine-tuning 순서.

## 데이터와 증거

- 비교 원본: `results/round1-cctv-comparison-34ad450/comparison.json`
- 패널 원본: 해시 검증된 `assets-manifest.json`의 19개 패널 중 공유용으로
  필요한 동일 장면 7개만 사용한다.
- canonical split SHA-256:
  `da1d947aad73d45be573d52a6462fa8022948fba2b1026a3bfff76b11a4b5c67`
- 모델 순서: STEERER, DM-Count, PET, MPCount, APGCC, CSRNet.
- 공유본 작성 중 inference, training, fine-tuning, test split 접근, 큐 변경을
  수행하지 않는다.

## 검증 기준

- PDF는 정확히 A4 세로 9페이지여야 한다.
- 여섯 모델명, UCF-QNRF, validation 36장, fine-tuning 없음,
  cross-domain compatibility smoke, PET 연구 비교군 문구가 추출되어야 한다.
- 모든 패널의 SHA-256이 manifest와 일치해야 한다.
- Poppler로 9페이지를 PNG로 렌더링한 뒤 전체 페이지를 시각 검수한다.
- 한글 깨짐, 표 잘림, 텍스트 겹침, 캡션 누락, 과도하게 작은 출력물,
  헤더·바닥글 누락이 하나라도 있으면 전달하지 않는다.

## 범위 밖

- 새로운 모델 평가나 재추론
- 재학습 또는 fine-tuning
- 제품 배포 승인
- 새로운 라이선스 법률 판단
- GitHub에 PDF 게시
