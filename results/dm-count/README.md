# DM-Count result index

## UCF-QNRF validation smoke (`cab50d1`)

- Model: official DM-Count UCF-QNRF checkpoint, pinned upstream commit
  `cc5f2132e0d1328909f31b6d665b8e0b15c30467`
- Input/output: RGB image -> non-negative density map -> density sum as count
- Split: 36 samples selected deterministically from the pinned 120-image
  DM-Count validation list, balanced as 12 low/12 medium/12 high-density images
- Result: MAE `154.2865`, RMSE `229.8056`, signed bias `+3.0316`
- Spatial result: GAME-L1 mean `168.8068` over four operator zones
- Runtime: median `69.52 ms` (`14.38 FPS`, batch 1); peak VRAM `8.98 GB`
- Coverage: 36/36 successful, 0 explicit inference failures
- Harness gate: `PASS_RESEARCH_ONLY`, `100/100`
- Rights gate: `PASS_COMMERCIAL_CANDIDATE`, `92/100`

The harness score measures protocol completeness against frozen smoke thresholds;
it is not a model-accuracy percentage. The paper's reported UCF-QNRF test result
(MAE 85.6/RMSE 148.3) is not directly comparable with this 36-image validation
smoke. The runner exposed no test-root argument and did not perform training or
fine-tuning.

The accepted Kaggle Apache-2.0 label authorizes the current asset-download and
frozen-evaluation lane under project policy. The uploader provenance caveat,
derived-weight rights, deployment permission, and production approval remain
separate gates.

Artifacts: [full small evidence bundle](ucf-qnrf-val-smoke-cab50d1/summary.md),
[score](ucf-qnrf-val-smoke-cab50d1/score.md),
[model structure and output contract](ucf-qnrf-val-smoke-cab50d1/model-brief.md),
and [selection manifest](ucf-qnrf-val-smoke-cab50d1/selection_manifest.json).
The 12 committed panels cover best/strong/typical cases for every density band
and three high-density failure cases.
