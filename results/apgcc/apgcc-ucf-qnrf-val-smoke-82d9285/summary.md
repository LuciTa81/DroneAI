# Evaluation summary: apgcc-ucf-qnrf-validation-smoke

- Model: `apgcc-official-shha-to-ucf-qnrf`
- Protocol: `apgcc-official-shha-to-qnrf-val-smoke-v1`
- Dataset/split: `ucf-qnrf-kaggle-apache` / `dm-count-upstream-qnrf-val-smoke-36-v1` (validation)
- Comparison scope: `compatibility_smoke`
- Checkpoint training split: `VERIFIED_DISJOINT`
- Ranking eligible: `False`
- Checkpoint split evidence: The official checkpoint is trained on ShanghaiTech Part A, so no UCF-QNRF validation sample belongs to checkpoint training; the result is cross-domain compatibility evidence only.
- Samples: 36 successful, 0 explicit failures
- MAE/RMSE/bias: 266.1944 / 431.2962 / -93.6389
- Spatial: `localization_f1` mean=0.6831052027940661 direction=maximize
- Rights scope: `PASS_COMMERCIAL_CANDIDATE`; see `rights-decision.json`
- This summary does not replace `score.md` or its blockers.
