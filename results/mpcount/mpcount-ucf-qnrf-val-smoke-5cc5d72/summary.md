# Evaluation summary: mpcount-ucf-qnrf-validation-smoke

- Model: `mpcount-official-sta-to-ucf-qnrf`
- Protocol: `mpcount-official-sta-to-qnrf-val-smoke-v1`
- Dataset/split: `ucf-qnrf-kaggle-apache` / `dm-count-upstream-qnrf-val-smoke-36-v1` (validation)
- Comparison scope: `compatibility_smoke`
- Checkpoint training split: `VERIFIED_DISJOINT`
- Ranking eligible: `False`
- Checkpoint split evidence: The deterministic checkpoint is trained on ShanghaiTech-A and evaluated on UCF-QNRF, so no UCF-QNRF validation sample belongs to checkpoint training.
- Samples: 36 successful, 0 explicit failures
- MAE/RMSE/bias: 213.6244 / 333.5529 / -17.6008
- Spatial: `game_l1` mean=239.27004881572398 direction=minimize
- Rights scope: `PASS_COMMERCIAL_CANDIDATE`; see `rights-decision.json`
- This summary does not replace `score.md` or its blockers.
