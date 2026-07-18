# Evaluation summary: csrnet-ucf-qnrf-validation-smoke

- Model: `csrnet-independent-shha-to-ucf-qnrf`
- Protocol: `csrnet-official-shha-to-qnrf-val-smoke-v1`
- Dataset/split: `ucf-qnrf-kaggle-apache` / `dm-count-upstream-qnrf-val-smoke-36-v1` (validation)
- Comparison scope: `compatibility_smoke`
- Checkpoint training split: `VERIFIED_DISJOINT`
- Ranking eligible: `False`
- Checkpoint split evidence: The rights-gated official checkpoint is trained on ShanghaiTech Part A and the frozen evaluation split is UCF-QNRF validation, so their samples are disjoint; the result remains a cross-domain compatibility smoke.
- Samples: 36 successful, 0 explicit failures
- MAE/RMSE/bias: 332.3291 / 522.0313 / -86.8039
- Spatial: `game_l1` mean=364.1291174650485 direction=minimize
- Rights scope: `PASS_COMMERCIAL_CANDIDATE`; see `rights-decision.json`
- This summary does not replace `score.md` or its blockers.
