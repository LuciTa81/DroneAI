# STEERER long-run progress (user ceiling: epoch 400)

This directory stores Git-safe milestone evidence from the authoritative
home5090 SSD run. The runner retained its audited T800 schedule identity, but
the user reduced the authorized execution ceiling to epoch 400 on 2026-08-07.
The epoch-400 validation, checkpoint, and manifest were completed before the
process received `SIGTERM`. No model weights are stored in Git.

At epoch 100, the complete fixed 240-image validation split produced MAE
150.773 and RMSE 389.616. Both best-MAE and best-RMSE advanced to epoch 100,
and `milestone-100.pth` was preserved. Every checkpoint hash in
`milestone-100.json` was recomputed from the SSD and matched the checkpoint
manifest.

The official UCF-QNRF Test remains sealed. The paper gap is diagnostic only
because this run uses a Train-derived holdout while the paper reports the
official Test after full training.

At epoch 200, MAE/RMSE were 154.786/537.058, so the epoch-175 best
129.808/379.434 remained selected. `milestone-200.pth` was still preserved as
an independent, hash-verified recovery point.

At epoch 300, MAE/RMSE were 130.746/464.236. The selected count checkpoints
therefore remained best-MAE epoch 250 and best-RMSE epoch 175, while
`milestone-300.pth` was preserved and hash verified.

At epoch 350, the fixed validation split produced a new best MAE of 120.409
with RMSE 429.212. At epoch 400, MAE/RMSE were 122.830/460.392, so the selected
checkpoints at the user ceiling were best-MAE epoch 350 and best-RMSE epoch
175. `milestone-400.pth`, `last.pth`, both best checkpoints, the checkpoint
manifest, and the run log were independently re-hashed after shutdown.

All four selected/recovery checkpoint files were then loaded into fresh model
instances and re-evaluated on all 240 validation samples. Their report MAE and
RMSE exactly matched the stored boundary metrics. See
`../reevaluation-epoch-400.json` and `../final-report-epoch-400.md`.
