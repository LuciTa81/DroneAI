# STEERER T800 progress

This directory stores Git-safe milestone evidence while the authoritative
800-epoch run continues on the home5090 SSD. It does not contain model weights.

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
