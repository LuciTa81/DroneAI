# STEERER T800 recovery evidence

The first real T800 attempt resumed the verified epoch-5 checkpoint with AMP.
It completed epoch 9 and stopped at epoch 10 when the harness detected a
non-finite gradient norm. No T800 boundary checkpoint had been promoted, so the
verified `milestone-005.pth` remains the authoritative predecessor.

The recovery is intentionally explicit: `--precision fp32` is accepted only
for an approved, resumed T800 run. Model, optimizer, scheduler, RNG, and
best-metric state are restored; only the AMP scaler state is discarded. The
original failed log remains immutable and the recovery writes
`run-result-t800-fp32.jsonl`.

The diagnosis remains provisional until the real FP32 run crosses epoch 10.
