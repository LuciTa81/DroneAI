# MPCount Frozen 36-Sample Benchmark Design

## Outcome

Add one MPCount-specific command that evaluates the already accepted deterministic
checkpoint on exactly the frozen 36-image UCF-QNRF validation subset. The command
does not expose a test path, training, fine-tuning, download, or resume option.

## Selected approach

Use a thin model-specific runner around the existing `run_evaluation` harness.
This matches the established DM-Count, STEERER, and PET pattern and preserves one
common artifact and scoring contract. A generic multi-model CLI would expand scope
without improving this gate, while invoking the one-sample runner 36 times would
reload the model and would not produce an atomic benchmark bundle.

## Inputs and validation

The runner accepts explicit paths for the evaluation config, UCF-QNRF Train root,
pinned MPCount checkout, pinned DM-Count split checkout and lists, deterministic
checkpoint and expected SHA-256, accepted rights decision and manifest, new output
directory, CUDA device, and patch size. Before constructing the adapter, it rejects
a non-empty output directory, a rights mismatch, a dirty or unpinned split checkout,
and any split/config/hash mismatch. The source test partition is never accepted.

## Evaluation flow

1. Load the frozen MPCount config and validate the accepted rights bundle.
2. Revalidate the pinned DM-Count train/validation lists and prepare exactly 36
   validation samples using seed `20260715`.
3. Instantiate one `MPCountAdapter` with the pinned checkpoint and official patch
   size `3584`.
4. Build the existing density compatibility protocol and call `run_evaluation`
   once for all samples.
5. Preserve the complete run on `/workspace/data/results`; copy only summaries,
   manifests, scores, CSV predictions, and 12 curated panels to Git.

## Outputs and success criteria

The common harness must produce `predictions.csv`, `metrics.json`,
`selection_manifest.json`, `evidence-manifest.json`, `model-brief.md`,
`environment-summary.json`, `score.json`, `score.md`, and exactly 12 review panels.
Metrics include MAE, RMSE, signed bias, MAPE reference, density GAME-L1, zone MAE,
PSNR, SSIM, latency, FPS, and peak VRAM. A successful technical gate requires all
36 samples, no inference failures, valid artifact hashes, the required density
bands, and the configured score checks. It remains a `compatibility_smoke` and is
not ranking-eligible or production approval.

## Error handling and recovery

The output directory is immutable once non-empty. Any failure remains recorded on
the SSD and is not overwritten; a corrected rerun uses a new run ID. CUDA OOM,
invalid density, rights failure, hash mismatch, and dirty upstream state fail
closed. No automatic patch-size change is allowed because it would change the
declared protocol. Long execution uses the existing container and is recoverable
by launching a new, separately named run; no checkpoint resume is relevant because
this is inference only.

## Testing

Runner tests enforce the allowed CLI surface, exact adapter arguments including
patch size, early rejection of non-empty output, fail-closed rights and split
checks, exactly 36 prepared samples, and delegation to the common evaluation
harness. Focused tests run before the complete repository suite. The real run is
accepted only after artifact counts, hashes, score, metrics, and panels are
independently re-read.
