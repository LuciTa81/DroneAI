# Home RTX 5090 Backend Design

## Goal

Keep the existing GitHub, harness, Colab notebooks, Drive artifacts, and Draft
PR history while making the `home5090_docker` profile the default backend for
new smoke, training, and evaluation work.

## Boundaries

- Historical notebooks and recorded Colab/Drive paths remain unchanged.
- New runtime code consumes an explicit JSON profile instead of assuming
  `/content` or `MyDrive`.
- The home profile uses `/workspace` for source and `/workspace/data` for large
  assets. Its host bind is limited to `/mnt/crowd-data/CrowdCounting`.
- No dataset transfer or long training is part of this change.
- ShanghaiTech and other restricted assets remain in the research-only lane.
  A successful runtime smoke is not a commercial or production approval.
- Existing files in `/home/lucita/crowd-counting-lab` are preserved.

## Selected architecture

Runtime profiles live under `configs/runtime/` and share one validated schema.
The two profiles are `colab_drive` for legacy/fallback execution and
`home5090_docker` for new default execution. A profile records the source root,
storage directories, container identity, execution policy, and license scope.

Stage 0 accepts either a profile or the legacy `--drive-root` argument. The
profile path is the primary interface; the legacy argument preserves every
existing Colab notebook. Stage 0 writes the environment, profile/config
snapshot, score, and a manifest whose artifact entries contain recomputed
SHA-256 values. Its CUDA blocker performs matrix multiplication, not just
device discovery.

DM-Count keeps explicit `--data-dir`, `--run-dir`, `--work-dir`, and
`--upstream-dir` inputs. Colab notebooks already provide the Colab path; the
home runbook provides `/workspace` paths. Model dependencies are installed in
a per-model venv that inherits PyTorch from the pinned NGC container and adds
only an exact overlay lock.

## Safety and evidence

Profile validation rejects relative storage paths and any datasets,
checkpoints, or results path outside the declared storage root. The home smoke
must capture the Git commit and dirty state, container tag, Python, PyTorch,
CUDA, GPU, seeds, research-only license scope, and hashes for persisted small
artifacts. Dataset and split fields are explicitly null during a synthetic
foundation smoke.

The Windows branch is verified and pushed before the server is initialized.
Because the server bind root already contains ignored `data/` and an unrelated
`test/startcodex.txt`, repository initialization preserves both and uses a
local `.git/info/exclude` entry for `test/`; no existing file is deleted or
moved.

## Test strategy

Unit tests cover profile parsing, containment guards, exact home paths,
manifest SHA-256 entries, legacy Stage 0 compatibility, and required explicit
Stage 3 upstream paths. The existing full suite must pass through the same
`pytest` command used by GitHub Actions. After push, the server runs the full
harness and a synthetic Stage 0 CUDA smoke inside `crowd-jupyter`. No dataset
or pretrained weight is needed for this gate.

## Deferred approval gates

Stage 3C rights/fit approval, dataset transfer and hash verification, DM-Count
compatibility smoke, and all long training remain separate approval points.
Checkpoint/resume is mandatory before any long run, but a tmux session alone
is not considered reboot recovery.
