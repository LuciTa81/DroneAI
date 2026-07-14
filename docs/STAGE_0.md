# Stage 0 review guide

## Purpose

Stage 0 proves only that one configured runtime can be traced to an exact Git
commit, use a CUDA GPU, write persistent artifacts and pass the repository
tests. It does not claim that a crowd-counting model is accurate or commercially
deployable.

## Pass rule

- Score: at least 85/100
- Every blocker: PASS
- Seeds: 17, 42 and 2026

The CUDA requirement is intentionally a blocker because the next stage is a
training smoke test. Stage 0 performs a deterministic CUDA matrix
multiplication and records the exact GPU, but the GPU model itself is not
graded.

## Score breakdown

| Group | Points | What is checked |
|---|---:|---|
| Runtime | 35 | Python, PyTorch, CUDA visibility, GPU identity, CUDA tensor operation |
| Storage | 25 | Profile layout, write/read roundtrip, persistent environment report |
| Reproducibility | 25 | Git SHA, clean tree, versions, three seeds, config snapshot |
| Automation | 15 | Unit tests and runtime-report schema |

## Run on the default home5090 backend

From the running container:

```bash
cd /workspace
. .venvs/harness/bin/activate
python scripts/run_stage0.py --profile configs/runtime/home5090_docker.json
```

The final artifacts are written to
`/workspace/data/results/stage-0/`, including `run_manifest.json` with the
profile hash, container image digest, research-only scope, and artifact hashes.

## Colab fallback

In Colab, execute `notebooks/01_stage0_gate.ipynb` from top to bottom. The private
repository bootstrap expects a Colab secret named `GITHUB_TOKEN`. Give the secret
read access to the notebook and use a fine-grained token scoped only to
`LuciTa81/DroneAI` with Contents read permission.

The final artifacts are written to:

```text
MyDrive/DroneAI/runs/stage-0/
├── environment.json
├── stage0_config.json
├── score.json
├── score.md
└── run_manifest.json
```

## Manual verification

Before approving Stage 0, confirm:

- `git_commit` is present and matches the GitHub commit used by the notebook.
- `git_dirty` is false.
- `torch.cuda_available` is true and a GPU name is shown.
- The report remains on the configured persistent storage after restarting the runtime.
- `run_manifest.json` says `production_approved=false` and contains 64-character artifact SHA-256 values for the home backend.
- `automation.tests` is PASS.
- There are no failed blockers even if the numeric score is 85 or higher.
