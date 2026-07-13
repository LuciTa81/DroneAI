# Stage 0 review guide

## Purpose

Stage 0 proves only that one Colab session can be traced to an exact Git commit,
use a CUDA GPU, write persistent artifacts to Google Drive and pass the repository
tests. It does not claim that a crowd-counting model is accurate.

## Pass rule

- Score: at least 85/100
- Every blocker: PASS
- Seeds: 17, 42 and 2026

The CUDA requirement is intentionally a blocker because the next stage is a
training smoke test. The exact GPU model is recorded but is not graded; Colab
may assign different GPUs between sessions.

## Score breakdown

| Group | Points | What is checked |
|---|---:|---|
| Runtime | 35 | Python, PyTorch, CUDA visibility, GPU identity, CUDA tensor operation |
| Storage | 25 | Drive layout, write/read roundtrip, persistent environment report |
| Reproducibility | 25 | Git SHA, clean tree, versions, three seeds, config snapshot |
| Automation | 15 | Unit tests and runtime-report schema |

## Run

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
└── score.md
```

## Manual verification

Before approving Stage 0, confirm:

- `git_commit` is present and matches the GitHub commit used by the notebook.
- `git_dirty` is false.
- `torch.cuda_available` is true and a GPU name is shown.
- The Drive report remains after restarting the Colab runtime.
- `automation.tests` is PASS.
- There are no failed blockers even if the numeric score is 85 or higher.
