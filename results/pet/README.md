# PET evaluation evidence

PET is being evaluated as a point-based research comparison only.

- Rights gate: `PASS_RESEARCH_ONLY`, 84/100
- Allowed: catalog, research reproduction, research asset download, research checkpoint evaluation
- Blocked: commercial fine-tuning, derived commercial weights, festival deployment
- Dataset lane: frozen UCF-QNRF validation subset; official test remains sealed
- Preflight: `PASS` on RTX 5090 with current NGC Torch/CUDA
- Checkpoint: strict load, 20,909,385 parameters, no missing/unexpected keys
- One-sample gate: `PASS` on frozen medium-density validation image `img_0067`
- One-sample result: GT 1453, predicted 1762, localization F1 0.6407,
  latency 231.28 ms, peak VRAM 3444.80 MB
- Current next gate: frozen 36-image benchmark

The accepted rights artifact is
`pet-rights-99bfc15/rights-decision.json` with SHA-256
`40d5f7c9af3417567ae55d8f1e634c144a68a23b44314057313455cf27c438d7`.
