# Common evaluation harness

The common fixture proves that the evaluation harness, evidence bundle, score,
and review-panel selection work end to end. It is not evidence of real-model
quality and does not grant commercial or production approval.

## Required review order

1. Model architecture brief and reviewed code paths.
2. Rights decision for the requested action.
3. Synthetic or approved non-test smoke.
4. Frozen protocol and complete labeled evaluation.
5. 100-point score, blockers, and 12 selected panels.
6. Explicit user approval before the next model.

Technical scoring and rights scope are separate gates. A technically successful
run can remain `PASS_RESEARCH_ONLY` or `PASS_COMMERCIAL_CANDIDATE`; only verified
component-level rights evidence can support `PRODUCTION_APPROVED`.

## Artifact split

- Git: model brief, sample and rights manifests, predictions table, score, and
  12 panels.
- home5090 SSD: checkpoints, full logs, native outputs, and large evidence.

Git review bundles must stay below 25 MiB. Native outputs, datasets, and weights
remain under `/workspace/data` and are referenced by versioned identity and
SHA-256 rather than committed.

## Commands

Run these inside the repository. `home5090_docker` is the default runtime; the
CPU fixture is available for local control-plane checks.

```bash
python -m pip install -e ".[dev,evaluation]"
python scripts/run_evaluation_fixture.py --config configs/evaluation/fixture_density.json --output-dir /tmp/droneai-fixture-cpu-$(date +%Y%m%d-%H%M%S) --device cpu
python scripts/run_evaluation_fixture.py --config configs/evaluation/fixture_density.json --output-dir /workspace/data/results/fixture-density/smoke --device cuda
python -m pytest -q
```

Use a new output directory for every fixture. The command refuses to overwrite
a non-empty destination. A complete fixture contains `score.md`, `score.json`,
`summary.md`, `predictions.csv`, frozen sample and rights evidence, environment
provenance, per-sample native-output/preprocessing metadata, and exactly 12
deterministic four-column review panels.

## Verified home5090 fixture

The 2026-07-15 verification used code commit `a1620a9` in
`nvcr.io/nvidia/pytorch:25.10-py3`. The full container suite passed 200 tests.
The CUDA fixture recorded the RTX 5090, PyTorch
`2.9.0a0+145a3a7bda.nv25.10`, CUDA 13.0, a clean Git tree, 36 prediction rows,
12 panels, and a 231630-byte bundle. Its technical gate was 100/100, while its
scope correctly remained `PASS_RESEARCH_ONLY` because it is a synthetic
harness proof.
