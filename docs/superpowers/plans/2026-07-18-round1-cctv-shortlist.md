# Round 1 Fixed-CCTV Technical Shortlist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, hash-gated 100-point fixed-CCTV technical scorecard for all six accepted Round 1 model runs while keeping commercial-rights status on an independent deployment axis.

**Architecture:** Split the work into a pure frozen-scorecard module, an SSD evidence loader, a comparison/shortlist builder, and an atomic report writer with a thin CLI. The loader accepts only the queue's hash-verified 36-image validation evidence, family adapters expose density/point/hybrid capabilities without imputing incompatible metrics, and the comparison builder produces a provisional top three plus a separate rights label.

**Tech Stack:** Python 3.12, standard-library `csv`, `dataclasses`, `hashlib`, `json`, `math`, `statistics`, `pathlib`, existing DroneAI integrity/artifact helpers, and pytest 8.

## Global Constraints

- Use exactly `ucf-qnrf-kaggle-apache` / `dm-count-upstream-qnrf-val-smoke-36-v1` / `validation` / 36 samples.
- Never accept or expose a test split or test path.
- Do not run inference, training, fine-tuning, or create derived weights.
- Keep every model `ranking_eligible=false`; label the result `PROVISIONAL_TECHNICAL_SHORTLIST`.
- Technical score and rights status are independent; rights changes cannot alter technical points.
- `PRODUCT_ELIGIBLE` requires `PRODUCTION_APPROVED`; `PASS_COMMERCIAL_CANDIDATE` maps to `PENDING_DUE_DILIGENCE`.
- Recompute missing aggregates only from the sibling hash-verified `predictions.csv`; never invent values.
- Fail closed on missing evidence, hash mismatch, mixed samples, non-finite required values, unknown families, or a Git review bundle above 25 MiB.
- Keep datasets, checkpoints, raw arrays, and full runs under `/workspace/data`; commit only small configuration, code, tests, summaries, manifests, and links to existing panels.
- Use `apply_patch` for repository edits, focused RED/GREEN tests for each task, frequent commits, safe push, and `git pull --ff-only` on home5090.

---

## File structure

- Create `configs/evaluation/round1_cctv_scorecard.json`: immutable model list, score weights, anchors, split identity, and tie-break rules.
- Create `src/droneai/round_scorecard.py`: validate the config and calculate pure category/technical scores.
- Create `src/droneai/round_evidence.py`: load queue entries, verify nested evidence, validate canonical samples, and derive allowed aggregates/capabilities.
- Create `src/droneai/round_comparison.py`: map rights labels, combine scores, build the provisional shortlist, and render serializable rows.
- Create `src/droneai/round_report.py`: write JSON/CSV/Markdown/config snapshot/evidence manifest atomically and enforce the Git budget.
- Create `scripts/build_round1_cctv_comparison.py`: parse CLI arguments, require clean Git, invoke the four modules, and print the comparison summary.
- Create `tests/test_round_scorecard.py`: formula, boundary, config, and weight tests.
- Create `tests/test_round_evidence.py`: accepted/nested hash, split, sample, aggregate, capability, and family tests.
- Create `tests/test_round_comparison.py`: shortlist, tie-break, rights independence, deployment-label, and limitation tests.
- Create `tests/test_round_report.py`: deterministic artifacts, atomic failure, manifest, Markdown, and 25 MiB tests.
- Create `tests/test_build_round1_cctv_comparison.py`: CLI success and fail-closed error JSON tests.
- Modify `docs/HOME5090_RUNBOOK.md`: add the approved Round 1 comparison command and review gate.
- Generate `results/round1-cctv-comparison-<implementation-commit>/`: reviewed small artifacts only.

---

### Task 1: Freeze and implement the pure CCTV scorecard

**Files:**
- Create: `configs/evaluation/round1_cctv_scorecard.json`
- Create: `src/droneai/round_scorecard.py`
- Create: `tests/test_round_scorecard.py`

**Interfaces:**
- Consumes: JSON configuration at `configs/evaluation/round1_cctv_scorecard.json`.
- Produces: `ScorecardConfig`, `load_scorecard(path: str | Path) -> ScorecardConfig`, `piecewise(value: float, full: float, zero: float) -> float`, and `score_technical(...) -> dict[str, object]`.

- [ ] **Step 1: Write failing config and score tests**

```python
from pathlib import Path

import pytest

from droneai.round_scorecard import load_scorecard, piecewise, score_technical


CONFIG = Path("configs/evaluation/round1_cctv_scorecard.json")


def _aggregates() -> dict[str, float]:
    return {
        "mean_zone_mae": 75.0,
        "spatial_metric_value": 250.0,
        "max_absolute_band_bias": 200.0,
        "mae": 125.0,
        "rmse": 200.0,
        "mape_reference": 20.0,
        "high_band_bias": -200.0,
        "median_latency_ms": 66.67,
        "peak_vram_mb": 8192.0,
    }


def test_repository_scorecard_is_frozen_and_totals_100() -> None:
    config = load_scorecard(CONFIG)
    assert config.required_model_ids == (
        "dm-count", "steerer", "pet", "mpcount", "apgcc", "csrnet"
    )
    assert sum(config.category_weights.values()) == 100
    assert config.split_role == "validation"
    assert config.expected_samples == 36


def test_piecewise_has_exact_full_linear_and_zero_boundaries() -> None:
    assert piecewise(50.0, 66.67, 500.0) == 1.0
    assert piecewise(500.0, 66.67, 500.0) == 0.0
    assert piecewise((66.67 + 500.0) / 2, 66.67, 500.0) == pytest.approx(0.5)


def test_density_score_uses_frozen_anchors() -> None:
    config = load_scorecard(CONFIG)
    result = score_technical(
        family="density",
        spatial_metric_name="game_l1",
        aggregates=_aggregates(),
        capabilities={
            "zone_aggregation": True,
            "density_map": True,
            "point_localization": False,
            "confidence": False,
        },
        checkpoint_training_split_status="VERIFIED_DISJOINT",
        coverage_complete=True,
        artifacts_verified=True,
        config=config,
    )
    assert result["categories"] == {
        "zone_spatial": pytest.approx(17.5),
        "count_stability": pytest.approx(12.5),
        "runtime_resources": pytest.approx(15.0),
        "operator_integration": pytest.approx(10.0),
        "evidence_quality": pytest.approx(10.0),
    }
    assert result["technical_score"] == pytest.approx(65.0)


def test_unknown_family_is_rejected() -> None:
    config = load_scorecard(CONFIG)
    with pytest.raises(ValueError, match="unsupported model family"):
        score_technical(
            family="detector",
            spatial_metric_name="game_l1",
            aggregates=_aggregates(),
            capabilities={},
            checkpoint_training_split_status="UNKNOWN",
            coverage_complete=True,
            artifacts_verified=True,
            config=config,
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
& 'C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\work\DroneAI\.venv\Scripts\python.exe' -m pytest tests/test_round_scorecard.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'droneai.round_scorecard'`.

- [ ] **Step 3: Add the exact frozen configuration**

```json
{
  "schema_version": 1,
  "round_id": "round-1-static-ucf-qnrf",
  "comparison_id": "round1-fixed-cctv-technical-shortlist-v1",
  "dataset_id": "ucf-qnrf-kaggle-apache",
  "split_id": "dm-count-upstream-qnrf-val-smoke-36-v1",
  "split_role": "validation",
  "expected_samples": 36,
  "required_model_ids": ["dm-count", "steerer", "pet", "mpcount", "apgcc", "csrnet"],
  "category_weights": {
    "zone_spatial": 35,
    "count_stability": 25,
    "runtime_resources": 15,
    "operator_integration": 15,
    "evidence_quality": 10
  },
  "anchors": {
    "zone_mae_zero": 150.0,
    "game_l1_zero": 500.0,
    "localization_f1_full": 0.8,
    "band_bias_zero": 400.0,
    "mae_zero": 250.0,
    "rmse_zero": 400.0,
    "mape_zero_percent": 40.0,
    "high_bias_zero": 400.0,
    "latency_full_ms": 66.67,
    "latency_zero_ms": 500.0,
    "vram_full_mb": 8192.0,
    "vram_zero_mb": 24576.0
  },
  "capability_points": {
    "zone_aggregation": 6,
    "density_map": 4,
    "point_localization": 3,
    "confidence": 2
  },
  "shortlist_size": 3,
  "tie_breakers": ["mean_zone_mae", "mae", "model_id"],
  "shortlist_label": "PROVISIONAL_TECHNICAL_SHORTLIST",
  "ranking_eligible": false,
  "review_budget_bytes": 26214400
}
```

- [ ] **Step 4: Implement config validation and pure formulas**

```python
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_FAMILIES = {"density", "points", "density_and_points"}


@dataclass(frozen=True)
class ScorecardConfig:
    round_id: str
    comparison_id: str
    dataset_id: str
    split_id: str
    split_role: str
    expected_samples: int
    required_model_ids: tuple[str, ...]
    category_weights: dict[str, int]
    anchors: dict[str, float]
    capability_points: dict[str, int]
    shortlist_size: int
    tie_breakers: tuple[str, ...]
    shortlist_label: str
    ranking_eligible: bool
    review_budget_bytes: int


def clamp(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("score inputs must be finite")
    return min(1.0, max(0.0, value))


def piecewise(value: float, full: float, zero: float) -> float:
    if not all(math.isfinite(item) for item in (value, full, zero)) or full >= zero:
        raise ValueError("piecewise requires finite bounds with full < zero")
    if value <= full:
        return 1.0
    if value >= zero:
        return 0.0
    return (zero - value) / (zero - full)


def load_scorecard(path: str | Path) -> ScorecardConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("scorecard schema_version must be 1")
    config = ScorecardConfig(
        round_id=str(payload["round_id"]),
        comparison_id=str(payload["comparison_id"]),
        dataset_id=str(payload["dataset_id"]),
        split_id=str(payload["split_id"]),
        split_role=str(payload["split_role"]),
        expected_samples=int(payload["expected_samples"]),
        required_model_ids=tuple(payload["required_model_ids"]),
        category_weights={str(k): int(v) for k, v in payload["category_weights"].items()},
        anchors={str(k): float(v) for k, v in payload["anchors"].items()},
        capability_points={str(k): int(v) for k, v in payload["capability_points"].items()},
        shortlist_size=int(payload["shortlist_size"]),
        tie_breakers=tuple(payload["tie_breakers"]),
        shortlist_label=str(payload["shortlist_label"]),
        ranking_eligible=bool(payload["ranking_eligible"]),
        review_budget_bytes=int(payload["review_budget_bytes"]),
    )
    if sum(config.category_weights.values()) != 100:
        raise ValueError("category weights must total 100")
    if sum(config.capability_points.values()) != config.category_weights["operator_integration"]:
        raise ValueError("capability points must match operator integration weight")
    if config.split_role != "validation" or config.ranking_eligible:
        raise ValueError("Round 1 scorecard must remain non-ranking validation evidence")
    if len(set(config.required_model_ids)) != len(config.required_model_ids):
        raise ValueError("required model IDs must be unique")
    return config


def score_technical(*, family: str, spatial_metric_name: str,
                    aggregates: dict[str, float], capabilities: dict[str, bool],
                    checkpoint_training_split_status: str, coverage_complete: bool,
                    artifacts_verified: bool, config: ScorecardConfig) -> dict[str, object]:
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported model family: {family}")
    a = config.anchors
    zone = 20 * clamp(1 - aggregates["mean_zone_mae"] / a["zone_mae_zero"])
    if spatial_metric_name == "game_l1":
        spatial = 10 * clamp(1 - aggregates["spatial_metric_value"] / a["game_l1_zero"])
    elif spatial_metric_name == "localization_f1":
        spatial = 10 * clamp(aggregates["spatial_metric_value"] / a["localization_f1_full"])
    else:
        raise ValueError(f"unsupported primary spatial metric: {spatial_metric_name}")
    robust = 5 * clamp(1 - aggregates["max_absolute_band_bias"] / a["band_bias_zero"])
    count = (
        10 * clamp(1 - aggregates["mae"] / a["mae_zero"])
        + 7 * clamp(1 - aggregates["rmse"] / a["rmse_zero"])
        + 4 * clamp(1 - aggregates["mape_reference"] / a["mape_zero_percent"])
        + 4 * clamp(1 - abs(aggregates["high_band_bias"]) / a["high_bias_zero"])
    )
    runtime = (
        9 * piecewise(aggregates["median_latency_ms"], a["latency_full_ms"], a["latency_zero_ms"])
        + 6 * piecewise(aggregates["peak_vram_mb"], a["vram_full_mb"], a["vram_zero_mb"])
    )
    operator = sum(
        config.capability_points[name]
        for name in config.capability_points
        if capabilities.get(name) is True
    )
    evidence = (4 if coverage_complete else 0) + (4 if artifacts_verified else 0)
    if checkpoint_training_split_status == "VERIFIED_DISJOINT":
        evidence += 2
    categories = {
        "zone_spatial": zone + spatial + robust,
        "count_stability": count,
        "runtime_resources": runtime,
        "operator_integration": float(operator),
        "evidence_quality": float(evidence),
    }
    return {"categories": categories, "technical_score": sum(categories.values())}
```

- [ ] **Step 5: Run focused GREEN tests**

Run the Task 1 pytest command again.

Expected: `4 passed`.

- [ ] **Step 6: Commit Task 1**

```powershell
git add configs/evaluation/round1_cctv_scorecard.json src/droneai/round_scorecard.py tests/test_round_scorecard.py
git commit -m "Add frozen Round 1 CCTV scorecard"
```

---

### Task 2: Load and verify all six accepted model runs

**Files:**
- Create: `src/droneai/round_evidence.py`
- Create: `tests/test_round_evidence.py`

**Interfaces:**
- Consumes: `ScorecardConfig`, `configs/evaluation/model_queue.json`, and accepted result directories.
- Produces: `ModelRunEvidence` and `load_verified_round(queue_path: str | Path, results_root: str | Path, config: ScorecardConfig) -> tuple[ModelRunEvidence, ...]`; every returned run carries the same `canonical_split_sha256` field.

- [ ] **Step 1: Write a fixture-bundle helper and failing happy-path test**

```python
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from droneai.round_evidence import load_verified_round
from droneai.round_scorecard import load_scorecard


def _json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(root: Path, model_id: str, family: str, rights: str) -> dict[str, object]:
    run = root / model_id / "run"
    run.mkdir(parents=True)
    fields = [
        "sample_id", "ground_truth_count", "density_band",
        "condition_source_partition", "normalized_error", "zone_mae",
        "game_l1", "localization_f1", "latency_ms", "peak_vram_mb",
        "output_type", "failure_state", "signed_error", "absolute_error",
    ]
    with (run / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index in range(36):
            band = ("low", "medium", "high")[index // 12]
            writer.writerow({
                "sample_id": f"img_{index:04d}", "ground_truth_count": 100 + index,
                "density_band": band, "condition_source_partition": "official_validation",
                "normalized_error": 0.1, "zone_mae": 10.0,
                "game_l1": 20.0 if family != "points" else "",
                "localization_f1": 0.7 if family != "density" else "",
                "latency_ms": 70.0, "peak_vram_mb": 4096.0,
                "output_type": {"density": "density", "points": "points",
                                "density_and_points": "hybrid"}[family],
                "failure_state": "", "signed_error": 1.0, "absolute_error": 1.0,
            })
    selection = _json(run / "selection_manifest.json", {
        "selected_count": 12, "shortfall": 0,
        "panel_paths": [f"figures/panel-{index:02d}.png" for index in range(12)],
    })
    for index in range(12):
        panel = run / f"figures/panel-{index:02d}.png"
        panel.parent.mkdir(exist_ok=True)
        panel.write_bytes(b"png fixture")
    metrics_hash = _json(run / "metrics.json", {
        "expected_samples": 36, "recorded_samples": 36, "successful_samples": 36,
        "explicit_failures": 0, "mae": 10.0, "rmse": 12.0, "signed_bias": 1.0,
        "median_latency_ms": 70.0, "throughput_fps_batch1": 14.2857,
        "peak_vram_mb": 4096.0,
        "band_signed_bias": {"low": 1.0, "medium": 1.0, "high": 1.0},
        "spatial_metric_name": "localization_f1" if family == "points" else "game_l1",
        "spatial_mean": 0.7 if family == "points" else 20.0,
    })
    rights_hash = _json(run / "rights-decision.json", {
        "status": rights,
        "component_ids": {
            "code": f"{model_id}-code", "dataset": "ucf-qnrf-kaggle-apache",
            "pretrained_weights": f"{model_id}-checkpoint",
            "derived_weights": "not-created", "deployment": "pending",
        },
        "ambiguous_components": ["pretrained_weights", "derived_weights", "deployment"],
        "restricted_components": [],
    })
    artifacts = []
    for path in [run / "predictions.csv", run / "selection_manifest.json", run / "metrics.json",
                 run / "rights-decision.json", *sorted((run / "figures").glob("*.png"))]:
        artifacts.append({"path": path.relative_to(run).as_posix(),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    evidence_hash = _json(
        run / "evidence-manifest.json", {"schema_version": 1, "artifacts": artifacts}
    )
    score_hash = _json(run / "score.json", {
        "status": "PASS_RESEARCH_ONLY", "decision_scope": "research_only", "score": 90.0,
        "checks": [{"check_id": "provenance.bundle",
                    "evidence": f"evidence-manifest.json@{evidence_hash}"}],
    })
    return {
        "model_id": model_id, "family": family, "queue_state": "completed",
        "rights_scope": rights, "dataset_id": "ucf-qnrf-kaggle-apache",
        "split_id": "dm-count-upstream-qnrf-val-smoke-36-v1", "split_role": "validation",
        "expected_samples": 36, "checkpoint_training_split_status": "VERIFIED_DISJOINT",
        "comparison_scope": "compatibility_smoke",
        "checkpoint_split_evidence": "fixture checkpoint split evidence",
        "gate_order": ["benchmark"],
        "accepted_evidence": [
            {"gate": "benchmark", "artifact_kind": "score",
             "path": f"{model_id}/run/score.json", "sha256": score_hash},
            {"gate": "benchmark", "artifact_kind": "metrics",
             "path": f"{model_id}/run/metrics.json", "sha256": metrics_hash},
        ],
    }


def test_load_verified_round_accepts_six_matching_runs(tmp_path: Path) -> None:
    config = load_scorecard("configs/evaluation/round1_cctv_scorecard.json")
    results = tmp_path / "results"
    families = ("density", "density_and_points", "points", "density", "points", "density")
    models = [_run(results, model_id, family, "PASS_COMMERCIAL_CANDIDATE")
              for model_id, family in zip(config.required_model_ids, families, strict=True)]
    models[-1]["queue_state"] = "active"
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"schema_version": 1, "round_id": config.round_id,
                                 "active_model": "csrnet", "models": models}), encoding="utf-8")

    runs = load_verified_round(queue, results, config)

    assert tuple(run.model_id for run in runs) == config.required_model_ids
    assert len({run.canonical_split_sha256 for run in runs}) == 1
    assert all(run.aggregates["mape_reference"] == pytest.approx(10.0) for run in runs)
    assert runs[0].capabilities["density_map"] is True
    assert runs[2].capabilities["point_localization"] is True
```

- [ ] **Step 2: Add failing mutation tests**

```python
def test_nested_prediction_hash_change_is_rejected(six_run_fixture) -> None:
    queue, results, config = six_run_fixture
    target = results / "dm-count/run/predictions.csv"
    target.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_verified_round(queue, results, config)


def test_mixed_sample_identity_is_rejected(six_run_fixture) -> None:
    queue, results, config = six_run_fixture
    target = results / "pet/run/predictions.csv"
    text = target.read_text(encoding="utf-8").replace("img_0000", "img_other", 1)
    target.write_text(text, encoding="utf-8")
    evidence = json.loads((results / "pet/run/evidence-manifest.json").read_text())
    for artifact in evidence["artifacts"]:
        if artifact["path"] == "predictions.csv":
            artifact["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    evidence_path = results / "pet/run/evidence-manifest.json"
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    score_path = results / "pet/run/score.json"
    score = json.loads(score_path.read_text(encoding="utf-8"))
    score["checks"][0]["evidence"] = f"evidence-manifest.json@{evidence_hash}"
    score_path.write_text(json.dumps(score, sort_keys=True) + "\n", encoding="utf-8")
    payload = json.loads(queue.read_text(encoding="utf-8"))
    pet = next(model for model in payload["models"] if model["model_id"] == "pet")
    accepted_score = next(item for item in pet["accepted_evidence"]
                          if item["artifact_kind"] == "score")
    accepted_score["sha256"] = hashlib.sha256(score_path.read_bytes()).hexdigest()
    queue.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical validation samples differ"):
        load_verified_round(queue, results, config)


def test_test_role_is_rejected_before_artifact_loading(six_run_fixture) -> None:
    queue, results, config = six_run_fixture
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["models"][0]["split_role"] = "test"
    queue.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test split"):
        load_verified_round(queue, results, config)
```

- [ ] **Step 3: Run Task 2 tests and verify RED**

Run the Windows pytest command from Task 1 with `tests/test_round_evidence.py`.

Expected: collection fails because `droneai.round_evidence` does not exist.

- [ ] **Step 4: Implement the evidence dataclass and loader**

```python
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from droneai.integrity import sha256_file, verify_artifact_reference
from droneai.model_workflow import load_model_queue
from droneai.round_scorecard import ScorecardConfig, SUPPORTED_FAMILIES


@dataclass(frozen=True)
class ModelRunEvidence:
    model_id: str
    family: str
    rights_scope: str
    checkpoint_training_split_status: str
    comparison_scope: str
    run_dir: Path
    run_relative_path: str
    score: dict[str, object]
    metrics: dict[str, object]
    rights_decision: dict[str, object]
    aggregates: dict[str, float | None]
    capabilities: dict[str, bool]
    panels: tuple[str, ...]
    canonical_split_sha256: str
    input_artifacts: tuple[dict[str, str], ...]


def _finite(rows: list[dict[str, str]], field: str) -> list[float]:
    values = [float(row[field]) for row in rows if row.get(field, "").strip()]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"non-finite prediction field: {field}")
    return values


def _accepted(model: dict[str, object], kind: str, root: Path) -> Path:
    matches = [item for item in model["accepted_evidence"]
               if item["gate"] == "benchmark" and item["artifact_kind"] == kind]
    if len(matches) != 1:
        raise ValueError(f"{model['model_id']} requires one accepted benchmark {kind}")
    target = root / matches[0]["path"]
    if sha256_file(target) != matches[0]["sha256"]:
        raise ValueError(f"accepted evidence hash mismatch: {matches[0]['path']}")
    return target


def _anchored_evidence_hash(score: dict[str, object]) -> str:
    anchors: set[str] = set()
    for check in score.get("checks", []):
        for token in str(check.get("evidence", "")).split(";"):
            token = token.strip()
            if token.startswith("evidence-manifest.json@"):
                anchors.add(token.rsplit("@", 1)[1])
    if len(anchors) != 1:
        raise ValueError("score must anchor exactly one evidence-manifest SHA-256")
    return anchors.pop()


def load_verified_round(queue_path: str | Path, results_root: str | Path,
                        config: ScorecardConfig) -> tuple[ModelRunEvidence, ...]:
    queue = load_model_queue(queue_path)
    root = Path(results_root).resolve()
    if queue["round_id"] != config.round_id:
        raise ValueError("queue round does not match scorecard")
    models = {model["model_id"]: model for model in queue["models"]}
    if tuple(model["model_id"] for model in queue["models"]) != config.required_model_ids:
        raise ValueError("queue model order does not match frozen scorecard")
    runs: list[ModelRunEvidence] = []
    for model_id in config.required_model_ids:
        model = models[model_id]
        if model["split_role"] == "test":
            raise ValueError("test split is forbidden")
        expected_identity = (config.dataset_id, config.split_id, config.split_role,
                             config.expected_samples)
        observed_identity = (model["dataset_id"], model["split_id"], model["split_role"],
                             model["expected_samples"])
        if observed_identity != expected_identity:
            raise ValueError(f"frozen evaluation identity mismatch: {model_id}")
        if model["family"] not in SUPPORTED_FAMILIES:
            raise ValueError(f"unsupported model family: {model['family']}")
        score_path = _accepted(model, "score", root)
        metrics_path = _accepted(model, "metrics", root)
        if score_path.parent != metrics_path.parent:
            raise ValueError(f"score and metrics must share one run directory: {model_id}")
        run_dir = metrics_path.parent
        score = json.loads(score_path.read_text(encoding="utf-8"))
        evidence_path = run_dir / "evidence-manifest.json"
        if sha256_file(evidence_path) != _anchored_evidence_hash(score):
            raise ValueError(f"evidence manifest hash mismatch: {model_id}")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        for reference in evidence.get("artifacts", []):
            verified, message = verify_artifact_reference(reference, base_dir=run_dir)
            if not verified:
                raise ValueError(message)
        predictions_path = run_dir / "predictions.csv"
        with predictions_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != config.expected_samples or any(row.get("failure_state") for row in rows):
            raise ValueError(f"incomplete prediction accounting: {model_id}")
        canonical = sorted((row["sample_id"], float(row["ground_truth_count"]),
                            row["density_band"], row["condition_source_partition"])
                           for row in rows)
        split_hash = hashlib.sha256(
            json.dumps(canonical, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        spatial_name = str(metrics["spatial_metric_name"])
        optional_metrics = {
            "game_l1": "mean_game_l1",
            "density_psnr": "density_psnr",
            "density_ssim": "density_ssim",
            "localization_precision": "localization_precision",
            "localization_recall": "localization_recall",
            "localization_f1": "localization_f1",
            "localization_mean_distance": "localization_mean_distance",
        }
        aggregates = {
            "mean_zone_mae": statistics.fmean(_finite(rows, "zone_mae")),
            "spatial_metric_value": float(metrics["spatial_mean"]),
            "max_absolute_band_bias": max(abs(float(value)) for value in metrics["band_signed_bias"].values()),
            "mae": float(metrics["mae"]), "rmse": float(metrics["rmse"]),
            "mape_reference": float(metrics.get("mape_reference",
                100 * statistics.fmean(_finite(rows, "normalized_error")))),
            "signed_bias": float(metrics["signed_bias"]),
            "high_band_bias": float(metrics["band_signed_bias"]["high"]),
            "median_latency_ms": float(metrics["median_latency_ms"]),
            "throughput_fps_batch1": float(metrics["throughput_fps_batch1"]),
            "peak_vram_mb": float(metrics["peak_vram_mb"]),
        }
        for column, output_name in optional_metrics.items():
            values = _finite(rows, column)
            aggregates[output_name] = statistics.fmean(values) if values else None
        capabilities = {
            "zone_aggregation": len(_finite(rows, "zone_mae")) == config.expected_samples,
            "density_map": len(_finite(rows, "game_l1")) == config.expected_samples,
            "point_localization": len(_finite(rows, "localization_f1")) == config.expected_samples,
            "confidence": len(_finite(rows, "confidence")) == config.expected_samples,
        }
        selection = json.loads((run_dir / "selection_manifest.json").read_text(encoding="utf-8"))
        panels = tuple(selection["panel_paths"])
        if selection["selected_count"] != 12 or selection["shortfall"] != 0 or len(panels) != 12:
            raise ValueError(f"invalid curated panel selection: {model_id}")
        rights_decision = json.loads((run_dir / "rights-decision.json").read_text(encoding="utf-8"))
        if rights_decision.get("status") != model["rights_scope"]:
            raise ValueError(f"rights scope mismatch: {model_id}")
        runs.append(ModelRunEvidence(
            model_id=model_id, family=str(model["family"]), rights_scope=str(model["rights_scope"]),
            checkpoint_training_split_status=str(model["checkpoint_training_split_status"]),
            comparison_scope=str(model["comparison_scope"]), run_dir=run_dir,
            run_relative_path=run_dir.relative_to(root).as_posix(),
            score=score, metrics=metrics, rights_decision=rights_decision,
            aggregates=aggregates, capabilities=capabilities, panels=panels,
            canonical_split_sha256=split_hash,
            input_artifacts=tuple(evidence["artifacts"]),
        ))
    if len({run.canonical_split_sha256 for run in runs}) != 1:
        raise ValueError("canonical validation samples differ across model runs")
    return tuple(runs)
```

- [ ] **Step 5: Run focused GREEN tests**

Run:

```powershell
& 'C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\work\DroneAI\.venv\Scripts\python.exe' -m pytest tests/test_round_evidence.py tests/test_round_scorecard.py -q
```

Expected: all Task 1 and Task 2 tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add src/droneai/round_evidence.py tests/test_round_evidence.py
git commit -m "Verify Round 1 comparison evidence"
```

---

### Task 3: Build the technical shortlist and independent rights axis

**Files:**
- Create: `src/droneai/round_comparison.py`
- Create: `tests/test_round_comparison.py`

**Interfaces:**
- Consumes: `tuple[ModelRunEvidence, ...]` and `ScorecardConfig`.
- Produces: `deployment_label(scope: str) -> str` and `build_round_comparison(runs: tuple[ModelRunEvidence, ...], config: ScorecardConfig, *, git_commit: str) -> dict[str, object]`.

- [ ] **Step 1: Write failing shortlist and rights-independence tests**

```python
from dataclasses import replace

import pytest

from droneai.round_comparison import build_round_comparison, deployment_label


def test_deployment_labels_never_infer_production_permission() -> None:
    assert deployment_label("PRODUCTION_APPROVED") == "PRODUCT_ELIGIBLE"
    assert deployment_label("PASS_COMMERCIAL_CANDIDATE") == "PENDING_DUE_DILIGENCE"
    assert deployment_label("PASS_RESEARCH_ONLY") == "RESEARCH_ONLY"
    assert deployment_label("BLOCKED_EXPLICIT_RESTRICTION") == "BLOCKED"


def test_rights_change_does_not_change_technical_score(verified_runs, scorecard) -> None:
    first = build_round_comparison(verified_runs, scorecard, git_commit="a" * 40)
    changed = tuple(replace(run, rights_scope="PASS_RESEARCH_ONLY") for run in verified_runs)
    second = build_round_comparison(changed, scorecard, git_commit="a" * 40)
    assert [row["technical_score"] for row in first["models"]] == [
        row["technical_score"] for row in second["models"]
    ]
    assert {row["deployment_label"] for row in second["models"]} == {"RESEARCH_ONLY"}


def test_top_three_uses_unrounded_score_then_frozen_ties(verified_runs, scorecard) -> None:
    comparison = build_round_comparison(verified_runs, scorecard, git_commit="b" * 40)
    assert comparison["shortlist"]["label"] == "PROVISIONAL_TECHNICAL_SHORTLIST"
    assert len(comparison["shortlist"]["model_ids"]) == 3
    assert comparison["ranking_eligible"] is False
    rows = {row["model_id"]: row for row in comparison["models"]}
    assert all("categories" in rows[model_id] for model_id in comparison["shortlist"]["model_ids"])
    assert set(rows["csrnet"]["rights_components"]) == {
        "code", "dataset", "pretrained_weights", "derived_weights", "deployment"
    }
    assert rows["csrnet"]["representative_panels"][0].startswith("../csrnet/")
```

- [ ] **Step 2: Run Task 3 tests and verify RED**

Run the Windows pytest command with `tests/test_round_comparison.py`.

Expected: collection fails because `droneai.round_comparison` does not exist.

- [ ] **Step 3: Implement rights labels and deterministic comparison**

```python
from __future__ import annotations

import re
from typing import Iterable

from droneai.round_evidence import ModelRunEvidence
from droneai.round_scorecard import ScorecardConfig, score_technical


_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def deployment_label(scope: str) -> str:
    if scope == "PRODUCTION_APPROVED":
        return "PRODUCT_ELIGIBLE"
    if scope == "PASS_COMMERCIAL_CANDIDATE":
        return "PENDING_DUE_DILIGENCE"
    if scope == "PASS_RESEARCH_ONLY":
        return "RESEARCH_ONLY"
    if scope.startswith("BLOCKED"):
        return "BLOCKED"
    raise ValueError(f"unsupported rights scope: {scope}")


def _component_rights(decision: dict[str, object]) -> dict[str, dict[str, str]]:
    component_ids = decision.get("component_ids", {})
    ambiguous = set(decision.get("ambiguous_components", []))
    restricted = set(decision.get("restricted_components", []))
    result: dict[str, dict[str, str]] = {}
    for name in ("code", "dataset", "pretrained_weights", "derived_weights", "deployment"):
        if name not in component_ids:
            raise ValueError(f"rights decision is missing component ID: {name}")
        state = "RESTRICTED" if name in restricted else (
            "PENDING" if name in ambiguous else "REVIEWED_NO_EXPLICIT_PROHIBITION"
        )
        result[name] = {"component_id": str(component_ids[name]), "status": state}
    return result


def _row(run: ModelRunEvidence, config: ScorecardConfig) -> dict[str, object]:
    required = (
        "mean_zone_mae", "spatial_metric_value", "max_absolute_band_bias",
        "mae", "rmse", "mape_reference", "high_band_bias",
        "median_latency_ms", "peak_vram_mb",
    )
    if any(run.aggregates[name] is None for name in required):
        raise ValueError(f"required technical aggregate is unavailable: {run.model_id}")
    score_inputs = {name: float(run.aggregates[name]) for name in required}
    scored = score_technical(
        family=run.family,
        spatial_metric_name=str(run.metrics["spatial_metric_name"]),
        aggregates=score_inputs,
        capabilities=run.capabilities,
        checkpoint_training_split_status=run.checkpoint_training_split_status,
        coverage_complete=(int(run.metrics["successful_samples"]) == config.expected_samples
                           and int(run.metrics["explicit_failures"]) == 0),
        artifacts_verified=True,
        config=config,
    )
    return {
        "model_id": run.model_id,
        "family": run.family,
        "technical_score": scored["technical_score"],
        "technical_score_display": round(float(scored["technical_score"]), 2),
        "categories": scored["categories"],
        "aggregates": run.aggregates,
        "primary_spatial_metric": str(run.metrics["spatial_metric_name"]),
        "capabilities": {name: ("DEMONSTRATED" if value else "NOT_DEMONSTRATED")
                         for name, value in run.capabilities.items()},
        "rights_scope": run.rights_scope,
        "deployment_label": deployment_label(run.rights_scope),
        "rights_components": _component_rights(run.rights_decision),
        "checkpoint_training_split_status": run.checkpoint_training_split_status,
        "comparison_scope": run.comparison_scope,
        "ranking_eligible": False,
        "representative_panels": [
            f"../{run.run_relative_path}/{panel}" for panel in run.panels
        ],
    }


def build_round_comparison(runs: tuple[ModelRunEvidence, ...], config: ScorecardConfig,
                           *, git_commit: str) -> dict[str, object]:
    if not _COMMIT.fullmatch(git_commit):
        raise ValueError("comparison requires a full clean Git commit")
    if tuple(run.model_id for run in runs) != config.required_model_ids:
        raise ValueError("verified run order does not match scorecard")
    rows = [_row(run, config) for run in runs]
    ordered = sorted(rows, key=lambda row: (
        -float(row["technical_score"]),
        float(row["aggregates"]["mean_zone_mae"]),
        float(row["aggregates"]["mae"]),
        str(row["model_id"]),
    ))
    shortlist = [str(row["model_id"]) for row in ordered[:config.shortlist_size]]
    return {
        "schema_version": 1,
        "comparison_id": config.comparison_id,
        "round_id": config.round_id,
        "git_commit": git_commit,
        "dataset": {"dataset_id": config.dataset_id, "split_id": config.split_id,
                    "split_role": config.split_role, "expected_samples": config.expected_samples,
                    "canonical_split_sha256": runs[0].canonical_split_sha256},
        "ranking_eligible": False,
        "comparison_scope": "compatibility_smoke",
        "models": rows,
        "shortlist": {"label": config.shortlist_label, "model_ids": shortlist,
                      "tie_breakers": list(config.tie_breakers)},
        "limitations": [
            "Official checkpoints do not share one training protocol.",
            "This is a 36-image cross-domain operational screen, not an official ranking.",
            "Technical score does not grant code, dataset, weight, derived-weight, or deployment rights.",
        ],
    }
```

- [ ] **Step 4: Run focused GREEN tests**

Run:

```powershell
& 'C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\work\DroneAI\.venv\Scripts\python.exe' -m pytest tests/test_round_scorecard.py tests/test_round_evidence.py tests/test_round_comparison.py -q
```

Expected: all three focused files pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/droneai/round_comparison.py tests/test_round_comparison.py
git commit -m "Build provisional CCTV technical shortlist"
```

---

### Task 4: Write an atomic small-artifact bundle and CLI

**Files:**
- Create: `src/droneai/round_report.py`
- Create: `scripts/build_round1_cctv_comparison.py`
- Create: `tests/test_round_report.py`
- Create: `tests/test_build_round1_cctv_comparison.py`

**Interfaces:**
- Consumes: comparison dictionary, `ScorecardConfig`, and paths to verified inputs.
- Produces: `write_comparison_bundle(output_dir: str | Path, comparison: dict[str, object], scorecard_path: str | Path, *, input_references: tuple[dict[str, str], ...]) -> Path` and CLI exit codes 0/2.

- [ ] **Step 1: Write failing atomic-writer tests**

```python
import hashlib
import json
from pathlib import Path

import pytest

from droneai.round_report import write_comparison_bundle


def test_writer_creates_exact_small_bundle(tmp_path: Path, comparison, scorecard_path: Path) -> None:
    output = tmp_path / "comparison"
    written = write_comparison_bundle(
        output, comparison, scorecard_path,
        input_references=({"path": "source/metrics.json", "sha256": "a" * 64},),
    )
    assert written == output
    assert {path.name for path in output.iterdir()} == {
        "comparison.json", "comparison.csv", "comparison.md",
        "evidence-manifest.json", "scorecard-config.snapshot.json",
    }
    manifest = json.loads((output / "evidence-manifest.json").read_text())
    assert manifest["inputs"][0]["sha256"] == "a" * 64
    assert {item["path"] for item in manifest["outputs"]} == {
        "comparison.json", "comparison.csv", "comparison.md",
        "scorecard-config.snapshot.json",
    }


def test_writer_failure_leaves_no_final_or_temporary_directory(
    tmp_path: Path, comparison, scorecard_path: Path, monkeypatch
) -> None:
    output = tmp_path / "comparison"
    monkeypatch.setattr("droneai.round_report.enforce_review_budget",
                        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("over budget")))
    with pytest.raises(ValueError, match="over budget"):
        write_comparison_bundle(output, comparison, scorecard_path, input_references=())
    assert not output.exists()
    assert not (tmp_path / ".comparison.tmp").exists()
```

- [ ] **Step 2: Write failing CLI error-contract test**

```python
import json

from scripts.build_round1_cctv_comparison import main


def test_cli_returns_structured_error_for_missing_queue(capsys, tmp_path) -> None:
    code = main([
        "--queue", str(tmp_path / "missing.json"),
        "--results-root", str(tmp_path / "results"),
        "--scorecard", "configs/evaluation/round1_cctv_scorecard.json",
        "--output-dir", str(tmp_path / "out"),
    ])
    error = json.loads(capsys.readouterr().err)
    assert code == 2
    assert error["status"] == "FAIL"
    assert error["error_type"] == "FileNotFoundError"
    assert not (tmp_path / "out").exists()
```

- [ ] **Step 3: Run Task 4 tests and verify RED**

Run the Windows pytest command with `tests/test_round_report.py tests/test_build_round1_cctv_comparison.py`.

Expected: collection fails because the report module and CLI do not exist.

- [ ] **Step 4: Implement atomic reporting**

```python
from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path

from droneai.evaluation_artifacts import enforce_review_budget, write_json
from droneai.integrity import sha256_file


def _markdown(comparison: dict[str, object]) -> str:
    lines = [
        "# Round 1 fixed-CCTV technical comparison", "",
        "**Scope: PROVISIONAL_TECHNICAL_SHORTLIST; compatibility smoke; not an official ranking.**", "",
        "| Model | Technical | Zone/spatial | Count | Runtime | Integration | Evidence | Rights | Deployment |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in comparison["models"]:
        category = row["categories"]
        lines.append(
            f"| {row['model_id']} | {row['technical_score_display']:.2f} | "
            f"{category['zone_spatial']:.2f} | {category['count_stability']:.2f} | "
            f"{category['runtime_resources']:.2f} | {category['operator_integration']:.2f} | "
            f"{category['evidence_quality']:.2f} | {row['rights_scope']} | {row['deployment_label']} |"
        )
    lines.extend([
        "", "## Raw validation metrics", "",
        "| Model | MAE | RMSE | MAPE % | Bias | High bias | Zone MAE | Spatial metric | Latency ms | FPS | VRAM MB |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ])
    for row in comparison["models"]:
        metric = row["aggregates"]
        lines.append(
            f"| {row['model_id']} | {metric['mae']:.3f} | {metric['rmse']:.3f} | "
            f"{metric['mape_reference']:.3f} | {metric['signed_bias']:.3f} | "
            f"{metric['high_band_bias']:.3f} | {metric['mean_zone_mae']:.3f} | "
            f"{row['primary_spatial_metric']}={metric['spatial_metric_value']:.3f} | "
            f"{metric['median_latency_ms']:.3f} | {metric['throughput_fps_batch1']:.3f} | "
            f"{metric['peak_vram_mb']:.3f} |"
        )
    lines.extend([
        "", "## Independent rights components", "",
        "| Model | Code | Dataset | Pretrained weight | Derived weight | Deployment | Overall |",
        "|---|---|---|---|---|---|---|",
    ])
    for row in comparison["models"]:
        rights = row["rights_components"]
        lines.append(
            f"| {row['model_id']} | {rights['code']['status']} | {rights['dataset']['status']} | "
            f"{rights['pretrained_weights']['status']} | {rights['derived_weights']['status']} | "
            f"{rights['deployment']['status']} | {row['deployment_label']} |"
        )
    lines.extend(["", "## Provisional technical shortlist", ""])
    lines.extend(f"{index}. `{model_id}`" for index, model_id in
                 enumerate(comparison["shortlist"]["model_ids"], start=1))
    lines.extend(["", "## Representative panels", ""])
    for row in comparison["models"]:
        links = ", ".join(
            f"[panel {index}]({path})"
            for index, path in enumerate(row["representative_panels"], start=1)
        )
        lines.append(f"- `{row['model_id']}`: {links}")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in comparison["limitations"])
    return "\n".join(lines) + "\n"


def write_comparison_bundle(output_dir: str | Path, comparison: dict[str, object],
                            scorecard_path: str | Path, *,
                            input_references: tuple[dict[str, str], ...]) -> Path:
    output = Path(output_dir)
    temporary = output.parent / f".{output.name}.tmp"
    if output.exists() or temporary.exists():
        raise FileExistsError("comparison output and temporary path must not exist")
    temporary.mkdir(parents=True)
    try:
        write_json(temporary / "comparison.json", comparison)
        scorecard = json.loads(Path(scorecard_path).read_text(encoding="utf-8"))
        write_json(temporary / "scorecard-config.snapshot.json", scorecard)
        fields = [
            "model_id", "family", "technical_score_display",
            "zone_spatial", "count_stability", "runtime_resources",
            "operator_integration", "evidence_quality",
            "mae", "rmse", "mape_reference", "signed_bias", "high_band_bias",
            "mean_zone_mae", "primary_spatial_metric", "spatial_metric_value",
            "median_latency_ms", "throughput_fps_batch1", "peak_vram_mb",
            "code_rights", "dataset_rights", "pretrained_weight_rights",
            "derived_weight_rights", "deployment_rights",
            "rights_scope", "deployment_label",
        ]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        rows = []
        for row in comparison["models"]:
            rows.append({
                "model_id": row["model_id"], "family": row["family"],
                "technical_score_display": row["technical_score_display"],
                **row["categories"],
                **{name: row["aggregates"][name] for name in (
                    "mae", "rmse", "mape_reference", "signed_bias", "high_band_bias",
                    "mean_zone_mae", "spatial_metric_value", "median_latency_ms",
                    "throughput_fps_batch1", "peak_vram_mb",
                )},
                "primary_spatial_metric": row["primary_spatial_metric"],
                "code_rights": row["rights_components"]["code"]["status"],
                "dataset_rights": row["rights_components"]["dataset"]["status"],
                "pretrained_weight_rights": row["rights_components"]["pretrained_weights"]["status"],
                "derived_weight_rights": row["rights_components"]["derived_weights"]["status"],
                "deployment_rights": row["rights_components"]["deployment"]["status"],
                "rights_scope": row["rights_scope"],
                "deployment_label": row["deployment_label"],
            })
        writer.writerows(rows)
        (temporary / "comparison.csv").write_text(buffer.getvalue(), encoding="utf-8", newline="")
        (temporary / "comparison.md").write_text(_markdown(comparison), encoding="utf-8")
        outputs = []
        for path in sorted(temporary.iterdir()):
            if path.name != "evidence-manifest.json":
                outputs.append({"path": path.name, "sha256": sha256_file(path)})
        write_json(temporary / "evidence-manifest.json", {
            "schema_version": 1, "inputs": list(input_references), "outputs": outputs,
        })
        enforce_review_budget(temporary)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output
```

- [ ] **Step 5: Implement the thin CLI**

```python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from droneai.integrity import sha256_file
from droneai.round_comparison import build_round_comparison
from droneai.round_evidence import load_verified_round
from droneai.round_report import write_comparison_bundle
from droneai.round_scorecard import load_scorecard


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Round 1 fixed-CCTV comparison")
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--scorecard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _git_identity() -> str:
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                            check=True, capture_output=True, text=True).stdout
    if status.strip():
        raise ValueError("comparison generation requires a clean Git worktree")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_scorecard(args.scorecard)
        runs = load_verified_round(args.queue, args.results_root, config)
        comparison = build_round_comparison(runs, config, git_commit=_git_identity())
        inputs = tuple(
            {"path": str(path), "sha256": sha256_file(path)}
            for run in runs
            for path in (run.run_dir / "score.json", run.run_dir / "metrics.json",
                         run.run_dir / "predictions.csv", run.run_dir / "evidence-manifest.json")
        )
        output = write_comparison_bundle(args.output_dir, comparison, args.scorecard,
                                         input_references=inputs)
        print(json.dumps({"status": "PASS_RESEARCH_ONLY", "output_dir": str(output),
                          "shortlist": comparison["shortlist"]}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"schema_version": 1, "status": "FAIL",
                          "error_type": type(exc).__name__, "error": str(exc)},
                         sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run focused GREEN tests and CLI help**

Run:

```powershell
& 'C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\work\DroneAI\.venv\Scripts\python.exe' -m pytest tests/test_round_report.py tests/test_build_round1_cctv_comparison.py -q
& 'C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\work\DroneAI\.venv\Scripts\python.exe' scripts/build_round1_cctv_comparison.py --help
```

Expected: report/CLI tests pass and help lists exactly `--queue`, `--results-root`, `--scorecard`, and `--output-dir`.

- [ ] **Step 7: Commit Task 4**

```powershell
git add src/droneai/round_report.py scripts/build_round1_cctv_comparison.py tests/test_round_report.py tests/test_build_round1_cctv_comparison.py
git commit -m "Generate atomic Round 1 CCTV comparison"
```

---

### Task 5: Verify, run on home5090, review, and record the comparison

**Files:**
- Modify: `docs/HOME5090_RUNBOOK.md`
- Create: `results/round1-cctv-comparison-<implementation-commit>/comparison.json`
- Create: `results/round1-cctv-comparison-<implementation-commit>/comparison.csv`
- Create: `results/round1-cctv-comparison-<implementation-commit>/comparison.md`
- Create: `results/round1-cctv-comparison-<implementation-commit>/evidence-manifest.json`
- Create: `results/round1-cctv-comparison-<implementation-commit>/scorecard-config.snapshot.json`

**Interfaces:**
- Consumes: all code from Tasks 1-4 and real accepted SSD evidence.
- Produces: the reviewed Round 1 comparison commit and a home5090 workflow status that remains at `review_and_advance` until the user accepts the generated shortlist.

- [ ] **Step 1: Add the exact runbook command and safety text**

```markdown
## Round 1 fixed-CCTV technical comparison

Run only when all six queue entries have accepted 36-image benchmark evidence and
workflow status reports `review_and_advance`. This command aggregates existing
validation evidence only; it performs no inference, training, fine-tuning, or test
access.

```bash
cd /workspace
/workspace/.venvs/harness/bin/python scripts/build_round1_cctv_comparison.py \
  --queue configs/evaluation/model_queue.json \
  --results-root /workspace/data/results \
  --scorecard configs/evaluation/round1_cctv_scorecard.json \
  --output-dir /workspace/data/results/round1-cctv-comparison-<implementation-commit>
```

Review all raw metrics, category subscores, limitations, input/output SHA-256
values, and the independent rights/deployment columns before accepting the
provisional shortlist. Do not advance the queue or start fine-tuning here.
```

- [ ] **Step 2: Run focused and full verification on Windows**

Run:

```powershell
& 'C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\work\DroneAI\.venv\Scripts\python.exe' -m pytest tests/test_round_scorecard.py tests/test_round_evidence.py tests/test_round_comparison.py tests/test_round_report.py tests/test_build_round1_cctv_comparison.py -q
& 'C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\work\DroneAI\.venv\Scripts\python.exe' -m pytest -q
git diff --check
git status --short
```

Expected: all focused tests pass, the full suite reports zero failures, diff check exits 0, and only the intended runbook change is uncommitted.

- [ ] **Step 3: Commit implementation documentation, push, and sync home5090**

```powershell
git add docs/HOME5090_RUNBOOK.md
git commit -m "Document Round 1 CCTV comparison workflow"
git push
ssh home5090-pop "cd /home/lucita/crowd-counting-lab && git status --short && git pull --ff-only && git rev-parse --short HEAD"
```

Expected: home5090 fast-forwards with no local changes. Record the resulting seven-character implementation commit for the output directory name.

- [ ] **Step 4: Reverify workflow evidence and run the comparison on home5090**

```powershell
ssh home5090-pop "docker exec crowd-jupyter bash -lc 'cd /workspace && /workspace/.venvs/harness/bin/python scripts/run_model_workflow.py status --queue configs/evaluation/model_queue.json --results-root /workspace/data/results --format markdown && /workspace/.venvs/harness/bin/python scripts/build_round1_cctv_comparison.py --queue configs/evaluation/model_queue.json --results-root /workspace/data/results --scorecard configs/evaluation/round1_cctv_scorecard.json --output-dir /workspace/data/results/round1-cctv-comparison-<implementation-commit>'"
```

Expected: status shows all six benchmark gates verified and `next_action=review_and_advance`; the comparison command returns `PASS_RESEARCH_ONLY`, exactly three provisional model IDs, and no training output.

- [ ] **Step 5: Pull only the small output bundle and verify every hash**

```powershell
scp -r home5090-pop:/mnt/crowd-data/CrowdCounting/results/round1-cctv-comparison-<implementation-commit> results/
$d='results/round1-cctv-comparison-<implementation-commit>'
$m=Get-Content "$d/evidence-manifest.json" -Raw | ConvertFrom-Json
$bad=@()
foreach($a in $m.outputs){
  $actual=(Get-FileHash -Algorithm SHA256 (Join-Path $d $a.path)).Hash.ToLower()
  if($actual -ne $a.sha256){$bad += $a.path}
}
if($bad.Count){$bad; exit 1}
$files=Get-ChildItem $d -Recurse -File
if((($files | Measure-Object Length -Sum).Sum) -gt 25MB){exit 1}
Get-Content "$d/comparison.md"
```

Expected: zero hash mismatches, five small files, total size below 25 MiB, all six models in the Markdown table, three provisional shortlist entries, and separate rights/deployment columns.

- [ ] **Step 6: Review score semantics before accepting the artifact**

Check `comparison.json` and reject the run unless all statements are true:

```text
ranking_eligible is false
comparison_scope is compatibility_smoke
dataset split_role is validation
dataset expected_samples is 36
all six model IDs are present once
shortlist label is PROVISIONAL_TECHNICAL_SHORTLIST
shortlist contains exactly three unique model IDs
no PRODUCT_ELIGIBLE label exists without rights_scope PRODUCTION_APPROVED
technical scores remain independent from rights labels
limitations contain checkpoint heterogeneity and non-ranking language
```

- [ ] **Step 7: Record the reviewed comparison and rerun verification**

```powershell
git add results/round1-cctv-comparison-<implementation-commit>
git commit -m "Record Round 1 CCTV technical comparison"
& 'C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\work\DroneAI\.venv\Scripts\python.exe' -m pytest -q
git status --short
git push
ssh home5090-pop "cd /home/lucita/crowd-counting-lab && git pull --ff-only && git rev-parse HEAD && git status --short"
```

Expected: the full test suite has zero failures; Windows, origin, and home5090 share one commit and both worktrees are clean.

- [ ] **Step 8: Report and stop at the next approval point**

Report in the DroneAI status-contract order: outcome, model families/outputs, frozen dataset/split and hashes, all raw metrics and subscores, evidence hashes/paths, technical scores plus independent rights scopes, provisional top three, limitations, and roadmap. Do not advance to calibration, drone-domain evaluation, or fine-tuning until the user reviews and approves the comparison.
