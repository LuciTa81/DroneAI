"""Verified, dataset-wise comparison for the Round 2 reference benchmark."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from droneai.integrity import is_sha256, sha256_file, verify_artifact_reference


MODEL_ORDER = ("steerer", "dm-count", "mpcount")
DATASET_ORDER = (
    "ucf-qnrf-kaggle-apache",
    "jhu-crowd-plus-v2",
    "up-count-v1",
)
MODEL_NOTES = {
    "steerer": {
        "family": "density_and_points",
        "native_output": "multi-resolution density maps and localization points",
        "count_derivation": "integral of the highest-resolution valid density map",
    },
    "dm-count": {
        "family": "density",
        "native_output": "non-negative density map",
        "count_derivation": "density-map sum",
    },
    "mpcount": {
        "family": "density_and_domain_generalization",
        "native_output": "non-negative density map",
        "count_derivation": "cropped density-map sum divided by log_para",
    },
}
REQUIRED_FILES = {
    "round2-shared-manifest.json",
    "round2-dataset-manifest.json",
    "predictions.csv",
    "metrics.json",
    "score.json",
    "selection_manifest.json",
    "environment-summary.json",
    "rights-decision.json",
    "sample-manifest.json",
}
REQUIRED_REFERENCES = REQUIRED_FILES - {"score.json"}
METRICS = (
    "mae",
    "rmse",
    "signed_bias",
    "mape_reference",
    "median_latency_ms",
    "throughput_fps_batch1",
    "peak_vram_mb",
    "spatial_mean",
)


@dataclass(frozen=True)
class Round2Run:
    model_id: str
    dataset_id: str
    expected_samples: int
    dataset_rights_scope: str
    path: Path

    def __post_init__(self) -> None:
        if self.model_id not in MODEL_ORDER or self.dataset_id not in DATASET_ORDER:
            raise ValueError("Round 2 run identity is invalid")
        if self.expected_samples <= 0:
            raise ValueError("Round 2 run expected_samples must be positive")
        if self.dataset_rights_scope not in {
            "PASS_COMMERCIAL_CANDIDATE",
            "PASS_RESEARCH_ONLY",
        }:
            raise ValueError("Round 2 dataset rights scope is invalid")


def _object(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _finite(payload: dict[str, object], field: str) -> float:
    value = payload.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"Round 2 metric is unavailable or non-finite: {field}")
    return float(value)


def _prediction_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _verify_run(run: Round2Run) -> tuple[str, dict[str, object]]:
    root = run.path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Round 2 run directory missing: {root}")
    missing_files = sorted(name for name in REQUIRED_FILES if not (root / name).is_file())
    if missing_files:
        raise FileNotFoundError(f"Round 2 run files are incomplete: {missing_files}")
    evidence_path = root / "evidence-manifest.json"
    evidence = _object(evidence_path, label="evidence manifest")
    references = evidence.get("artifacts")
    if not isinstance(references, list) or not references:
        raise ValueError("Round 2 evidence manifest has no artifact references")
    if any(
        not isinstance(reference, dict)
        or not verify_artifact_reference(reference, base_dir=root)[0]
        for reference in references
    ):
        raise ValueError(f"artifact verification failed: {run.model_id}/{run.dataset_id}")
    referenced = {
        str(reference["path"])
        for reference in references
        if isinstance(reference, dict) and "path" in reference
    }
    if not REQUIRED_REFERENCES <= referenced:
        missing = sorted(REQUIRED_REFERENCES - referenced)
        raise ValueError(f"Round 2 evidence manifest is incomplete: {missing}")

    shared = _object(root / "round2-shared-manifest.json", label="shared manifest")
    shared_hash = shared.get("sample_manifest_sha256")
    if not isinstance(shared_hash, str) or not is_sha256(shared_hash):
        raise ValueError("shared manifest sample identity is invalid")
    sample_manifest = _object(root / "sample-manifest.json", label="sample manifest")
    if (
        sample_manifest.get("dataset_id") != run.dataset_id
        or sample_manifest.get("expected_samples") != run.expected_samples
        or sample_manifest.get("comparison_scope") != "research_reference_only"
        or sample_manifest.get("ranking_eligible") is not False
    ):
        raise ValueError("Round 2 sample manifest identity or scope mismatch")
    protocol = evidence.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("dataset_id") != run.dataset_id:
        raise ValueError("Round 2 evidence protocol identity mismatch")

    metrics = _object(root / "metrics.json", label="metrics")
    for field in ("expected_samples", "recorded_samples", "successful_samples"):
        if metrics.get(field) != run.expected_samples:
            raise ValueError(f"Round 2 sample coverage mismatch: {field}")
    if metrics.get("explicit_failures") != 0:
        raise ValueError("Round 2 run contains explicit failures")
    if _prediction_count(root / "predictions.csv") != run.expected_samples:
        raise ValueError("Round 2 prediction row count mismatch")
    aggregates = {field: _finite(metrics, field) for field in METRICS}
    spatial_name = metrics.get("spatial_metric_name")
    if not isinstance(spatial_name, str) or not spatial_name:
        raise ValueError("Round 2 spatial metric name is unavailable")

    rights = _object(root / "rights-decision.json", label="rights decision")
    declared_dataset_scope = rights.get(
        "dataset_rights_scope", rights.get("rights_scope")
    )
    if declared_dataset_scope != run.dataset_rights_scope:
        raise ValueError("Round 2 dataset rights scope mismatch")
    effective_scope = rights.get("rights_scope")
    if effective_scope not in {
        "PASS_COMMERCIAL_CANDIDATE",
        "PASS_RESEARCH_ONLY",
    }:
        raise ValueError("Round 2 effective rights scope is invalid")
    if (
        run.dataset_rights_scope == "PASS_RESEARCH_ONLY"
        and effective_scope != "PASS_RESEARCH_ONLY"
    ):
        raise ValueError("restricted dataset lane was promoted beyond research-only")

    score = _object(root / "score.json", label="score")
    environment = _object(root / "environment-summary.json", label="environment")
    row = {
        **MODEL_NOTES[run.model_id],
        "model_id": run.model_id,
        "dataset_id": run.dataset_id,
        "expected_samples": run.expected_samples,
        "dataset_rights_scope": run.dataset_rights_scope,
        "effective_rights_scope": effective_scope,
        "comparison_scope": "research_reference_only",
        "ranking_eligible": False,
        "metrics": aggregates,
        "spatial_metric_name": spatial_name,
        "technical_gate_status": score.get("status"),
        "technical_gate_score": score.get("score"),
        "git_commit": environment.get("git_commit"),
        "evidence": {
            "path": str(evidence_path),
            "sha256": sha256_file(evidence_path),
        },
    }
    return shared_hash, row


def build_round2_comparison(runs: Sequence[Round2Run]) -> dict[str, object]:
    keyed = {(run.model_id, run.dataset_id): run for run in runs}
    expected_keys = {(model, dataset) for model in MODEL_ORDER for dataset in DATASET_ORDER}
    if len(runs) != 9 or set(keyed) != expected_keys:
        raise ValueError("Round 2 comparison requires exactly the 3 × 3 run matrix")
    verified = {
        key: _verify_run(keyed[key])
        for key in ((model, dataset) for model in MODEL_ORDER for dataset in DATASET_ORDER)
    }
    shared_hashes = {value[0] for value in verified.values()}
    if len(shared_hashes) != 1:
        raise ValueError("Round 2 runs do not share one shared manifest")
    datasets: dict[str, object] = {}
    for dataset_id in DATASET_ORDER:
        descriptor = keyed[(MODEL_ORDER[0], dataset_id)]
        datasets[dataset_id] = {
            "samples_per_model": descriptor.expected_samples,
            "rights_scope": descriptor.dataset_rights_scope,
            "models": {
                model_id: verified[(model_id, dataset_id)][1]
                for model_id in MODEL_ORDER
            },
        }
    return {
        "schema_version": 1,
        "comparison_id": "round-2-reference-comparison-v1",
        "rights_scope": "PASS_RESEARCH_ONLY",
        "comparison_scope": "research_reference_only",
        "ranking_eligible": False,
        "fine_tuning": False,
        "shared_sample_manifest_sha256": next(iter(shared_hashes)),
        "models": {model_id: MODEL_NOTES[model_id] for model_id in MODEL_ORDER},
        "datasets": datasets,
        "limitations": [
            "Metrics are reported per dataset and are not pooled into a commercial score.",
            "JHU-CROWD++ and UP-COUNT evidence is non-commercial research reference only.",
            "STEERER checkpoint terms remain research-only pending independent clearance.",
            "Frozen checkpoints use different training domains; this comparison does not authorize production selection.",
        ],
    }


def _markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Round 2 reference comparison",
        "",
        "- Scope: `PASS_RESEARCH_ONLY`",
        "- Comparison: `research_reference_only`",
        "- Ranking eligible: `false`",
        f"- Shared sample manifest: `{payload['shared_sample_manifest_sha256']}`",
        "",
    ]
    datasets = payload["datasets"]
    assert isinstance(datasets, dict)
    for dataset_id in DATASET_ORDER:
        dataset = datasets[dataset_id]
        assert isinstance(dataset, dict)
        models = dataset["models"]
        assert isinstance(models, dict)
        lines.extend(
            [
                f"## {dataset_id}",
                "",
                f"Rights: `{dataset['rights_scope']}`; samples/model: `{dataset['samples_per_model']}`",
                "",
                "| Model | MAE | RMSE | Bias | MAPE ref. (%) | Median ms | FPS | Peak VRAM MB | Spatial | Gate |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for model_id in MODEL_ORDER:
            row = models[model_id]
            assert isinstance(row, dict)
            metrics = row["metrics"]
            assert isinstance(metrics, dict)
            lines.append(
                f"| {model_id} | {metrics['mae']:.2f} | {metrics['rmse']:.2f} | "
                f"{metrics['signed_bias']:.2f} | {metrics['mape_reference']:.2f} | "
                f"{metrics['median_latency_ms']:.2f} | {metrics['throughput_fps_batch1']:.2f} | "
                f"{metrics['peak_vram_mb']:.2f} | {row['spatial_metric_name']}="
                f"{metrics['spatial_mean']:.3f} | {row['technical_gate_status']} |"
            )
        lines.append("")
    lines.extend(["## Limitations", ""])
    limitations = payload["limitations"]
    assert isinstance(limitations, list)
    lines.extend(f"- {value}" for value in limitations)
    lines.append("")
    return "\n".join(lines)


def write_round2_comparison(
    output_dir: str | Path, runs: Sequence[Round2Run]
) -> dict[str, Path]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"comparison output already exists: {output}")
    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        raise FileExistsError(f"comparison staging output already exists: {staging}")
    payload = build_round2_comparison(runs)
    staging.mkdir(parents=True)
    (staging / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (staging / "comparison.md").write_text(_markdown(payload), encoding="utf-8")
    staging.replace(output)
    return {
        "json": output / "comparison.json",
        "markdown": output / "comparison.md",
    }


__all__ = [
    "DATASET_ORDER",
    "MODEL_ORDER",
    "Round2Run",
    "build_round2_comparison",
    "write_round2_comparison",
]
