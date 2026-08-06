"""Evidence-backed 100-point gates for STEERER training stages.

The gate deliberately awards no production status.  It only establishes whether
the supplied non-Test training evidence is a commercial candidate for review.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from droneai.integrity import is_sha256, sha256_file
from droneai.scoring import CheckResult, StageReport, score_stage


STAGE_THRESHOLDS = {"T0": 70, "T1": 75, "T5": 80, "T50": 85}
SUCCESS_STATUS = "PASS_COMMERCIAL_CANDIDATE"
APPROVED_UPSTREAM_COMMIT = "5b1854dbc2d280f2326d67c65515d8baf9083810"

REQUIRED_ARTIFACTS = frozenset(
    {
        "environment",
        "dataset_inventory",
        "train_split",
        "validation_split",
        "backbone",
        "input_checkpoint",
        "output_checkpoint",
        "config",
        "rights_decision",
    }
)
REQUIRED_T1_METRICS = frozenset(
    {
        "mae",
        "rmse",
        "signed_bias",
        "reference_mape",
        "game_l1",
        "quadrant_zone_mae",
        "localization_precision_16px",
        "localization_recall_16px",
        "localization_f1_16px",
        "median_latency_ms",
        "p95_latency_ms",
        "fps",
        "peak_vram_mb",
        "density_sum_count_difference",
        "train_loss",
        "validation_loss",
        "training_window_improvement",
    }
)
_T0_METRICS = frozenset({"train_loss"})
_COMMIT = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def _mapping(value: Any, *, name: str, keys: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    if actual != set(keys):
        missing = sorted(set(keys).difference(actual))
        extra = sorted(actual.difference(keys))
        raise ValueError(f"{name} keys are invalid: missing={missing} extra={extra}")
    return value


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _integer(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _metric_names(stage: str) -> frozenset[str]:
    return _T0_METRICS if stage == "T0" else REQUIRED_T1_METRICS


def validate_training_evidence(evidence: Any, *, stage: str) -> Mapping[str, Any]:
    """Validate the closed evidence schema before it is scored or persisted."""

    if stage not in STAGE_THRESHOLDS:
        raise ValueError(f"unsupported training stage: {stage}")
    root = _mapping(
        evidence,
        name="training evidence",
        keys=frozenset(
            {
                "schema_version",
                "run_id",
                "stage",
                "provenance",
                "split",
                "initialization",
                "stability",
                "metrics",
                "artifacts",
            }
        ),
    )
    if root["schema_version"] != 1:
        raise ValueError("training evidence schema_version must be 1")
    _nonempty_string(root["run_id"], name="run_id")
    if root["stage"] != stage:
        raise ValueError("evidence stage must match requested stage")

    provenance = _mapping(
        root["provenance"],
        name="provenance",
        keys=frozenset(
            {
                "git_commit",
                "upstream_commit",
                "container_image_digest",
                "python",
                "torch",
                "cuda",
                "gpu",
                "upstream_clean",
            }
        ),
    )
    for field in ("git_commit", "upstream_commit"):
        value = _nonempty_string(provenance[field], name=f"provenance.{field}")
        if _COMMIT.fullmatch(value) is None:
            raise ValueError(f"provenance.{field} must be a full Git commit")
    digest = _nonempty_string(
        provenance["container_image_digest"], name="provenance.container_image_digest"
    )
    if not digest.startswith("sha256:") or not is_sha256(digest.removeprefix("sha256:")):
        raise ValueError("provenance.container_image_digest must be a SHA-256 digest")
    for field in ("python", "torch", "cuda", "gpu"):
        _nonempty_string(provenance[field], name=f"provenance.{field}")
    _boolean(provenance["upstream_clean"], name="provenance.upstream_clean")

    split = _mapping(
        root["split"],
        name="split",
        keys=frozenset(
            {
                "dataset_population",
                "train_count",
                "validation_count",
                "overlap",
                "complete",
                "official_test_accessed",
            }
        ),
    )
    for field in ("dataset_population", "train_count", "validation_count", "overlap"):
        value = _integer(split[field], name=f"split.{field}")
        if value < 0:
            raise ValueError(f"split.{field} must be non-negative")
    _boolean(split["complete"], name="split.complete")
    _boolean(split["official_test_accessed"], name="split.official_test_accessed")

    initialization = _mapping(
        root["initialization"],
        name="initialization",
        keys=frozenset({"model_checkpoint_loaded"}),
    )
    _boolean(
        initialization["model_checkpoint_loaded"],
        name="initialization.model_checkpoint_loaded",
    )
    stability = _mapping(
        root["stability"],
        name="stability",
        keys=frozenset({"finite", "checkpoint_round_trip"}),
    )
    _boolean(stability["finite"], name="stability.finite")
    _boolean(stability["checkpoint_round_trip"], name="stability.checkpoint_round_trip")

    metrics = root["metrics"]
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be an object")
    required_metrics = _metric_names(stage)
    unknown_metrics = set(metrics).difference(required_metrics)
    if unknown_metrics:
        raise ValueError(f"metrics has unknown key: {sorted(unknown_metrics)[0]}")
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"metrics.{name} must be numeric")

    artifacts = root["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != REQUIRED_ARTIFACTS:
        raise ValueError("artifacts must contain exactly the required artifact references")
    for name, reference in artifacts.items():
        parsed = _mapping(
            reference,
            name=f"artifacts.{name}",
            keys=frozenset({"path", "sha256"}),
        )
        _nonempty_string(parsed["path"], name=f"artifacts.{name}.path")
        if not is_sha256(parsed["sha256"]):
            raise ValueError(f"artifacts.{name}.sha256 must be a SHA-256")
        if not Path(str(parsed["path"])).is_absolute():
            raise ValueError(f"artifacts.{name}.path must be absolute")
    environment = artifacts["environment"]
    assert isinstance(environment, Mapping)
    if Path(str(environment["path"])).name != f"environment.{environment['sha256']}.json":
        raise ValueError("environment evidence must use Task 6's content-addressed filename")
    return root


def verify_referenced_artifacts(evidence: Any, *, stage: str) -> dict[str, str]:
    """Hash every referenced regular file, returning concise observations."""

    validated = validate_training_evidence(evidence, stage=stage)
    observations: dict[str, str] = {}
    artifacts = validated["artifacts"]
    assert isinstance(artifacts, Mapping)
    for name in sorted(REQUIRED_ARTIFACTS):
        reference = artifacts[name]
        assert isinstance(reference, Mapping)
        path = Path(str(reference["path"]))
        if path.is_symlink() or not path.is_file():
            observations[name] = f"artifact missing or not a regular file: {path}"
            continue
        observed = sha256_file(path)
        expected = str(reference["sha256"]).lower()
        observations[name] = (
            f"verified: {path}; sha256={observed}"
            if observed == expected
            else f"artifact hash mismatch: {path}; expected={expected} observed={observed}"
        )
    return observations


def _artifacts_verified(observations: Mapping[str, str]) -> bool:
    return len(observations) == len(REQUIRED_ARTIFACTS) and all(
        observation.startswith("verified:") for observation in observations.values()
    )


def _metrics_are_finite(metrics: Mapping[str, Any], *, stage: str) -> bool:
    return set(metrics) == set(_metric_names(stage)) and all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (isinstance(value, int) or math.isfinite(value))
        for value in metrics.values()
    )


def score_training_stage(evidence: Any, *, stage: str) -> StageReport:
    """Score one stage; every safety failure is a non-averageable blocker."""

    try:
        validated = validate_training_evidence(evidence, stage=stage)
    except ValueError as error:
        # A malformed schema has no trustworthy sub-score, so fail before a report.
        raise ValueError(f"invalid {stage} training evidence: {error}") from error

    provenance = validated["provenance"]
    split = validated["split"]
    initialization = validated["initialization"]
    stability = validated["stability"]
    metrics = validated["metrics"]
    assert all(isinstance(value, Mapping) for value in (provenance, split, initialization, stability, metrics))
    artifact_observations = verify_referenced_artifacts(validated, stage=stage)
    artifacts_ok = _artifacts_verified(artifact_observations)
    metrics_ok = _metrics_are_finite(metrics, stage=stage)
    finite_ok = bool(stability["finite"]) and metrics_ok
    validation_observed = (
        "T0 intentionally records only its one-update training loss"
        if stage == "T0"
        else f"all {len(REQUIRED_T1_METRICS)} required T1 validation metrics recorded"
    )

    checks = (
        CheckResult(
            "provenance.pinned_upstream", "provenance", "Pinned, clean upstream",
            15, bool(provenance["upstream_clean"])
            and str(provenance["upstream_commit"]).lower() == APPROVED_UPSTREAM_COMMIT,
            blocker=True, expected=f"clean {APPROVED_UPSTREAM_COMMIT}",
            observed=f"clean={provenance['upstream_clean']}; commit={provenance['upstream_commit']}",
        ),
        CheckResult(
            "provenance.no_official_model_checkpoint", "provenance", "ImageNet-backbone-only initialization",
            10, not bool(initialization["model_checkpoint_loaded"]), blocker=True,
            expected="official model checkpoint not loaded",
            observed=str(initialization["model_checkpoint_loaded"]),
        ),
        CheckResult(
            "split.no_overlap", "split", "Train and validation do not overlap",
            10, int(split["overlap"]) == 0, blocker=True, expected="overlap=0",
            observed=f"overlap={split['overlap']}",
        ),
        CheckResult(
            "split.complete", "split", "Approved split is complete",
            5, bool(split["complete"])
            and (int(split["dataset_population"]), int(split["train_count"]), int(split["validation_count"]))
            == (1201, 961, 240), blocker=True, expected="complete 1201/961/240 split",
            observed=(f"complete={split['complete']}; population={split['dataset_population']}; "
                      f"train={split['train_count']}; validation={split['validation_count']}"),
        ),
        CheckResult(
            "split.no_official_test_access", "split", "Official Test remains sealed",
            5, not bool(split["official_test_accessed"]), blocker=True,
            expected="official_test_accessed=false", observed=str(split["official_test_accessed"]),
        ),
        CheckResult(
            "stability.finite_metrics", "stability", "All required values are finite",
            12, finite_ok, blocker=True, expected="finite evidence and metrics",
            observed=f"finite={stability['finite']}; metrics_finite={metrics_ok}",
        ),
        CheckResult(
            "stability.resume_round_trip", "stability", "Checkpoint round trip succeeds",
            8, bool(stability["checkpoint_round_trip"]), blocker=True,
            expected="checkpoint_round_trip=true", observed=str(stability["checkpoint_round_trip"]),
        ),
        CheckResult(
            "validation.stage_evidence", "validation", "Required stage validation evidence",
            20, metrics_ok, expected="all required metrics present and finite",
            observed=validation_observed if metrics_ok else "missing or non-finite metric",
        ),
        CheckResult(
            "artifacts.hashes", "artifacts", "All referenced artifacts hash-verify from disk",
            15, artifacts_ok, blocker=True, expected="every artifact SHA-256 matches",
            observed="; ".join(artifact_observations[name] for name in sorted(artifact_observations)),
        ),
    )
    return score_stage(
        stage_id=stage,
        stage_name=f"STEERER UCF-QNRF {stage} training",
        threshold=STAGE_THRESHOLDS[stage],
        checks=checks,
        success_status=SUCCESS_STATUS,
    )


def weights_by_category(report: StageReport) -> dict[str, float]:
    """Return category totals for review tests and human scorecards."""

    totals: defaultdict[str, float] = defaultdict(float)
    for check in report.checks:
        totals[check.category] += check.weight
    return dict(totals)


__all__ = [
    "APPROVED_UPSTREAM_COMMIT",
    "REQUIRED_ARTIFACTS",
    "REQUIRED_T1_METRICS",
    "STAGE_THRESHOLDS",
    "score_training_stage",
    "validate_training_evidence",
    "verify_referenced_artifacts",
    "weights_by_category",
]
