from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import PurePosixPath

from droneai.integrity import is_sha256
from droneai.scoring import CheckResult, StageReport, score_stage


@dataclass(frozen=True)
class EvidenceArtifact:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("artifact reference path is required")
        normalized = PurePosixPath(self.path.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError("artifact reference path must be contained and relative")
        if not is_sha256(self.sha256):
            raise ValueError("artifact reference requires SHA-256")


@dataclass(frozen=True)
class EvaluationEvidence:
    run_id: str
    expected_samples: int
    recorded_samples: int
    successful_samples: int
    split_verified: bool
    leakage_free: bool
    mae: float
    mae_max: float
    rmse: float
    rmse_max: float
    absolute_bias: float
    bias_max: float
    spatial_pass: bool
    robustness_pass: bool
    median_latency_ms: float
    latency_max_ms: float
    peak_vram_mb: float
    vram_max_mb: float
    provenance_verified: bool
    model_brief_approved: bool
    review_artifacts_verified: bool
    review_budget_ok: bool
    explicit_failures: int
    artifact_references: tuple[EvidenceArtifact, ...]
    spatial_target: str
    spatial_observed: str
    robustness_target: str
    robustness_observed: str

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id is required")
        counts = (
            self.expected_samples,
            self.recorded_samples,
            self.successful_samples,
            self.explicit_failures,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise ValueError("sample counts must be integers")
        if self.expected_samples <= 0 or any(value < 0 for value in counts[1:]):
            raise ValueError("sample counts must be non-negative and expected positive")

        boolean_fields = (
            "split_verified",
            "leakage_free",
            "spatial_pass",
            "robustness_pass",
            "provenance_verified",
            "model_brief_approved",
            "review_artifacts_verified",
            "review_budget_ok",
        )
        if any(type(getattr(self, name)) is not bool for name in boolean_fields):
            raise ValueError("gate flags must be boolean values")

        threshold_fields = (
            "mae_max",
            "rmse_max",
            "bias_max",
            "latency_max_ms",
            "vram_max_mb",
        )
        if any(
            not isinstance(getattr(self, name), Real)
            or isinstance(getattr(self, name), bool)
            or not math.isfinite(getattr(self, name))
            or getattr(self, name) < 0
            for name in threshold_fields
        ):
            raise ValueError(
                "every frozen threshold must be a finite non-negative real number"
            )

        observed_fields = (
            "mae",
            "rmse",
            "absolute_bias",
            "median_latency_ms",
            "peak_vram_mb",
        )
        if any(
            not isinstance(getattr(self, name), Real)
            or isinstance(getattr(self, name), bool)
            or not math.isfinite(getattr(self, name))
            or getattr(self, name) < 0
            for name in observed_fields
        ):
            raise ValueError(
                "observed metrics must be finite non-negative real numbers"
            )
        if self.median_latency_ms <= 0:
            raise ValueError("median latency must be positive")

        if (
            not isinstance(self.artifact_references, tuple)
            or not self.artifact_references
            or any(
                not isinstance(reference, EvidenceArtifact)
                for reference in self.artifact_references
            )
            or len({reference.path for reference in self.artifact_references})
            != len(self.artifact_references)
        ):
            raise ValueError("artifact references must be a non-empty unique tuple")
        audit_text = (
            self.spatial_target,
            self.spatial_observed,
            self.robustness_target,
            self.robustness_observed,
        )
        if any(not isinstance(value, str) or not value.strip() for value in audit_text):
            raise ValueError("spatial and robustness audit text is required")


def build_evaluation_checks(evidence: EvaluationEvidence) -> list[CheckResult]:
    artifact_evidence = "; ".join(
        f"{reference.path}@{reference.sha256}"
        for reference in evidence.artifact_references
    )
    return [
        CheckResult(
            "data.split",
            "data and split integrity",
            "Frozen split and no leakage",
            15,
            evidence.split_verified and evidence.leakage_free,
            True,
            observed=(
                f"split={evidence.split_verified}; "
                f"leakage_free={evidence.leakage_free}"
            ),
        ),
        CheckResult(
            "coverage.accounting",
            "evaluation coverage",
            "Every expected sample has a success or explicit failure record",
            10,
            evidence.recorded_samples == evidence.expected_samples
            and evidence.successful_samples + evidence.explicit_failures
            == evidence.expected_samples,
            True,
            expected=(
                f"records={evidence.expected_samples}; "
                f"successes+failures={evidence.expected_samples}"
            ),
            observed=(
                f"records={evidence.recorded_samples}; successes+failures="
                f"{evidence.successful_samples + evidence.explicit_failures}"
            ),
        ),
        CheckResult(
            "count.quality",
            "count quality",
            "MAE, RMSE, and absolute bias meet frozen targets",
            20,
            evidence.mae <= evidence.mae_max
            and evidence.rmse <= evidence.rmse_max
            and evidence.absolute_bias <= evidence.bias_max,
            False,
            observed=(
                f"mae={evidence.mae}/{evidence.mae_max}; "
                f"rmse={evidence.rmse}/{evidence.rmse_max}; "
                f"bias={evidence.absolute_bias}/{evidence.bias_max}"
            ),
        ),
        CheckResult(
            "spatial.quality",
            "spatial output quality",
            "Family-specific spatial targets pass",
            15,
            evidence.spatial_pass,
            False,
            expected=evidence.spatial_target,
            observed=evidence.spatial_observed,
            evidence=artifact_evidence,
        ),
        CheckResult(
            "condition.robustness",
            "condition robustness",
            "Density and condition targets pass",
            10,
            evidence.robustness_pass,
            False,
            expected=evidence.robustness_target,
            observed=evidence.robustness_observed,
            evidence=artifact_evidence,
        ),
        CheckResult(
            "runtime.resources",
            "runtime and resources",
            "Latency and peak VRAM meet frozen targets",
            10,
            evidence.median_latency_ms <= evidence.latency_max_ms
            and evidence.peak_vram_mb <= evidence.vram_max_mb,
            False,
            observed=(
                f"latency={evidence.median_latency_ms}/{evidence.latency_max_ms}; "
                f"vram={evidence.peak_vram_mb}/{evidence.vram_max_mb}"
            ),
        ),
        CheckResult(
            "provenance.bundle",
            "reproducibility and provenance",
            "Code, data, checkpoint, config, and environment are verified",
            10,
            evidence.provenance_verified,
            True,
            observed=str(evidence.provenance_verified),
            evidence=artifact_evidence,
        ),
        CheckResult(
            "review.bundle",
            "review artifacts",
            "Approved model brief and verified budgeted review bundle exist",
            10,
            evidence.model_brief_approved
            and evidence.review_artifacts_verified
            and evidence.review_budget_ok,
            True,
            observed=(
                f"brief={evidence.model_brief_approved}; "
                f"artifacts={evidence.review_artifacts_verified}; "
                f"budget={evidence.review_budget_ok}"
            ),
            evidence=artifact_evidence,
        ),
    ]


def run_evaluation_gate(evidence: EvaluationEvidence) -> StageReport:
    return score_stage(
        stage_id="model-evaluation",
        stage_name=f"Model-dataset evaluation: {evidence.run_id}",
        threshold=80,
        checks=build_evaluation_checks(evidence),
        success_status="PASS_RESEARCH_ONLY",
    )
