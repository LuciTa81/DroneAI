"""Reusable score gates for DroneAI experiment stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


@dataclass(frozen=True)
class CheckResult:
    """One independently reviewable item in a stage gate."""

    check_id: str
    category: str
    description: str
    weight: float
    passed: bool
    blocker: bool = False
    expected: str = ""
    observed: str = ""
    evidence: str = ""

    @property
    def earned(self) -> float:
        return self.weight if self.passed else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["earned"] = self.earned
        return payload


@dataclass(frozen=True)
class StageReport:
    """Machine-readable and human-reviewable result for one experiment stage."""

    stage_id: str
    stage_name: str
    threshold: float
    score: float
    status: str
    checks: tuple[CheckResult, ...]
    generated_at: str

    @property
    def failed_blockers(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if check.blocker and not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "threshold": self.threshold,
            "score": self.score,
            "status": self.status,
            "generated_at": self.generated_at,
            "failed_blockers": [check.check_id for check in self.failed_blockers],
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.stage_id}: {self.stage_name}",
            "",
            f"- Status: **{self.status}**",
            f"- Score: **{self.score:.1f}/100**",
            f"- Pass threshold: **{self.threshold:.1f}**",
            f"- Generated at: `{self.generated_at}`",
            "",
            "| Result | Category | Check | Points | Blocker | Observed |",
            "|---|---|---|---:|:---:|---|",
        ]
        for check in self.checks:
            result = "PASS" if check.passed else "FAIL"
            blocker = "Y" if check.blocker else ""
            observed = check.observed.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {result} | {check.category} | {check.description} "
                f"| {check.earned:.1f}/{check.weight:.1f} | {blocker} | {observed} |"
            )

        if self.failed_blockers:
            lines.extend(["", "## Failed blockers", ""])
            lines.extend(
                f"- `{check.check_id}`: expected {check.expected}; observed {check.observed}"
                for check in self.failed_blockers
            )

        lines.extend(
            [
                "",
                "## Decision rule",
                "",
                "A stage passes only when the score reaches the threshold and every blocker passes.",
                "",
            ]
        )
        return "\n".join(lines)


def score_stage(
    *,
    stage_id: str,
    stage_name: str,
    threshold: float,
    checks: Iterable[CheckResult],
) -> StageReport:
    """Calculate a 100-point gate without allowing blockers to be averaged away."""

    normalized = tuple(checks)
    if not normalized:
        raise ValueError("at least one check is required")
    if len({check.check_id for check in normalized}) != len(normalized):
        raise ValueError("check_id values must be unique")
    if any(check.weight <= 0 for check in normalized):
        raise ValueError("every check weight must be positive")

    total_weight = sum(check.weight for check in normalized)
    if abs(total_weight - 100.0) > 1e-6:
        raise ValueError(f"check weights must total 100, got {total_weight}")
    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")

    score = round(sum(check.earned for check in normalized), 2)
    blocker_failed = any(check.blocker and not check.passed for check in normalized)
    if blocker_failed:
        status = "BLOCKED"
    elif score >= threshold:
        status = "PASS"
    else:
        status = "REVIEW"

    return StageReport(
        stage_id=stage_id,
        stage_name=stage_name,
        threshold=threshold,
        score=score,
        status=status,
        checks=normalized,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
