from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from droneai.integrity import is_sha256


@dataclass(frozen=True)
class ModelBrief:
    model_id: str
    paper: str
    role: str
    family: str
    backbone: str
    parameter_count: int | None
    blocks: tuple[str, ...]
    feature_scales: tuple[str, ...]
    input_contract: str
    preprocessing_policy: str
    coordinate_transform: str
    native_output: str
    count_derivation: str
    zone_derivation: str
    original_losses: tuple[str, ...]
    official_protocol: str
    official_reported_metrics: tuple[str, ...]
    strengths: tuple[str, ...]
    failure_modes: tuple[str, ...]
    runtime_risks: tuple[str, ...]
    rights_status: str
    code_rights_status: str
    dataset_rights_status: str
    checkpoint_rights_status: str
    deployment_rights_status: str
    upstream_commit: str
    checkpoint_path: str
    checkpoint_sha256: str
    reviewed_paths: tuple[str, ...]
    review_status: str

    def __post_init__(self) -> None:
        required_text = (
            self.model_id,
            self.paper,
            self.role,
            self.family,
            self.backbone,
            self.input_contract,
            self.preprocessing_policy,
            self.coordinate_transform,
            self.native_output,
            self.count_derivation,
            self.zone_derivation,
            self.official_protocol,
            self.rights_status,
            self.code_rights_status,
            self.dataset_rights_status,
            self.checkpoint_rights_status,
            self.deployment_rights_status,
            self.checkpoint_path,
        )
        if any(not value.strip() for value in required_text):
            raise ValueError("model brief fields cannot be empty")
        required_lists = (
            self.blocks,
            self.feature_scales,
            self.original_losses,
            self.official_reported_metrics,
            self.strengths,
            self.failure_modes,
            self.runtime_risks,
        )
        if any(
            not values or any(not value.strip() for value in values)
            for values in required_lists
        ):
            raise ValueError(
                "model brief architecture and review lists cannot be empty"
            )
        if self.parameter_count is not None and self.parameter_count < 0:
            raise ValueError("parameter count must be non-negative when measured")
        if (
            re.fullmatch(r"[0-9a-f]{40}", self.upstream_commit) is None
            or not is_sha256(self.checkpoint_sha256)
        ):
            raise ValueError("model brief requires commit and checkpoint SHA-256")

    def require_full_run_approval(self) -> None:
        if (
            self.review_status != "approved"
            or not self.reviewed_paths
            or any(not path.strip() for path in self.reviewed_paths)
        ):
            raise ValueError(
                "full run requires an approved model brief and reviewed code paths"
            )


def _bullets(values: tuple[str, ...]) -> list[str]:
    return [f"- {value}" for value in values]


def render_model_brief(brief: ModelBrief) -> str:
    lines = [
        f"# Model brief: {brief.model_id}",
        "",
        f"- Paper: {brief.paper}",
        f"- Role: {brief.role}",
        f"- Family: {brief.family}",
        f"- Backbone: {brief.backbone}",
        f"- Parameter count: {brief.parameter_count if brief.parameter_count is not None else 'not measured'}",
        f"- Input: {brief.input_contract}",
        f"- Preprocessing: {brief.preprocessing_policy}",
        f"- Coordinate transform: {brief.coordinate_transform}",
        f"- Native output: {brief.native_output}",
        f"- Count derivation: {brief.count_derivation}",
        f"- Zone derivation: {brief.zone_derivation}",
        f"- Official protocol: {brief.official_protocol}",
        f"- Rights status: {brief.rights_status}",
        f"- Code rights: {brief.code_rights_status}",
        f"- Dataset rights: {brief.dataset_rights_status}",
        f"- Checkpoint rights: {brief.checkpoint_rights_status}",
        f"- Deployment rights: {brief.deployment_rights_status}",
        f"- Upstream commit: `{brief.upstream_commit}`",
        f"- Checkpoint path: `{brief.checkpoint_path}`",
        f"- Checkpoint SHA-256: `{brief.checkpoint_sha256}`",
        "",
        "## Major blocks",
        "",
        *_bullets(brief.blocks),
        "",
        "## Feature scales",
        "",
        *_bullets(brief.feature_scales),
        "",
        "## Original losses",
        "",
        *_bullets(brief.original_losses),
        "",
        "## Official reported metrics",
        "",
        *_bullets(brief.official_reported_metrics),
        "",
        "## Strengths",
        "",
        *_bullets(brief.strengths),
        "",
        "## Failure modes",
        "",
        *_bullets(brief.failure_modes),
        "",
        "## Runtime risks",
        "",
        *_bullets(brief.runtime_risks),
        "",
        "## Reviewed code",
        "",
        *_bullets(brief.reviewed_paths),
        "",
        f"Review status: **{brief.review_status}**",
        "",
    ]
    return "\n".join(lines)


def write_model_brief(path: str | Path, brief: ModelBrief) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_model_brief(brief), encoding="utf-8")
    return target
