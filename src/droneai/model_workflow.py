from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from droneai.comparison_claim import is_ranking_eligible, validate_comparison_claim


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GATES = ("rights", "preflight", "one_sample", "benchmark")
_QUEUE_STATES = {"completed", "active", "queued", "blocked"}


def _nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_relative_path(value: object, *, field: str) -> str:
    text = _nonempty(value, field=field)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must stay relative to the results root")
    return path.as_posix()


def _validate_model(raw: object, *, index: int) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"models[{index}] must be an object")
    model = dict(raw)
    prefix = f"models[{index}]"
    for field in ("model_id", "family", "rights_scope", "dataset_id", "split_id"):
        model[field] = _nonempty(model.get(field), field=f"{prefix}.{field}")

    state = _nonempty(model.get("queue_state"), field=f"{prefix}.queue_state")
    if state not in _QUEUE_STATES:
        raise ValueError(f"{prefix}.queue_state is invalid")
    model["queue_state"] = state

    split_role = _nonempty(model.get("split_role"), field=f"{prefix}.split_role")
    if split_role == "test":
        raise ValueError("model queue cannot authorize a test split")
    if split_role not in {"train", "validation", "smoke"}:
        raise ValueError(f"{prefix}.split_role is invalid")
    model["split_role"] = split_role

    checkpoint_status = _nonempty(
        model.get("checkpoint_training_split_status"),
        field=f"{prefix}.checkpoint_training_split_status",
    )
    comparison_scope = _nonempty(
        model.get("comparison_scope"),
        field=f"{prefix}.comparison_scope",
    )
    checkpoint_evidence = _nonempty(
        model.get("checkpoint_split_evidence"),
        field=f"{prefix}.checkpoint_split_evidence",
    )
    validate_comparison_claim(
        checkpoint_status,
        comparison_scope,
        checkpoint_evidence,
    )
    model["checkpoint_training_split_status"] = checkpoint_status
    model["comparison_scope"] = comparison_scope
    model["checkpoint_split_evidence"] = checkpoint_evidence
    model["ranking_eligible"] = is_ranking_eligible(
        checkpoint_status,
        comparison_scope,
    )

    expected = model.get("expected_samples")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
        raise ValueError(f"{prefix}.expected_samples must be a positive integer")

    order = model.get("gate_order")
    if (
        not isinstance(order, list)
        or not order
        or any(gate not in _GATES for gate in order)
        or len(set(order)) != len(order)
    ):
        raise ValueError(f"{prefix}.gate_order must contain unique supported gates")
    positions = [_GATES.index(gate) for gate in order]
    if positions != sorted(positions):
        raise ValueError(f"{prefix}.gate_order must preserve the safety order")

    accepted = model.get("accepted_evidence")
    if not isinstance(accepted, list):
        raise ValueError(f"{prefix}.accepted_evidence must be a list")
    normalized_evidence: list[dict[str, str]] = []
    for evidence_index, raw_evidence in enumerate(accepted):
        evidence_prefix = f"{prefix}.accepted_evidence[{evidence_index}]"
        if not isinstance(raw_evidence, dict):
            raise ValueError(f"{evidence_prefix} must be an object")
        gate = _nonempty(raw_evidence.get("gate"), field=f"{evidence_prefix}.gate")
        if gate not in order:
            raise ValueError(f"{evidence_prefix}.gate is not in gate_order")
        digest = _nonempty(
            raw_evidence.get("sha256"), field=f"{evidence_prefix}.sha256"
        ).lower()
        if _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{evidence_prefix}.sha256 is invalid")
        normalized_evidence.append(
            {
                "gate": gate,
                "artifact_kind": _nonempty(
                    raw_evidence.get("artifact_kind"),
                    field=f"{evidence_prefix}.artifact_kind",
                ),
                "path": _validate_relative_path(
                    raw_evidence.get("path"), field=f"{evidence_prefix}.path"
                ),
                "sha256": digest,
            }
        )
    model["accepted_evidence"] = normalized_evidence
    return model


def load_model_queue(path: str | Path) -> dict[str, object]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("model queue schema_version must be 1")

    queue = dict(payload)
    queue["round_id"] = _nonempty(queue.get("round_id"), field="round_id")
    active_model = _nonempty(queue.get("active_model"), field="active_model")
    raw_models = queue.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models must be a non-empty list")
    models = [_validate_model(raw, index=index) for index, raw in enumerate(raw_models)]
    identifiers = [str(model["model_id"]) for model in models]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("model_id values must be unique")
    if active_model not in identifiers:
        raise ValueError("active_model must identify a queued model")
    active_entries = [model for model in models if model["queue_state"] == "active"]
    if len(active_entries) != 1 or active_entries[0]["model_id"] != active_model:
        raise ValueError("active_model and active queue_state must identify one model")
    queue["active_model"] = active_model
    queue["models"] = models
    return queue


def _verified_gates(model: dict[str, object], results_root: Path) -> list[str]:
    evidence_by_gate: dict[str, list[dict[str, str]]] = {}
    for evidence in model["accepted_evidence"]:
        target = results_root / evidence["path"]
        if not target.is_file():
            raise FileNotFoundError(f"accepted evidence is missing: {target}")
        observed = _sha256(target)
        if observed != evidence["sha256"]:
            raise ValueError(
                f"evidence SHA-256 mismatch: {evidence['path']} "
                f"expected={evidence['sha256']} observed={observed}"
            )
        evidence_by_gate.setdefault(evidence["gate"], []).append(evidence)
    return [gate for gate in model["gate_order"] if evidence_by_gate.get(gate)]


def _metric_names(family: str) -> list[str]:
    metrics = ["mae", "rmse", "signed_bias", "mape_reference", "latency", "fps", "peak_vram"]
    if family in {"density", "density_and_points"}:
        metrics.extend(["game_l1", "density_zone_mae", "psnr", "ssim"])
    if family in {"points", "density_and_points"}:
        metrics.extend(["localization_precision", "localization_recall", "localization_f1", "mean_distance"])
    return metrics


def build_workflow_status(
    queue_path: str | Path,
    results_root: str | Path,
) -> dict[str, object]:
    queue = load_model_queue(queue_path)
    root = Path(results_root).resolve()
    roadmap: list[dict[str, object]] = []
    active: dict[str, object] | None = None
    active_gates: list[str] = []
    for model in queue["models"]:
        gates = _verified_gates(model, root)
        roadmap.append(
            {
                "model_id": model["model_id"],
                "family": model["family"],
                "queue_state": model["queue_state"],
                "rights_scope": model["rights_scope"],
                "checkpoint_training_split_status": model[
                    "checkpoint_training_split_status"
                ],
                "comparison_scope": model["comparison_scope"],
                "checkpoint_split_evidence": model["checkpoint_split_evidence"],
                "ranking_eligible": model["ranking_eligible"],
                "verified_gates": gates,
            }
        )
        if model["model_id"] == queue["active_model"]:
            active = model
            active_gates = gates
    if active is None:
        raise RuntimeError("validated queue lost its active model")

    next_action = next(
        (gate for gate in active["gate_order"] if gate not in active_gates),
        "review_and_advance",
    )
    return {
        "schema_version": 1,
        "active_model": active["model_id"],
        "model_family": active["family"],
        "current_stage": queue["round_id"],
        "completed_gates": active_gates,
        "next_action": next_action,
        "approval_required": next_action in {"benchmark", "review_and_advance"},
        "rights_scope": active["rights_scope"],
        "evaluation_scope": "research_comparison_only",
        "checkpoint_training_split_status": active[
            "checkpoint_training_split_status"
        ],
        "comparison_scope": active["comparison_scope"],
        "checkpoint_split_evidence": active["checkpoint_split_evidence"],
        "ranking_eligible": active["ranking_eligible"],
        "dataset": {
            "dataset_id": active["dataset_id"],
            "split_id": active["split_id"],
            "split_role": active["split_role"],
        },
        "validation": {
            "expected_samples": active["expected_samples"],
            "metrics": _metric_names(str(active["family"])),
            "fine_tuning": False,
        },
        "roadmap": roadmap,
    }


def assert_action_allowed(
    status: dict[str, object],
    requested_action: str,
    *,
    user_approved: bool,
) -> None:
    next_action = status.get("next_action")
    if requested_action != next_action:
        raise PermissionError(
            f"requested action {requested_action} is not allowed; "
            f"next safe action is {next_action}"
        )
    if status.get("approval_required") is True and not user_approved:
        raise PermissionError(f"{requested_action} requires explicit user approval")
