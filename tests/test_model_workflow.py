from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from droneai.model_workflow import (
    assert_action_allowed,
    build_workflow_status,
    load_model_queue,
)


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comparison_claim(status: str = "UNKNOWN") -> dict[str, str]:
    return {
        "checkpoint_training_split_status": status,
        "comparison_scope": "compatibility_smoke",
        "checkpoint_split_evidence": "pinned fixture evidence",
    }


def _queue(
    tmp_path: Path,
    *,
    active_model: str = "steerer",
    split_role: str = "validation",
    include_one_sample: bool = True,
) -> Path:
    results = tmp_path / "results"
    rights_hash = _write_json(results / "steerer/rights/rights-decision.json", {"status": "PASS"})
    preflight_hash = _write_json(results / "steerer/environment/verify-only.json", {"status": "PASS"})
    evidence = [
        {
            "gate": "rights",
            "artifact_kind": "rights_decision",
            "path": "steerer/rights/rights-decision.json",
            "sha256": rights_hash,
        },
        {
            "gate": "preflight",
            "artifact_kind": "environment_verification",
            "path": "steerer/environment/verify-only.json",
            "sha256": preflight_hash,
        },
    ]
    if include_one_sample:
        sample_hash = _write_json(
            results / "steerer/one-sample/result.json",
            {
                "status": "PASS",
                "dataset_id": "ucf-qnrf-kaggle-apache",
                "split_id": "dm-count-upstream-qnrf-val-smoke-36-v1",
            },
        )
        evidence.append(
            {
                "gate": "one_sample",
                "artifact_kind": "one_sample_result",
                "path": "steerer/one-sample/result.json",
                "sha256": sample_hash,
            }
        )

    dm_score_hash = _write_json(results / "dm-count/run/score.json", {"status": "PASS_RESEARCH_ONLY"})
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "round_id": "round-1-static-ucf-qnrf",
                "active_model": active_model,
                "models": [
                    {
                        **_comparison_claim("VERIFIED_DISJOINT"),
                        "model_id": "dm-count",
                        "family": "density",
                        "queue_state": "completed",
                        "rights_scope": "PASS_COMMERCIAL_CANDIDATE",
                        "dataset_id": "ucf-qnrf-kaggle-apache",
                        "split_id": "dm-count-upstream-qnrf-val-smoke-36-v1",
                        "split_role": "validation",
                        "expected_samples": 36,
                        "gate_order": ["benchmark"],
                        "accepted_evidence": [
                            {
                                "gate": "benchmark",
                                "artifact_kind": "score",
                                "path": "dm-count/run/score.json",
                                "sha256": dm_score_hash,
                            }
                        ],
                    },
                    {
                        **_comparison_claim(),
                        "model_id": "steerer",
                        "family": "density_and_points",
                        "queue_state": "active",
                        "rights_scope": "PASS_COMMERCIAL_CANDIDATE",
                        "dataset_id": "ucf-qnrf-kaggle-apache",
                        "split_id": "dm-count-upstream-qnrf-val-smoke-36-v1",
                        "split_role": split_role,
                        "expected_samples": 36,
                        "gate_order": ["rights", "preflight", "one_sample", "benchmark"],
                        "accepted_evidence": evidence,
                    },
                    {
                        **_comparison_claim(),
                        "model_id": "pet",
                        "family": "points",
                        "queue_state": "queued",
                        "rights_scope": "PENDING",
                        "dataset_id": "ucf-qnrf-kaggle-apache",
                        "split_id": "dm-count-upstream-qnrf-val-smoke-36-v1",
                        "split_role": "validation",
                        "expected_samples": 36,
                        "gate_order": ["rights", "preflight", "one_sample", "benchmark"],
                        "accepted_evidence": [],
                    },
                    {
                        **_comparison_claim(),
                        "model_id": "csrnet",
                        "family": "density",
                        "queue_state": "blocked",
                        "rights_scope": "NOT_EVALUATED_NO_ELIGIBLE_WEIGHT",
                        "dataset_id": "ucf-qnrf-kaggle-apache",
                        "split_id": "dm-count-upstream-qnrf-val-smoke-36-v1",
                        "split_role": "validation",
                        "expected_samples": 36,
                        "gate_order": ["rights", "preflight", "one_sample", "benchmark"],
                        "accepted_evidence": [],
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return queue_path


def test_status_uses_hash_verified_evidence_to_select_steerer_benchmark(tmp_path: Path) -> None:
    queue_path = _queue(tmp_path)

    status = build_workflow_status(queue_path, tmp_path / "results")

    assert status["active_model"] == "steerer"
    assert status["current_stage"] == "round-1-static-ucf-qnrf"
    assert status["completed_gates"] == ["rights", "preflight", "one_sample"]
    assert status["next_action"] == "benchmark"
    assert status["approval_required"] is True
    assert status["dataset"]["dataset_id"] == "ucf-qnrf-kaggle-apache"
    assert status["validation"]["expected_samples"] == 36
    assert status["validation"]["fine_tuning"] is False
    assert status["checkpoint_training_split_status"] == "UNKNOWN"
    assert status["comparison_scope"] == "compatibility_smoke"
    assert status["ranking_eligible"] is False
    assert status["roadmap"][1]["ranking_eligible"] is False
    assert [entry["model_id"] for entry in status["roadmap"]] == [
        "dm-count",
        "steerer",
        "pet",
        "csrnet",
    ]


def test_changed_evidence_hash_is_rejected(tmp_path: Path) -> None:
    queue_path = _queue(tmp_path)
    result = tmp_path / "results/steerer/one-sample/result.json"
    result.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="evidence SHA-256 mismatch"):
        build_workflow_status(queue_path, tmp_path / "results")


def test_unknown_active_model_is_rejected(tmp_path: Path) -> None:
    queue_path = _queue(tmp_path, active_model="unknown")

    with pytest.raises(ValueError, match="active_model"):
        load_model_queue(queue_path)


def test_missing_checkpoint_split_evidence_is_rejected(tmp_path: Path) -> None:
    queue_path = _queue(tmp_path)
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    del payload["models"][1]["checkpoint_split_evidence"]
    queue_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint_split_evidence"):
        load_model_queue(queue_path)


def test_unknown_checkpoint_cannot_claim_held_out_performance(tmp_path: Path) -> None:
    queue_path = _queue(tmp_path)
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    payload["models"][1]["comparison_scope"] = "held_out_performance"
    queue_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="held-out"):
        load_model_queue(queue_path)


def test_test_split_is_rejected(tmp_path: Path) -> None:
    queue_path = _queue(tmp_path, split_role="test")

    with pytest.raises(ValueError, match="test split"):
        load_model_queue(queue_path)


def test_benchmark_cannot_be_authorized_before_one_sample(tmp_path: Path) -> None:
    queue_path = _queue(tmp_path, include_one_sample=False)
    status = build_workflow_status(queue_path, tmp_path / "results")

    assert status["next_action"] == "one_sample"
    with pytest.raises(PermissionError, match="next safe action is one_sample"):
        assert_action_allowed(status, "benchmark", user_approved=True)


def test_benchmark_requires_explicit_user_approval(tmp_path: Path) -> None:
    queue_path = _queue(tmp_path)
    status = build_workflow_status(queue_path, tmp_path / "results")

    with pytest.raises(PermissionError, match="explicit user approval"):
        assert_action_allowed(status, "benchmark", user_approved=False)
    assert_action_allowed(status, "benchmark", user_approved=True)


def test_repository_queue_records_apgcc_rights_and_advances_to_preflight() -> None:
    queue = load_model_queue(Path("configs/evaluation/model_queue.json"))
    models = {model["model_id"]: model for model in queue["models"]}

    assert queue["active_model"] == "apgcc"
    assert models["mpcount"]["queue_state"] == "completed"
    assert models["apgcc"]["queue_state"] == "active"
    assert models["apgcc"]["family"] == "points"
    assert models["apgcc"]["rights_scope"] == "PASS_COMMERCIAL_CANDIDATE"
    assert models["apgcc"]["checkpoint_training_split_status"] == "VERIFIED_DISJOINT"
    assert models["apgcc"]["comparison_scope"] == "compatibility_smoke"
    assert models["apgcc"]["gate_order"] == [
        "rights",
        "preflight",
        "one_sample",
        "benchmark",
    ]
    assert [
        (evidence["gate"], evidence["artifact_kind"])
        for evidence in models["apgcc"]["accepted_evidence"]
    ] == [
        ("rights", "rights_decision"),
        ("rights", "manifest_snapshot"),
    ]
