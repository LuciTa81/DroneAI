from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import scripts.run_steerer_a_g1_official_checkpoint as g1_cli

from droneai.steerer_official_reproduction import (
    build_g1_protocol,
    evaluate_g1_compatibility,
    load_official_reproduction_profile,
    validate_official_checkpoint_manifest,
    validate_official_reproduction_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    REPO_ROOT
    / "configs/evaluation/steerer_a_official_checkpoint_test334.json"
)


def _payload() -> dict[str, object]:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def test_a_profile_freezes_official_code_protocol_and_isolated_roots() -> None:
    profile = load_official_reproduction_profile(PROFILE)

    assert profile.train_samples == 1201
    assert profile.test_samples == 334
    assert profile.result_label == "official Test, test-selected"
    assert profile.checkpoint_origin == "research_checkpoint"
    assert profile.official_weight_scope == "PASS_RESEARCH_ONLY"
    assert profile.upstream_commit == (
        "5b1854dbc2d280f2326d67c65515d8baf9083810"
    )
    assert profile.checkpoint.sha256 == (
        "85f71763595527e419257d459288e4a784766f41687ff0b859dcff65a151bbec"
    )
    assert profile.checkpoint.size_bytes == 259607784
    assert all("steerer-official-code-reproduction" in path for path in profile.roots)
    assert all("steerer-qnrf-imagenet-20260806-t1-e" not in path for path in profile.roots)
    assert profile.g1_prepared_test_root.endswith("/g1-test-v1")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p.update({"train_samples": 1200}), "1,201"),
        (lambda p: p.update({"test_samples": 333}), "334"),
        (
            lambda p: p.update({"result_label": "official Test"}),
            "test-selected",
        ),
        (
            lambda p: p.update({"official_weight_scope": "PASS_COMMERCIAL_CANDIDATE"}),
            "research",
        ),
        (
            lambda p: p["roots"].update(
                {"results": "/workspace/data/results/steerer-qnrf-imagenet-20260806-t1-e"}
            ),
            "isolated",
        ),
        (
            lambda p: p["checkpoint"].update({"sha256": "0" * 64}),
            "checkpoint",
        ),
        (lambda p: p.update({"test_access_approved": False}), "Test access"),
    ],
)
def test_a_profile_rejects_protocol_drift(mutation, message: str) -> None:
    payload = copy.deepcopy(_payload())
    mutation(payload)

    with pytest.raises((PermissionError, ValueError), match=message):
        validate_official_reproduction_profile(payload)


def test_g1_protocol_is_explicit_research_checkpoint_test_evaluation(
    tmp_path: Path,
) -> None:
    profile = load_official_reproduction_profile(PROFILE)
    rights = tmp_path / "rights.json"
    rights.write_text('{"scope":"research comparison only"}', encoding="utf-8")

    protocol = build_g1_protocol(profile, rights_path=rights)

    assert protocol.split_role == "test"
    assert protocol.expected_samples == 334
    assert protocol.sealed_test_access_approved is True
    assert protocol.checkpoint_training_split_status == "UNKNOWN"
    assert protocol.comparison_scope == "compatibility_smoke"
    assert protocol.leakage_free is True


def test_g1_cli_exposes_no_training_or_tuning_controls() -> None:
    options = {
        option
        for action in g1_cli._parser()._actions
        for option in action.option_strings
    }

    assert {
        "--config",
        "--prepared-test-root",
        "--prepared-manifest",
        "--upstream-dir",
        "--checkpoint",
        "--checkpoint-manifest",
        "--rights-decision",
        "--output-dir",
        "--device",
    }.issubset(options)
    assert {
        "--test-root",
        "--train-root",
        "--resume",
        "--epochs",
        "--optimizer",
        "--checkpoint-origin",
    }.isdisjoint(options)


def test_official_checkpoint_manifest_matches_frozen_research_artifact(
    tmp_path: Path,
) -> None:
    profile = load_official_reproduction_profile(PROFILE)
    manifest = tmp_path / "artifact-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "allowed_scope": "research_comparison_only",
                "artifact_id": profile.checkpoint.artifact_id,
                "byte_count": profile.checkpoint.size_bytes,
                "filename": profile.checkpoint.filename,
                "sha256": profile.checkpoint.sha256,
                "upstream_commit": profile.upstream_commit,
            }
        ),
        encoding="utf-8",
    )

    verified = validate_official_checkpoint_manifest(profile, manifest)
    assert verified["allowed_scope"] == "research_comparison_only"

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["allowed_scope"] = "commercial_deployment"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="research"):
        validate_official_checkpoint_manifest(profile, manifest)


def test_g1_compatibility_gate_uses_prefrozen_repository_tolerance() -> None:
    profile = load_official_reproduction_profile(PROFILE)

    passed = evaluate_g1_compatibility(
        profile,
        {
            "accounting_complete": True,
            "expected_samples": 334,
            "recorded_samples": 334,
            "successful_samples": 334,
            "explicit_failures": 0,
            "mae": 80.0,
            "rmse": 145.0,
        },
    )
    failed = evaluate_g1_compatibility(
        profile,
        {
            "accounting_complete": True,
            "expected_samples": 334,
            "recorded_samples": 334,
            "successful_samples": 334,
            "explicit_failures": 0,
            "mae": 83.0,
            "rmse": 149.0,
        },
    )

    assert passed["status"] == "PASS_RESEARCH_ONLY"
    assert passed["training_authorized"] is True
    assert passed["absolute_difference"] == {"mae": 2.2, "rmse": 7.0}
    assert failed["status"] == "BLOCKED"
    assert failed["training_authorized"] is False
