import json
from pathlib import Path

from droneai.stage3c import REQUIRED_COMPONENTS, classify_rights, run_stage3c


def _component(
    component_type: str,
    *,
    status: str = "verified",
    commercial_use: bool | None = True,
    research_use: bool | None = True,
    basis: str = "explicit_terms",
) -> dict:
    return {
        "component_type": component_type,
        "component_id": f"fixture-{component_type}",
        "source_url": f"https://example.test/{component_type}",
        "rights": {
            "status": status,
            "license_id": "FIXTURE-LICENSE" if status == "verified" else None,
            "evidence_url": (
                f"https://example.test/{component_type}/terms" if status == "verified" else None
            ),
            "commercial_use": commercial_use,
            "research_use": research_use,
            "basis": basis,
        },
    }


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "candidate_id": "fixture-bundle",
        "intended_use": "commercial_product_rnd",
        "components": [_component(kind) for kind in sorted(REQUIRED_COMPONENTS)],
    }


def test_silent_terms_are_candidate_but_not_permission() -> None:
    manifest = _manifest()
    dataset = next(c for c in manifest["components"] if c["component_type"] == "dataset")
    dataset["rights"].update(
        {
            "status": "unverified",
            "license_id": None,
            "evidence_url": None,
            "commercial_use": None,
            "research_use": None,
            "basis": "not_stated",
        }
    )
    decision = classify_rights(manifest)
    assert decision.status == "PASS_COMMERCIAL_CANDIDATE"
    assert "dataset" in decision.ambiguous_components
    assert set(decision.allowed_actions) == {
        "catalog",
        "license_due_diligence",
        "synthetic_compatibility_smoke",
    }
    assert not {"asset_download", "commercial_training", "weight_reuse", "deployment"}.intersection(
        decision.allowed_actions
    )


def test_candidate_with_ambiguous_code_cannot_run_synthetic_smoke() -> None:
    manifest = _manifest()
    code = next(c for c in manifest["components"] if c["component_type"] == "code")
    code["rights"].update(
        {
            "status": "unverified",
            "license_id": None,
            "evidence_url": None,
            "commercial_use": None,
            "research_use": None,
            "basis": "not_stated",
        }
    )
    decision = classify_rights(manifest)
    assert decision.status == "PASS_COMMERCIAL_CANDIDATE"
    assert decision.allowed_actions == ("catalog", "license_due_diligence")


def test_verified_input_chain_allows_frozen_evaluation_but_not_training() -> None:
    manifest = _manifest()
    for component in manifest["components"]:
        if component["component_type"] in {"derived_weights", "deployment"}:
            component["rights"].update(
                {
                    "status": "pending",
                    "license_id": None,
                    "evidence_url": None,
                    "commercial_use": None,
                    "research_use": None,
                    "basis": "pending_review",
                }
            )

    decision = classify_rights(manifest)

    assert decision.status == "PASS_COMMERCIAL_CANDIDATE"
    assert {
        "asset_download",
        "synthetic_compatibility_smoke",
        "frozen_checkpoint_evaluation",
    }.issubset(decision.allowed_actions)
    assert not {
        "commercial_training",
        "derived_weight_use",
        "weight_reuse",
        "deployment",
    }.intersection(decision.allowed_actions)


def test_explicit_noncommercial_bundle_is_research_only() -> None:
    manifest = _manifest()
    for component in manifest["components"]:
        component["rights"].update(
            {"commercial_use": False, "research_use": True, "basis": "explicit_terms"}
        )
    decision = classify_rights(manifest)
    assert decision.status == "PASS_RESEARCH_ONLY"
    assert decision.allowed_actions == ("catalog", "research_reproduction")


def test_all_verified_components_are_production_approved() -> None:
    decision = classify_rights(_manifest())
    assert decision.status == "PRODUCTION_APPROVED"
    assert "asset_download" in decision.allowed_actions
    assert "synthetic_compatibility_smoke" in decision.allowed_actions
    assert "frozen_checkpoint_evaluation" in decision.allowed_actions
    assert "commercial_training" in decision.allowed_actions
    assert "deployment" in decision.allowed_actions


def test_prohibited_or_inconsistent_rights_are_blocked() -> None:
    prohibited = _manifest()
    dataset = next(c for c in prohibited["components"] if c["component_type"] == "dataset")
    dataset["rights"].update(
        {
            "status": "prohibited",
            "commercial_use": False,
            "research_use": False,
            "basis": "explicit_terms",
        }
    )
    assert classify_rights(prohibited).status == "BLOCKED"

    inconsistent = _manifest()
    dataset = next(c for c in inconsistent["components"] if c["component_type"] == "dataset")
    dataset["rights"].update({"commercial_use": False, "basis": "not_stated"})
    assert classify_rights(inconsistent).status == "BLOCKED"

    claimed_without_terms = _manifest()
    dataset = next(
        c for c in claimed_without_terms["components"] if c["component_type"] == "dataset"
    )
    dataset["rights"].update({"commercial_use": True, "basis": "not_stated"})
    assert classify_rights(claimed_without_terms).status == "BLOCKED"

    numeric_boolean = _manifest()
    dataset = next(c for c in numeric_boolean["components"] if c["component_type"] == "dataset")
    dataset["rights"]["commercial_use"] = 1
    assert classify_rights(numeric_boolean).status == "BLOCKED"


def test_stage3c_persists_candidate_decision_and_score(tmp_path: Path) -> None:
    manifest = _manifest()
    dataset = next(c for c in manifest["components"] if c["component_type"] == "dataset")
    dataset["rights"].update(
        {
            "status": "pending",
            "license_id": None,
            "evidence_url": None,
            "commercial_use": None,
            "research_use": None,
            "basis": "pending_review",
        }
    )
    manifest_path = tmp_path / "candidate.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = run_stage3c(manifest_path=manifest_path, output_dir=tmp_path / "run")
    assert result.report.status == "PASS_COMMERCIAL_CANDIDATE"
    assert result.decision.status == result.report.status
    for filename in (
        "manifest.snapshot.json",
        "rights-decision.json",
        "rights-decision.md",
        "score.json",
        "score.md",
    ):
        assert (tmp_path / "run" / filename).is_file()
    persisted = json.loads(
        (tmp_path / "run" / "rights-decision.json").read_text(encoding="utf-8")
    )
    assert persisted["candidate_id"] == "fixture-bundle"
    assert set(persisted["component_ids"]) == REQUIRED_COMPONENTS
    assert len(persisted["manifest_semantic_sha256"]) == 64


def test_dm_count_ucf_qnrf_manifest_authorizes_only_frozen_evaluation() -> None:
    manifest_path = (
        Path(__file__).parents[1]
        / "configs"
        / "candidates"
        / "dm_count_ucf_qnrf.candidate.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    decision = classify_rights(manifest)

    assert decision.status == "PASS_COMMERCIAL_CANDIDATE"
    assert "frozen_checkpoint_evaluation" in decision.allowed_actions
    assert "commercial_training" not in decision.allowed_actions
    dataset = next(
        component
        for component in manifest["components"]
        if component["component_type"] == "dataset"
    )
    assert dataset["rights"]["license_id"] == "Apache-2.0"
    assert dataset["rights"]["commercial_use"] is True
    assert "not the dataset owner" in dataset["rights"]["notes"]
