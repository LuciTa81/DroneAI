import json
from pathlib import Path

from droneai.stage3c import classify_rights


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_apgcc_rights_keep_silent_checkpoint_in_research_comparison_lane() -> None:
    manifest_path = REPO_ROOT / "configs/candidates/apgcc_shha_to_ucf_qnrf.candidate.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    components = {
        component["component_type"]: component for component in manifest["components"]
    }

    code = components["code"]
    assert code["component_id"] == "apgcc-official-e3e997bf"
    assert code["rights"] == {
        "status": "verified",
        "license_id": "MIT",
        "evidence_url": (
            "https://github.com/AaronCIH/APGCC/blob/"
            "e3e997bf592a70fd34233a432a974200d8c0c847/LICENSE"
        ),
        "commercial_use": True,
        "research_use": True,
        "basis": "explicit_terms",
    }

    checkpoint = components["pretrained_weights"]
    assert checkpoint["component_id"] == "apgcc-official-shha-checkpoint"
    assert checkpoint["source_url"] == (
        "https://drive.google.com/file/d/1pEvn5RrvmDqVJUDZ4c9-rCJcl2I7bRhu/"
        "view?usp=sharing"
    )
    assert checkpoint["rights"]["status"] == "unverified"
    assert checkpoint["rights"]["commercial_use"] is None
    assert checkpoint["rights"]["research_use"] is True
    assert checkpoint["rights"]["basis"] == "internal_approval"

    decision = classify_rights(manifest)
    assert decision.status == "PASS_COMMERCIAL_CANDIDATE"
    assert {
        "synthetic_compatibility_smoke",
        "research_asset_download",
        "research_checkpoint_evaluation",
    }.issubset(decision.allowed_actions)
    assert {
        "frozen_checkpoint_evaluation",
        "commercial_training",
        "deployment",
    }.isdisjoint(decision.allowed_actions)

