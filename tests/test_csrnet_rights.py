import json
from pathlib import Path

from droneai.stage3c import classify_rights


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs/models/csrnet.json"
RIGHTS_MANIFEST = (
    ROOT / "configs/candidates/csrnet_shha_to_ucf_qnrf.candidate.json"
)


def test_csrnet_reference_uses_a_real_upstream_commit() -> None:
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))

    assert config["reference_model_sha"] == (
        "ed29d895989c188cb913a9503721271c6cf1ab1f"
    )
    assert config["independent_implementation_commit"] == (
        "5c9dd5cabb91664767da93732c365b6f9940b0f2"
    )
    assert config["reference_license_status"] == "no_explicit_license_file_found"
    assert config["checkpoint"]["raw_sha256"] == (
        "68383c1053be371ad54b0061ab99fed08c2bfff39de73937f2a4388041ab0128"
    )
    assert config["checkpoint"]["safe_state_dict_sha256"] == (
        "7093f29f1469fb4e3a80781fda03665848689ada59c5a13d1b49940b230755e6"
    )
    assert config["checkpoint"]["training_dataset"] == "ShanghaiTech Part A"


def test_csrnet_rights_keep_the_silent_official_weight_research_only() -> None:
    assert RIGHTS_MANIFEST.is_file()
    manifest = json.loads(RIGHTS_MANIFEST.read_text(encoding="utf-8"))
    components = {
        component["component_type"]: component for component in manifest["components"]
    }

    code = components["code"]
    assert code["component_id"] == "csrnet-droneai-independent-5c9dd5ca"
    assert code["rights"]["status"] == "verified"
    assert code["rights"]["basis"] == "internal_approval"
    assert code["rights"]["commercial_use"] is True
    assert "no reference source" in code["rights"]["notes"]

    checkpoint = components["pretrained_weights"]
    assert checkpoint["component_id"] == "csrnet-official-shha-mae66.4"
    assert checkpoint["source_url"] == (
        "https://drive.google.com/open?id=1Z-atzS5Y2pOd-nEWqZRVBDMYJDreGWHH"
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
        "derived_weight_use",
        "deployment",
    }.isdisjoint(decision.allowed_actions)
