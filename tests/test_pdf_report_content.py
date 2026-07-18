from __future__ import annotations

import json
from pathlib import Path

import pytest

from droneai.pdf_report_content import load_report_content


CONTENT_PATH = Path("configs/reporting/round1_model_analysis_ko.json")


def test_content_covers_exact_six_models() -> None:
    content = load_report_content(CONTENT_PATH)

    assert tuple(model.model_id for model in content.models) == (
        "dm-count",
        "steerer",
        "pet",
        "mpcount",
        "apgcc",
        "csrnet",
    )
    assert all(len(model.strengths) >= 2 for model in content.models)
    assert all(len(model.limitations) >= 2 for model in content.models)


def test_content_keeps_pet_research_only_and_avoids_production_claims() -> None:
    content = load_report_content(CONTENT_PATH)
    payload = CONTENT_PATH.read_text(encoding="utf-8")
    pet = next(model for model in content.models if model.model_id == "pet")

    assert "연구 비교군" in pet.cctv_interpretation
    assert "배포 승인 완료" not in payload
    assert "공식 모델 순위" not in payload
    assert "PRODUCTION_APPROVED" not in payload


def test_content_contains_decision_and_prefinetuning_roadmap() -> None:
    content = load_report_content(CONTENT_PATH)

    assert any("STEERER" in item and "DM-Count" in item for item in content.conclusion)
    assert any("PET" in item and "연구" in item for item in content.conclusion)
    assert content.roadmap[-1].startswith("권리 적격 최종 1개 모델")


def test_loader_rejects_production_approval_claim(tmp_path: Path) -> None:
    payload = json.loads(CONTENT_PATH.read_text(encoding="utf-8"))
    payload["conclusion"].append("PRODUCTION_APPROVED")
    target = tmp_path / "invalid.json"
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="production approval"):
        load_report_content(target)
