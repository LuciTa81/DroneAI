from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest


pytest.importorskip("reportlab")
Image = pytest.importorskip("PIL.Image")
PdfReader = pytest.importorskip("pypdf").PdfReader
A4 = pytest.importorskip("reportlab.lib.pagesizes").A4
canvas_module = pytest.importorskip("reportlab.pdfgen.canvas")

from droneai.integrity import sha256_file
from droneai.round1_pdf_report import DISPLAY_NAMES, register_korean_fonts
from droneai.round1_share_report import (
    MODEL_PAGE_ORDER,
    build_round1_share_pdf,
    draw_report_table,
    model_page_asset_contract,
    model_output_note,
    model_table_height_contract,
    readability_contract,
    report_page_contract,
)
from scripts.build_round1_share_report import main


COMPARISON_PATH = Path("results/round1-cctv-comparison-34ad450/comparison.json")
CONTENT_PATH = Path("configs/reporting/round1_model_analysis_ko.json")
MODELS = ("dm-count", "steerer", "pet", "mpcount", "apgcc", "csrnet")


def test_readability_contract_has_no_small_report_text() -> None:
    contract = readability_contract()
    assert contract["body_size"] >= 10.5
    assert contract["body_leading"] >= 15
    assert contract["table_header_size"] >= 8.5
    assert contract["table_body_size"] >= 8.5
    assert contract["caption_size"] >= 8.5
    assert contract["caption_leading"] >= 11
    assert contract["header_footer_size"] >= 8


def test_report_page_contract_has_two_scenarios_per_model() -> None:
    expected = [("cover",), ("summary",), ("evaluation",)]
    for model_id in MODEL_PAGE_ORDER:
        expected.extend(((model_id, "moderate"), (model_id, "high_density")))
    expected.extend((("performance",), ("rights",), ("conclusion",)))
    assert report_page_contract() == tuple(expected)


def test_model_page_asset_contract_covers_every_model() -> None:
    assert model_page_asset_contract() == {
        model_id: ("moderate", "high_density") for model_id in MODEL_PAGE_ORDER
    }


def test_model_table_uses_compact_header_and_readable_body() -> None:
    contract = model_table_height_contract()
    assert contract == {"header": 42.0, "body": 126.0}


def test_report_table_can_size_header_independently() -> None:
    buffer = BytesIO()
    pdf = canvas_module.Canvas(buffer, pagesize=A4)
    end_y = draw_report_table(
        pdf,
        [["Header"], ["Body 1"], ["Body 2"]],
        (120,),
        42,
        700,
        row_height=60,
        header_row_height=30,
        fonts=register_korean_fonts(),
    )
    assert end_y == 550


def test_every_model_has_a_concise_output_note() -> None:
    for model_id in MODELS:
        note = model_output_note(model_id)
        assert note.startswith(f"출력 해석({DISPLAY_NAMES[model_id]}):")
        assert len(note) <= 115


def make_assets(tmp_path: Path) -> Path:
    root = tmp_path / "assets"
    root.mkdir(parents=True)
    assets: list[dict[str, str]] = []
    scenarios: dict[str, object] = {}
    scenario_specs = {
        "moderate": ("img_0775", 195.0, "적정 인원 공통 장면"),
        "high_density": ("img_0221", 1762.0, "고밀도 공통 장면"),
    }
    for scenario_index, (scenario_id, (sample_id, gt, label)) in enumerate(
        scenario_specs.items()
    ):
        model_rows: dict[str, object] = {}
        for model_index, model_id in enumerate(MODEL_PAGE_ORDER):
            relative = f"panels/{model_id}/{scenario_id}.png"
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new(
                "RGB",
                (1200, 520),
                (28 + model_index * 22, 70 + scenario_index * 60, 120),
            )
            image.save(target)
            digest = sha256_file(target)
            assets.append({"path": relative, "sha256": digest})
            pred = gt + model_index + scenario_index + 1
            model_rows[model_id] = {
                "model_id": model_id,
                "sample_id": sample_id,
                "ground_truth_count": gt,
                "predicted_count": pred,
                "absolute_error": abs(pred - gt),
                "normalized_error": abs(pred - gt) / gt,
                "latency_ms": 10.0 + model_index,
                "peak_vram_mb": 1000.0 + model_index,
                "spatial_metric_name": "localization_f1"
                if model_id in {"pet", "apgcc"}
                else "game_l1",
                "spatial_metric_value": 0.75
                if model_id in {"pet", "apgcc"}
                else 12.0,
                "rights_scope": "PASS_RESEARCH_ONLY"
                if model_id == "pet"
                else "PASS_COMMERCIAL_CANDIDATE",
                "comparison_rights_scope": "PASS_RESEARCH_ONLY"
                if model_id == "pet"
                else "PASS_COMMERCIAL_CANDIDATE",
                "source_sha256": "a" * 64,
                "annotation_sha256": "b" * 64,
                "checkpoint_sha256": "c" * 64,
                "rights_decision_sha256": "d" * 64,
                "split_manifest_sha256": "e" * 64,
                "environment_sha256": "f" * 64,
                "panel_sha256": digest,
                "result_sha256": str(model_index) * 64,
                "packaged_panel_path": relative,
            }
        scenarios[scenario_id] = {
            "sample_id": sample_id,
            "ground_truth_count": gt,
            "label": label,
            "models": model_rows,
        }
    manifest = root / "assets-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "PASS_RESEARCH_ONLY",
                "comparison_sha256": sha256_file(COMPARISON_PATH),
                "model_order": list(MODEL_PAGE_ORDER),
                "scenario_order": ["moderate", "high_density"],
                "scenarios": scenarios,
                "assets": assets,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def test_share_report_is_exact_eighteen_page_a4_portrait(tmp_path: Path) -> None:
    manifest = make_assets(tmp_path)
    output = build_round1_share_pdf(
        COMPARISON_PATH,
        manifest,
        CONTENT_PATH,
        tmp_path / "share.pdf",
        report_date="2026-07-18",
    )

    reader = PdfReader(output)
    assert len(reader.pages) == 18
    for page in reader.pages:
        assert float(page.mediabox.height) > float(page.mediabox.width)
        assert abs(float(page.mediabox.width) - A4[0]) < 1
        assert abs(float(page.mediabox.height) - A4[1]) < 1


def test_each_model_scenario_page_has_common_image_metrics_and_no_5090_cover(
    tmp_path: Path,
) -> None:
    manifest = make_assets(tmp_path)
    output = build_round1_share_pdf(
        COMPARISON_PATH,
        manifest,
        CONTENT_PATH,
        tmp_path / "share.pdf",
        report_date="2026-07-19",
    )

    reader = PdfReader(output)
    assert "5090" not in (reader.pages[0].extract_text() or "")
    expected = [
        (model_id, scenario_id)
        for model_id in MODEL_PAGE_ORDER
        for scenario_id in ("moderate", "high_density")
    ]
    for page, (model_id, scenario_id) in zip(reader.pages[3:15], expected, strict=True):
        text = page.extract_text() or ""
        normalized = " ".join(text.split())
        sample_id = "img_0775" if scenario_id == "moderate" else "img_0221"
        assert DISPLAY_NAMES[model_id] in text
        assert sample_id in text
        assert " ".join(model_output_note(model_id).split()) in normalized
        assert "GT" in text
        assert "Pred" in text
        assert "절대오차" in text
        assert "오차율" in text
        assert "PASS_" in text
        assert "..." not in text
        assert len(page.images) >= 1
    all_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for forbidden in ("강점 사례", "best case", "winner"):
        assert forbidden not in all_text


def test_share_report_contains_required_claim_boundaries(tmp_path: Path) -> None:
    manifest = make_assets(tmp_path)
    output = build_round1_share_pdf(
        COMPARISON_PATH,
        manifest,
        CONTENT_PATH,
        tmp_path / "share.pdf",
        report_date="2026-07-18",
    )

    text = "\n".join(page.extract_text() or "" for page in PdfReader(output).pages)
    for token in (
        "STEERER",
        "DM-Count",
        "PET",
        "MPCount",
        "APGCC",
        "CSRNet",
        "UCF-QNRF",
        "validation 36장",
        "fine-tuning 없음",
        "cross-domain compatibility smoke",
        "PET는 연구 비교군",
        "공식 모델 순위가 아님",
    ):
        assert token in text


def test_share_report_contains_every_model_output_note(tmp_path: Path) -> None:
    manifest = make_assets(tmp_path)
    output = build_round1_share_pdf(
        COMPARISON_PATH,
        manifest,
        CONTENT_PATH,
        tmp_path / "share.pdf",
        report_date="2026-07-19",
    )

    text = "\n".join(page.extract_text() or "" for page in PdfReader(output).pages)
    normalized = " ".join(text.split())
    for model_id in MODELS:
        assert " ".join(model_output_note(model_id).split()) in normalized


def test_share_report_rejects_changed_asset_hash(tmp_path: Path) -> None:
    manifest = make_assets(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    target = manifest.parent / payload["assets"][0]["path"]
    target.write_bytes(b"changed")

    with pytest.raises(ValueError, match="asset hash mismatch"):
        build_round1_share_pdf(
            COMPARISON_PATH,
            manifest,
            CONTENT_PATH,
            tmp_path / "share.pdf",
            report_date="2026-07-18",
        )
    assert not (tmp_path / "share.pdf").exists()


def test_share_report_cli_returns_structured_result(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    manifest = make_assets(tmp_path)
    output = tmp_path / "share.pdf"
    code = main(
        [
            "--comparison",
            str(COMPARISON_PATH),
            "--assets-manifest",
            str(manifest),
            "--content",
            str(CONTENT_PATH),
            "--output",
            str(output),
            "--report-date",
            "2026-07-18",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["status"] == "PASS_RESEARCH_ONLY"
    assert result["pages"] == 18
    assert result["sha256"] == sha256_file(output)
