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
    build_round1_share_pdf,
    draw_report_table,
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


def test_report_page_contract_splits_dense_content() -> None:
    pages = report_page_contract()
    assert len(pages) == 12
    assert pages[3] == ("steerer", "dm-count", "pet")
    assert pages[4] == ("mpcount", "apgcc", "csrnet")
    assert pages[6] == ("dm-count", "steerer")
    assert pages[7] == ("mpcount", "csrnet")
    assert pages[8] == ("steerer", "pet")
    assert pages[9] == ("apgcc",)


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


def _panel(
    model_id: str,
    role: str,
    path: str,
    digest: str,
    *,
    sample_id: str,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "role": role,
        "sample_id": sample_id,
        "category": "failure" if role == "limitation" else "best",
        "density_band": "low",
        "ground_truth_count": 254.0,
        "predicted_count": 250.0,
        "normalized_error": 0.0157,
        "spatial_metric_name": "game_l1",
        "spatial_metric_value": 5.0,
        "source_path": f"{model_id}/run/{path}",
        "source_sha256": digest,
        "packaged_path": path,
    }


def make_assets(tmp_path: Path) -> Path:
    root = tmp_path / "assets"
    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    assets: list[dict[str, str]] = []
    models: dict[str, object] = {}
    records: dict[tuple[str, str], dict[str, object]] = {}
    for model_index, model_id in enumerate(MODELS):
        model_records = {}
        for role_index, role in enumerate(("lower_error", "limitation")):
            relative = f"images/{model_id}--{role}.png"
            target = root / relative
            image = Image.new(
                "RGB",
                (1200, 260),
                (28 + model_index * 22, 70 + role_index * 60, 120),
            )
            image.save(target)
            digest = sha256_file(target)
            assets.append(
                {
                    "packaged_path": relative,
                    "packaged_sha256": digest,
                    "source_path": f"{model_id}/run/{relative}",
                    "source_sha256": digest,
                }
            )
            record = _panel(
                model_id,
                role,
                relative,
                digest,
                sample_id=f"{model_id}_{role}",
            )
            model_records[role] = record
            records[(model_id, role)] = record
        models[model_id] = {
            **model_records,
            "failure_candidates": [model_records["limitation"]],
        }

    density = {
        model_id: {
            **records[(model_id, "lower_error")],
            "role": "density_comparison",
            "sample_id": "img_0097",
        }
        for model_id in ("dm-count", "steerer", "mpcount", "csrnet")
    }
    points = {
        model_id: {
            **records[(model_id, "limitation")],
            "role": "points_comparison",
            "sample_id": "img_0062",
        }
        for model_id in ("steerer", "pet", "apgcc")
    }
    manifest = root / "assets-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "comparison_sha256": sha256_file(COMPARISON_PATH),
                "models": models,
                "comparisons": {
                    "density_sample_id": "img_0097",
                    "density": density,
                    "points_sample_id": "img_0062",
                    "points": points,
                },
                "assets": assets,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest


def test_share_report_is_exact_twelve_page_a4_portrait(tmp_path: Path) -> None:
    manifest = make_assets(tmp_path)
    output = build_round1_share_pdf(
        COMPARISON_PATH,
        manifest,
        CONTENT_PATH,
        tmp_path / "share.pdf",
        report_date="2026-07-18",
    )

    reader = PdfReader(output)
    assert len(reader.pages) == 12
    for page in reader.pages:
        assert float(page.mediabox.height) > float(page.mediabox.width)
        assert abs(float(page.mediabox.width) - A4[0]) < 1
        assert abs(float(page.mediabox.height) - A4[1]) < 1


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
    target = manifest.parent / payload["assets"][0]["packaged_path"]
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
    assert result["pages"] == 12
    assert result["sha256"] == sha256_file(output)
