from __future__ import annotations

import json
from pathlib import Path

import pytest


pytest.importorskip("reportlab")
Image = pytest.importorskip("PIL.Image")
PdfReader = pytest.importorskip("pypdf").PdfReader

from droneai.integrity import sha256_file
from droneai.round1_pdf_report import (
    _comparison_layout,
    _evaluation_output_layout,
    _fit_text_size,
    build_round1_pdf,
)
from scripts.build_round1_model_analysis_pdf import main


COMPARISON_PATH = Path(
    "results/round1-cctv-comparison-34ad450/comparison.json"
)
CONTENT_PATH = Path("configs/reporting/round1_model_analysis_ko.json")
MODELS = ("dm-count", "steerer", "pet", "mpcount", "apgcc", "csrnet")


def test_comparison_layout_uses_large_fixed_order_cards() -> None:
    density = _comparison_layout("density")
    assert tuple(item[0] for item in density) == (
        "dm-count",
        "steerer",
        "mpcount",
        "csrnet",
    )
    assert len({item[1] for item in density}) == 2
    assert len({item[2] for item in density}) == 2
    assert all(item[3] >= 370 and item[4] >= 170 for item in density)

    points = _comparison_layout("points")
    assert tuple(item[0] for item in points) == ("steerer", "pet", "apgcc")
    assert all(item[3] >= 370 and item[4] >= 170 for item in points)
    assert points[-1][3] >= 770


def test_evaluation_page_keeps_long_values_and_rights_note_in_bounds() -> None:
    size = _fit_text_size(
        "low 12 / medium 12 / high 12",
        "Helvetica-Bold",
        156,
        preferred=14,
        minimum=9,
    )
    assert 9 <= size < 14

    row_positions, rights_y = _evaluation_output_layout()
    assert row_positions[-1] - 44 >= 102
    assert rights_y < 102


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
        "density_band": "high",
        "ground_truth_count": 1200.0,
        "predicted_count": 1100.0,
        "normalized_error": 0.0833,
        "spatial_metric_name": "game_l1",
        "spatial_metric_value": 90.0,
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
            "sample_id": "img_common_density",
        }
        for model_id in ("dm-count", "steerer", "mpcount", "csrnet")
    }
    points = {
        model_id: {
            **records[(model_id, "limitation")],
            "role": "points_comparison",
            "sample_id": "img_common_points",
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
                    "density_sample_id": "img_common_density",
                    "density": density,
                    "points_sample_id": "img_common_points",
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


def test_report_has_exact_pages_and_required_text(tmp_path: Path) -> None:
    manifest = make_assets(tmp_path)
    output = build_round1_pdf(
        COMPARISON_PATH,
        manifest,
        CONTENT_PATH,
        tmp_path / "report.pdf",
        report_date="2026-07-18",
    )

    reader = PdfReader(output)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert len(reader.pages) == 15
    for token in ("STEERER", "DM-Count", "PET", "MPCount", "APGCC", "CSRNet"):
        assert token in text
    assert "공식 모델 순위가 아님" in text
    assert "PROVISIONAL_TECHNICAL_SHORTLIST" in text
    assert "UCF-QNRF" in text


def test_report_rejects_changed_asset_hash(tmp_path: Path) -> None:
    manifest = make_assets(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    target = manifest.parent / payload["assets"][0]["packaged_path"]
    target.write_bytes(b"changed")

    with pytest.raises(ValueError, match="asset hash mismatch"):
        build_round1_pdf(
            COMPARISON_PATH,
            manifest,
            CONTENT_PATH,
            tmp_path / "report.pdf",
            report_date="2026-07-18",
        )
    assert not (tmp_path / "report.pdf").exists()


def test_cli_returns_structured_error_for_missing_assets(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    output = tmp_path / "report.pdf"
    code = main(
        [
            "--comparison",
            str(COMPARISON_PATH),
            "--assets-manifest",
            str(tmp_path / "missing.json"),
            "--content",
            str(CONTENT_PATH),
            "--output",
            str(output),
            "--report-date",
            "2026-07-18",
        ]
    )

    error = json.loads(capsys.readouterr().err)
    assert code == 2
    assert error["status"] == "FAIL"
    assert not output.exists()


def test_cli_success_reports_page_count_and_hash(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    manifest = make_assets(tmp_path)
    output = tmp_path / "report.pdf"
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
    assert result["pages"] == 15
    assert result["sha256"] == sha256_file(output)
