from __future__ import annotations

import json
from pathlib import Path

import pytest


pytest.importorskip("reportlab")
Image = pytest.importorskip("PIL.Image")
PdfReader = pytest.importorskip("pypdf").PdfReader
A4 = pytest.importorskip("reportlab.lib.pagesizes").A4

from droneai.integrity import sha256_file
from droneai.round1_share_report import build_round1_share_pdf
from scripts.build_round1_share_report import main


COMPARISON_PATH = Path("results/round1-cctv-comparison-34ad450/comparison.json")
CONTENT_PATH = Path("configs/reporting/round1_model_analysis_ko.json")
MODELS = ("dm-count", "steerer", "pet", "mpcount", "apgcc", "csrnet")


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


def test_share_report_is_exact_nine_page_a4_portrait(tmp_path: Path) -> None:
    manifest = make_assets(tmp_path)
    output = build_round1_share_pdf(
        COMPARISON_PATH,
        manifest,
        CONTENT_PATH,
        tmp_path / "share.pdf",
        report_date="2026-07-18",
    )

    reader = PdfReader(output)
    assert len(reader.pages) == 9
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
    assert result["pages"] == 9
    assert result["sha256"] == sha256_file(output)
