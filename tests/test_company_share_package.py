from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from droneai.company_share_package import (
    CompanyShareSpec,
    build_company_share,
)
from droneai.integrity import sha256_file


MODELS = ("STEERER", "PET", "APGCC", "MPCount", "DMCount", "CSRNet")
ROUND2_IDS = {
    "STEERER": "steerer",
    "PET": "pet",
    "APGCC": "apgcc",
}
ROUND1_IDS = {
    "MPCount": "mpcount",
    "DMCount": "dm-count",
    "CSRNet": "csrnet",
}


def _fixture_spec(tmp_path: Path) -> CompanyShareSpec:
    full = tmp_path / "full"
    round2 = tmp_path / "round2"
    round1_roots: dict[str, Path] = {}
    code_files: dict[str, tuple[Path, ...]] = {}
    expected: dict[str, dict[str, int]] = {}
    for display, model_id in ROUND2_IDS.items():
        panel = full / model_id / "ucf-qnrf-kaggle-apache" / "panels"
        panel.mkdir(parents=True)
        (panel / f"{model_id}.png").write_bytes(f"{model_id}-panel".encode())
        lane = round2 / model_id / "ucf-qnrf-kaggle-apache"
        lane.mkdir(parents=True)
        (lane / "metrics.json").write_text("{}\n", encoding="utf-8")
        (lane / "predictions.csv").write_text(
            "sample_id,predicted_count\nsample,1\n",
            encoding="utf-8",
        )
        (lane / "score.md").write_text("PASS_RESEARCH_ONLY\n", encoding="utf-8")
        (full / model_id / "ucf-qnrf-kaggle-apache" / "export-summary.json").write_text(
            json.dumps({"panel_count": 1}),
            encoding="utf-8",
        )
        expected[model_id] = {"ucf-qnrf-kaggle-apache": 1}
    for display, model_id in ROUND1_IDS.items():
        root = tmp_path / "round1" / model_id
        root.mkdir(parents=True)
        (root / "README.md").write_text("36 image smoke\n", encoding="utf-8")
        (root / "selected.png").write_bytes(b"round1-panel")
        round1_roots[display] = root
    for display in MODELS:
        source = tmp_path / "code" / display / f"{display.lower()}_adapter.py"
        source.parent.mkdir(parents=True)
        source.write_text("# integration\n", encoding="utf-8")
        code_files[display] = (source,)
    report = tmp_path / "report.pdf"
    report.write_bytes(b"%PDF-report")
    compact = tmp_path / "comparison.md"
    compact.write_text("comparison\n", encoding="utf-8")
    return CompanyShareSpec(
        staging_root=tmp_path / "staging",
        zip_path=tmp_path / "DroneAI_회사공유용_v1.zip",
        report_pdf=report,
        full_panels_root=full,
        round2_results_root=round2,
        round2_expected=expected,
        round1_result_roots=round1_roots,
        model_code_files=code_files,
        interpretation_files=(compact,),
    )


def test_build_company_share_creates_seven_folders_and_utf8_zip(
    tmp_path: Path,
) -> None:
    spec = _fixture_spec(tmp_path)

    report = build_company_share(spec)

    share = spec.staging_root / "Drone AI"
    assert {path.name for path in share.iterdir() if path.is_dir()} == {
        "결과해석",
        *MODELS,
    }
    assert report.round2_panel_count == 3
    assert report.scope == "PASS_RESEARCH_ONLY"
    assert report.zip_path == spec.zip_path
    assert report.zip_sha256 == sha256_file(spec.zip_path)
    assert report.sidecar_path.read_text(encoding="utf-8").startswith(
        report.zip_sha256
    )
    with zipfile.ZipFile(spec.zip_path) as archive:
        names = set(archive.namelist())
        assert any("결과해석/00_먼저읽기.md" in name for name in names)
        assert any("STEERER" in name and name.endswith(".png") for name in names)
        assert all(not name.startswith("/") and "../" not in name for name in names)


def test_content_manifest_covers_every_non_manifest_file(tmp_path: Path) -> None:
    spec = _fixture_spec(tmp_path)

    build_company_share(spec)

    share = spec.staging_root / "Drone AI"
    validation = share / "결과해석" / "패키지_검증정보"
    excluded = {
        (validation / "files.sha256").resolve(),
        (validation / "package-manifest.json").resolve(),
    }
    content_files = {
        path.resolve()
        for path in share.rglob("*")
        if path.is_file() and path.resolve() not in excluded
    }
    rows = (validation / "files.sha256").read_text(encoding="utf-8").splitlines()
    manifest_paths = {
        (share / row.split("  ", 1)[1]).resolve()
        for row in rows
    }

    assert manifest_paths == content_files
    for row in rows:
        digest, relative = row.split("  ", 1)
        assert sha256_file(share / relative) == digest


def test_package_rejects_missing_round2_panel(tmp_path: Path) -> None:
    spec = _fixture_spec(tmp_path)
    spec.round2_expected["steerer"]["ucf-qnrf-kaggle-apache"] = 2

    with pytest.raises(ValueError, match="panel count"):
        build_company_share(spec)


def test_package_rejects_forbidden_model_code_suffix(tmp_path: Path) -> None:
    spec = _fixture_spec(tmp_path)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weight")
    spec.model_code_files["STEERER"] = (checkpoint,)

    with pytest.raises(ValueError, match="forbidden"):
        build_company_share(spec)


def test_package_rejects_preexisting_staging_or_zip(tmp_path: Path) -> None:
    spec = _fixture_spec(tmp_path)
    spec.staging_root.mkdir()

    with pytest.raises(FileExistsError, match="staging"):
        build_company_share(spec)

    spec = _fixture_spec(tmp_path / "other")
    spec.zip_path.parent.mkdir(parents=True, exist_ok=True)
    spec.zip_path.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="ZIP"):
        build_company_share(spec)


def test_package_rejects_symlink_source(tmp_path: Path) -> None:
    spec = _fixture_spec(tmp_path)
    target = tmp_path / "target.py"
    target.write_text("content\n", encoding="utf-8")
    link = tmp_path / "link.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    spec.model_code_files["STEERER"] = (link,)

    with pytest.raises(ValueError, match="link"):
        build_company_share(spec)
