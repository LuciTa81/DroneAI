import hashlib
import json
from pathlib import Path

from droneai.scoring import score_stage
from droneai.stage1 import build_stage1_checks, run_stage1


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, list[dict]]:
    rows = []
    for index, split in enumerate(("train", "val", "test")):
        image = tmp_path / f"image-{index}.bin"
        annotation = tmp_path / f"annotation-{index}.json"
        image.write_bytes(f"image-{split}".encode())
        annotation.write_text(json.dumps({"points": [[1, 2], [7, 8]]}), encoding="utf-8")
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "split": split,
                "group_id": f"sequence-{index}",
                "image_path": image.name,
                "annotation_path": annotation.name,
                "image_sha256": _sha(image),
                "annotation_sha256": _sha(annotation),
                "width": 10,
                "height": 10,
                "point_count": 2,
                "condition_tags": {"altitude_band": "mid", "density_band": "low"},
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset_id": "fixture",
        "intended_use": "noncommercial_research",
        "inventory_path": "inventory.jsonl",
        "source": {"official_url": "https://example.test", "citation": "Fixture et al."},
        "rights": {
            "status": "verified",
            "evidence_url": "https://example.test/license",
            "allowed_uses": ["noncommercial_research"],
            "commercial_use": False,
            "restriction_basis": "explicit_terms",
        },
        "annotation": {
            "format": "normalized-point-json-v1",
            "point_semantics": "person_head",
            "coordinate_system": "zero_based_xy",
        },
        "split": {
            "unit": "sequence",
            "group_key": "sequence_id",
            "required_splits": ["train", "val", "test"],
        },
        "coverage": {"required_condition_tags": ["altitude_band", "density_band"]},
    }
    return manifest, rows


def test_stage1_passes_intact_legal_group_split_fixture(tmp_path: Path) -> None:
    manifest, rows = _fixture(tmp_path)
    checks = build_stage1_checks(manifest=manifest, inventory=rows, dataset_root=tmp_path)
    report = score_stage(
        stage_id="stage-1",
        stage_name="Dataset readiness and split integrity",
        threshold=85,
        checks=checks,
    )
    assert sum(check.weight for check in checks) == 100
    assert report.score == 100
    assert report.status == "PASS"


def test_stage1_blocks_unverified_rights(tmp_path: Path) -> None:
    manifest, rows = _fixture(tmp_path)
    manifest["rights"]["status"] = "unverified"
    report = score_stage(
        stage_id="stage-1",
        stage_name="Dataset readiness and split integrity",
        threshold=85,
        checks=build_stage1_checks(manifest=manifest, inventory=rows, dataset_root=tmp_path),
    )
    assert report.status == "BLOCKED"
    assert "access.rights" in report.to_dict()["failed_blockers"]


def test_stage1_blocks_group_leakage(tmp_path: Path) -> None:
    manifest, rows = _fixture(tmp_path)
    rows[1]["group_id"] = rows[0]["group_id"]
    report = score_stage(
        stage_id="stage-1",
        stage_name="Dataset readiness and split integrity",
        threshold=85,
        checks=build_stage1_checks(manifest=manifest, inventory=rows, dataset_root=tmp_path),
    )
    assert report.status == "BLOCKED"
    assert "split.groups" in report.to_dict()["failed_blockers"]


def test_run_stage1_persists_review_artifacts(tmp_path: Path) -> None:
    manifest, rows = _fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "inventory.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    output_dir = tmp_path / "run"
    report = run_stage1(
        manifest_path=manifest_path, dataset_root=tmp_path, output_dir=output_dir
    )
    assert report.status == "PASS_RESEARCH_ONLY"
    assert (output_dir / "score.json").is_file()
    assert (output_dir / "score.md").is_file()
    assert (output_dir / "manifest.snapshot.json").is_file()


def test_stage1_blocks_commercial_claim_when_commercial_use_is_false(tmp_path: Path) -> None:
    manifest, rows = _fixture(tmp_path)
    manifest["intended_use"] = "commercial_product_rnd"
    manifest["rights"]["allowed_uses"] = ["commercial_product_rnd"]
    checks = build_stage1_checks(manifest=manifest, inventory=rows, dataset_root=tmp_path)
    report = score_stage(
        stage_id="stage-1",
        stage_name="Dataset readiness and split integrity",
        threshold=85,
        checks=checks,
    )
    assert report.status == "BLOCKED"
    assert "access.intended_use" in report.to_dict()["failed_blockers"]


def test_stage1_commercial_claim_requires_explicit_verified_terms(tmp_path: Path) -> None:
    manifest, rows = _fixture(tmp_path)
    manifest["intended_use"] = "commercial_product_rnd"
    manifest["rights"].update(
        {
            "allowed_uses": ["commercial_product_rnd"],
            "commercial_use": True,
            "restriction_basis": "not_stated",
        }
    )
    report = score_stage(
        stage_id="stage-1",
        stage_name="Dataset readiness and split integrity",
        threshold=85,
        checks=build_stage1_checks(manifest=manifest, inventory=rows, dataset_root=tmp_path),
    )
    assert report.status == "BLOCKED"

    manifest["rights"].update(
        {"license_id": "COMMERCIAL-TERMS", "restriction_basis": "explicit_terms"}
    )
    approved_checks = build_stage1_checks(
        manifest=manifest, inventory=rows, dataset_root=tmp_path
    )
    assert next(c for c in approved_checks if c.check_id == "access.intended_use").passed


def test_stage1_accepts_independent_image_as_split_unit(tmp_path: Path) -> None:
    manifest, rows = _fixture(tmp_path)
    manifest["split"]["unit"] = "image"
    checks = build_stage1_checks(
        manifest=manifest,
        inventory=rows,
        dataset_root=tmp_path,
    )

    assert next(check for check in checks if check.check_id == "split.unit").passed
