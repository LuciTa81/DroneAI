from pathlib import Path

import pytest

from droneai.scoring import score_stage
from droneai.stage3b import build_clean_split, build_stage3b_checks, write_split_manifest
from scripts.run_stage3b_clean_smoke import build_parser


def test_clean_smoke_parser_requires_explicit_upstream_dir() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--data-dir",
                "/data/part_A",
                "--run-dir",
                "/results/seed-2026",
                "--work-dir",
                "/work/seed-2026",
            ]
        )


def _dataset(root: Path) -> Path:
    part_a = root / "ShanghaiTech" / "part_A"
    images = part_a / "train_data" / "images"
    ground_truth = part_a / "train_data" / "ground-truth"
    images.mkdir(parents=True)
    ground_truth.mkdir(parents=True)
    for index in range(1, 301):
        (images / f"IMG_{index}.jpg").write_bytes(f"image-{index}".encode())
        (ground_truth / f"GT_IMG_{index}.mat").write_bytes(f"gt-{index}".encode())
    return part_a


def _passing_evidence() -> dict:
    return {
        "split": {
            "seed": 2026,
            "counts": {"total": 300, "train": 240, "validation": 60},
            "overlap": 0,
            "split_hash": "a" * 64,
            "repeat_hash": "a" * 64,
        },
        "test_isolation": {
            "validation_source": "training_partition_only",
            "test_paths_seen": 0,
            "test_evaluations": 0,
            "split_frozen_before_smoke": True,
        },
        "smoke": {
            "epochs_completed": 1,
            "validation_samples": 60,
            "validation_mae": 250.0,
            "validation_rmse": 350.0,
            "checkpoint_exists": True,
            "best_model_exists": True,
        },
        "traceability": {
            "upstream_commit": "cc5f2132e0d1328909f31b6d665b8e0b15c30467",
            "gpu": "Tesla T4",
            "checkpoint_sha256": "b" * 64,
            "config_snapshot": "clean_config.json",
        },
        "artifacts": {
            "manifest_csv": "split_manifest.csv",
            "split_json": "split_manifest.json",
            "evidence_json": "evidence.snapshot.json",
            "score_json": "score.json",
            "score_md": "score.md",
        },
    }


def test_clean_split_is_stable_and_uses_only_train_data(tmp_path: Path) -> None:
    part_a = _dataset(tmp_path)
    first = build_clean_split(part_a, seed=2026)
    second = build_clean_split(part_a, seed=2026)
    assert first["counts"] == {"total": 300, "train": 240, "validation": 60}
    assert first["split_hash"] == second["split_hash"]
    assert {row["source_partition"] for row in first["records"]} == {"train_data"}
    assert all("test_data" not in row["image"] for row in first["records"])
    assert len({row["image"] for row in first["records"]}) == 300


def test_clean_split_changes_with_seed(tmp_path: Path) -> None:
    part_a = _dataset(tmp_path)
    assert build_clean_split(part_a, seed=2026)["split_hash"] != build_clean_split(
        part_a, seed=2027
    )["split_hash"]


def test_manifest_has_header_and_300_records(tmp_path: Path) -> None:
    split = build_clean_split(_dataset(tmp_path), seed=2026)
    output = tmp_path / "split_manifest.csv"
    write_split_manifest(split, output)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 301
    assert output.read_text(encoding="utf-8").splitlines()[0].startswith("image,ground_truth,split")


def test_stage3b_gate_passes_complete_clean_smoke() -> None:
    checks = build_stage3b_checks(_passing_evidence())
    report = score_stage(
        stage_id="stage-3b", stage_name="clean smoke", threshold=90, checks=checks
    )
    assert sum(check.weight for check in checks) == 100
    assert report.status == "PASS"
    assert report.score == 100


def test_stage3b_gate_blocks_test_access() -> None:
    evidence = _passing_evidence()
    evidence["test_isolation"]["test_paths_seen"] = 1
    report = score_stage(
        stage_id="stage-3b",
        stage_name="clean smoke",
        threshold=90,
        checks=build_stage3b_checks(evidence),
    )
    assert report.status == "BLOCKED"
    assert "evaluation.test_isolation" in report.to_dict()["failed_blockers"]


def test_stage3b_gate_blocks_split_overlap() -> None:
    evidence = _passing_evidence()
    evidence["split"]["overlap"] = 1
    report = score_stage(
        stage_id="stage-3b",
        stage_name="clean smoke",
        threshold=90,
        checks=build_stage3b_checks(evidence),
    )
    assert report.status == "BLOCKED"
    assert "split.counts" in report.to_dict()["failed_blockers"]
