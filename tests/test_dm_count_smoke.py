from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

from droneai.dm_count_smoke import (
    build_protocol,
    load_smoke_config,
    prepare_smoke_samples,
    validate_official_split_paths,
    validate_rights_decision,
    write_split_source_manifest,
)
from droneai.integrity import sha256_file


CONFIG = Path(__file__).parents[1] / "configs" / "evaluation" / "dm_count_ucf_qnrf_smoke.json"


def _write_sample(root: Path, index: int, count: int) -> None:
    image = root / f"img_{index:04d}.jpg"
    Image.new("RGB", (80, 64), color=(index % 255, 30, 40)).save(image)
    points = np.asarray(
        [[float(point % 80), float((point * 3) % 64)] for point in range(count)],
        dtype=np.float32,
    ).reshape((-1, 2))
    savemat(root / f"img_{index:04d}_ann.mat", {"annPoints": points})


def _dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    train_root = tmp_path / "Train"
    test_root = tmp_path / "Test"
    train_root.mkdir()
    test_root.mkdir()
    counts = (100, 700, 1700)
    for index in range(1, 40):
        _write_sample(train_root, index, counts[(index - 1) // 13])
    (test_root / "must-not-be-read.jpg").write_bytes(b"not an image")
    train_list = tmp_path / "qnrf_train.txt"
    validation_list = tmp_path / "qnrf_val.txt"
    train_list.write_text(" ".join(f"img_{index:04d}.jpg" for index in range(1, 4)), encoding="utf-8")
    validation_list.write_text(
        "\n".join(f"img_{index:04d}.jpg" for index in range(4, 40)),
        encoding="utf-8",
    )
    return train_root, train_list, validation_list


def _rights(root: Path, *, allowed: bool = True) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "candidate.json"
    manifest = {
        "schema_version": 1,
        "candidate_id": "dm-count-official-ucf-qnrf-apache",
        "intended_use": "commercial_product_rnd",
        "components": [
            {
                "component_type": kind,
                "component_id": f"fixture-{kind}",
            }
            for kind in (
                "code",
                "dataset",
                "pretrained_weights",
                "derived_weights",
                "deployment",
            )
        ],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    import hashlib

    decision_path = root / "rights.json"
    decision_path.write_text(
        json.dumps(
            {
                "status": "PASS_COMMERCIAL_CANDIDATE",
                "candidate_id": manifest["candidate_id"],
                "manifest_semantic_sha256": hashlib.sha256(canonical).hexdigest(),
                "component_ids": {
                    component["component_type"]: component["component_id"]
                    for component in manifest["components"]
                },
                "allowed_actions": [
                    "catalog",
                    "frozen_checkpoint_evaluation" if allowed else "license_due_diligence",
                ],
            }
        ),
        encoding="utf-8",
    )
    return decision_path, manifest_path


def test_smoke_config_freezes_official_non_test_protocol() -> None:
    config = load_smoke_config(CONFIG)

    assert config["expected_samples"] == 36
    assert config["split_role"] == "validation"
    assert config["checkpoint_training_split_status"] == "VERIFIED_DISJOINT"
    assert config["comparison_scope"] == "compatibility_smoke"
    assert config["sealed_test_access_approved"] is False
    assert config["upstream_commit"] == "cc5f2132e0d1328909f31b6d665b8e0b15c30467"
    assert config["required_action"] == "frozen_checkpoint_evaluation"
    assert config["source_train_images"] == 1201
    assert config["official_train_samples"] == 1081
    assert config["official_validation_samples"] == 120
    assert config["train_list_sha256"] == "e8a926b35e55f9e7bf38498ec722d9a7f051169f03dd994024a9e5c757694cb0"
    assert config["validation_list_sha256"] == "835f660ed3373118f4d76a36ebf013ef28d90fcbcc835ddb5c180a2b26524126"


def test_rights_decision_must_authorize_frozen_checkpoint_evaluation(tmp_path: Path) -> None:
    blocked_decision, blocked_manifest = _rights(tmp_path / "blocked", allowed=False)
    with pytest.raises(PermissionError, match="frozen_checkpoint_evaluation"):
        validate_rights_decision(
            blocked_decision,
            manifest_path=blocked_manifest,
            expected_candidate_id="dm-count-official-ucf-qnrf-apache",
        )

    allowed_decision, allowed_manifest = _rights(tmp_path / "allowed")
    decision = validate_rights_decision(
        allowed_decision,
        manifest_path=allowed_manifest,
        expected_candidate_id="dm-count-official-ucf-qnrf-apache",
    )
    assert decision["status"] == "PASS_COMMERCIAL_CANDIDATE"


def test_prepare_smoke_uses_only_official_validation_and_writes_hashes(tmp_path: Path) -> None:
    train_root, train_list, validation_list = _dataset(tmp_path)
    config = dict(load_smoke_config(CONFIG))
    config.update(
        {
            "source_train_images": 39,
            "official_train_samples": 3,
            "official_validation_samples": 36,
            "train_list_sha256": sha256_file(train_list),
            "validation_list_sha256": sha256_file(validation_list),
        }
    )

    prepared = prepare_smoke_samples(
        config=config,
        train_root=train_root,
        train_list_path=train_list,
        validation_list_path=validation_list,
    )
    manifest_path = write_split_source_manifest(
        tmp_path / "split-source-manifest.json",
        prepared=prepared,
        train_list_path=train_list,
        validation_list_path=validation_list,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(prepared.samples) == 36
    assert {sample.condition_tags["source_partition"] for sample in prepared.samples} == {
        "official_validation"
    }
    assert {sample.condition_tags["density_band"] for sample in prepared.samples} == {
        "low",
        "medium",
        "high",
    }
    assert prepared.train_count == 3
    assert prepared.validation_count == 36
    assert manifest["test_root_argument_exposed"] is False
    assert manifest["test_data_access"] == "not_possible_through_adapter_api"
    assert len(manifest["train_list_sha256"]) == 64
    assert len(manifest["validation_list_sha256"]) == 64
    assert all(len(row["annotation_sha256"]) == 64 for row in manifest["selected_samples"])
    assert all("Test" not in row["image_path"] for row in manifest["selected_samples"])


def test_protocol_references_the_exact_rights_decision(tmp_path: Path) -> None:
    config = load_smoke_config(CONFIG)
    rights_path, manifest_path = _rights(tmp_path)

    protocol = build_protocol(
        config,
        rights_decision_path=rights_path,
        rights_manifest_path=manifest_path,
        split_verified=True,
    )

    assert protocol.split_role == "validation"
    assert protocol.checkpoint_training_split_status == "VERIFIED_DISJOINT"
    assert protocol.comparison_scope == "compatibility_smoke"
    assert protocol.expected_samples == 36
    assert protocol.sealed_test_access_approved is False
    assert len(protocol.rights_decision_sha256) == 64


def test_rights_decision_rejects_an_unrelated_candidate_manifest(tmp_path: Path) -> None:
    decision_path, manifest_path = _rights(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate_id"] = "unrelated-candidate"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PermissionError, match="candidate or manifest"):
        validate_rights_decision(
            decision_path,
            manifest_path=manifest_path,
            expected_candidate_id="dm-count-official-ucf-qnrf-apache",
        )


def test_prepare_smoke_rejects_modified_official_split_lists(tmp_path: Path) -> None:
    train_root, train_list, validation_list = _dataset(tmp_path)
    config = dict(load_smoke_config(CONFIG))
    config.update(
        {
            "source_train_images": 39,
            "official_train_samples": 3,
            "official_validation_samples": 36,
            "train_list_sha256": sha256_file(train_list),
            "validation_list_sha256": sha256_file(validation_list),
        }
    )
    validation_list.write_text(
        validation_list.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation split-list SHA-256"):
        prepare_smoke_samples(
            config=config,
            train_root=train_root,
            train_list_path=train_list,
            validation_list_path=validation_list,
        )


def test_split_paths_must_be_the_pinned_upstream_files(tmp_path: Path) -> None:
    upstream = tmp_path / "DM-Count"
    preprocess = upstream / "preprocess"
    preprocess.mkdir(parents=True)
    official_train = preprocess / "qnrf_train.txt"
    official_validation = preprocess / "qnrf_val.txt"
    official_train.write_text("train", encoding="utf-8")
    official_validation.write_text("validation", encoding="utf-8")

    validate_official_split_paths(
        upstream_dir=upstream,
        train_list_path=official_train,
        validation_list_path=official_validation,
    )
    with pytest.raises(ValueError, match="pinned upstream preprocess files"):
        validate_official_split_paths(
            upstream_dir=upstream,
            train_list_path=tmp_path / "other-train.txt",
            validation_list_path=official_validation,
        )
