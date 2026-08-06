import json
from pathlib import Path

import pytest

from droneai.steerer_training_profile import (
    ArtifactReference,
    load_training_profile,
    validate_initialization,
    validate_training_profile,
)


PROFILE = Path("configs/training/steerer_ucf_qnrf_imagenet.home5090.json")


def test_profile_freezes_approved_training_contract() -> None:
    profile = load_training_profile(PROFILE)
    assert profile.model_upstream_commit == "5b1854dbc2d280f2326d67c65515d8baf9083810"
    assert profile.dataset_population == 1201
    assert (profile.train_count, profile.validation_count) == (961, 240)
    assert profile.seed == 3035
    assert profile.initialization == "imagenet_backbone_only"
    assert profile.sealed_test_access is False
    assert profile.success_scope == "PASS_COMMERCIAL_CANDIDATE"


def test_profile_rejects_official_steerer_checkpoint() -> None:
    payload = json.loads(PROFILE.read_text())
    payload["initialization"]["model_checkpoint"] = "/weights/QNRF_mae_78.4.pth"
    with pytest.raises(PermissionError, match="official STEERER checkpoint"):
        validate_training_profile(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"unexpected": True}), "unknown key"),
        (lambda payload: payload.update({"dataset_population": 1200}), "population"),
        (lambda payload: payload["split"].update({"train": 960}), "split"),
        (lambda payload: payload["rights"].update({"success_scope": "PASS"}), "success scope"),
        (lambda payload: payload["storage"].update({"result_root": "/tmp/results"}), "/workspace/data"),
    ],
)
def test_profile_rejects_policy_boundary_mutations(mutation, message: str) -> None:
    payload = json.loads(PROFILE.read_text())
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        validate_training_profile(payload)


def test_initialization_requires_https_source_and_sha256() -> None:
    backbone = ArtifactReference(source_url="http://example.test/backbone", sha256="x" * 64)

    with pytest.raises(PermissionError, match="ImageNet backbone"):
        validate_initialization(backbone=backbone, model_checkpoint=None)
