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
    assert profile.imagenet_backbone.filename == "hrnetv2_w48_imagenet_pretrained.pth"
    assert profile.imagenet_backbone.source_url == (
        "https://github.com/hsfzxjy/models.storage/releases/download/"
        "openseg-pytorch-pretrained/hrnetv2_w48_imagenet_pretrained.pth"
    )
    assert profile.imagenet_backbone.provenance_url == (
        "https://onedrive.live.com/?action=locate&authkey=%21AKvqI6pBZlifgJk&"
        "cid=F7FD0B7F26543CEB&id=F7FD0B7F26543CEB%21116&"
        "parId=F7FD0B7F26543CEB%21105"
    )
    assert profile.imagenet_backbone.sha256 == (
        "0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8"
    )
    assert profile.imagenet_backbone.byte_size == 310643500


def test_profile_rejects_model_checkpoint_loading() -> None:
    payload = json.loads(PROFILE.read_text())
    payload["initialization"]["model_checkpoint"] = "/weights/QNRF_mae_78.4.pth"
    with pytest.raises(PermissionError, match="model checkpoint loading is forbidden"):
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


@pytest.mark.parametrize(
    "backbone",
    [
        ArtifactReference(
            filename="hrnetv2_w48_imagenet_pretrained.pth",
            source_url="https://github.com/hsfzxjy/models.storage/releases/download/openseg-pytorch-pretrained/hrnetv2_w48_imagenet_pretrained.pth",
            provenance_url="https://onedrive.live.com/?action=locate&authkey=%21AKvqI6pBZlifgJk&cid=F7FD0B7F26543CEB&id=F7FD0B7F26543CEB%21116&parId=F7FD0B7F26543CEB%21105",
            sha256="invalid",
            byte_size=310643500,
        ),
        ArtifactReference(
            filename="hrnetv2_w48_imagenet_pretrained.pth",
            source_url="https://example.test/hrnetv2_w48_imagenet_pretrained.pth",
            provenance_url="https://onedrive.live.com/?action=locate&authkey=%21AKvqI6pBZlifgJk&cid=F7FD0B7F26543CEB&id=F7FD0B7F26543CEB%21116&parId=F7FD0B7F26543CEB%21105",
            sha256="0efec102d97f2ef58f0e258b2c3076b3704b93ffc2b73f64c8da5462c0037ef8",
            byte_size=310643500,
        ),
        ArtifactReference(
            filename="hrnetv2_w48_imagenet_pretrained.pth",
            source_url="https://github.com/hsfzxjy/models.storage/releases/download/openseg-pytorch-pretrained/hrnetv2_w48_imagenet_pretrained.pth",
            provenance_url="https://onedrive.live.com/?action=locate&authkey=%21AKvqI6pBZlifgJk&cid=F7FD0B7F26543CEB&id=F7FD0B7F26543CEB%21116&parId=F7FD0B7F26543CEB%21105",
            sha256="f" * 64,
            byte_size=310643500,
        ),
    ],
)
def test_initialization_rejects_non_pinned_backbone(backbone: ArtifactReference) -> None:

    with pytest.raises(PermissionError, match="ImageNet backbone"):
        validate_initialization(backbone=backbone, model_checkpoint=None)
