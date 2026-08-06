from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from droneai.integrity import sha256_file
from droneai.steerer_training_profile import load_training_profile
from droneai.steerer_training_upstream import (
    audit_upstream,
    initialize_steerer_model,
    synthesize_official_config,
)


PROFILE_PATH = Path("configs/training/steerer_ucf_qnrf_imagenet.home5090.json")


OFFICIAL_CONFIG = """
gpus = (0, 1,)
log_dir = 'exp'
workers = 6
print_freq = 20
seed = 3035
network = dict(
    backbone='MocHRBackbone', sub_arch='hrnet48', counter_type='withMOE',
    resolution_num=[0, 1, 2, 3], loss_weight=[1.0, 0.5, 0.25, 0.125],
    sigma=[4], gau_kernel_size=15, baseline_loss=False,
    pretrained_backbone='../PretrainedModels/hrnetv2_w48_imagenet_pretrained.pth',
    head=dict(type='CountingHead', fuse_method='cat', in_channels=96,
              stages_channel=[384, 192, 96, 48], inter_layer=[64, 32, 16],
              out_channels=1),
)
dataset = dict(
    name='QNRF', root='../ProcessedData/QNRF/', test_set='test.txt',
    train_set='train.txt', loc_gt='test_gt_loc.txt',
    num_classes=len(network['resolution_num']), den_factor=100,
    extra_train_set=None,
)
optimizer = dict(
    NAME='adamw', BASE_LR=1e-4, BETAS=(0.9, 0.999), WEIGHT_DECAY=1e-4,
    EPS=1.0e-08, MOMENTUM=0.9, AMSGRAD=False, NESTEROV=True,
)
lr_config = dict(
    NAME='cosine', WARMUP_METHOD='linear', DECAY_EPOCHS=250,
    DECAY_RATE=0.1, WARMUP_EPOCHS=10, WARMUP_LR=5.0e-07,
    MIN_LR=1.0e-07,
)
total_epochs = 210
train = dict(
    counter='normal', image_size=(768, 768), route_size=(256, 256),
    base_size=None, batch_size_per_gpu=8, shuffle=True, begin_epoch=0,
    end_epoch=800, extra_epoch=0, extra_lr=0, resume_path=None, flip=True,
    multi_scale=True, scale_factor=(0.5, 2.0),
    val_span=[-800, -600, -400, -200, -200, -100, -100],
    downsamplerate=1, ignore_label=255,
)
test = dict(
    image_size=(1024, 2048), base_size=3072, loc_base_size=3072,
    loc_threshold=0.15, batch_size_per_gpu=1, patch_batch_size=16,
    flip_test=False, multi_scale=False, model_file='./exp/official.pth',
)
CUDNN = dict(BENCHMARK=True, DETERMINISTIC=False, ENABLED=True)
"""


def _write_upstream(root: Path, config_source: str = OFFICIAL_CONFIG) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / "LICENSE").write_text("official license fixture\n", encoding="utf-8")
    (root / "configs" / "QNRF_final.py").write_text(config_source, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=DroneAI tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    return root


def _profile(tmp_path: Path):
    return replace(
        load_training_profile(PROFILE_PATH),
        checkpoint_root=tmp_path / "checkpoints",
        result_root=tmp_path / "results",
    )


def test_audit_records_clean_pinned_upstream(tmp_path: Path) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    audit = audit_upstream(
        upstream,
        expected_commit=commit,
        expected_license_sha256=sha256_file(upstream / "LICENSE"),
    )

    assert audit.commit == commit
    assert audit.clean is True
    assert audit.license_sha256 == sha256_file(upstream / "LICENSE")
    assert audit.config_sha256 == sha256_file(upstream / "configs" / "QNRF_final.py")


@pytest.mark.parametrize("defect", ["wrong-commit", "dirty"])
def test_audit_rejects_dirty_or_wrong_upstream(tmp_path: Path, defect: str) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=upstream, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if defect == "dirty":
        (upstream / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    else:
        commit = "a" * 40

    with pytest.raises(ValueError, match="pinned|clean"):
        audit_upstream(
            upstream,
            expected_commit=commit,
            expected_license_sha256=sha256_file(upstream / "LICENSE"),
        )


@pytest.mark.parametrize("stage", ["T0", "T1", "T5", "T50"])
def test_config_uses_validation_and_keeps_official_800_epoch_horizon(
    tmp_path: Path, stage: str
) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")
    profile = _profile(tmp_path)
    backbone = tmp_path / "hrnetv2_w48_imagenet_pretrained.pth"
    backbone.write_bytes(b"backbone")

    config = synthesize_official_config(
        profile=profile,
        upstream_dir=upstream,
        processed_root=tmp_path / "processed",
        backbone_path=backbone,
        stage=stage,
        physical_batch=4,
        accumulation_steps=2,
        run_id="run-a",
    )

    assert config["dataset"]["root"] == str(tmp_path / "processed")
    assert config["dataset"]["train_set"] == "train.txt"
    assert config["dataset"]["test_set"] == "val.txt"
    assert "test" not in config["dataset"]["test_set"].lower()
    assert config["network"]["pretrained_backbone"] == str(backbone)
    assert config["gpus"] == (0,)
    assert config["train"]["batch_size_per_gpu"] == 4
    assert config["train"]["end_epoch"] == 800
    assert config["log_dir"] == str(tmp_path / "results" / "run-a")
    assert config["droneai"]["stage_stop_epoch"] == {
        "T0": 0, "T1": 1, "T5": 5, "T50": 50
    }[stage]
    assert config["droneai"]["schedule_horizon_epochs"] == 800
    assert config["droneai"]["checkpoint_run_dir"] == str(
        tmp_path / "checkpoints" / "run-a"
    )


@pytest.mark.parametrize(
    "run_id",
    ["", ".", "..", "../escape", "nested/run", "nested\\run", "run:a", "run id"],
)
def test_config_rejects_unsafe_run_id(tmp_path: Path, run_id: str) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")

    with pytest.raises(ValueError, match="run_id"):
        synthesize_official_config(
            profile=_profile(tmp_path), upstream_dir=upstream,
            processed_root=tmp_path / "processed", backbone_path=tmp_path / "backbone.pth",
            stage="T1", physical_batch=8, accumulation_steps=1, run_id=run_id,
        )


@pytest.mark.parametrize(
    ("physical_batch", "accumulation_steps"),
    [("8", 1), (8, "1"), (2, 4), (8, 2)],
)
def test_config_rejects_non_integer_or_unapproved_batch_plan(
    tmp_path: Path,
    physical_batch: object,
    accumulation_steps: object,
) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")

    with pytest.raises(ValueError, match="batch|accumulation"):
        synthesize_official_config(
            profile=_profile(tmp_path), upstream_dir=upstream,
            processed_root=tmp_path / "processed", backbone_path=tmp_path / "backbone.pth",
            stage="T1", physical_batch=physical_batch,  # type: ignore[arg-type]
            accumulation_steps=accumulation_steps,  # type: ignore[arg-type]
            run_id="run-a",
        )


def test_config_rejects_resume_outside_current_run_lineage(tmp_path: Path) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")
    with pytest.raises(ValueError, match="resume.*run"):
        synthesize_official_config(
            profile=_profile(tmp_path), upstream_dir=upstream,
            processed_root=tmp_path / "processed", backbone_path=tmp_path / "backbone.pth",
            stage="T5", physical_batch=8, accumulation_steps=1, run_id="run-a",
            resume_path=tmp_path / "checkpoints" / "another-run" / "last.pth",
        )


def test_config_rejects_resume_for_t0_or_t1(tmp_path: Path) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")
    with pytest.raises(ValueError, match="T0|T1"):
        synthesize_official_config(
            profile=_profile(tmp_path), upstream_dir=upstream,
            processed_root=tmp_path / "processed", backbone_path=tmp_path / "backbone.pth",
            stage="T1", physical_batch=8, accumulation_steps=1, run_id="run-a",
            resume_path=tmp_path / "checkpoints" / "run-a" / "last.pth",
        )


def test_config_rejects_mutated_official_training_contract(tmp_path: Path) -> None:
    mutated = OFFICIAL_CONFIG.replace("end_epoch=800", "end_epoch=50")
    upstream = _write_upstream(tmp_path / "STEERER", mutated)

    with pytest.raises(ValueError, match="official.*end_epoch|end_epoch.*official"):
        synthesize_official_config(
            profile=_profile(tmp_path), upstream_dir=upstream,
            processed_root=tmp_path / "processed", backbone_path=tmp_path / "backbone.pth",
            stage="T50", physical_batch=8, accumulation_steps=1, run_id="run-a",
        )


class _FakeModel:
    def __init__(self, parameters: dict[str, np.ndarray], loaded_paths: tuple[Path, ...]):
        self._parameters = parameters
        self.loaded_weight_paths = loaded_paths

    def named_parameters(self):
        return tuple(self._parameters.items())


class _FakeModelFactory:
    def __init__(self, *, mutate_head: bool = False, extra_weight: Path | None = None):
        self.mutate_head = mutate_head
        self.extra_weight = extra_weight
        self.calls: list[Path | None] = []

    def __call__(self, *, pretrained_backbone: Path | None):
        self.calls.append(pretrained_backbone)
        parameters = {
            "backbone.conv.weight": np.random.standard_normal(4).astype(np.float32),
            "backbone.bn.weight": np.ones(2, dtype=np.float32),
            "multi_counters.weight": np.random.standard_normal(3).astype(np.float32),
            "upsample_module.fsia.weight": np.random.standard_normal(2).astype(np.float32),
        }
        loaded_paths: tuple[Path, ...] = ()
        if pretrained_backbone is not None:
            parameters["backbone.conv.weight"] += 10
            loaded_paths = (pretrained_backbone,)
            if self.mutate_head:
                parameters["multi_counters.weight"] += 1
            if self.extra_weight is not None:
                loaded_paths += (self.extra_weight,)
        return _FakeModel(parameters, loaded_paths)


def test_initialize_model_loads_backbone_only(tmp_path: Path) -> None:
    backbone = tmp_path / "backbone.pth"
    backbone.write_bytes(b"verified ImageNet backbone")
    factory = _FakeModelFactory()

    audit = initialize_steerer_model(
        factory, backbone, expected_sha256=sha256_file(backbone)
    )

    assert factory.calls == [None, backbone.resolve()]
    assert audit.loaded_scope == "backbone_only"
    assert audit.model_checkpoint_loaded is False
    assert audit.loaded_backbone_keys == ("backbone.conv.weight",)
    assert audit.unmatched_backbone_keys == ("backbone.bn.weight",)
    assert audit.random_head_parameter_count == 2
    assert audit.randomly_initialized_non_backbone_keys == (
        "multi_counters.weight", "upsample_module.fsia.weight"
    )


def test_initialize_model_rejects_head_changes_or_other_weight_reads(tmp_path: Path) -> None:
    backbone = tmp_path / "backbone.pth"
    backbone.write_bytes(b"verified ImageNet backbone")
    other = tmp_path / "model-checkpoint.pth"
    other.write_bytes(b"forbidden")

    with pytest.raises(ValueError, match="head|non-backbone"):
        initialize_steerer_model(
            _FakeModelFactory(mutate_head=True),
            backbone,
            expected_sha256=sha256_file(backbone),
        )
    with pytest.raises(PermissionError, match="other weight|backbone"):
        initialize_steerer_model(
            _FakeModelFactory(extra_weight=other),
            backbone,
            expected_sha256=sha256_file(backbone),
        )


def test_initialize_model_checks_hash_before_constructor(tmp_path: Path) -> None:
    backbone = tmp_path / "backbone.pth"
    backbone.write_bytes(b"wrong bytes")
    factory = _FakeModelFactory()

    with pytest.raises(ValueError, match="SHA-256"):
        initialize_steerer_model(factory, backbone, expected_sha256="0" * 64)

    assert factory.calls == []
