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
PINNED_ORIGIN = "https://github.com/taohan10200/STEERER.git"


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


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_upstream(
    root: Path,
    config_source: str = OFFICIAL_CONFIG,
    *,
    origin_url: str = PINNED_ORIGIN,
) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / "LICENSE").write_text("official license fixture\n", encoding="utf-8")
    (root / "configs" / "QNRF_final.py").write_text(config_source, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(
        root,
        "-c", "user.name=DroneAI tests",
        "-c", "user.email=tests@example.invalid",
        "commit", "-qm", "fixture",
    )
    _git(root, "remote", "add", "origin", origin_url)
    return root


def _profile(tmp_path: Path, upstream: Path, backbone: Path | None = None):
    profile = load_training_profile(PROFILE_PATH)
    if backbone is None:
        backbone = tmp_path / "backbones" / profile.imagenet_backbone.filename
        backbone.parent.mkdir(parents=True, exist_ok=True)
        backbone.write_bytes(b"verified ImageNet backbone fixture")
    upstream_ref = replace(
        profile.model_upstream,
        commit=_git(upstream, "rev-parse", "HEAD"),
        license_sha256=sha256_file(upstream / "LICENSE"),
    )
    backbone_ref = replace(
        profile.imagenet_backbone,
        path=backbone.resolve(),
        filename=backbone.name,
        sha256=sha256_file(backbone),
        byte_size=backbone.stat().st_size,
    )
    return replace(
        profile,
        model_upstream=upstream_ref,
        imagenet_backbone=backbone_ref,
        checkpoint_root=tmp_path / "checkpoints",
        result_root=tmp_path / "results",
    )


def test_audit_records_profile_pinned_origin_checkout_and_files(tmp_path: Path) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")
    profile = _profile(tmp_path, upstream)

    audit = audit_upstream(profile, upstream)

    assert audit.origin_url == PINNED_ORIGIN
    assert audit.commit == profile.model_upstream.commit
    assert audit.clean is True
    assert audit.license_sha256 == profile.model_upstream.license_sha256
    assert audit.config_path == (upstream / "configs" / "QNRF_final.py").resolve()
    assert audit.config_sha256 == sha256_file(audit.config_path)


@pytest.mark.parametrize("defect", ["wrong-commit", "dirty"])
def test_audit_rejects_dirty_or_wrong_upstream(tmp_path: Path, defect: str) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")
    profile = _profile(tmp_path, upstream)
    if defect == "dirty":
        (upstream / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    else:
        profile = replace(
            profile,
            model_upstream=replace(profile.model_upstream, commit="a" * 40),
        )

    with pytest.raises(ValueError, match="pinned|clean"):
        audit_upstream(profile, upstream)


def test_audit_rejects_wrong_origin(tmp_path: Path) -> None:
    upstream = _write_upstream(
        tmp_path / "STEERER", origin_url="https://example.test/attacker.git"
    )
    profile = _profile(tmp_path, upstream)

    with pytest.raises(ValueError, match="origin"):
        audit_upstream(profile, upstream)


def test_audit_rejects_config_path_escape(tmp_path: Path) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")
    outside = tmp_path / "outside.py"
    outside.write_text(OFFICIAL_CONFIG, encoding="utf-8")
    profile = _profile(tmp_path, upstream)
    profile = replace(
        profile,
        model_upstream=replace(profile.model_upstream, config_path=Path("../outside.py")),
    )

    with pytest.raises(ValueError, match="config.*inside|path"):
        audit_upstream(profile, upstream)


def test_untrusted_config_never_executes_before_origin_audit(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
    upstream = _write_upstream(
        tmp_path / "STEERER",
        source,
        origin_url="https://example.test/attacker.git",
    )
    profile = _profile(tmp_path, upstream)

    with pytest.raises(ValueError, match="origin"):
        synthesize_official_config(
            profile=profile,
            upstream_dir=upstream,
            processed_root=tmp_path / "processed",
            backbone_path=profile.imagenet_backbone.path,
            stage="T1",
            physical_batch=8,
            accumulation_steps=1,
            run_id="run-a",
        )

    assert not marker.exists()


def test_config_executes_the_single_audited_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")
    profile = _profile(tmp_path, upstream)
    config_path = (upstream / "configs" / "QNRF_final.py").resolve()
    expected_config_sha256 = sha256_file(config_path)
    original_read_bytes = Path.read_bytes
    config_reads = 0

    def read_bytes_once(path: Path) -> bytes:
        nonlocal config_reads
        payload = original_read_bytes(path)
        if path.resolve() == config_path:
            config_reads += 1
            path.write_text("raise RuntimeError('unaudited reread')\n", encoding="utf-8")
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_bytes_once)

    config = synthesize_official_config(
        profile=profile,
        upstream_dir=upstream,
        processed_root=tmp_path / "processed",
        backbone_path=profile.imagenet_backbone.path,
        stage="T1",
        physical_batch=8,
        accumulation_steps=1,
        run_id="run-a",
    )

    assert config_reads == 1
    assert config["seed"] == 3035
    assert config["droneai"]["upstream_config_sha256"] == expected_config_sha256


@pytest.mark.parametrize("stage", ["T0", "T1", "T5", "T50", "T800"])
def test_config_uses_validation_and_keeps_official_800_epoch_horizon(
    tmp_path: Path, stage: str
) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")
    profile = _profile(tmp_path, upstream)

    config = synthesize_official_config(
        profile=profile,
        upstream_dir=upstream,
        processed_root=tmp_path / "processed",
        backbone_path=profile.imagenet_backbone.path,
        stage=stage,
        physical_batch=4,
        accumulation_steps=2,
        run_id="run-a",
    )

    assert config["dataset"]["root"] == f"{tmp_path / 'processed'}/"
    assert config["dataset"]["train_set"] == "train.txt"
    assert config["dataset"]["test_set"] == "val.txt"
    assert Path(config["dataset"]["root"] + config["dataset"]["train_set"]) == (
        tmp_path / "processed" / "train.txt"
    )
    assert config["network"]["pretrained_backbone"] == str(
        profile.imagenet_backbone.path.resolve()
    )
    assert config["gpus"] == (0,)
    assert config["train"]["batch_size_per_gpu"] == 4
    assert config["train"]["end_epoch"] == 800
    assert config["log_dir"] == str(tmp_path / "results" / "run-a")
    assert config["droneai"]["stage_stop_epoch"] == {
        "T0": 0, "T1": 1, "T5": 5, "T50": 50, "T800": 800
    }[stage]
    assert config["droneai"]["schedule_horizon_epochs"] == 800


@pytest.mark.parametrize(
    "run_id",
    ["", ".", "..", "../escape", "nested/run", "nested\\run", "run:a", "run id"],
)
def test_config_rejects_unsafe_run_id(tmp_path: Path, run_id: str) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")
    profile = _profile(tmp_path, upstream)
    with pytest.raises(ValueError, match="run_id"):
        synthesize_official_config(
            profile=profile, upstream_dir=upstream,
            processed_root=tmp_path / "processed",
            backbone_path=profile.imagenet_backbone.path,
            stage="T1", physical_batch=8, accumulation_steps=1, run_id=run_id,
        )


@pytest.mark.parametrize(
    ("physical_batch", "accumulation_steps"),
    [("8", 1), (8, "1"), (2, 4), (8, 2)],
)
def test_config_rejects_non_integer_or_unapproved_batch_plan(
    tmp_path: Path, physical_batch: object, accumulation_steps: object
) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")
    profile = _profile(tmp_path, upstream)
    with pytest.raises(ValueError, match="batch|accumulation"):
        synthesize_official_config(
            profile=profile, upstream_dir=upstream,
            processed_root=tmp_path / "processed",
            backbone_path=profile.imagenet_backbone.path,
            stage="T1", physical_batch=physical_batch,  # type: ignore[arg-type]
            accumulation_steps=accumulation_steps,  # type: ignore[arg-type]
            run_id="run-a",
        )


def test_config_rejects_resume_outside_current_run_lineage(tmp_path: Path) -> None:
    upstream = _write_upstream(tmp_path / "STEERER")
    profile = _profile(tmp_path, upstream)
    with pytest.raises(ValueError, match="resume.*run"):
        synthesize_official_config(
            profile=profile, upstream_dir=upstream,
            processed_root=tmp_path / "processed",
            backbone_path=profile.imagenet_backbone.path,
            stage="T5", physical_batch=8, accumulation_steps=1, run_id="run-a",
            resume_path=tmp_path / "checkpoints" / "another-run" / "last.pth",
        )


def test_config_rejects_mutated_official_training_contract(tmp_path: Path) -> None:
    upstream = _write_upstream(
        tmp_path / "STEERER", OFFICIAL_CONFIG.replace("end_epoch=800", "end_epoch=50")
    )
    profile = _profile(tmp_path, upstream)
    with pytest.raises(ValueError, match="official.*end_epoch|end_epoch.*official"):
        synthesize_official_config(
            profile=profile, upstream_dir=upstream,
            processed_root=tmp_path / "processed",
            backbone_path=profile.imagenet_backbone.path,
            stage="T50", physical_batch=8, accumulation_steps=1, run_id="run-a",
        )


class _FakeModel:
    def __init__(self, parameters: dict[str, np.ndarray]):
        self._parameters = parameters

    def named_parameters(self):
        return tuple(self._parameters.items())


class _FakeCuda:
    def manual_seed_all(self, _seed: int) -> None:
        return None


class _FakeTorch:
    def __init__(self):
        self.calls: list[Path] = []
        self.cuda = _FakeCuda()

        def original_load(path: str | Path, *args: object, **kwargs: object):
            del args, kwargs
            self.calls.append(Path(path).resolve())
            return {"fixture": True}

        self.load = original_load

    def manual_seed(self, _seed: int) -> None:
        return None


class _FakeModelFactory:
    def __init__(
        self,
        torch_module: _FakeTorch,
        *,
        read_backbone: bool = True,
        read_path: Path | None = None,
        mutate_head: bool = False,
        raise_after_load: bool = False,
        reference_weight: Path | None = None,
    ):
        self.torch = torch_module
        self.read_backbone = read_backbone
        self.read_path = read_path
        self.mutate_head = mutate_head
        self.raise_after_load = raise_after_load
        self.reference_weight = reference_weight
        self.calls: list[Path | None] = []

    def __call__(self, *, pretrained_backbone: Path | None):
        self.calls.append(pretrained_backbone)
        parameters = {
            "backbone.conv.weight": np.random.standard_normal(4).astype(np.float32),
            "backbone.bn.weight": np.ones(2, dtype=np.float32),
            "multi_counters.weight": np.random.standard_normal(3).astype(np.float32),
            "upsample_module.fsia.weight": np.random.standard_normal(2).astype(np.float32),
        }
        if pretrained_backbone is None and self.reference_weight is not None:
            self.torch.load(self.reference_weight)
        if pretrained_backbone is not None:
            if self.read_backbone:
                self.torch.load(self.read_path or pretrained_backbone)
            if self.raise_after_load:
                raise RuntimeError("constructor failed")
            parameters["backbone.conv.weight"] += 10
            if self.mutate_head:
                parameters["multi_counters.weight"] += 1
        return _FakeModel(parameters)


def _model_fixture(tmp_path: Path):
    upstream = _write_upstream(tmp_path / "STEERER")
    backbone = tmp_path / "backbones" / "hrnetv2_w48_imagenet_pretrained.pth"
    backbone.parent.mkdir(parents=True)
    backbone.write_bytes(b"verified ImageNet backbone")
    return _profile(tmp_path, upstream, backbone), backbone


def test_initialize_model_requires_profile_canonical_path_name_size_and_hash(
    tmp_path: Path,
) -> None:
    profile, backbone = _model_fixture(tmp_path)
    copied = tmp_path / "elsewhere" / backbone.name
    copied.parent.mkdir()
    copied.write_bytes(backbone.read_bytes())

    defects = (
        (profile, copied, "canonical|path"),
        (
            replace(
                profile,
                imagenet_backbone=replace(profile.imagenet_backbone, filename="other.pth"),
            ),
            backbone,
            "filename|name",
        ),
        (
            replace(
                profile,
                imagenet_backbone=replace(
                    profile.imagenet_backbone,
                    byte_size=profile.imagenet_backbone.byte_size + 1,
                ),
            ),
            backbone,
            "size",
        ),
        (
            replace(
                profile,
                imagenet_backbone=replace(profile.imagenet_backbone, sha256="0" * 64),
            ),
            backbone,
            "SHA-256",
        ),
    )
    for mutated_profile, supplied_path, message in defects:
        with pytest.raises(ValueError, match=message):
            initialize_steerer_model(
                mutated_profile,
                _FakeModelFactory(_FakeTorch()),
                supplied_path,
                torch_module=_FakeTorch(),
            )


def test_initialize_model_requires_observed_approved_torch_load(tmp_path: Path) -> None:
    profile, backbone = _model_fixture(tmp_path)
    torch_module = _FakeTorch()
    factory = _FakeModelFactory(torch_module, read_backbone=False)

    with pytest.raises(ValueError, match="torch.load|read"):
        initialize_steerer_model(
            profile, factory, backbone, torch_module=torch_module
        )


def test_initialize_model_rejects_different_weight_read(tmp_path: Path) -> None:
    profile, backbone = _model_fixture(tmp_path)
    other = tmp_path / "model-checkpoint.pth"
    other.write_bytes(b"forbidden")
    torch_module = _FakeTorch()

    with pytest.raises(PermissionError, match="other weight|backbone"):
        initialize_steerer_model(
            profile,
            _FakeModelFactory(torch_module, read_path=other),
            backbone,
            torch_module=torch_module,
        )


def test_initialize_model_records_approved_path_read_and_backbone_scope(
    tmp_path: Path,
) -> None:
    profile, backbone = _model_fixture(tmp_path)
    torch_module = _FakeTorch()
    factory = _FakeModelFactory(torch_module)

    audit = initialize_steerer_model(
        profile, factory, backbone, torch_module=torch_module
    )

    assert factory.calls == [None, backbone.resolve()]
    assert audit.loaded_scope == "backbone_only"
    assert audit.model_checkpoint_loaded is False
    assert audit.observed_weight_load_paths == (backbone.resolve(),)
    assert audit.observed_weight_load_count == 1
    assert audit.loaded_backbone_keys == ("backbone.conv.weight",)
    assert audit.unmatched_backbone_keys == ("backbone.bn.weight",)
    assert audit.random_head_parameter_count == 2


def test_initialize_model_rejects_head_changes_and_reference_checkpoint(
    tmp_path: Path,
) -> None:
    profile, backbone = _model_fixture(tmp_path)
    torch_module = _FakeTorch()
    with pytest.raises(ValueError, match="head|non-backbone"):
        initialize_steerer_model(
            profile,
            _FakeModelFactory(torch_module, mutate_head=True),
            backbone,
            torch_module=torch_module,
        )

    other = tmp_path / "reference-checkpoint.pth"
    other.write_bytes(b"forbidden")
    with pytest.raises(PermissionError, match="reference|checkpoint|weight"):
        initialize_steerer_model(
            profile,
            _FakeModelFactory(torch_module, reference_weight=other),
            backbone,
            torch_module=torch_module,
        )


def test_torch_load_is_restored_when_constructor_raises(tmp_path: Path) -> None:
    profile, backbone = _model_fixture(tmp_path)
    torch_module = _FakeTorch()
    original_load = torch_module.load

    with pytest.raises(RuntimeError, match="constructor failed"):
        initialize_steerer_model(
            profile,
            _FakeModelFactory(torch_module, raise_after_load=True),
            backbone,
            torch_module=torch_module,
        )

    assert torch_module.load is original_load
