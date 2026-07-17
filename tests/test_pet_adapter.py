from __future__ import annotations

import subprocess
import inspect
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from droneai.evaluation_contract import EvaluationSample
from droneai.integrity import sha256_file
from droneai.pet_adapter import (
    PETAdapter,
    _pet_test_forward_current_torch,
    calculate_pet_size,
)


class _Backend:
    parameter_count = 21_000_000
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.observed: np.ndarray | None = None

    def infer(self, normalized_chw: np.ndarray):
        self.observed = normalized_chw
        if self.error:
            raise self.error
        return (
            ((0.25, 0.5), (1.05, -0.1)),
            (0.9, 0.75),
            (2, 3),
            12.5,
            2048.0,
        )


def _git_repo(root: Path) -> str:
    reviewed = (
        "README.md",
        "LICENSE",
        "models/pet.py",
        "models/backbones/backbone_vgg.py",
        "models/backbones/vgg.py",
        "models/transformer/prog_win_transformer.py",
        "util/misc.py",
        "preprocess_dataset.py",
        "engine.py",
    )
    for relative in reviewed:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    for command in (
        ("git", "init"),
        ("git", "config", "user.email", "fixture@example.test"),
        ("git", "config", "user.name", "Fixture"),
        ("git", "add", "."),
        ("git", "commit", "-m", "fixture"),
    ):
        subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _adapter(tmp_path: Path, backend: _Backend) -> PETAdapter:
    upstream = tmp_path / "upstream"
    commit = _git_repo(upstream)
    checkpoint = tmp_path / "pet.pth"
    checkpoint.write_bytes(b"fixture checkpoint")
    return PETAdapter(
        upstream_dir=upstream,
        expected_upstream_commit=commit,
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        device="cuda:0",
        backend=backend,
    )


def _sample(tmp_path: Path, width: int = 4000, height: int = 2000) -> EvaluationSample:
    image = tmp_path / "sample.png"
    Image.new("RGB", (width, height), (255, 0, 0)).save(image)
    return EvaluationSample(
        sample_id="sample",
        dataset_id="ucf-qnrf-kaggle-apache",
        split_id="dm-count-upstream-qnrf-val-smoke-36-v1",
        image_path=image,
        source_sha256=sha256_file(image),
        width=width,
        height=height,
        ground_truth_count=2.0,
        ground_truth_points=((10.0, 10.0), (20.0, 20.0)),
        has_point_annotations=True,
        condition_tags={"density_band": "low"},
    )


def test_pet_size_uses_official_1536_long_side_flooring() -> None:
    assert calculate_pet_size(4000, 2000) == (1536, 768, pytest.approx(0.384))
    assert calculate_pet_size(1000, 700) == (1000, 700, 1.0)
    with pytest.raises(ValueError, match="dimensions"):
        calculate_pet_size(0, 100)


def test_current_torch_compatibility_keeps_boolean_masks_on_device() -> None:
    source = inspect.getsource(_pet_test_forward_current_torch)
    assert ".cpu()" not in source
    assert "sparse_scores > threshold" in source
    assert "dense_scores > threshold" in source
    assert "threshold = 0.5" in source


def test_adapter_returns_filtered_points_in_original_xy_coordinates(tmp_path: Path) -> None:
    backend = _Backend()
    prediction = _adapter(tmp_path, backend).predict(_sample(tmp_path), retain_native=True)

    assert prediction.failure_state is None
    assert prediction.output_type == "points"
    assert prediction.predicted_count == 2.0
    assert prediction.points[0] == pytest.approx((2000.0, 500.0))
    assert prediction.points[1][0] == 0.0
    assert 0 <= prediction.points[1][1] < 2000
    assert prediction.point_confidences == pytest.approx((0.9, 0.75))
    assert prediction.metadata["native_density_available"] is False
    assert prediction.metadata["split_map_height"] == 2
    assert prediction.metadata["split_map_width"] == 3
    assert prediction.metadata["clipped_point_count"] == 1
    assert backend.observed is not None
    assert backend.observed.shape == (3, 768, 1536)
    assert backend.observed[0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229)


def test_brief_blocks_commercial_use_and_describes_point_count(tmp_path: Path) -> None:
    brief = _adapter(tmp_path, _Backend()).brief()
    assert brief.family == "point-query quadtree"
    assert brief.rights_status == "PASS_RESEARCH_ONLY"
    assert brief.count_derivation == "number of test-time point queries retained above probability 0.5"
    assert "academic purposes only" in brief.code_rights_status
    assert brief.parameter_count == 21_000_000
    assert brief.require_full_run_approval() is None


def test_adapter_fails_closed_on_dirty_upstream_or_bad_checkpoint_hash(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    commit = _git_repo(upstream)
    checkpoint = tmp_path / "pet.pth"
    checkpoint.write_bytes(b"fixture")
    (upstream / "untracked").write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="working tree must be clean"):
        PETAdapter(
            upstream_dir=upstream,
            expected_upstream_commit=commit,
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            device="cuda:0",
            backend=_Backend(),
        )

    (upstream / "untracked").unlink()
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        PETAdapter(
            upstream_dir=upstream,
            expected_upstream_commit=commit,
            checkpoint_path=checkpoint,
            checkpoint_sha256="0" * 64,
            device="cuda:0",
            backend=_Backend(),
        )


def test_backend_error_becomes_explicit_failed_prediction(tmp_path: Path) -> None:
    prediction = _adapter(tmp_path, _Backend(error=RuntimeError("boom"))).predict(
        _sample(tmp_path, 32, 32), retain_native=False
    )
    assert prediction.failure_state == "RuntimeError: boom"
    assert prediction.predicted_count is None
    assert prediction.points == ()
    assert "RuntimeError: boom" in prediction.metadata["traceback"]
