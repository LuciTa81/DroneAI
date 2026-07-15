from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from droneai.dm_count_adapter import (
    DMCountAdapter,
    calculate_qnrf_size,
    resize_density_preserve_mass,
)
from droneai.evaluation_contract import EvaluationSample
from droneai.integrity import sha256_file


class _Backend:
    def __init__(self, density: np.ndarray | None = None, error: Exception | None = None):
        self.density = density
        self.error = error
        self.observed: np.ndarray | None = None

    def infer(self, normalized_chw: np.ndarray) -> tuple[np.ndarray, float, float]:
        self.observed = normalized_chw
        if self.error is not None:
            raise self.error
        assert self.density is not None
        return self.density, 1.25, 64.0


def _git_repo(root: Path) -> str:
    root.mkdir()
    reviewed = (
        "models.py",
        "test.py",
        "datasets/crowd.py",
        "preprocess_dataset.py",
        "preprocess/preprocess_dataset_qnrf.py",
    )
    for relative in reviewed:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# reviewed fixture\n", encoding="utf-8")
    for command in (
        ["git", "init"],
        ["git", "config", "user.email", "fixture@example.test"],
        ["git", "config", "user.name", "Fixture"],
        ["git", "add", "."],
        ["git", "commit", "-m", "fixture"],
    ):
        subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sample(tmp_path: Path) -> EvaluationSample:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(image_path)
    density = np.ones((2, 2), dtype=np.float32)
    return EvaluationSample(
        sample_id="sample",
        dataset_id="ucf-qnrf-kaggle-apache",
        split_id="official-qnrf-val-smoke-v1",
        image_path=image_path,
        source_sha256=sha256_file(image_path),
        width=16,
        height=16,
        ground_truth_count=4.0,
        ground_truth_density=density,
        condition_tags={"density_band": "low"},
    )


def _adapter(tmp_path: Path, backend: _Backend) -> DMCountAdapter:
    upstream = tmp_path / "upstream"
    commit = _git_repo(upstream)
    checkpoint = tmp_path / "model_qnrf.pth"
    checkpoint.write_bytes(b"checkpoint fixture")
    return DMCountAdapter(
        upstream_dir=upstream,
        expected_upstream_commit=commit,
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        device="cpu",
        backend=backend,
        min_size=16,
        max_size=32,
    )


def test_qnrf_size_matches_official_short_side_policy() -> None:
    assert calculate_qnrf_size(width=400, height=200, min_size=512, max_size=2048) == (
        1024,
        512,
        2.56,
    )
    assert calculate_qnrf_size(width=5000, height=2500, min_size=512, max_size=2048) == (
        4096,
        2048,
        0.8192,
    )
    assert calculate_qnrf_size(width=1000, height=800, min_size=512, max_size=2048) == (
        1000,
        800,
        1.0,
    )


def test_density_resize_preserves_nonnegative_mass() -> None:
    density = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    resized = resize_density_preserve_mass(density, target_shape=(5, 7))

    assert resized.shape == (5, 7)
    assert np.all(resized >= 0)
    assert float(resized.sum()) == pytest.approx(6.0, rel=1e-6)


def test_adapter_returns_count_preserving_density_and_model_brief(tmp_path: Path) -> None:
    backend = _Backend(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    adapter = _adapter(tmp_path, backend)

    prediction = adapter.predict(_sample(tmp_path), retain_native=True)
    brief = adapter.brief()

    assert prediction.failure_state is None
    assert prediction.output_type == "density"
    assert prediction.predicted_count == pytest.approx(10.0)
    assert prediction.density is not None
    assert prediction.density.shape == (2, 2)
    assert float(prediction.density.sum()) == pytest.approx(10.0)
    assert prediction.latency_ms == 1.25
    assert prediction.peak_vram_mb == 64.0
    assert prediction.metadata == {
        "original_width": 16,
        "original_height": 16,
        "processed_width": 16,
        "processed_height": 16,
        "resize_ratio": 1.0,
        "native_density_width": 2,
        "native_density_height": 2,
        "evaluation_density_width": 2,
        "evaluation_density_height": 2,
        "coordinate_transform": "mass-preserving bilinear resize to original-image /8 grid",
    }
    assert backend.observed is not None
    assert backend.observed.shape == (3, 16, 16)
    assert backend.observed[0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229)
    assert backend.observed[1, 0, 0] == pytest.approx((0.0 - 0.456) / 0.224)
    assert brief.native_output == "non-negative density map"
    assert brief.count_derivation == "sum of density-map mass"
    assert brief.rights_status == "PASS_COMMERCIAL_CANDIDATE"
    assert any(Path(path).name == "models.py" for path in brief.reviewed_paths)
    assert all(Path(path).is_file() for path in brief.reviewed_paths)


def test_adapter_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    commit = _git_repo(upstream)
    checkpoint = tmp_path / "model_qnrf.pth"
    checkpoint.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        DMCountAdapter(
            upstream_dir=upstream,
            expected_upstream_commit=commit,
            checkpoint_path=checkpoint,
            checkpoint_sha256="0" * 64,
            device="cpu",
            backend=_Backend(np.ones((2, 2), dtype=np.float32)),
        )


@pytest.mark.parametrize("dirty_path", ("models.py", "untracked.py"))
def test_adapter_rejects_modified_or_untracked_upstream(
    tmp_path: Path, dirty_path: str
) -> None:
    upstream = tmp_path / "upstream"
    commit = _git_repo(upstream)
    (upstream / dirty_path).write_text("# dirty\n", encoding="utf-8")
    checkpoint = tmp_path / "model_qnrf.pth"
    checkpoint.write_bytes(b"checkpoint fixture")

    with pytest.raises(ValueError, match="working tree must be clean"):
        DMCountAdapter(
            upstream_dir=upstream,
            expected_upstream_commit=commit,
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            device="cpu",
            backend=_Backend(np.ones((2, 2), dtype=np.float32)),
        )


def test_adapter_turns_backend_error_into_explicit_failure(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, _Backend(error=RuntimeError("inference failed")))

    prediction = adapter.predict(_sample(tmp_path), retain_native=True)

    assert prediction.predicted_count is None
    assert prediction.density is None
    assert prediction.failure_state == "RuntimeError: inference failed"
    assert prediction.latency_ms > 0


def test_official_model_config_freezes_density_structure_and_qnrf_checkpoint() -> None:
    config_path = (
        Path(__file__).parents[1] / "configs" / "models" / "dm_count.official.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["architecture"]["output_type"] == "density"
    assert config["architecture"]["output_stride"] == 8
    assert config["checkpoint"]["repository_path"] == "pretrained_models/model_qnrf.pth"
    assert config["reported_results"]["ucf_qnrf"] == {"mae": 85.6, "rmse": 148.3}
