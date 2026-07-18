from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from droneai.evaluation_contract import EvaluationSample
from droneai.integrity import sha256_file


class _Backend:
    parameter_count = 12345
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()

    def __init__(
        self, density: np.ndarray | None = None, error: Exception | None = None
    ) -> None:
        self.density = density
        self.error = error
        self.observed: np.ndarray | None = None
        self.patch_size: int | None = None

    def infer(
        self, normalized_bchw: np.ndarray, patch_size: int
    ) -> tuple[np.ndarray, float, float]:
        self.observed = normalized_bchw
        self.patch_size = patch_size
        if self.error is not None:
            raise self.error
        assert self.density is not None
        return self.density, 2.5, 96.0


def _module():
    return importlib.import_module("droneai.mpcount_adapter")


def _git_repo(root: Path) -> str:
    reviewed = (
        "README.md",
        "LICENSE",
        "inference.py",
        "models/models.py",
        "utils/misc.py",
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
    image = tmp_path / "sample.png"
    Image.new("RGB", (17, 18), color=(255, 0, 0)).save(image)
    ground_truth_density = np.ones((3, 3), dtype=np.float32)
    return EvaluationSample(
        sample_id="sample",
        dataset_id="ucf-qnrf-kaggle-apache",
        split_id="dm-count-upstream-qnrf-val-smoke-36-v1",
        image_path=image,
        source_sha256=sha256_file(image),
        width=17,
        height=18,
        ground_truth_count=9.0,
        ground_truth_density=ground_truth_density,
        condition_tags={"density_band": "low"},
    )


def _adapter(tmp_path: Path, backend: _Backend):
    adapter_module = _module()
    upstream = tmp_path / "MPCount"
    commit = _git_repo(upstream)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint fixture")
    return adapter_module.MPCountAdapter(
        upstream_dir=upstream,
        expected_upstream_commit=commit,
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        device="cpu",
        backend=backend,
        unit_size=16,
        patch_size=3584,
        log_para=1000.0,
    )


def test_adapter_preserves_official_padding_normalization_and_count(
    tmp_path: Path,
) -> None:
    backend = _Backend(np.full((32, 32), 1000.0, dtype=np.float32))
    adapter = _adapter(tmp_path, backend)

    prediction = adapter.predict(_sample(tmp_path), retain_native=True)
    brief = adapter.brief()

    assert prediction.failure_state is None
    assert prediction.output_type == "density"
    assert prediction.predicted_count == pytest.approx(17 * 18)
    assert prediction.density is not None
    assert prediction.density.shape == (3, 3)
    assert float(prediction.density.sum()) == pytest.approx(17 * 18)
    assert prediction.latency_ms == 2.5
    assert prediction.peak_vram_mb == 96.0
    assert prediction.metadata["padding_left"] == 7
    assert prediction.metadata["padding_top"] == 7
    assert prediction.metadata["padding_right"] == 8
    assert prediction.metadata["padding_bottom"] == 7
    assert prediction.metadata["log_para"] == 1000.0
    assert backend.observed is not None
    assert backend.observed.shape == (1, 3, 32, 32)
    assert backend.observed[0, 0, 7, 7] == pytest.approx(1.0)
    assert backend.observed[0, 1, 7, 7] == pytest.approx(-1.0)
    assert backend.observed[0, :, 0, 0].tolist() == pytest.approx([-1.0] * 3)
    assert backend.patch_size == 3584
    assert brief.parameter_count == 12345
    assert brief.native_output == "non-negative full-resolution density map"
    assert brief.count_derivation == "sum of cropped density-map mass divided by 1000"


@pytest.mark.parametrize(
    "density",
    (
        np.full((32, 32), np.nan, dtype=np.float32),
        np.full((32, 32), -1.0, dtype=np.float32),
        np.ones((31, 32), dtype=np.float32),
    ),
)
def test_adapter_fails_closed_on_invalid_density(
    tmp_path: Path, density: np.ndarray
) -> None:
    prediction = _adapter(tmp_path, _Backend(density)).predict(
        _sample(tmp_path), retain_native=True
    )

    assert prediction.predicted_count is None
    assert prediction.density is None
    assert prediction.failure_state is not None


def test_adapter_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    module = _module()
    upstream = tmp_path / "MPCount"
    commit = _git_repo(upstream)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        module.MPCountAdapter(
            upstream_dir=upstream,
            expected_upstream_commit=commit,
            checkpoint_path=checkpoint,
            checkpoint_sha256="0" * 64,
            device="cpu",
            backend=_Backend(np.ones((32, 32), dtype=np.float32)),
        )


def test_adapter_rejects_dirty_upstream(tmp_path: Path) -> None:
    module = _module()
    upstream = tmp_path / "MPCount"
    commit = _git_repo(upstream)
    (upstream / "models/models.py").write_text("# dirty\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="working tree must be clean"):
        module.MPCountAdapter(
            upstream_dir=upstream,
            expected_upstream_commit=commit,
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            device="cpu",
            backend=_Backend(np.ones((32, 32), dtype=np.float32)),
        )


def test_adapter_turns_backend_error_into_explicit_failure(tmp_path: Path) -> None:
    prediction = _adapter(
        tmp_path, _Backend(error=RuntimeError("inference failed"))
    ).predict(_sample(tmp_path), retain_native=True)

    assert prediction.predicted_count is None
    assert prediction.failure_state == "RuntimeError: inference failed"
    assert prediction.latency_ms > 0


def test_official_model_config_records_density_and_domain_transfer() -> None:
    config = json.loads(
        (Path(__file__).parents[1] / "configs/models/mpcount.official.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["architecture"]["output_type"] == "density_map"
    assert config["checkpoint"]["training_dataset"] == "ShanghaiTech Part A"
    assert config["reported_results"]["sta_to_ucf_qnrf"] == {
        "mae": 112.8,
        "rmse": 193.8,
    }
