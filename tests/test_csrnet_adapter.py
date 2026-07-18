from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from droneai.evaluation_contract import EvaluationSample
from droneai.integrity import sha256_file


class _Backend:
    parameter_count = 16_263_489
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()

    def __init__(self, density: np.ndarray) -> None:
        self.density = density
        self.observed: np.ndarray | None = None

    def infer(self, normalized_bchw: np.ndarray) -> tuple[np.ndarray, float, float]:
        self.observed = normalized_bchw
        return self.density, 3.5, 128.0


def _module():
    return importlib.import_module("droneai.csrnet_adapter")


def _git_repo(root: Path) -> str:
    for relative in ("README.md", "model.py"):
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
    return EvaluationSample(
        sample_id="sample",
        dataset_id="ucf-qnrf-kaggle-apache",
        split_id="dm-count-upstream-qnrf-val-smoke-36-v1",
        image_path=image,
        source_sha256=sha256_file(image),
        width=17,
        height=18,
        ground_truth_count=9.0,
        ground_truth_density=np.ones((3, 3), dtype=np.float32),
        condition_tags={"density_band": "low"},
    )


def _adapter(
    tmp_path: Path,
    density: np.ndarray,
    *,
    negative_density_policy: str = "fail",
):
    upstream = tmp_path / "CSRNet-pytorch"
    commit = _git_repo(upstream)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"safe state dict fixture")
    backend = _Backend(density)
    adapter = _module().CSRNetAdapter(
        upstream_dir=upstream,
        expected_upstream_commit=commit,
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        device="cpu",
        backend=backend,
        negative_density_policy=negative_density_policy,
    )
    return adapter, backend


def test_adapter_preserves_official_normalization_density_mass_and_count(
    tmp_path: Path,
) -> None:
    adapter, backend = _adapter(tmp_path, np.ones((2, 2), dtype=np.float32))

    prediction = adapter.predict(_sample(tmp_path), retain_native=True)
    brief = adapter.brief()

    assert prediction.failure_state is None
    assert prediction.predicted_count == pytest.approx(4.0)
    assert prediction.density is not None
    assert prediction.density.shape == (3, 3)
    assert float(prediction.density.sum()) == pytest.approx(4.0)
    assert prediction.latency_ms == 3.5
    assert prediction.peak_vram_mb == 128.0
    assert backend.observed is not None
    assert backend.observed.shape == (1, 3, 18, 17)
    assert backend.observed[0, :, 0, 0].tolist() == pytest.approx(
        [(1.0 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    )
    assert brief.parameter_count == 16_263_489
    assert brief.native_output == "raw signed stride-8 density plus audited non-negative operational density"
    assert brief.count_derivation == "sum of max(raw_density, 0); raw signed sum retained for audit"


def test_adapter_clips_operational_density_and_preserves_raw_audit(
    tmp_path: Path,
) -> None:
    raw = np.asarray([[1.0, -1.0], [2.0, -0.5]], dtype=np.float32)
    adapter, _ = _adapter(
        tmp_path,
        raw,
        negative_density_policy="clip_zero_preserve_raw_audit",
    )

    prediction = adapter.predict(_sample(tmp_path), retain_native=True)
    audit = adapter.raw_density_audit()

    assert prediction.failure_state is None
    assert prediction.predicted_count == pytest.approx(3.0)
    assert prediction.density is not None
    assert np.all(prediction.density >= 0)
    assert float(prediction.density.sum()) == pytest.approx(3.0)
    assert prediction.metadata["negative_density_policy"] == (
        "clip_zero_preserve_raw_audit"
    )
    assert prediction.metadata["native_raw_sum"] == pytest.approx(1.5)
    assert prediction.metadata["native_negative_mass"] == pytest.approx(1.5)
    assert prediction.metadata["operational_density_sum"] == pytest.approx(3.0)
    assert prediction.metadata["operational_clipped_values"] == 2
    assert audit is not None
    assert np.array_equal(audit, raw)


@pytest.mark.parametrize(
    "density",
    (
        np.full((2, 2), np.nan, dtype=np.float32),
        np.full((2, 2), -1.0, dtype=np.float32),
        np.ones((3, 2), dtype=np.float32),
    ),
)
def test_adapter_fails_closed_on_invalid_native_density(
    tmp_path: Path, density: np.ndarray
) -> None:
    adapter, _ = _adapter(tmp_path, density)

    prediction = adapter.predict(_sample(tmp_path), retain_native=True)

    assert prediction.predicted_count is None
    assert prediction.density is None
    assert prediction.failure_state is not None
    assert prediction.latency_ms == 3.5
    assert prediction.peak_vram_mb == 128.0
    assert prediction.metadata["forward_completed"] is True
    assert prediction.metadata["checkpoint_missing_key_count"] == 0
    assert prediction.metadata["checkpoint_unexpected_key_count"] == 0


def test_adapter_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    upstream = tmp_path / "CSRNet-pytorch"
    commit = _git_repo(upstream)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        _module().CSRNetAdapter(
            upstream_dir=upstream,
            expected_upstream_commit=commit,
            checkpoint_path=checkpoint,
            checkpoint_sha256="0" * 64,
            device="cpu",
            backend=_Backend(np.ones((2, 2), dtype=np.float32)),
        )
