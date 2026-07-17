from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from droneai.evaluation_contract import EvaluationSample, ZoneBox
from droneai.integrity import sha256_file
from droneai.steerer_adapter import (
    STEERERAdapter,
    calculate_steerer_size,
    extract_steerer_points,
)


class _Backend:
    def __init__(
        self,
        densities: tuple[np.ndarray, np.ndarray, np.ndarray],
        *,
        points: tuple[tuple[float, float], ...] = (),
        error: Exception | None = None,
    ):
        self.densities = densities
        self.points = points
        self.error = error
        self.observed: np.ndarray | None = None
        self.missing_keys = ("fixture.missing",)
        self.unexpected_keys = ("fixture.unexpected",)

    def infer(
        self, normalized_chw: np.ndarray
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[tuple[float, float], ...],
        float,
        float,
    ]:
        self.observed = normalized_chw
        if self.error is not None:
            raise self.error
        return self.densities, self.points, 2.5, 128.0


def _git_repo(root: Path) -> str:
    root.mkdir()
    reviewed = (
        "configs/QNRF_final.py",
        "lib/models/build_counter.py",
        "lib/utils/points_from_den.py",
        "lib/core/cc_function.py",
        "lib/datasets/base_dataset.py",
        "lib/datasets/nwpu.py",
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


def _adapter(tmp_path: Path, backend: _Backend) -> STEERERAdapter:
    upstream = tmp_path / "upstream"
    commit = _git_repo(upstream)
    checkpoint = tmp_path / "steerer_qnrf.pth"
    checkpoint.write_bytes(b"checkpoint fixture")
    return STEERERAdapter(
        upstream_dir=upstream,
        expected_upstream_commit=commit,
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        device="cuda:0",
        backend=backend,
    )


def _sample(
    tmp_path: Path,
    *,
    width: int,
    height: int,
    ground_truth_density: np.ndarray | None = None,
    zones: tuple[ZoneBox, ...] = (),
) -> EvaluationSample:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (width, height), color=(255, 0, 0)).save(image_path)
    ground_truth_count = (
        float(np.asarray(ground_truth_density).sum())
        if ground_truth_density is not None
        else 0.0
    )
    return EvaluationSample(
        sample_id="sample",
        dataset_id="ucf-qnrf-kaggle-apache",
        split_id="official-qnrf-val-smoke-v1",
        image_path=image_path,
        source_sha256=sha256_file(image_path),
        width=width,
        height=height,
        ground_truth_count=ground_truth_count,
        ground_truth_density=ground_truth_density,
        condition_tags={"density_band": "low"},
        zones=zones,
    )


def test_size_caps_long_side_and_pads_to_32() -> None:
    assert calculate_steerer_size(4000, 2000) == (
        3072,
        1536,
        pytest.approx(0.768),
    )
    assert calculate_steerer_size(1000, 769) == (1024, 800, 1.0)

    with pytest.raises(ValueError, match="dimensions"):
        calculate_steerer_size(0, 100)


def test_extract_points_merges_scales_and_clips_original_coordinates() -> None:
    x1 = np.zeros((256, 256), dtype=np.float32)
    x4 = np.zeros((64, 64), dtype=np.float32)
    x8 = np.zeros((32, 32), dtype=np.float32)
    x1[20, 20] = 1.0
    x4[5, 6] = 1.0  # (24, 20), suppressed within 32 px of x1
    x4[30, 30] = 1.0  # (120, 120), retained
    x8[31, 0] = 1.0  # (0, 248), retained then clipped after /0.5

    points = extract_steerer_points(
        (x1, x4, x8),
        gaussian_maximum=1.0,
        resize_ratio=0.5,
        original_size=(400, 300),
    )

    assert np.asarray(points[:2]) == pytest.approx(
        np.asarray(((40.0, 40.0), (240.0, 240.0)))
    )
    assert points[2][0] == pytest.approx(0.0)
    assert 0.0 <= points[2][1] < 300.0
    assert points[2][1] == pytest.approx(300.0)


def test_rgb_imagenet_normalization_and_zero_padding(tmp_path: Path) -> None:
    density = np.ones((4, 4), dtype=np.float32)
    backend = _Backend((density, density, density))
    adapter = _adapter(tmp_path, backend)

    prediction = adapter.predict(
        _sample(tmp_path, width=33, height=31), retain_native=True
    )

    assert prediction.failure_state is None
    assert backend.observed is not None
    assert backend.observed.shape == (3, 32, 64)
    assert backend.observed[0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229)
    assert backend.observed[1, 0, 0] == pytest.approx((0.0 - 0.456) / 0.224)
    assert backend.observed[0, 31, 63] == pytest.approx((0.0 - 0.485) / 0.229)
    assert prediction.metadata["resized_width"] == 33
    assert prediction.metadata["resized_height"] == 31
    assert prediction.metadata["processed_width"] == 64
    assert prediction.metadata["processed_height"] == 32
    assert prediction.metadata["resize_ratio"] == 1.0


def test_adapter_rejects_dirty_upstream_and_checkpoint_hash_mismatch(
    tmp_path: Path,
) -> None:
    dirty_upstream = tmp_path / "dirty-upstream"
    dirty_commit = _git_repo(dirty_upstream)
    (dirty_upstream / "untracked.py").write_text("# dirty\n", encoding="utf-8")
    checkpoint = tmp_path / "dirty-checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint fixture")

    with pytest.raises(ValueError, match="working tree must be clean"):
        STEERERAdapter(
            upstream_dir=dirty_upstream,
            expected_upstream_commit=dirty_commit,
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            device="cuda:0",
            backend=_Backend((np.ones((1, 1), np.float32),) * 3),
        )

    clean_upstream = tmp_path / "clean-upstream"
    clean_commit = _git_repo(clean_upstream)
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        STEERERAdapter(
            upstream_dir=clean_upstream,
            expected_upstream_commit=clean_commit,
            checkpoint_path=checkpoint,
            checkpoint_sha256="0" * 64,
            device="cuda:0",
            backend=_Backend((np.ones((1, 1), np.float32),) * 3),
        )


def test_hybrid_prediction_keeps_density_count_separate_from_point_count(
    tmp_path: Path,
) -> None:
    x1 = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    backend = _Backend(
        (x1, np.ones((1, 1), np.float32), np.ones((1, 1), np.float32)),
        points=((1.0, 1.0), (20.0, 20.0)),
    )
    adapter = _adapter(tmp_path, backend)
    zones = (
        ZoneBox("nw", 0, 0, 16, 16),
        ZoneBox("ne", 16, 0, 32, 16),
        ZoneBox("sw", 0, 16, 16, 32),
        ZoneBox("se", 16, 16, 32, 32),
    )

    prediction = adapter.predict(
        _sample(
            tmp_path,
            width=32,
            height=32,
            ground_truth_density=np.ones((4, 4), dtype=np.float32),
            zones=zones,
        ),
        retain_native=True,
    )
    brief = adapter.brief()

    assert prediction.output_type == "hybrid"
    assert prediction.density is not None
    assert prediction.predicted_count == pytest.approx(float(prediction.density.sum()))
    assert prediction.predicted_count == pytest.approx(10.0)
    assert prediction.metadata["localization_point_count"] == len(prediction.points)
    assert prediction.metadata["localization_point_count"] == 2
    assert prediction.coordinate_space == "original_pixels"
    assert prediction.latency_ms == 2.5
    assert prediction.peak_vram_mb == 128.0
    assert prediction.metadata["missing_checkpoint_key_count"] == 1
    assert prediction.metadata["unexpected_checkpoint_key_count"] == 1
    assert brief.model_id == "steerer-official-ucf-qnrf"
    assert brief.native_output == "multi-resolution density maps and merged localization points"
    assert brief.count_derivation == "sum of highest-resolution density-map mass"


def test_adapter_turns_backend_error_into_explicit_failed_prediction(
    tmp_path: Path,
) -> None:
    density = np.ones((1, 1), dtype=np.float32)
    adapter = _adapter(
        tmp_path,
        _Backend((density, density, density), error=RuntimeError("inference failed")),
    )

    prediction = adapter.predict(
        _sample(tmp_path, width=32, height=32), retain_native=True
    )

    assert prediction.output_type == "hybrid"
    assert prediction.predicted_count is None
    assert prediction.density is None
    assert prediction.points == ()
    assert prediction.failure_state == "RuntimeError: inference failed"
    assert prediction.latency_ms > 0
