from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from droneai.evaluation_contract import EvaluationSample
from droneai.integrity import sha256_file
from droneai.apgcc_adapter import APGCCAdapter, official_eval_dimensions


class _Backend:
    parameter_count = 18_000_000
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
            ((20.0, 10.0), (110.0, 90.0)),
            (0.9, 0.8),
            12.5,
            2048.0,
            {
                "resized_width": 100,
                "resized_height": 80,
                "padded_width": 128,
                "padded_height": 128,
                "scale_x": 1.0,
                "scale_y": 1.0,
            },
        )


class _ScaledBackend(_Backend):
    def infer(self, normalized_chw: np.ndarray):
        self.observed = normalized_chw
        return (
            ((10.0, 5.0), (55.0, 10.0)),
            (0.9, 0.8),
            12.5,
            2048.0,
            {
                "resized_width": 50,
                "resized_height": 40,
                "padded_width": 128,
                "padded_height": 128,
                "scale_x": 0.5,
                "scale_y": 0.5,
            },
        )


def _git_repo(root: Path) -> str:
    reviewed = (
        "README.md",
        "LICENSE",
        "apgcc/configs/SHHA_test.yml",
        "apgcc/models/APGCC.py",
        "apgcc/models/Decoder.py",
        "apgcc/models/Encoder.py",
        "apgcc/models/backbones/vgg.py",
        "apgcc/engine.py",
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


def _adapter(tmp_path: Path, backend: _Backend) -> APGCCAdapter:
    upstream = tmp_path / "APGCC"
    commit = _git_repo(upstream)
    checkpoint = tmp_path / "SHHA_best.pth"
    checkpoint.write_bytes(b"fixture checkpoint")
    return APGCCAdapter(
        upstream_dir=upstream,
        expected_upstream_commit=commit,
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        device="cuda:0",
        backend=backend,
    )


def _sample(tmp_path: Path) -> EvaluationSample:
    image = tmp_path / "sample.png"
    Image.new("RGB", (100, 80), (255, 0, 0)).save(image)
    return EvaluationSample(
        sample_id="sample",
        dataset_id="ucf-qnrf-kaggle-apache",
        split_id="dm-count-upstream-qnrf-val-smoke-36-v1",
        image_path=image,
        source_sha256=sha256_file(image),
        width=100,
        height=80,
        ground_truth_count=1.0,
        ground_truth_points=((20.0, 10.0),),
        has_point_annotations=True,
        condition_tags={"density_band": "low"},
    )


def test_adapter_normalizes_native_image_and_filters_out_of_bounds_points(
    tmp_path: Path,
) -> None:
    backend = _Backend()
    prediction = _adapter(tmp_path, backend).predict(_sample(tmp_path), retain_native=True)

    assert prediction.failure_state is None
    assert prediction.output_type == "points"
    assert prediction.predicted_count == 1.0
    assert prediction.points == ((20.0, 10.0),)
    assert prediction.point_confidences == pytest.approx((0.9,))
    assert prediction.metadata["discarded_out_of_bounds_point_count"] == 1
    assert prediction.metadata["resized_width"] == 100
    assert prediction.metadata["padded_width"] == 128
    assert prediction.metadata["padding_right"] == 28
    assert prediction.metadata["padding_bottom"] == 48
    assert prediction.metadata["point_probability_threshold"] == 0.5
    assert prediction.metadata["native_density_available"] is False
    assert backend.observed is not None
    assert backend.observed.shape == (3, 80, 100)
    assert backend.observed[0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229)


def test_official_eval_dimensions_resize_and_pad_to_128() -> None:
    assert official_eval_dimensions(width=800, height=533) == (800, 533, 896, 640)
    assert official_eval_dimensions(width=4288, height=2848) == (
        2560,
        1700,
        2560,
        1792,
    )


def test_adapter_maps_resized_points_back_and_discards_padding(tmp_path: Path) -> None:
    prediction = _adapter(tmp_path, _ScaledBackend()).predict(
        _sample(tmp_path), retain_native=True
    )

    assert prediction.failure_state is None
    assert prediction.predicted_count == 1.0
    assert prediction.points == ((20.0, 10.0),)
    assert prediction.metadata["discarded_padding_point_count"] == 1
    assert prediction.metadata["scale_x"] == 0.5


def test_brief_marks_cross_domain_research_candidate(tmp_path: Path) -> None:
    brief = _adapter(tmp_path, _Backend()).brief()
    assert brief.family == "point-based APGCC"
    assert brief.rights_status == "PASS_COMMERCIAL_CANDIDATE"
    assert brief.count_derivation == (
        "number of confidence-thresholded in-bounds point predictions"
    )
    assert "commercial checkpoint reuse remains unverified" in (
        brief.checkpoint_rights_status
    )
    assert "maximum side 2560" in brief.preprocessing_policy
    assert "pad right/bottom to a multiple of 128" in brief.preprocessing_policy
    assert brief.parameter_count == 18_000_000
    assert brief.require_full_run_approval() is None


def test_adapter_fails_closed_on_dirty_upstream_or_checkpoint_hash(tmp_path: Path) -> None:
    upstream = tmp_path / "APGCC"
    commit = _git_repo(upstream)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    (upstream / "untracked").write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="working tree must be clean"):
        APGCCAdapter(
            upstream_dir=upstream,
            expected_upstream_commit=commit,
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            device="cuda:0",
            backend=_Backend(),
        )

    (upstream / "untracked").unlink()
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        APGCCAdapter(
            upstream_dir=upstream,
            expected_upstream_commit=commit,
            checkpoint_path=checkpoint,
            checkpoint_sha256="0" * 64,
            device="cuda:0",
            backend=_Backend(),
        )


def test_backend_error_becomes_explicit_failed_prediction(tmp_path: Path) -> None:
    prediction = _adapter(tmp_path, _Backend(error=RuntimeError("boom"))).predict(
        _sample(tmp_path), retain_native=False
    )
    assert prediction.failure_state == "RuntimeError: boom"
    assert prediction.predicted_count is None
    assert prediction.points == ()
    assert "RuntimeError: boom" in prediction.metadata["traceback"]
