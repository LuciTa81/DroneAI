from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from droneai.evaluation_contract import EvaluationSample, ScalarEvaluation
from droneai.full_panel_export import AcceptedPrediction, PanelExportIdentity
from scripts.export_round2_full_panels import (
    _parser,
    export_lane,
    validate_output_boundary,
)


def _sample(tmp_path: Path, sample_id: str = "img_0001") -> EvaluationSample:
    image = tmp_path / f"{sample_id}.jpg"
    image.write_bytes(b"image")
    return EvaluationSample(
        sample_id=sample_id,
        dataset_id="ucf-qnrf-kaggle-apache",
        split_id="frozen",
        image_path=image,
        source_sha256="e" * 64,
        width=10,
        height=10,
        ground_truth_count=11.0,
    )


def _record(
    sample_id: str = "img_0001",
    predicted_count: float = 10.0,
) -> ScalarEvaluation:
    return ScalarEvaluation(
        sample_id=sample_id,
        ground_truth_count=11.0,
        predicted_count=predicted_count,
        signed_error=predicted_count - 11.0,
        absolute_error=abs(predicted_count - 11.0),
        normalized_error=abs(predicted_count - 11.0) / 11.0,
        density_band="low",
        latency_ms=2.0,
        peak_vram_mb=3.0,
        output_type="density",
        failure_state=None,
    )


def _identity() -> PanelExportIdentity:
    return PanelExportIdentity(
        model_id="steerer",
        dataset_id="ucf-qnrf-kaggle-apache",
        git_commit="a" * 40,
        upstream_commit="b" * 40,
        checkpoint_sha256="c" * 64,
        shared_manifest_sha256="d" * 64,
    )


class FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, sample: EvaluationSample, *, retain_native: bool):
        assert retain_native is True
        self.calls.append(sample.sample_id)
        return SimpleNamespace(sample_id=sample.sample_id)


def _render(*args, **kwargs) -> Path:
    target = Path(args[3])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"panel")
    return target


def test_parser_has_no_training_download_or_checkpoint_override_flags() -> None:
    actions = {action.dest for action in _parser()._actions}

    assert "train" not in actions
    assert "download" not in actions
    assert "checkpoint" not in actions
    assert {
        "config",
        "runtime_config",
        "manifest",
        "model",
        "dataset",
        "result_lane",
        "output_dir",
    } <= actions


def test_output_must_not_be_inside_accepted_result_lane(tmp_path: Path) -> None:
    result_lane = tmp_path / "accepted"
    result_lane.mkdir()

    with pytest.raises(ValueError, match="outside"):
        validate_output_boundary(
            result_lane=result_lane,
            output_dir=result_lane / "export",
        )


def test_export_lane_renders_and_records_one_verified_panel(
    tmp_path: Path,
) -> None:
    sample = _sample(tmp_path)
    adapter = FakeAdapter()

    summary = export_lane(
        adapter=adapter,
        samples=(sample,),
        protocol=SimpleNamespace(
            localization_radius=16.0,
            zone_warning_count=50.0,
            zone_critical_count=150.0,
        ),
        accepted_predictions={
            sample.sample_id: AcceptedPrediction(
                sample.sample_id, 10.0, "density"
            )
        },
        identity=_identity(),
        output_dir=tmp_path / "export",
        evaluate=lambda *_args, **_kwargs: _record(),
        render_panel=_render,
    )

    assert adapter.calls == ["img_0001"]
    assert summary["panel_count"] == 1
    assert summary["count_mismatches"] == 0
    assert len(list((tmp_path / "export" / "panels").glob("*.png"))) == 1


def test_export_lane_blocks_count_mismatch_before_accepting_panel(
    tmp_path: Path,
) -> None:
    sample = _sample(tmp_path)
    rendered: list[bool] = []

    def render(*args, **kwargs):
        rendered.append(True)
        return _render(*args, **kwargs)

    with pytest.raises(ValueError, match="accepted predicted count"):
        export_lane(
            adapter=FakeAdapter(),
            samples=(sample,),
            protocol=SimpleNamespace(
                localization_radius=16.0,
                zone_warning_count=50.0,
                zone_critical_count=150.0,
            ),
            accepted_predictions={
                sample.sample_id: AcceptedPrediction(
                    sample.sample_id, 10.0, "density"
                )
            },
            identity=_identity(),
            output_dir=tmp_path / "export",
            evaluate=lambda *_args, **_kwargs: _record(predicted_count=9.0),
            render_panel=render,
        )

    assert rendered == []


def test_export_lane_resume_does_not_repeat_completed_inference(
    tmp_path: Path,
) -> None:
    sample = _sample(tmp_path)
    accepted = {
        sample.sample_id: AcceptedPrediction(sample.sample_id, 10.0, "density")
    }
    first = FakeAdapter()
    kwargs = {
        "samples": (sample,),
        "protocol": SimpleNamespace(
            localization_radius=16.0,
            zone_warning_count=50.0,
            zone_critical_count=150.0,
        ),
        "accepted_predictions": accepted,
        "identity": _identity(),
        "output_dir": tmp_path / "export",
        "evaluate": lambda *_args, **_kwargs: _record(),
        "render_panel": _render,
    }
    export_lane(adapter=first, **kwargs)
    resumed = FakeAdapter()

    summary = export_lane(adapter=resumed, **kwargs)

    assert first.calls == ["img_0001"]
    assert resumed.calls == []
    assert summary["panel_count"] == 1


def test_export_lane_requires_exact_prediction_sample_identity(
    tmp_path: Path,
) -> None:
    sample = _sample(tmp_path)

    with pytest.raises(ValueError, match="sample identities"):
        export_lane(
            adapter=FakeAdapter(),
            samples=(sample,),
            protocol=SimpleNamespace(
                localization_radius=16.0,
                zone_warning_count=50.0,
                zone_critical_count=150.0,
            ),
            accepted_predictions={
                "different": AcceptedPrediction("different", 10.0, "density")
            },
            identity=_identity(),
            output_dir=tmp_path / "export",
            evaluate=lambda *_args, **_kwargs: _record(),
            render_panel=_render,
        )


def test_company_share_runbook_uses_point_shortlist_manifest() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runbook = (project_root / "docs" / "HOME5090_RUNBOOK.md").read_text(
        encoding="utf-8"
    )
    company_share_section = runbook.split(
        "## Company-share full-panel export",
        maxsplit=1,
    )[1]

    assert (
        "/workspace/data/results/round2-point-reference-v1/sample-manifest.json"
        in company_share_section
    )
    assert (
        "/workspace/data/results/round2-reference-v1/sample-manifest.json"
        not in company_share_section
    )
