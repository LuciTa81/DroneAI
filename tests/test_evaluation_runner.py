from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import droneai.evaluation_runner as evaluation_runner
from droneai.evaluation_contract import EvaluationSample, NativePrediction
from droneai.evaluation_runner import EvaluationProtocol, run_evaluation
from droneai.integrity import sha256_file
from droneai.model_brief import ModelBrief


class FixtureAdapter:
    def __init__(self, brief: ModelBrief) -> None:
        self._brief = brief
        self.calls: list[tuple[str, bool]] = []

    def brief(self) -> ModelBrief:
        return self._brief

    def predict(
        self,
        sample: EvaluationSample,
        *,
        retain_native: bool,
    ) -> NativePrediction:
        self.calls.append((sample.sample_id, retain_native))
        index = int(sample.sample_id.rsplit("-", 1)[1])
        error = (-0.003, 0.0, 0.003)[index % 3]
        density = np.asarray(sample.ground_truth_density, dtype=np.float64)
        predicted = density * ((sample.ground_truth_count + error) / sample.ground_truth_count)
        return NativePrediction(
            sample_id=sample.sample_id,
            output_type="density",
            predicted_count=float(predicted.sum()),
            latency_ms=2.0,
            peak_vram_mb=1.0,
            density=predicted,
            metadata={
                "original_width": sample.width,
                "original_height": sample.height,
                "processed_width": sample.width,
                "processed_height": sample.height,
                "coordinate_transform": "identity",
            },
        )


class NondeterministicRerunAdapter(FixtureAdapter):
    def predict(
        self,
        sample: EvaluationSample,
        *,
        retain_native: bool,
    ) -> NativePrediction:
        prediction = super().predict(sample, retain_native=retain_native)
        if not retain_native:
            return prediction
        changed_density = np.asarray(prediction.density) * 1.01
        return replace(
            prediction,
            predicted_count=float(changed_density.sum()),
            density=changed_density,
        )


class SameCountDifferentDensityAdapter(FixtureAdapter):
    def predict(
        self,
        sample: EvaluationSample,
        *,
        retain_native: bool,
    ) -> NativePrediction:
        prediction = super().predict(sample, retain_native=retain_native)
        if not retain_native:
            return prediction
        changed_density = np.asarray(prediction.density).copy()
        delta = min(0.01, float(changed_density[0, 0]) / 2)
        changed_density[0, 0] -= delta
        changed_density[0, 1] += delta
        return replace(prediction, density=changed_density)


def _fixture(tmp_path: Path) -> tuple[FixtureAdapter, list[EvaluationSample], EvaluationProtocol]:
    checkpoint = tmp_path / "fixture-checkpoint.bin"
    checkpoint.write_bytes(b"fixture checkpoint")
    reviewed = tmp_path / "fixture_adapter.py"
    reviewed.write_text("# reviewed fixture adapter\n", encoding="utf-8")
    rights = tmp_path / "fixture-rights.json"
    rights.write_text(
        '{"scope":"research_only","production_approved":false}',
        encoding="utf-8",
    )
    brief = ModelBrief(
        model_id="fixture-density",
        paper="Fixture Density Paper",
        role="density baseline",
        family="density",
        backbone="deterministic fixture",
        parameter_count=1,
        blocks=("density fixture",),
        feature_scales=("native resolution",),
        input_contract="RGB image",
        preprocessing_policy="none",
        coordinate_transform="identity",
        native_output="density map",
        count_derivation="density sum",
        zone_derivation="density integration",
        original_losses=("fixture loss",),
        official_protocol="synthetic validation",
        official_reported_metrics=("fixture MAE 0",),
        strengths=("deterministic",),
        failure_modes=("fixture only",),
        runtime_risks=("none",),
        rights_status="PASS_RESEARCH_ONLY",
        code_rights_status="project fixture",
        dataset_rights_status="project fixture",
        checkpoint_rights_status="project fixture",
        deployment_rights_status="not production evidence",
        upstream_commit="a" * 40,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        reviewed_paths=(str(reviewed),),
        review_status="approved",
    )
    samples: list[EvaluationSample] = []
    for band, count in (("low", 10.0), ("medium", 50.0), ("high", 120.0)):
        for index in range(12):
            sample_id = f"{band}-{index:02d}"
            image_path = tmp_path / "images" / f"{sample_id}.png"
            image_path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (32, 32), (index, index, index)).save(image_path)
            density = np.full((8, 8), count / 64.0, dtype=np.float64)
            samples.append(
                EvaluationSample(
                    sample_id=sample_id,
                    dataset_id="synthetic-fixture-v1",
                    split_id="validation",
                    image_path=image_path,
                    source_sha256=sha256_file(image_path),
                    width=32,
                    height=32,
                    ground_truth_count=count,
                    ground_truth_density=density,
                    condition_tags={"density_band": band, "source": "synthetic"},
                )
            )
    protocol = EvaluationProtocol(
        run_id="fixture-density-validation",
        protocol_id="common-evaluation-fixture-v1",
        dataset_id="synthetic-fixture-v1",
        split_id="validation",
        split_role="validation",
        expected_samples=36,
        split_verified=True,
        leakage_free=True,
        sealed_test_access_approved=False,
        require_clean_git=False,
        rights_decision_path=str(rights),
        rights_decision_sha256=sha256_file(rights),
        localization_radius=4.0,
        mae_max=0.01,
        rmse_max=0.01,
        bias_max=0.01,
        band_bias_max=0.01,
        condition_bias_max=0.01,
        latency_max_ms=10.0,
        vram_max_mb=10.0,
        spatial_metric_name="game_l1",
        spatial_direction="minimize",
        spatial_target=0.01,
        required_density_bands=("low", "medium", "high"),
        density_band_rules=(
            "low: count < 20",
            "medium: 20 <= count < 100",
            "high: count >= 100",
        ),
        required_condition_keys=("source",),
        zone_warning_count=25.0,
        zone_critical_count=75.0,
    )
    return FixtureAdapter(brief), samples, protocol


def test_end_to_end_runner_writes_small_review_bundle(tmp_path: Path) -> None:
    adapter, samples, protocol = _fixture(tmp_path)
    output = tmp_path / "run"

    report = run_evaluation(
        adapter=adapter,
        samples=samples,
        protocol=protocol,
        output_dir=output,
    )

    assert report.status == "PASS_RESEARCH_ONLY"
    for name in (
        "model-brief.md",
        "summary.md",
        "sample-manifest.json",
        "native-output-metadata.json",
        "rights-decision.json",
        "predictions.csv",
        "metrics.json",
        "selection_manifest.json",
        "environment-summary.json",
        "evidence-manifest.json",
        "score.json",
        "score.md",
    ):
        assert (output / name).is_file()
    assert len(list((output / "figures").glob("*.png"))) == 12
    assert not (output / "native-outputs").exists()
    assert adapter.calls.count(("low-00", False)) == 1
    assert sum(retained for _, retained in adapter.calls) == 12
    assert json.loads((output / "score.json").read_text())["score"] == 100
    metadata = json.loads(
        (output / "native-output-metadata.json").read_text(encoding="utf-8")
    )
    assert len(metadata["samples"]) == 36
    assert metadata["samples"][0]["metadata"]["coordinate_transform"] == "identity"


def test_runner_hashes_additional_provenance_inside_output(tmp_path: Path) -> None:
    adapter, samples, protocol = _fixture(tmp_path)
    output = tmp_path / "run-with-split-source"
    output.mkdir()
    split_source = output / "split-source-manifest.json"
    split_source.write_text('{"split_verified":true}\n', encoding="utf-8")

    report = run_evaluation(
        adapter=adapter,
        samples=samples,
        protocol=protocol,
        output_dir=output,
        provenance_artifacts=(split_source,),
    )

    evidence = json.loads(
        (output / "evidence-manifest.json").read_text(encoding="utf-8")
    )
    reference = next(
        item
        for item in evidence["artifacts"]
        if item["path"] == "split-source-manifest.json"
    )
    assert reference["sha256"] == sha256_file(split_source)
    assert report.status == "PASS_RESEARCH_ONLY"


def test_runner_rejects_provenance_outside_output_before_inference(
    tmp_path: Path,
) -> None:
    adapter, samples, protocol = _fixture(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="inside the evaluation output"):
        run_evaluation(
            adapter=adapter,
            samples=samples,
            protocol=protocol,
            output_dir=output,
            provenance_artifacts=(outside,),
        )

    assert adapter.calls == []


def test_sealed_test_requires_explicit_approval_before_output(tmp_path: Path) -> None:
    adapter, samples, protocol = _fixture(tmp_path)
    output = tmp_path / "unapproved-test"

    with pytest.raises(PermissionError, match="sealed test"):
        run_evaluation(
            adapter=adapter,
            samples=samples,
            protocol=replace(protocol, split_role="test"),
            output_dir=output,
        )

    assert not output.exists()
    assert adapter.calls == []


def test_protocol_rejects_missing_or_nonfinite_frozen_targets(tmp_path: Path) -> None:
    _, _, protocol = _fixture(tmp_path)
    with pytest.raises(ValueError, match="finite and non-negative"):
        replace(protocol, mae_max=float("nan"))
    with pytest.raises(ValueError, match="density bands"):
        replace(protocol, required_density_bands=())


def test_protocol_accepts_zero_as_a_frozen_nonnegative_target(tmp_path: Path) -> None:
    _, _, protocol = _fixture(tmp_path)

    frozen = replace(protocol, localization_radius=0.0, latency_max_ms=0.0)

    assert frozen.localization_radius == 0.0
    assert frozen.latency_max_ms == 0.0


def test_required_condition_coverage_must_include_every_sample(tmp_path: Path) -> None:
    adapter, samples, protocol = _fixture(tmp_path)
    samples[0] = replace(
        samples[0],
        condition_tags={"density_band": samples[0].condition_tags["density_band"]},
    )

    report = run_evaluation(
        adapter=adapter,
        samples=samples,
        protocol=protocol,
        output_dir=tmp_path / "missing-condition-run",
    )

    robustness = next(
        check for check in report.checks if check.check_id == "condition.robustness"
    )
    assert not robustness.passed


def test_selected_sample_rerun_must_match_first_pass_count(tmp_path: Path) -> None:
    adapter, samples, protocol = _fixture(tmp_path)
    nondeterministic = NondeterministicRerunAdapter(adapter.brief())
    output = tmp_path / "nondeterministic-run"

    with pytest.raises(RuntimeError, match="rerun is not deterministic"):
        run_evaluation(
            adapter=nondeterministic,
            samples=samples,
            protocol=protocol,
            output_dir=output,
        )

    assert not list((output / "figures").glob("*.png"))
    assert not (output / "score.json").exists()


def test_runner_refuses_existing_report_artifacts_before_inference(
    tmp_path: Path,
) -> None:
    adapter, samples, protocol = _fixture(tmp_path)
    output = tmp_path / "existing-run"
    output.mkdir()
    (output / "score.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already contains report artifacts"):
        run_evaluation(
            adapter=adapter,
            samples=samples,
            protocol=protocol,
            output_dir=output,
        )

    assert adapter.calls == []


def test_checkpoint_hash_failure_blocks_before_inference_and_output(
    tmp_path: Path,
) -> None:
    adapter, samples, protocol = _fixture(tmp_path)
    Path(adapter.brief().checkpoint_path).write_bytes(b"tampered")
    output = tmp_path / "bad-checkpoint-run"

    with pytest.raises(ValueError, match="checkpoint hash"):
        run_evaluation(
            adapter=adapter,
            samples=samples,
            protocol=protocol,
            output_dir=output,
        )

    assert adapter.calls == []
    assert not output.exists()


def test_missing_reviewed_code_blocks_before_inference_and_output(
    tmp_path: Path,
) -> None:
    adapter, samples, protocol = _fixture(tmp_path)
    Path(adapter.brief().reviewed_paths[0]).unlink()
    output = tmp_path / "missing-reviewed-code-run"

    with pytest.raises(ValueError, match="reviewed code path"):
        run_evaluation(
            adapter=adapter,
            samples=samples,
            protocol=protocol,
            output_dir=output,
        )

    assert adapter.calls == []
    assert not output.exists()


def test_selected_rerun_must_match_first_pass_native_density(
    tmp_path: Path,
) -> None:
    adapter, samples, protocol = _fixture(tmp_path)
    changed = SameCountDifferentDensityAdapter(adapter.brief())

    with pytest.raises(RuntimeError, match="native output is not deterministic"):
        run_evaluation(
            adapter=changed,
            samples=samples,
            protocol=protocol,
            output_dir=tmp_path / "changed-native-run",
        )


def test_native_fingerprint_distinguishes_same_count_point_outputs() -> None:
    first = NativePrediction(
        "points",
        "points",
        2.0,
        1.0,
        1.0,
        points=((1.0, 1.0), (2.0, 2.0)),
        point_confidences=(0.9, 0.8),
    )
    changed = replace(first, points=((1.0, 2.0), (2.0, 1.0)))

    assert evaluation_runner._native_output_fingerprint(first) != (
        evaluation_runner._native_output_fingerprint(changed)
    )


def test_sample_order_is_canonicalized_before_artifact_writes(tmp_path: Path) -> None:
    first_adapter, samples, protocol = _fixture(tmp_path)
    second_adapter = FixtureAdapter(first_adapter.brief())
    first_output = tmp_path / "ordered-run"
    second_output = tmp_path / "reversed-run"

    run_evaluation(
        adapter=first_adapter,
        samples=samples,
        protocol=protocol,
        output_dir=first_output,
    )
    run_evaluation(
        adapter=second_adapter,
        samples=list(reversed(samples)),
        protocol=protocol,
        output_dir=second_output,
    )

    assert (first_output / "predictions.csv").read_bytes() == (
        second_output / "predictions.csv"
    ).read_bytes()
    first_manifest = json.loads(
        (first_output / "sample-manifest.json").read_text(encoding="utf-8")
    )
    second_manifest = json.loads(
        (second_output / "sample-manifest.json").read_text(encoding="utf-8")
    )
    assert first_manifest["samples"] == second_manifest["samples"]


def test_panel_filename_cannot_escape_with_untrusted_sample_id() -> None:
    filename = evaluation_runner._panel_filename(
        1,
        "../../../outside\\also-outside",
    )

    assert Path(filename).name == filename
    assert "/" not in filename
    assert "\\" not in filename
    assert ".." not in filename
    assert filename.startswith("selected-01-")
    assert filename.endswith(".png")
