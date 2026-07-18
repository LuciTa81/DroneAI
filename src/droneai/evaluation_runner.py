from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from droneai.comparison_claim import is_ranking_eligible, validate_comparison_claim
from droneai.evaluation_artifacts import (
    REVIEW_BUDGET_BYTES,
    artifact_reference,
    enforce_review_budget,
    write_json,
    write_predictions_csv,
)
from droneai.evaluation_contract import (
    EvaluationSample,
    ModelAdapter,
    NativePrediction,
    ScalarEvaluation,
)
from droneai.evaluation_curation import select_review_samples, selection_manifest
from droneai.evaluation_gate import (
    EvidenceArtifact,
    EvaluationEvidence,
    run_evaluation_gate,
)
from droneai.evaluation_metrics import evaluate_sample, summarize_records
from droneai.evaluation_panels import render_review_panel
from droneai.integrity import is_sha256, sha256_file, verify_artifact_reference
from droneai.model_brief import write_model_brief
from droneai.runtime_probe import collect_environment
from droneai.scoring import StageReport


@dataclass(frozen=True)
class EvaluationProtocol:
    run_id: str
    protocol_id: str
    dataset_id: str
    split_id: str
    split_role: Literal["train", "validation", "test", "smoke"]
    expected_samples: int
    split_verified: bool
    leakage_free: bool
    checkpoint_training_split_status: str
    comparison_scope: str
    checkpoint_split_evidence: str
    sealed_test_access_approved: bool
    require_clean_git: bool
    rights_decision_path: str
    rights_decision_sha256: str
    localization_radius: float
    mae_max: float
    rmse_max: float
    bias_max: float
    band_bias_max: float
    condition_bias_max: float
    latency_max_ms: float
    vram_max_mb: float
    spatial_metric_name: str
    spatial_direction: Literal["minimize", "maximize"]
    spatial_target: float
    required_density_bands: tuple[str, ...]
    density_band_rules: tuple[str, ...]
    required_condition_keys: tuple[str, ...]
    zone_warning_count: float
    zone_critical_count: float

    def __post_init__(self) -> None:
        identity = (
            self.run_id,
            self.protocol_id,
            self.dataset_id,
            self.split_id,
            self.spatial_metric_name,
        )
        if (
            any(not isinstance(value, str) or not value.strip() for value in identity)
            or isinstance(self.expected_samples, bool)
            or not isinstance(self.expected_samples, int)
            or self.expected_samples <= 0
        ):
            raise ValueError(
                "evaluation protocol requires run identity and positive sample count"
            )
        if (
            not isinstance(self.rights_decision_path, str)
            or not self.rights_decision_path.strip()
            or not is_sha256(self.rights_decision_sha256)
        ):
            raise ValueError("evaluation protocol requires a rights decision and SHA-256")
        if self.split_role not in {"train", "validation", "test", "smoke"}:
            raise ValueError("evaluation protocol has an invalid split role")
        if self.spatial_direction not in {"minimize", "maximize"}:
            raise ValueError("evaluation protocol has an invalid spatial direction")
        flags = (
            self.split_verified,
            self.leakage_free,
            self.sealed_test_access_approved,
            self.require_clean_git,
        )
        if any(type(value) is not bool for value in flags):
            raise ValueError("evaluation protocol flags must be boolean values")
        required_groups = (
            self.required_density_bands,
            self.density_band_rules,
            self.required_condition_keys,
        )
        if any(
            not isinstance(values, tuple)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            for values in required_groups
        ):
            raise ValueError(
                "evaluation protocol requires density bands, rules, and condition keys"
            )
        numeric_targets = (
            self.localization_radius,
            self.mae_max,
            self.rmse_max,
            self.bias_max,
            self.band_bias_max,
            self.condition_bias_max,
            self.latency_max_ms,
            self.vram_max_mb,
            self.spatial_target,
            self.zone_warning_count,
            self.zone_critical_count,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in numeric_targets
        ):
            raise ValueError(
                "evaluation protocol targets must be finite and non-negative"
            )
        if self.zone_critical_count < self.zone_warning_count:
            raise ValueError(
                "critical zone threshold must be at least the warning threshold"
            )
        validate_comparison_claim(
            self.checkpoint_training_split_status,
            self.comparison_scope,
            self.checkpoint_split_evidence,
        )


def _spatial_pass(
    records: Sequence[ScalarEvaluation],
    protocol: EvaluationProtocol,
) -> tuple[bool, float | None]:
    successful = [row for row in records if row.predicted_count is not None]
    values = [
        float(row.spatial_metric_value)
        for row in successful
        if row.spatial_metric_name == protocol.spatial_metric_name
        and row.spatial_metric_value is not None
        and math.isfinite(row.spatial_metric_value)
    ]
    if len(values) != len(successful) or not values:
        return False, None
    mean_value = float(np.mean(values))
    if protocol.spatial_direction == "minimize":
        return mean_value <= protocol.spatial_target, mean_value
    return mean_value >= protocol.spatial_target, mean_value


def _group_has_complete_success(coverage: object) -> bool:
    if not isinstance(coverage, dict):
        return False
    expected = coverage.get("expected_samples")
    recorded = coverage.get("recorded_samples")
    successful = coverage.get("successful_samples")
    failures = coverage.get("explicit_failures")
    return (
        isinstance(expected, int)
        and not isinstance(expected, bool)
        and expected > 0
        and recorded == expected
        and successful == expected
        and failures == 0
    )


def _robustness_pass(
    summary: dict[str, object],
    protocol: EvaluationProtocol,
) -> bool:
    band_bias = summary.get("band_signed_bias")
    condition_bias = summary.get("condition_signed_bias")
    band_coverage = summary.get("band_coverage")
    condition_coverage = summary.get("condition_coverage")
    if not all(
        isinstance(value, dict)
        for value in (band_bias, condition_bias, band_coverage, condition_coverage)
    ):
        return False
    expected_samples = summary.get("expected_samples")
    if (
        isinstance(expected_samples, bool)
        or not isinstance(expected_samples, int)
        or expected_samples <= 0
    ):
        return False

    bands_pass = all(
        band in band_bias
        and math.isfinite(float(band_bias[band]))
        and abs(float(band_bias[band])) <= protocol.band_bias_max
        and band in band_coverage
        and _group_has_complete_success(band_coverage[band])
        for band in protocol.required_density_bands
    )
    if bands_pass:
        bands_pass = sum(
            int(band_coverage[band]["expected_samples"])
            for band in protocol.required_density_bands
        ) == expected_samples
    conditions_pass = True
    for key in protocol.required_condition_keys:
        bias_groups = condition_bias.get(key)
        coverage_groups = condition_coverage.get(key)
        if (
            not isinstance(bias_groups, dict)
            or not bias_groups
            or not isinstance(coverage_groups, dict)
            or set(bias_groups) != set(coverage_groups)
        ):
            conditions_pass = False
            break
        if not all(
            math.isfinite(float(value))
            and abs(float(value)) <= protocol.condition_bias_max
            and _group_has_complete_success(coverage_groups[group])
            for group, value in bias_groups.items()
        ):
            conditions_pass = False
            break
        if sum(
            int(group["expected_samples"])
            for group in coverage_groups.values()
        ) != expected_samples:
            conditions_pass = False
            break
    return bands_pass and conditions_pass


def _json_size(payload: object) -> int:
    return len(
        (
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )


def _native_output_fingerprint(prediction: NativePrediction) -> str:
    digest = hashlib.sha256()
    metadata = {
        "sample_id": prediction.sample_id,
        "output_type": prediction.output_type,
        "predicted_count": prediction.predicted_count,
        "points": prediction.points,
        "point_confidences": prediction.point_confidences,
        "confidence": prediction.confidence,
        "failure_state": prediction.failure_state,
        "coordinate_space": prediction.coordinate_space,
        "metadata": prediction.metadata,
    }
    digest.update(
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    if prediction.density is not None:
        density = np.ascontiguousarray(prediction.density)
        digest.update(density.dtype.str.encode("ascii"))
        digest.update(json.dumps(density.shape).encode("ascii"))
        digest.update(density.tobytes(order="C"))
    return digest.hexdigest()


def _panel_filename(panel_index: int, sample_id: str) -> str:
    sample_digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
    return f"selected-{panel_index:02d}-{sample_digest}.png"


def run_evaluation(
    *,
    adapter: ModelAdapter,
    samples: Sequence[EvaluationSample],
    protocol: EvaluationProtocol,
    output_dir: str | Path,
    provenance_artifacts: Sequence[str | Path] = (),
) -> StageReport:
    if protocol.split_role == "test" and not protocol.sealed_test_access_approved:
        raise PermissionError("sealed test evaluation requires explicit approval")

    output = Path(output_dir)
    reserved_outputs = (
        "predictions.csv",
        "metrics.json",
        "selection_manifest.json",
        "model-brief.md",
        "summary.md",
        "sample-manifest.json",
        "native-output-metadata.json",
        "rights-decision.json",
        "environment-summary.json",
        "evidence-manifest.json",
        "score.json",
        "score.md",
        "figures",
    )
    if any((output / name).exists() for name in reserved_outputs):
        raise FileExistsError(
            f"evaluation output already contains report artifacts: {output}"
        )
    output_root = output.resolve()
    provenance_references: list[dict[str, str]] = []
    seen_provenance: set[Path] = set()
    for requested_path in provenance_artifacts:
        provenance_path = Path(requested_path).resolve()
        try:
            provenance_path.relative_to(output_root)
        except ValueError as error:
            raise ValueError(
                "provenance artifacts must stay inside the evaluation output"
            ) from error
        if provenance_path in seen_provenance:
            raise ValueError("provenance artifact paths must be unique")
        if not provenance_path.is_file():
            raise FileNotFoundError(
                f"provenance artifact must be an existing file: {provenance_path}"
            )
        seen_provenance.add(provenance_path)
        provenance_references.append(
            artifact_reference(provenance_path, base_dir=output_root)
        )

    brief = adapter.brief()
    brief.require_full_run_approval()
    checkpoint_path = Path(brief.checkpoint_path)
    checkpoint_verified = (
        checkpoint_path.is_file()
        and sha256_file(checkpoint_path) == brief.checkpoint_sha256
    )
    reviewed_paths_verified = all(Path(path).is_file() for path in brief.reviewed_paths)
    if not checkpoint_verified:
        raise ValueError("checkpoint hash verification failed before inference")
    if not reviewed_paths_verified:
        raise ValueError("reviewed code path verification failed before inference")
    rights_source = Path(protocol.rights_decision_path)
    rights_verified = (
        rights_source.is_file()
        and sha256_file(rights_source) == protocol.rights_decision_sha256
    )
    if not rights_verified:
        raise ValueError("rights decision hash verification failed before inference")
    if len(samples) != protocol.expected_samples:
        raise ValueError("sample count differs from frozen protocol")
    samples_by_id = {sample.sample_id: sample for sample in samples}
    if len(samples_by_id) != len(samples):
        raise ValueError("evaluation sample IDs must be unique")
    if any(
        sample.dataset_id != protocol.dataset_id
        or sample.split_id != protocol.split_id
        for sample in samples
    ):
        raise ValueError("sample dataset or split differs from frozen protocol")
    if any(
        not sample.image_path.is_file()
        or sha256_file(sample.image_path) != sample.source_sha256
        for sample in samples
    ):
        raise ValueError("sample source hash verification failed before inference")
    ordered_samples = tuple(
        sorted(samples, key=lambda sample: sample.sample_id)
    )

    output.mkdir(parents=True, exist_ok=True)
    rights_payload = json.loads(rights_source.read_text(encoding="utf-8"))
    rights_path = write_json(output / "rights-decision.json", rights_payload)
    sample_manifest_path = write_json(
        output / "sample-manifest.json",
        {
            "schema_version": 1,
            "protocol_id": protocol.protocol_id,
            "dataset_id": protocol.dataset_id,
            "split_id": protocol.split_id,
            "split_role": protocol.split_role,
            "expected_samples": protocol.expected_samples,
            "checkpoint_training_split_status": (
                protocol.checkpoint_training_split_status
            ),
            "comparison_scope": protocol.comparison_scope,
            "checkpoint_split_evidence": protocol.checkpoint_split_evidence,
            "ranking_eligible": is_ranking_eligible(
                protocol.checkpoint_training_split_status,
                protocol.comparison_scope,
            ),
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "source_sha256": sample.source_sha256,
                    "annotation_sha256": sample.annotation_sha256,
                }
                for sample in ordered_samples
            ],
        },
    )

    records: list[ScalarEvaluation] = []
    first_pass_fingerprints: dict[str, str] = {}
    native_metadata: list[dict[str, object]] = []
    for sample in ordered_samples:
        prediction = adapter.predict(sample, retain_native=False)
        first_pass_fingerprints[sample.sample_id] = _native_output_fingerprint(
            prediction
        )
        native_metadata.append(
            {"sample_id": sample.sample_id, "metadata": prediction.metadata}
        )
        records.append(
            evaluate_sample(
                sample,
                prediction,
                localization_radius=protocol.localization_radius,
            )
        )

    predictions_path = write_predictions_csv(output / "predictions.csv", records)
    native_metadata_path = write_json(
        output / "native-output-metadata.json",
        {"schema_version": 1, "samples": native_metadata},
    )
    predictions_ref = artifact_reference(predictions_path, base_dir=output)
    summary = summarize_records(records, expected_samples=protocol.expected_samples)
    spatial_pass, spatial_mean = _spatial_pass(records, protocol)
    robustness_pass = _robustness_pass(summary, protocol)
    summary.update(
        {
            "spatial_metric_name": protocol.spatial_metric_name,
            "spatial_direction": protocol.spatial_direction,
            "spatial_target": protocol.spatial_target,
            "spatial_mean": spatial_mean,
            "robustness_pass": robustness_pass,
        }
    )
    metrics_path = write_json(output / "metrics.json", summary)

    selections = select_review_samples(
        records,
        spatial_direction=protocol.spatial_direction,
    )
    panel_paths: list[Path] = []
    records_by_id = {record.sample_id: record for record in records}
    for panel_index, selection in enumerate(selections, start=1):
        sample = samples_by_id[selection.sample_id]
        retained = adapter.predict(sample, retain_native=True)
        first = records_by_id[selection.sample_id]
        if retained.failure_state or first.failure_state:
            raise RuntimeError(
                f"selected sample cannot be rendered: {selection.sample_id}"
            )
        if (
            retained.output_type != first.output_type
            or retained.predicted_count is None
            or first.predicted_count is None
            or abs(float(retained.predicted_count) - float(first.predicted_count)) > 1e-6
        ):
            raise RuntimeError(
                f"selected-sample rerun is not deterministic: {selection.sample_id}"
            )
        if _native_output_fingerprint(retained) != first_pass_fingerprints[
            selection.sample_id
        ]:
            raise RuntimeError(
                "selected-sample native output is not deterministic: "
                f"{selection.sample_id}"
            )
        panel_paths.append(
            render_review_panel(
                sample,
                retained,
                first,
                output
                / "figures"
                / _panel_filename(panel_index, selection.sample_id),
                model_id=brief.model_id,
                checkpoint_sha256=brief.checkpoint_sha256,
                zone_warning_count=protocol.zone_warning_count,
                zone_critical_count=protocol.zone_critical_count,
                category=selection.category,
            )
        )

    selection_payload = selection_manifest(
        selections,
        predictions_sha256=predictions_ref["sha256"],
        spatial_direction=protocol.spatial_direction,
        density_band_rules=protocol.density_band_rules,
        expected_panels=12,
    )
    selection_payload["panel_paths"] = [
        path.relative_to(output).as_posix() for path in panel_paths
    ]
    selection_path = write_json(
        output / "selection_manifest.json",
        selection_payload,
    )
    brief_path = write_model_brief(output / "model-brief.md", brief)
    summary_path = output / "summary.md"
    summary_path.write_text(
        "\n".join(
            (
                f"# Evaluation summary: {protocol.run_id}",
                "",
                f"- Model: `{brief.model_id}`",
                f"- Protocol: `{protocol.protocol_id}`",
                f"- Dataset/split: `{protocol.dataset_id}` / `{protocol.split_id}` ({protocol.split_role})",
                f"- Comparison scope: `{protocol.comparison_scope}`",
                f"- Checkpoint training split: `{protocol.checkpoint_training_split_status}`",
                f"- Ranking eligible: `{is_ranking_eligible(protocol.checkpoint_training_split_status, protocol.comparison_scope)}`",
                f"- Checkpoint split evidence: {protocol.checkpoint_split_evidence}",
                f"- Samples: {summary['successful_samples']} successful, {summary['failures']} explicit failures",
                f"- MAE/RMSE/bias: {float(summary['mae']):.4f} / {float(summary['rmse']):.4f} / {float(summary['signed_bias']):.4f}",
                f"- Spatial: `{protocol.spatial_metric_name}` mean={summary['spatial_mean']} direction={protocol.spatial_direction}",
                f"- Rights scope: `{brief.rights_status}`; see `rights-decision.json`",
                "- This summary does not replace `score.md` or its blockers.",
                "",
            )
        ),
        encoding="utf-8",
    )
    environment = collect_environment(Path.cwd())
    environment_path = write_json(output / "environment-summary.json", environment)

    referenced_paths = [
        sample_manifest_path,
        rights_path,
        predictions_path,
        native_metadata_path,
        metrics_path,
        selection_path,
        brief_path,
        summary_path,
        environment_path,
        *panel_paths,
    ]
    if not all(
        verify_artifact_reference(reference, base_dir=output)[0]
        for reference in provenance_references
    ):
        raise ValueError("provenance artifact changed during evaluation")
    references = [
        *provenance_references,
        *[
            artifact_reference(path, base_dir=output)
            for path in referenced_paths
        ],
    ]
    references_verified = all(
        verify_artifact_reference(reference, base_dir=output)[0]
        for reference in references
    )
    evidence_path = write_json(
        output / "evidence-manifest.json",
        {
            "schema_version": 1,
            "protocol": asdict(protocol),
            "comparison_claim": {
                "checkpoint_training_split_status": (
                    protocol.checkpoint_training_split_status
                ),
                "comparison_scope": protocol.comparison_scope,
                "checkpoint_split_evidence": protocol.checkpoint_split_evidence,
                "ranking_eligible": is_ranking_eligible(
                    protocol.checkpoint_training_split_status,
                    protocol.comparison_scope,
                ),
            },
            "checkpoint": {
                "path": brief.checkpoint_path,
                "sha256": brief.checkpoint_sha256,
                "verified": checkpoint_verified,
            },
            "reviewed_paths": {
                "paths": list(brief.reviewed_paths),
                "verified": reviewed_paths_verified,
            },
            "rights_decision": {
                "source_path": protocol.rights_decision_path,
                "source_sha256": protocol.rights_decision_sha256,
                "verified": rights_verified,
            },
            "artifacts": references,
        },
    )
    references.append(artifact_reference(evidence_path, base_dir=output))
    references_verified = references_verified and all(
        verify_artifact_reference(reference, base_dir=output)[0]
        for reference in references
    )
    gate_artifacts = tuple(EvidenceArtifact(**reference) for reference in references)
    spatial_target = (
        f"mean {protocol.spatial_metric_name} {protocol.spatial_direction} "
        f"target={protocol.spatial_target}"
    )
    spatial_observed = (
        f"mean={spatial_mean}" if spatial_mean is not None else "mean=unavailable"
    )
    robustness_target = (
        f"bands={list(protocol.required_density_bands)} abs_bias<={protocol.band_bias_max}; "
        f"condition_keys={list(protocol.required_condition_keys)} "
        f"abs_bias<={protocol.condition_bias_max}; complete successful group coverage"
    )
    robustness_observed = json.dumps(
        {
            "band_signed_bias": summary["band_signed_bias"],
            "band_coverage": summary["band_coverage"],
            "condition_signed_bias": summary["condition_signed_bias"],
            "condition_coverage": summary["condition_coverage"],
        },
        sort_keys=True,
    )
    evidence = EvaluationEvidence(
        run_id=protocol.run_id,
        expected_samples=protocol.expected_samples,
        recorded_samples=int(summary["recorded_samples"]),
        successful_samples=int(summary["successful_samples"]),
        split_verified=protocol.split_verified,
        leakage_free=protocol.leakage_free,
        mae=float(summary["mae"]),
        mae_max=protocol.mae_max,
        rmse=float(summary["rmse"]),
        rmse_max=protocol.rmse_max,
        absolute_bias=abs(float(summary["signed_bias"])),
        bias_max=protocol.bias_max,
        spatial_pass=spatial_pass,
        robustness_pass=robustness_pass,
        median_latency_ms=float(summary["median_latency_ms"]),
        latency_max_ms=protocol.latency_max_ms,
        peak_vram_mb=float(summary["peak_vram_mb"]),
        vram_max_mb=protocol.vram_max_mb,
        provenance_verified=(
            references_verified
            and checkpoint_verified
            and reviewed_paths_verified
            and bool(environment.get("git_commit"))
            and (
                not protocol.require_clean_git
                or environment.get("git_dirty") is False
            )
        ),
        model_brief_approved=True,
        review_artifacts_verified=(
            len(panel_paths) == 12 and selection_payload["shortfall"] == 0
        ),
        review_budget_ok=True,
        explicit_failures=int(summary["explicit_failures"]),
        artifact_references=gate_artifacts,
        spatial_target=spatial_target,
        spatial_observed=spatial_observed,
        robustness_target=robustness_target,
        robustness_observed=robustness_observed,
    )

    candidate_report = run_evaluation_gate(evidence)
    candidate_markdown = candidate_report.to_markdown()
    projected_score_size = _json_size(candidate_report.to_dict()) + len(
        candidate_markdown.encode("utf-8")
    )
    try:
        enforce_review_budget(
            output,
            limit_bytes=REVIEW_BUDGET_BYTES - projected_score_size,
        )
        review_budget_ok = True
    except ValueError:
        review_budget_ok = False
    report = run_evaluation_gate(
        replace(evidence, review_budget_ok=review_budget_ok)
    )
    write_json(output / "score.json", report.to_dict())
    (output / "score.md").write_text(report.to_markdown(), encoding="utf-8")
    if review_budget_ok:
        enforce_review_budget(output)
    return report
