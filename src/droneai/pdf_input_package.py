from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from droneai.evaluation_artifacts import write_json
from droneai.integrity import sha256_file, verify_artifact_reference


MODEL_IDS = ("dm-count", "steerer", "pet", "mpcount", "apgcc", "csrnet")
DENSITY_MODELS = ("dm-count", "steerer", "mpcount", "csrnet")
POINT_MODELS = ("steerer", "pet", "apgcc")
DENSITY_SUFFIX = "fd9373c130ac1077"
POINT_SUFFIX = "8e0a4dbe072725a1"
CANONICAL_SPLIT_SHA256 = (
    "da1d947aad73d45be573d52a6462fa8022948fba2b1026a3bfff76b11a4b5c67"
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class SelectedPanel:
    model_id: str
    role: str
    sample_id: str
    category: str
    density_band: str
    ground_truth_count: float
    predicted_count: float
    normalized_error: float
    spatial_metric_name: str
    spatial_metric_value: float | None
    source_path: str
    source_sha256: str
    packaged_path: str


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _finite(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"panel metric is unavailable: {field}") from exc
    if not math.isfinite(number):
        raise ValueError(f"panel metric must be finite: {field}")
    return number


def _optional_finite(value: object, *, field: str) -> float | None:
    if value is None or value == "":
        return None
    return _finite(value, field=field)


def _resolve_panel(root: Path, model_id: str, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"representative panel path is missing: {model_id}")
    pure = PurePosixPath(raw_path)
    if pure.is_absolute() or not pure.parts or pure.parts[0] != "..":
        raise ValueError(f"representative panel path is invalid: {raw_path}")
    remainder = pure.parts[1:]
    if not remainder or ".." in remainder or remainder[0] != model_id:
        raise ValueError(f"representative panel path escapes model run: {raw_path}")
    target = (root / Path(*remainder)).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"representative panel path escapes results root: {raw_path}")
    return target


def _bundle_anchor(comparison_path: Path) -> dict[str, object]:
    manifest = _read_object(
        comparison_path.parent / "evidence-manifest.json",
        label="comparison evidence manifest",
    )
    outputs = manifest.get("outputs")
    inputs = manifest.get("inputs")
    if not isinstance(outputs, list) or not isinstance(inputs, list):
        raise ValueError("comparison evidence manifest lists are invalid")
    matches = [
        item
        for item in outputs
        if isinstance(item, dict) and item.get("path") == comparison_path.name
    ]
    if len(matches) != 1 or matches[0].get("sha256") != sha256_file(comparison_path):
        raise ValueError("comparison JSON hash mismatch")
    return manifest


def _anchored_run_evidence(
    run_dir: Path,
    root: Path,
    bundle_manifest: dict[str, object],
    *,
    model_id: str,
) -> dict[str, object]:
    evidence_path = run_dir / "evidence-manifest.json"
    if not evidence_path.is_file():
        raise ValueError(f"run evidence manifest is missing: {model_id}")
    suffix = evidence_path.relative_to(root).as_posix()
    matches = [
        item
        for item in bundle_manifest["inputs"]
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and item["path"].replace("\\", "/").endswith(suffix)
    ]
    if len(matches) != 1 or matches[0].get("sha256") != sha256_file(evidence_path):
        raise ValueError(f"run evidence manifest hash mismatch: {model_id}")
    evidence = _read_object(evidence_path, label=f"{model_id} run evidence")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"run evidence artifacts are invalid: {model_id}")
    for reference in artifacts:
        verified, message = verify_artifact_reference(reference, base_dir=run_dir)
        if not verified:
            raise ValueError(message)
    return evidence


def _prediction_rows(path: Path, *, model_id: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 36:
        raise ValueError(f"predictions must contain 36 samples: {model_id}")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id or sample_id in by_id:
            raise ValueError(f"prediction sample IDs are invalid: {model_id}")
        by_id[sample_id] = row
    return by_id


def _panel_record(
    *,
    model_id: str,
    role: str,
    panel_path: Path,
    selection: dict[str, object],
    prediction: dict[str, str],
    root: Path,
) -> SelectedPanel:
    sample_id = str(selection.get("sample_id", "")).strip()
    category = str(selection.get("category", "")).strip()
    density_band = str(selection.get("density_band", "")).strip()
    if not _SAFE_NAME.fullmatch(sample_id):
        raise ValueError(f"selected sample ID is invalid: {model_id}")
    if density_band not in {"low", "medium", "high"}:
        raise ValueError(f"selected density band is invalid: {model_id}")
    predicted_band = (prediction.get("density_band") or "").strip()
    if predicted_band and predicted_band != density_band:
        raise ValueError(f"selection density band mismatch: {model_id}.{sample_id}")
    source_hash = sha256_file(panel_path)
    packaged_name = f"{model_id}--{sample_id}--{source_hash[:12]}.png"
    return SelectedPanel(
        model_id=model_id,
        role=role,
        sample_id=sample_id,
        category=category,
        density_band=density_band,
        ground_truth_count=_finite(
            prediction.get("ground_truth_count"), field="ground_truth_count"
        ),
        predicted_count=_finite(
            prediction.get("predicted_count"), field="predicted_count"
        ),
        normalized_error=_finite(
            prediction.get("normalized_error"), field="normalized_error"
        ),
        spatial_metric_name=(prediction.get("spatial_metric_name") or "N/A").strip()
        or "N/A",
        spatial_metric_value=_optional_finite(
            prediction.get("spatial_metric_value"), field="spatial_metric_value"
        ),
        source_path=panel_path.relative_to(root).as_posix(),
        source_sha256=source_hash,
        packaged_path=f"images/{packaged_name}",
    )


def _load_model_panels(
    *,
    model_id: str,
    raw_panel_paths: object,
    root: Path,
    bundle_manifest: dict[str, object],
) -> list[SelectedPanel]:
    if not isinstance(raw_panel_paths, list) or len(raw_panel_paths) != 12:
        raise ValueError(f"model must expose exactly 12 panels: {model_id}")
    panel_paths = [_resolve_panel(root, model_id, item) for item in raw_panel_paths]
    if any(not path.is_file() for path in panel_paths):
        raise ValueError(f"representative panel is missing: {model_id}")
    run_dirs = {path.parent.parent for path in panel_paths}
    if len(run_dirs) != 1:
        raise ValueError(f"representative panels span multiple runs: {model_id}")
    run_dir = run_dirs.pop()
    evidence = _anchored_run_evidence(
        run_dir, root, bundle_manifest, model_id=model_id
    )
    evidence_paths = {
        str(item["path"]).replace("\\", "/")
        for item in evidence["artifacts"]
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    required_paths = {
        "predictions.csv",
        "selection_manifest.json",
        *(path.relative_to(run_dir).as_posix() for path in panel_paths),
    }
    if not required_paths.issubset(evidence_paths):
        raise ValueError(f"run evidence is missing PDF inputs: {model_id}")

    selection_manifest = _read_object(
        run_dir / "selection_manifest.json", label=f"{model_id} selection manifest"
    )
    selected_paths = selection_manifest.get("panel_paths")
    selections = selection_manifest.get("selections")
    if (
        selection_manifest.get("selected_count") != 12
        or selection_manifest.get("shortfall") != 0
        or not isinstance(selected_paths, list)
        or not isinstance(selections, list)
        or len(selected_paths) != 12
        or len(selections) != 12
    ):
        raise ValueError(f"selection manifest is incomplete: {model_id}")
    observed_relative = [path.relative_to(run_dir).as_posix() for path in panel_paths]
    if selected_paths != observed_relative:
        raise ValueError(f"comparison panels differ from selection manifest: {model_id}")

    predictions = _prediction_rows(run_dir / "predictions.csv", model_id=model_id)
    records: list[SelectedPanel] = []
    for panel_path, selection in zip(panel_paths, selections, strict=True):
        if not isinstance(selection, dict):
            raise ValueError(f"selection record is invalid: {model_id}")
        sample_id = str(selection.get("sample_id", "")).strip()
        if sample_id not in predictions:
            raise ValueError(f"selected sample is absent from predictions: {model_id}")
        records.append(
            _panel_record(
                model_id=model_id,
                role="curated",
                panel_path=panel_path,
                selection=selection,
                prediction=predictions[sample_id],
                root=root,
            )
        )
    return records


def _with_role(panel: SelectedPanel, role: str) -> SelectedPanel:
    values = asdict(panel)
    values["role"] = role
    return SelectedPanel(**values)


def _select_model_examples(records: list[SelectedPanel]) -> dict[str, object]:
    band_rank = {"low": 0, "medium": 1, "high": 2}
    best = [item for item in records if item.category == "best"]
    failures = [item for item in records if item.category == "failure"]
    if not best or not failures:
        raise ValueError(f"curated best/failure panels are missing: {records[0].model_id}")
    lower = sorted(
        best,
        key=lambda item: (
            -band_rank[item.density_band],
            item.normalized_error,
            item.sample_id,
        ),
    )[0]
    limitation = sorted(
        failures,
        key=lambda item: (-item.normalized_error, item.sample_id),
    )[0]
    return {
        "lower_error": asdict(_with_role(lower, "lower_error")),
        "limitation": asdict(_with_role(limitation, "limitation")),
        "failure_candidates": [asdict(item) for item in failures],
    }


def _common_group(
    records_by_model: dict[str, list[SelectedPanel]],
    model_ids: tuple[str, ...],
    suffix: str,
    *,
    label: str,
) -> tuple[str, dict[str, dict[str, object]]]:
    selected: dict[str, SelectedPanel] = {}
    for model_id in model_ids:
        matches = [
            item
            for item in records_by_model[model_id]
            if Path(item.source_path).stem.endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(f"common {label} panel is not unique: {model_id}")
        selected[model_id] = _with_role(matches[0], f"{label}_comparison")
    sample_ids = {item.sample_id for item in selected.values()}
    if len(sample_ids) != 1:
        raise ValueError(f"common {label} sample differs across models")
    return sample_ids.pop(), {
        model_id: asdict(panel) for model_id, panel in selected.items()
    }


def _copy_assets(
    temporary: Path,
    root: Path,
    payload: dict[str, object],
) -> list[dict[str, str]]:
    panel_dicts: list[dict[str, object]] = []
    for model in payload["models"].values():
        panel_dicts.extend((model["lower_error"], model["limitation"]))
    panel_dicts.extend(payload["comparisons"]["density"].values())
    panel_dicts.extend(payload["comparisons"]["points"].values())

    assets: dict[str, dict[str, str]] = {}
    for panel in panel_dicts:
        source = root / str(panel["source_path"])
        packaged_path = str(panel["packaged_path"])
        if packaged_path not in assets:
            destination = temporary / packaged_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            packaged_hash = sha256_file(destination)
            if packaged_hash != panel["source_sha256"]:
                raise ValueError(f"copied panel hash mismatch: {packaged_path}")
            assets[packaged_path] = {
                "packaged_path": packaged_path,
                "packaged_sha256": packaged_hash,
                "source_path": str(panel["source_path"]),
                "source_sha256": str(panel["source_sha256"]),
            }
    return [assets[path] for path in sorted(assets)]


def build_pdf_input_package(
    comparison_path: str | Path,
    results_root: str | Path,
    output_dir: str | Path,
) -> Path:
    comparison_source = Path(comparison_path).resolve()
    root = Path(results_root).resolve()
    output = Path(output_dir)
    temporary = output.parent / f".{output.name}.tmp"
    if output.exists() or temporary.exists():
        raise FileExistsError("PDF input output and temporary paths must not exist")
    comparison = _read_object(comparison_source, label="comparison")
    dataset = comparison.get("dataset")
    if not isinstance(dataset, dict) or (
        dataset.get("dataset_id"),
        dataset.get("split_id"),
        dataset.get("split_role"),
        dataset.get("expected_samples"),
        dataset.get("canonical_split_sha256"),
    ) != (
        "ucf-qnrf-kaggle-apache",
        "dm-count-upstream-qnrf-val-smoke-36-v1",
        "validation",
        36,
        CANONICAL_SPLIT_SHA256,
    ):
        raise ValueError("comparison does not use the frozen validation split")
    raw_models = comparison.get("models")
    if not isinstance(raw_models, list) or tuple(
        item.get("model_id") if isinstance(item, dict) else None for item in raw_models
    ) != MODEL_IDS:
        raise ValueError("comparison model order does not match the frozen six models")

    bundle_manifest = _bundle_anchor(comparison_source)
    records_by_model: dict[str, list[SelectedPanel]] = {}
    model_payload: dict[str, object] = {}
    for model in raw_models:
        model_id = str(model["model_id"])
        records = _load_model_panels(
            model_id=model_id,
            raw_panel_paths=model.get("representative_panels"),
            root=root,
            bundle_manifest=bundle_manifest,
        )
        records_by_model[model_id] = records
        model_payload[model_id] = _select_model_examples(records)

    density_sample, density_group = _common_group(
        records_by_model,
        DENSITY_MODELS,
        DENSITY_SUFFIX,
        label="density",
    )
    points_sample, points_group = _common_group(
        records_by_model,
        POINT_MODELS,
        POINT_SUFFIX,
        label="points",
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "comparison_sha256": sha256_file(comparison_source),
        "dataset": dataset,
        "models": model_payload,
        "comparisons": {
            "density_sample_id": density_sample,
            "density": density_group,
            "points_sample_id": points_sample,
            "points": points_group,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        payload["assets"] = _copy_assets(temporary, root, payload)
        write_json(temporary / "assets-manifest.json", payload)
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output

