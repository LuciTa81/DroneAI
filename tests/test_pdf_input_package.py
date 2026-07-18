from __future__ import annotations

import base64
import csv
import json
from pathlib import Path

import pytest

from droneai.integrity import sha256_file
from droneai.pdf_input_package import build_pdf_input_package
from scripts.build_round1_pdf_inputs import main


MODELS = ("dm-count", "steerer", "pet", "mpcount", "apgcc", "csrnet")
DENSITY_MODELS = {"dm-count", "steerer", "mpcount", "csrnet"}
POINT_MODELS = {"steerer", "pet", "apgcc"}
DENSITY_SUFFIX = "fd9373c130ac1077"
POINT_SUFFIX = "8e0a4dbe072725a1"
SPLIT_SHA = "da1d947aad73d45be573d52a6462fa8022948fba2b1026a3bfff76b11a4b5c67"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _prediction_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(36):
        if index == 0:
            sample_id = "img_common_density"
        elif index == 1:
            sample_id = "img_common_points"
        else:
            sample_id = f"img_{index:04d}"
        band = ("low", "medium", "high")[index // 12]
        normalized_error = 0.05 + index / 100
        if index == 24:
            normalized_error = 0.01
        if index in {25, 26, 27}:
            normalized_error = {25: 0.6, 26: 0.9, 27: 0.7}[index]
        rows.append(
            {
                "sample_id": sample_id,
                "ground_truth_count": 100 + index,
                "predicted_count": 101 + index,
                "normalized_error": normalized_error,
                "density_band": band,
                "spatial_metric_name": "game_l1",
                "spatial_metric_value": 10 + index,
            }
        )
    return rows


def _write_predictions(path: Path) -> None:
    rows = _prediction_rows()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _panel_names(model_id: str) -> list[str]:
    names: list[str] = []
    for index in range(12):
        suffix = f"{index + 1:016x}"
        if index == 0 and model_id in DENSITY_MODELS:
            suffix = DENSITY_SUFFIX
        if index == 1 and model_id in POINT_MODELS:
            suffix = POINT_SUFFIX
        names.append(f"figures/selected-{index + 1:02d}-{suffix}.png")
    return names


def _selections(model_id: str) -> list[dict[str, str]]:
    rows = _prediction_rows()
    sample_indexes = (0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27)
    selections: list[dict[str, str]] = []
    for index, sample_index in enumerate(sample_indexes):
        sample_id = str(rows[sample_index]["sample_id"])
        if index == 0 and model_id not in DENSITY_MODELS:
            sample_id = "img_0002"
        if index == 1 and model_id not in POINT_MODELS:
            sample_id = "img_0003"
        category = "typical"
        if index in {2, 5, 8}:
            category = "best"
        if index in {9, 10, 11}:
            category = "failure"
        selections.append(
            {
                "sample_id": sample_id,
                "category": category,
                "density_band": ("low", "medium", "high")[index // 4],
                "reason": f"fixture {category}",
            }
        )
    return selections


def make_pdf_fixture(tmp_path: Path) -> tuple[Path, Path]:
    results = tmp_path / "results"
    comparison_models = []
    comparison_inputs = []
    for model_id in MODELS:
        run = results / model_id / "run"
        run.mkdir(parents=True)
        predictions = run / "predictions.csv"
        _write_predictions(predictions)
        panel_paths = _panel_names(model_id)
        for panel_path in panel_paths:
            target = run / panel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(PNG + model_id.encode("ascii") + panel_path.encode("ascii"))
        selection = run / "selection_manifest.json"
        _write_json(
            selection,
            {
                "schema_version": 1,
                "selected_count": 12,
                "shortfall": 0,
                "panel_paths": panel_paths,
                "selections": _selections(model_id),
            },
        )
        artifacts = [predictions, selection, *(run / item for item in panel_paths)]
        evidence = run / "evidence-manifest.json"
        _write_json(
            evidence,
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "path": item.relative_to(run).as_posix(),
                        "sha256": sha256_file(item),
                    }
                    for item in artifacts
                ],
            },
        )
        comparison_inputs.append(
            {
                "path": evidence.resolve().as_posix(),
                "sha256": sha256_file(evidence),
            }
        )
        comparison_models.append(
            {
                "model_id": model_id,
                "representative_panels": [
                    f"../{model_id}/run/{item}" for item in panel_paths
                ],
            }
        )

    comparison_dir = tmp_path / "comparison"
    comparison = comparison_dir / "comparison.json"
    _write_json(
        comparison,
        {
            "schema_version": 1,
            "dataset": {
                "dataset_id": "ucf-qnrf-kaggle-apache",
                "split_id": "dm-count-upstream-qnrf-val-smoke-36-v1",
                "split_role": "validation",
                "expected_samples": 36,
                "canonical_split_sha256": SPLIT_SHA,
            },
            "models": comparison_models,
        },
    )
    _write_json(
        comparison_dir / "evidence-manifest.json",
        {
            "schema_version": 1,
            "inputs": comparison_inputs,
            "outputs": [
                {"path": "comparison.json", "sha256": sha256_file(comparison)}
            ],
        },
    )
    return comparison, results


def test_package_selects_high_density_best_and_largest_failure(
    tmp_path: Path,
) -> None:
    comparison, results = make_pdf_fixture(tmp_path)

    output = build_pdf_input_package(comparison, results, tmp_path / "package")

    manifest = json.loads(
        (output / "assets-manifest.json").read_text(encoding="utf-8")
    )
    steerer = manifest["models"]["steerer"]
    assert steerer["lower_error"]["category"] == "best"
    assert steerer["lower_error"]["density_band"] == "high"
    assert steerer["lower_error"]["normalized_error"] == pytest.approx(0.01)
    assert steerer["limitation"]["category"] == "failure"
    assert steerer["limitation"]["normalized_error"] == pytest.approx(0.9)
    assert all(
        sha256_file(output / item["packaged_path"]) == item["packaged_sha256"]
        for item in manifest["assets"]
    )


def test_package_requires_frozen_common_samples(tmp_path: Path) -> None:
    comparison, results = make_pdf_fixture(tmp_path)

    output = build_pdf_input_package(comparison, results, tmp_path / "package")

    manifest = json.loads(
        (output / "assets-manifest.json").read_text(encoding="utf-8")
    )
    groups = manifest["comparisons"]
    assert set(groups["density"]) == DENSITY_MODELS
    assert {item["sample_id"] for item in groups["density"].values()} == {
        groups["density_sample_id"]
    }
    assert set(groups["points"]) == POINT_MODELS
    assert {item["sample_id"] for item in groups["points"].values()} == {
        groups["points_sample_id"]
    }


def test_package_rejects_panel_hash_change(tmp_path: Path) -> None:
    comparison, results = make_pdf_fixture(tmp_path)
    target = results / "dm-count/run" / _panel_names("dm-count")[0]
    target.write_bytes(b"changed")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        build_pdf_input_package(comparison, results, tmp_path / "package")


def test_package_rejects_common_suffix_with_different_sample(tmp_path: Path) -> None:
    comparison, results = make_pdf_fixture(tmp_path)
    selection_path = results / "csrnet/run/selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["selections"][0]["sample_id"] = "img_0002"
    _write_json(selection_path, selection)
    evidence_path = results / "csrnet/run/evidence-manifest.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    next(
        item
        for item in evidence["artifacts"]
        if item["path"] == "selection_manifest.json"
    )["sha256"] = sha256_file(selection_path)
    _write_json(evidence_path, evidence)
    bundle = comparison.parent / "evidence-manifest.json"
    bundle_payload = json.loads(bundle.read_text(encoding="utf-8"))
    next(
        item
        for item in bundle_payload["inputs"]
        if item["path"].endswith("csrnet/run/evidence-manifest.json")
    )["sha256"] = sha256_file(evidence_path)
    _write_json(bundle, bundle_payload)

    with pytest.raises(ValueError, match="common density sample differs"):
        build_pdf_input_package(comparison, results, tmp_path / "package")


def test_cli_returns_structured_error_for_missing_comparison(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code = main(
        [
            "--comparison",
            str(tmp_path / "missing.json"),
            "--results-root",
            str(tmp_path / "results"),
            "--output-dir",
            str(tmp_path / "package"),
        ]
    )

    error = json.loads(capsys.readouterr().err)
    assert code == 2
    assert error["status"] == "FAIL"
    assert error["error_type"] == "ValueError"
    assert not (tmp_path / "package").exists()
