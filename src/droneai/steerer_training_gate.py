"""Evidence-backed 100-point gates for STEERER training stages.

The gate deliberately awards no production status.  It only establishes whether
the supplied non-Test training evidence is a commercial candidate for review.
"""

from __future__ import annotations

import copy
import math
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from droneai.dm_count_adapter import _git_head, _git_status
from droneai.integrity import is_sha256, sha256_file
from droneai.scoring import CheckResult, StageReport, score_stage
from droneai.steerer_training_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    ResumeExpectations,
    load_training_checkpoint,
    verify_resume,
)
from droneai.steerer_training_profile import (
    SteererTrainingProfile,
    load_training_profile,
)
from droneai.steerer_training_runner import _canonical_config_sha256
from droneai.steerer_training_upstream import audit_upstream, synthesize_official_config
from droneai.ucf_qnrf import density_band


STAGE_THRESHOLDS = {"T0": 70, "T1": 75, "T5": 80, "T50": 85}
SUCCESS_STATUS = "PASS_COMMERCIAL_CANDIDATE"
APPROVED_UPSTREAM_COMMIT = "5b1854dbc2d280f2326d67c65515d8baf9083810"

REQUIRED_ARTIFACTS = frozenset(
    {
        "environment",
        "dataset_inventory",
        "train_split",
        "validation_split",
        "backbone",
        "input_checkpoint",
        "output_checkpoint",
        "config",
        "rights_decision",
    }
)
REQUIRED_T1_METRICS = frozenset(
    {
        "mae",
        "rmse",
        "signed_bias",
        "reference_mape",
        "game_l1",
        "quadrant_zone_mae",
        "localization_precision_16px",
        "localization_recall_16px",
        "localization_f1_16px",
        "median_latency_ms",
        "p95_latency_ms",
        "fps",
        "peak_vram_mb",
        "density_sum_count_difference",
        "train_loss",
        "validation_loss",
        "training_window_improvement",
    }
)
_T0_METRICS = frozenset({"train_loss"})
_COMMIT = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_AUTHORITY_TOKEN = object()
_CANONICAL_PROFILE = Path(
    "configs/training/steerer_ucf_qnrf_imagenet.home5090.json"
)
_STAGE_EPOCHS = {"T0": 0, "T1": 1, "T5": 5, "T50": 50}


@dataclass(frozen=True)
class AuthoritativeTrainingInputs:
    """Concrete paths needed to recompute one stage's complete trust boundary."""

    stage: str
    run_id: str
    profile_path: Path
    project_repo_root: Path
    upstream_dir: Path
    processed_root: Path
    backbone_path: Path
    checkpoint_path: Path
    checkpoint_manifest_path: Path
    environment_path: Path
    metrics_path: Path


class AuthoritativeTrainingEvidence:
    """Opaque evidence created only after authoritative disk verification."""

    __slots__ = ("_evidence", "_token")

    def __init__(self, evidence: Mapping[str, Any], *, _token: object) -> None:
        if _token is not _AUTHORITY_TOKEN:
            raise TypeError(
                "AuthoritativeTrainingEvidence must come from authoritative verification"
            )
        self._evidence = copy.deepcopy(dict(evidence))
        self._token = _token

    @property
    def evidence(self) -> dict[str, Any]:
        """Return a defensive copy so verified facts cannot be changed post-check."""

        return copy.deepcopy(self._evidence)


def _strict_json(path: Path, *, name: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains forbidden non-finite constant {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid strict JSON: {path}") from error


def _regular_file(path: str | Path, *, name: str) -> Path:
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")
    try:
        resolved = supplied.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{name} is required: {supplied}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")
    return resolved


def _directory(path: str | Path, *, name: str) -> Path:
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError(f"{name} must be a regular directory")
    try:
        resolved = supplied.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{name} is required: {supplied}") from error
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{name} must be a regular directory")
    return resolved


def _inside(path: Path, root: Path, *, name: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must stay inside {root}") from error
    return path


def _exact_mapping(value: Any, *, name: str, keys: frozenset[str]) -> Mapping[str, Any]:
    return _mapping(value, name=name, keys=keys)


def _git_toplevel(root: Path) -> Path:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("project repo root must be a Git worktree") from error
    return Path(completed.stdout.strip()).resolve(strict=True)


def _verify_project_and_profile(
    inputs: AuthoritativeTrainingInputs,
    *,
    profile_override: SteererTrainingProfile | None,
) -> tuple[Path, Path, str, str, SteererTrainingProfile]:
    project_root = _directory(inputs.project_repo_root, name="project repo root")
    if _git_toplevel(project_root) != project_root:
        raise ValueError("project repo root must be the Git worktree root")
    profile_path = _regular_file(inputs.profile_path, name="canonical training profile")
    expected_profile = (project_root / _CANONICAL_PROFILE).resolve(strict=False)
    if profile_path != expected_profile:
        raise ValueError("profile must be the checked-in canonical Task 1 profile")
    project_commit = _git_head(project_root)
    if _git_status(project_root):
        raise ValueError("project working tree must be clean")
    loaded = load_training_profile(profile_path)
    if profile_override is not None:
        if not isinstance(profile_override, SteererTrainingProfile):
            raise TypeError("_profile_override must be a validated training profile")
        profile = profile_override
    else:
        profile = loaded
    return project_root, profile_path, project_commit, sha256_file(profile_path), profile


def _verify_backbone(profile: SteererTrainingProfile, supplied: Path) -> Path:
    backbone = _regular_file(supplied, name="ImageNet backbone")
    expected = profile.imagenet_backbone.path.resolve(strict=False)
    if backbone != expected or backbone.name != profile.imagenet_backbone.filename:
        raise ValueError("backbone must use the profile canonical path and filename")
    if backbone.stat().st_size != profile.imagenet_backbone.byte_size:
        raise ValueError("backbone byte size does not match the profile")
    if sha256_file(backbone) != profile.imagenet_backbone.sha256.lower():
        raise ValueError("backbone SHA-256 does not match the profile")
    return backbone


def _jsonl_rows(path: Path, *, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite {value}")
            ))
            if not isinstance(row, dict):
                raise ValueError
            rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is invalid") from error
    return rows


def _sha256_lines(lines: list[str]) -> str:
    return __import__("hashlib").sha256("\n".join(lines).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _ProcessedAuthority:
    train_path: Path
    validation_path: Path
    inventory_path: Path
    split_manifest_path: Path
    test_sealed_path: Path
    train_sha256: str
    validation_sha256: str
    inventory_sha256: str
    population_sha256: str
    split_manifest_sha256: str
    test_sealed_sha256: str


def _verify_processed_dataset(
    profile: SteererTrainingProfile, supplied_root: Path
) -> _ProcessedAuthority:
    root = _directory(supplied_root, name="processed split root")
    if root != profile.processed_root.resolve(strict=False):
        raise ValueError("processed root must match the authoritative profile")
    train_path = _regular_file(root / "train.txt", name="train split")
    validation_path = _regular_file(root / "val.txt", name="validation split")
    split_manifest_path = _regular_file(
        root / "split-manifest.json", name="split manifest"
    )
    test_sealed_path = _regular_file(root / "test-sealed.json", name="sealed Test marker")
    input_inventory_path = _regular_file(
        root / "manifests" / "input-inventory.jsonl", name="input inventory"
    )
    inventory_path = _regular_file(
        root / "manifests" / "output-inventory.jsonl", name="dataset inventory"
    )
    training_manifest_path = _regular_file(
        root / "manifests" / "training-manifest.json", name="training manifest"
    )

    train_ids = train_path.read_text(encoding="utf-8").splitlines()
    validation_ids = validation_path.read_text(encoding="utf-8").splitlines()
    if (len(train_ids), len(validation_ids)) != (
        profile.train_count,
        profile.validation_count,
    ):
        raise ValueError("prepared split must contain exactly 961/240 samples")
    if (
        any(not sample_id for sample_id in train_ids + validation_ids)
        or len(set(train_ids)) != len(train_ids)
        or len(set(validation_ids)) != len(validation_ids)
        or set(train_ids).intersection(validation_ids)
        or len(set(train_ids).union(validation_ids)) != profile.dataset_population
    ):
        raise ValueError("prepared split must be disjoint and complete for 1201 samples")
    train_sha = sha256_file(train_path)
    validation_sha = sha256_file(validation_path)

    input_rows = _jsonl_rows(input_inventory_path, name="input inventory")
    output_rows = _jsonl_rows(inventory_path, name="dataset inventory")
    input_keys = frozenset(
        {"annotation_sha256", "count", "height", "image_sha256", "sample_id", "width"}
    )
    output_keys = frozenset(
        {"count", "height", "image_sha256", "json_sha256", "sample_id", "width"}
    )
    if len(input_rows) != profile.dataset_population or len(output_rows) != profile.dataset_population:
        raise ValueError("dataset inventories must each contain 1201 rows")
    inputs_by_id: dict[str, Mapping[str, Any]] = {}
    outputs_by_id: dict[str, Mapping[str, Any]] = {}
    population_lines: list[str] = []
    band_by_id: dict[str, str] = {}
    for row in input_rows:
        parsed = _exact_mapping(row, name="input inventory row", keys=input_keys)
        sample_id = _nonempty_string(parsed["sample_id"], name="input sample_id")
        count = _integer(parsed["count"], name="input count")
        if sample_id in inputs_by_id or count < 0:
            raise ValueError("input inventory contains duplicate or invalid rows")
        for field in ("annotation_sha256", "image_sha256"):
            if not is_sha256(parsed[field]):
                raise ValueError(f"input inventory {field} must be a SHA-256")
        band = density_band(count)
        band_by_id[sample_id] = band
        population_lines.append(
            f"{sample_id},{parsed['image_sha256']},{parsed['annotation_sha256']},{count},{band}"
        )
        inputs_by_id[sample_id] = parsed
    for row in output_rows:
        parsed = _exact_mapping(row, name="output inventory row", keys=output_keys)
        sample_id = _nonempty_string(parsed["sample_id"], name="output sample_id")
        if sample_id in outputs_by_id:
            raise ValueError("output inventory contains duplicate rows")
        source = inputs_by_id.get(sample_id)
        if source is None:
            raise ValueError("input and output inventories have different populations")
        for field in ("image_sha256", "json_sha256"):
            if not is_sha256(parsed[field]):
                raise ValueError(f"output inventory {field} must be a SHA-256")
        for field in ("count", "height", "image_sha256", "width"):
            if parsed[field] != source[field]:
                raise ValueError("input and output inventory lineage mismatch")
        image_path = _regular_file(root / "images" / f"{sample_id}.jpg", name="source image")
        annotation_path = _regular_file(
            root / "jsons" / f"{sample_id}.json", name="converted annotation"
        )
        if sha256_file(image_path) != str(parsed["image_sha256"]).lower():
            raise ValueError("source image hash mismatch")
        if sha256_file(annotation_path) != str(parsed["json_sha256"]).lower():
            raise ValueError("converted annotation hash mismatch")
        annotation = _strict_json(annotation_path, name="converted annotation")
        if (
            not isinstance(annotation, Mapping)
            or annotation.get("sample_id") != sample_id
            or annotation.get("source_partition") != "official_train_only"
            or annotation.get("source_sha256") != parsed["image_sha256"]
            or annotation.get("annotation_sha256") != source["annotation_sha256"]
            or annotation.get("human_num") != source["count"]
        ):
            raise ValueError("converted annotation source lineage mismatch")
        outputs_by_id[sample_id] = parsed
    population = set(train_ids).union(validation_ids)
    if set(inputs_by_id) != population or set(outputs_by_id) != population:
        raise ValueError("prepared split must bind the complete inventory population")
    population_sha = _sha256_lines(sorted(population_lines))

    split = _strict_json(split_manifest_path, name="split manifest")
    split = _exact_mapping(
        split,
        name="split manifest",
        keys=frozenset(
            {"seed", "train_ids", "validation_ids", "band_counts", "population_sha256", "train_sha256", "validation_sha256"}
        ),
    )
    expected_band_counts: dict[str, dict[str, int]] = {}
    for band in sorted(set(band_by_id.values())):
        expected_band_counts[band] = {
            "population": sum(value == band for value in band_by_id.values()),
            "train": sum(band_by_id[sample_id] == band for sample_id in train_ids),
            "validation": sum(
                band_by_id[sample_id] == band for sample_id in validation_ids
            ),
        }
    if (
        split["seed"] != profile.seed
        or split["train_ids"] != train_ids
        or split["validation_ids"] != validation_ids
        or split["band_counts"] != expected_band_counts
        or split["population_sha256"] != population_sha
        or split["train_sha256"] != train_sha
        or split["validation_sha256"] != validation_sha
    ):
        raise ValueError("split manifest hashes, counts, or population binding mismatch")

    sealed = _strict_json(test_sealed_path, name="sealed Test marker")
    if sealed != {"role": "sealed", "sample_count": 334, "accessed": False}:
        raise ValueError("official Test must remain sealed and unaccessed")
    training = _strict_json(training_manifest_path, name="training manifest")
    training = _exact_mapping(
        training,
        name="training manifest",
        keys=frozenset(
            {"converted_sample_count", "count_mismatches", "input_inventory_sha256", "out_of_bounds_boxes", "out_of_bounds_points", "output_inventory_sha256", "sample_count", "source_partition", "train_count", "validation_count"}
        ),
    )
    if (
        training["converted_sample_count"] != profile.dataset_population
        or training["sample_count"] != profile.dataset_population
        or training["train_count"] != profile.train_count
        or training["validation_count"] != profile.validation_count
        or training["count_mismatches"] != 0
        or training["out_of_bounds_boxes"] != 0
        or training["out_of_bounds_points"] != 0
        or training["source_partition"] != "official_train_only"
        or training["input_inventory_sha256"] != sha256_file(input_inventory_path)
        or training["output_inventory_sha256"] != sha256_file(inventory_path)
    ):
        raise ValueError("training manifest source/inventory binding mismatch")
    return _ProcessedAuthority(
        train_path=train_path,
        validation_path=validation_path,
        inventory_path=inventory_path,
        split_manifest_path=split_manifest_path,
        test_sealed_path=test_sealed_path,
        train_sha256=train_sha,
        validation_sha256=validation_sha,
        inventory_sha256=sha256_file(inventory_path),
        population_sha256=population_sha,
        split_manifest_sha256=sha256_file(split_manifest_path),
        test_sealed_sha256=sha256_file(test_sealed_path),
    )


@dataclass(frozen=True)
class _CheckpointAuthority:
    checkpoint_path: Path
    checkpoint_manifest_path: Path
    checkpoint_sha256: str
    environment_sha256: str
    state: Mapping[str, Any]


def _verify_checkpoint_manifest(
    inputs: AuthoritativeTrainingInputs,
    profile: SteererTrainingProfile,
    *,
    torch_module: Any | None,
) -> _CheckpointAuthority:
    checkpoint_path = _regular_file(inputs.checkpoint_path, name="training checkpoint")
    checkpoint_dir = (
        profile.checkpoint_root.resolve(strict=False) / inputs.run_id
    ).resolve(strict=False)
    if checkpoint_path.parent != checkpoint_dir:
        raise ValueError("checkpoint must be inside the approved current run directory")
    manifest_path = _regular_file(
        inputs.checkpoint_manifest_path, name="checkpoint policy manifest"
    )
    if manifest_path != checkpoint_dir / "checkpoint-manifest.json":
        raise ValueError("checkpoint policy manifest must be the current run manifest")
    manifest = _strict_json(manifest_path, name="checkpoint policy manifest")
    manifest = _exact_mapping(
        manifest,
        name="checkpoint policy manifest",
        keys=frozenset({"schema_version", "run_id", "checkpoints"}),
    )
    entries = manifest["checkpoints"]
    if (
        manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION
        or manifest["run_id"] != inputs.run_id
        or not isinstance(entries, list)
    ):
        raise ValueError("checkpoint policy manifest lineage is invalid")
    entry_keys = frozenset(
        {"filename", "sha256", "byte_count", "epoch", "metric_reason", "parent_checkpoint_sha256", "environment_manifest_sha256"}
    )
    normalized_entries: list[Mapping[str, Any]] = []
    for raw_entry in entries:
        entry = _exact_mapping(
            raw_entry, name="checkpoint policy entry", keys=entry_keys
        )
        parent_sha = entry["parent_checkpoint_sha256"]
        if (
            not isinstance(entry["filename"], str)
            or not entry["filename"]
            or not is_sha256(entry["sha256"])
            or not is_sha256(entry["environment_manifest_sha256"])
            or (parent_sha is not None and not is_sha256(parent_sha))
            or not isinstance(entry["byte_count"], int)
            or isinstance(entry["byte_count"], bool)
            or entry["byte_count"] < 0
            or not isinstance(entry["epoch"], int)
            or isinstance(entry["epoch"], bool)
            or entry["epoch"] < 0
            or not isinstance(entry["metric_reason"], str)
            or not entry["metric_reason"]
        ):
            raise ValueError("checkpoint policy entry is invalid")
        normalized_entries.append(entry)
    filenames = [str(entry["filename"]) for entry in normalized_entries]
    if len(filenames) != len(set(filenames)):
        raise ValueError("checkpoint policy entry filenames must be unique")
    matches = [
        entry
        for entry in normalized_entries
        if entry["filename"] == checkpoint_path.name
    ]
    if len(matches) != 1:
        raise ValueError("checkpoint must have one unique policy manifest entry")
    entry = matches[0]
    expected_epoch = _STAGE_EPOCHS[inputs.stage]
    parent_sha = entry["parent_checkpoint_sha256"]
    if (
        not is_sha256(entry["sha256"])
        or not is_sha256(entry["environment_manifest_sha256"])
        or (parent_sha is not None and not is_sha256(parent_sha))
        or entry["byte_count"] != checkpoint_path.stat().st_size
        or entry["epoch"] != expected_epoch
        or not isinstance(entry["metric_reason"], str)
        or not entry["metric_reason"]
    ):
        raise ValueError("checkpoint policy manifest entry is invalid")
    observed_sha = sha256_file(checkpoint_path)
    if observed_sha != str(entry["sha256"]).lower():
        raise ValueError("checkpoint policy manifest SHA-256 mismatch")
    state = load_training_checkpoint(
        checkpoint_path,
        expected_sha256=observed_sha,
        torch_module=torch_module,
    )
    if (
        state["run_id"] != inputs.run_id
        or state["stage"] != inputs.stage
        or state["epoch"] != expected_epoch
        or state["environment_manifest_sha256"].lower()
        != str(entry["environment_manifest_sha256"]).lower()
    ):
        raise ValueError("checkpoint stage/run/environment lineage mismatch")
    return _CheckpointAuthority(
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_path=manifest_path,
        checkpoint_sha256=observed_sha,
        environment_sha256=str(entry["environment_manifest_sha256"]).lower(),
        state=state,
    )


def _environment_sections(payload: Any) -> tuple[Mapping[str, Any], ...]:
    root = _exact_mapping(
        payload,
        name="immutable environment evidence",
        keys=frozenset(
            {"schema_version", "run_id", "stage", "project", "upstream", "dataset", "initialization", "training", "runtime", "batch", "metrics"}
        ),
    )
    if root["schema_version"] != 2:
        raise ValueError("immutable environment evidence schema_version must be 2")
    specs = (
        ("project", frozenset({"git_commit", "profile_sha256"})),
        ("upstream", frozenset({"origin_url", "commit", "clean", "license_sha256", "config_sha256"})),
        ("dataset", frozenset({"processed_root", "source_partition", "official_test_accessed", "train_sha256", "validation_sha256", "dataset_inventory_sha256", "population_sha256", "test_sealed_sha256", "split_manifest_sha256"})),
        ("initialization", frozenset({"mode", "model_checkpoint_loaded", "backbone_path", "backbone_sha256", "backbone_byte_size"})),
        ("training", frozenset({"config_sha256"})),
        ("runtime", frozenset({"container_image_digest", "python", "torch", "cuda", "gpu"})),
        ("batch", frozenset({"physical_batch", "accumulation_steps", "effective_batch"})),
        ("metrics", frozenset({"path", "sha256"})),
    )
    sections = tuple(
        _exact_mapping(root[name], name=f"environment.{name}", keys=keys)
        for name, keys in specs
    )
    return (root, *sections)


def verify_authoritative_training_evidence(
    inputs: AuthoritativeTrainingInputs,
    *,
    torch_module: Any | None = None,
    _profile_override: SteererTrainingProfile | None = None,
) -> AuthoritativeTrainingEvidence:
    """Recompute every trust fact before creating opaque scoreable evidence.

    ``_profile_override`` exists only for local tests whose Windows filesystem cannot
    represent the canonical home5090 storage paths.  The production CLI never exposes
    or supplies it.
    """

    if not isinstance(inputs, AuthoritativeTrainingInputs):
        raise TypeError("AuthoritativeTrainingInputs are required")
    if inputs.stage not in STAGE_THRESHOLDS:
        raise ValueError("stage must be T0, T1, T5, or T50")
    if not isinstance(inputs.run_id, str) or re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", inputs.run_id
    ) is None:
        raise ValueError("run_id must be one safe path component")

    (
        project_root,
        profile_path,
        project_commit,
        profile_sha256,
        profile,
    ) = _verify_project_and_profile(inputs, profile_override=_profile_override)
    del project_root
    upstream_root = _directory(inputs.upstream_dir, name="pinned upstream directory")
    upstream = audit_upstream(profile, upstream_root)
    backbone = _verify_backbone(profile, inputs.backbone_path)
    processed = _verify_processed_dataset(profile, inputs.processed_root)
    checkpoint = _verify_checkpoint_manifest(
        inputs, profile, torch_module=torch_module
    )

    environment = _regular_file(
        inputs.environment_path, name="checkpoint-bound immutable environment"
    )
    result_run_root = (profile.result_root.resolve(strict=False) / inputs.run_id).resolve(
        strict=False
    )
    if (
        environment.parent != result_run_root
        or environment.name != f"environment.{checkpoint.environment_sha256}.json"
    ):
        raise ValueError("environment path must be content-addressed by the checkpoint hash")
    if sha256_file(environment) != checkpoint.environment_sha256:
        raise ValueError("environment evidence SHA-256 mismatch")
    (
        environment_root,
        project_section,
        upstream_section,
        dataset_section,
        initialization_section,
        training_section,
        runtime_section,
        batch_section,
        metrics_section,
    ) = _environment_sections(_strict_json(environment, name="immutable environment evidence"))
    if environment_root["run_id"] != inputs.run_id or environment_root["stage"] != inputs.stage:
        raise ValueError("environment stage/run lineage mismatch")
    if project_section != {
        "git_commit": project_commit,
        "profile_sha256": profile_sha256,
    }:
        raise ValueError("environment project Git/profile audit mismatch")
    if upstream_section != {
        "origin_url": upstream.origin_url,
        "commit": upstream.commit,
        "clean": True,
        "license_sha256": upstream.license_sha256,
        "config_sha256": upstream.config_sha256,
    }:
        raise ValueError("environment pinned upstream audit mismatch")
    if dataset_section != {
        "processed_root": str(profile.processed_root.resolve(strict=False)),
        "source_partition": "official_train_only",
        "official_test_accessed": False,
        "train_sha256": processed.train_sha256,
        "validation_sha256": processed.validation_sha256,
        "dataset_inventory_sha256": processed.inventory_sha256,
        "population_sha256": processed.population_sha256,
        "test_sealed_sha256": processed.test_sealed_sha256,
        "split_manifest_sha256": processed.split_manifest_sha256,
    }:
        raise ValueError("environment split/source/Test audit mismatch")
    if initialization_section != {
        "mode": "imagenet_backbone_only",
        "model_checkpoint_loaded": False,
        "backbone_path": str(backbone),
        "backbone_sha256": profile.imagenet_backbone.sha256.lower(),
        "backbone_byte_size": profile.imagenet_backbone.byte_size,
    }:
        raise ValueError("environment initialization/backbone audit mismatch")
    container_digest = runtime_section["container_image_digest"]
    if (
        not isinstance(container_digest, str)
        or not container_digest.startswith("sha256:")
        or not is_sha256(container_digest.removeprefix("sha256:"))
    ):
        raise ValueError("environment container image digest is invalid")
    for field in ("python", "torch", "cuda", "gpu"):
        _nonempty_string(runtime_section[field], name=f"environment.runtime.{field}")

    physical_batch = _integer(batch_section["physical_batch"], name="physical batch")
    accumulation_steps = _integer(
        batch_section["accumulation_steps"], name="accumulation steps"
    )
    if batch_section["effective_batch"] != physical_batch * accumulation_steps:
        raise ValueError("environment effective batch audit mismatch")
    synthesized = synthesize_official_config(
        profile=profile,
        upstream_dir=upstream_root,
        processed_root=profile.processed_root,
        backbone_path=backbone,
        stage=inputs.stage,
        physical_batch=physical_batch,
        accumulation_steps=accumulation_steps,
        run_id=inputs.run_id,
        resume_path=(
            checkpoint.checkpoint_path if inputs.stage in {"T5", "T50"} else None
        ),
    )
    config_sha256 = _canonical_config_sha256(synthesized)
    if training_section != {"config_sha256": config_sha256}:
        raise ValueError("environment synthesized config audit mismatch")

    metrics_path = _regular_file(inputs.metrics_path, name="runner metrics artifact")
    if metrics_path.parent != result_run_root:
        raise ValueError("metrics artifact must stay inside the approved result run")
    if metrics_section != {"path": str(metrics_path), "sha256": sha256_file(metrics_path)}:
        raise ValueError("metrics SHA-256 is not checkpoint-bound by the environment")
    metrics_payload = _strict_json(metrics_path, name="runner metrics artifact")
    metrics_payload = _exact_mapping(
        metrics_payload,
        name="runner metrics artifact",
        keys=frozenset({"schema_version", "run_id", "stage", "metrics"}),
    )
    if (
        metrics_payload["schema_version"] != 1
        or metrics_payload["run_id"] != inputs.run_id
        or metrics_payload["stage"] != inputs.stage
        or not isinstance(metrics_payload["metrics"], Mapping)
    ):
        raise ValueError("runner metrics stage/run lineage mismatch")

    expectations = ResumeExpectations(
        run_id=inputs.run_id,
        config_sha256=config_sha256,
        split_sha256s={
            "train": processed.train_sha256,
            "validation": processed.validation_sha256,
        },
        dataset_inventory_sha256=processed.inventory_sha256,
        backbone_sha256=profile.imagenet_backbone.sha256,
        upstream_commit=upstream.commit,
        environment_manifest_sha256=checkpoint.environment_sha256,
    )
    verify_resume(
        checkpoint.checkpoint_path,
        expected_sha256=checkpoint.checkpoint_sha256,
        expectations=expectations,
        torch_module=torch_module,
    )
    artifact = lambda path: {"path": str(path), "sha256": sha256_file(path)}
    evidence = {
        "schema_version": 1,
        "run_id": inputs.run_id,
        "stage": inputs.stage,
        "provenance": {
            "git_commit": project_commit,
            "upstream_commit": upstream.commit,
            "container_image_digest": runtime_section["container_image_digest"],
            "python": runtime_section["python"],
            "torch": runtime_section["torch"],
            "cuda": runtime_section["cuda"],
            "gpu": runtime_section["gpu"],
            "upstream_clean": True,
        },
        "split": {
            "dataset_population": profile.dataset_population,
            "train_count": len(processed.train_path.read_text(encoding="utf-8").splitlines()),
            "validation_count": len(
                processed.validation_path.read_text(encoding="utf-8").splitlines()
            ),
            "overlap": 0,
            "complete": True,
            "official_test_accessed": False,
        },
        "initialization": {"model_checkpoint_loaded": False},
        "stability": {"finite": True, "checkpoint_round_trip": True},
        "metrics": dict(metrics_payload["metrics"]),
        "artifacts": {
            "environment": artifact(environment),
            "dataset_inventory": artifact(processed.inventory_path),
            "train_split": artifact(processed.train_path),
            "validation_split": artifact(processed.validation_path),
            "backbone": artifact(backbone),
            "input_checkpoint": artifact(checkpoint.checkpoint_path),
            "output_checkpoint": artifact(checkpoint.checkpoint_path),
            "config": artifact(upstream.config_path),
            "rights_decision": artifact(profile_path),
        },
    }
    validate_training_evidence(evidence, stage=inputs.stage)
    return AuthoritativeTrainingEvidence(evidence, _token=_AUTHORITY_TOKEN)


def _mapping(value: Any, *, name: str, keys: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    if actual != set(keys):
        missing = sorted(set(keys).difference(actual))
        extra = sorted(actual.difference(keys))
        raise ValueError(f"{name} keys are invalid: missing={missing} extra={extra}")
    return value


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _integer(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _metric_names(stage: str) -> frozenset[str]:
    return _T0_METRICS if stage == "T0" else REQUIRED_T1_METRICS


def validate_training_evidence(evidence: Any, *, stage: str) -> Mapping[str, Any]:
    """Validate the closed evidence schema before it is scored or persisted."""

    if stage not in STAGE_THRESHOLDS:
        raise ValueError(f"unsupported training stage: {stage}")
    root = _mapping(
        evidence,
        name="training evidence",
        keys=frozenset(
            {
                "schema_version",
                "run_id",
                "stage",
                "provenance",
                "split",
                "initialization",
                "stability",
                "metrics",
                "artifacts",
            }
        ),
    )
    if root["schema_version"] != 1:
        raise ValueError("training evidence schema_version must be 1")
    _nonempty_string(root["run_id"], name="run_id")
    if root["stage"] != stage:
        raise ValueError("evidence stage must match requested stage")

    provenance = _mapping(
        root["provenance"],
        name="provenance",
        keys=frozenset(
            {
                "git_commit",
                "upstream_commit",
                "container_image_digest",
                "python",
                "torch",
                "cuda",
                "gpu",
                "upstream_clean",
            }
        ),
    )
    for field in ("git_commit", "upstream_commit"):
        value = _nonempty_string(provenance[field], name=f"provenance.{field}")
        if _COMMIT.fullmatch(value) is None:
            raise ValueError(f"provenance.{field} must be a full Git commit")
    digest = _nonempty_string(
        provenance["container_image_digest"], name="provenance.container_image_digest"
    )
    if not digest.startswith("sha256:") or not is_sha256(digest.removeprefix("sha256:")):
        raise ValueError("provenance.container_image_digest must be a SHA-256 digest")
    for field in ("python", "torch", "cuda", "gpu"):
        _nonempty_string(provenance[field], name=f"provenance.{field}")
    _boolean(provenance["upstream_clean"], name="provenance.upstream_clean")

    split = _mapping(
        root["split"],
        name="split",
        keys=frozenset(
            {
                "dataset_population",
                "train_count",
                "validation_count",
                "overlap",
                "complete",
                "official_test_accessed",
            }
        ),
    )
    for field in ("dataset_population", "train_count", "validation_count", "overlap"):
        value = _integer(split[field], name=f"split.{field}")
        if value < 0:
            raise ValueError(f"split.{field} must be non-negative")
    _boolean(split["complete"], name="split.complete")
    _boolean(split["official_test_accessed"], name="split.official_test_accessed")

    initialization = _mapping(
        root["initialization"],
        name="initialization",
        keys=frozenset({"model_checkpoint_loaded"}),
    )
    _boolean(
        initialization["model_checkpoint_loaded"],
        name="initialization.model_checkpoint_loaded",
    )
    stability = _mapping(
        root["stability"],
        name="stability",
        keys=frozenset({"finite", "checkpoint_round_trip"}),
    )
    _boolean(stability["finite"], name="stability.finite")
    _boolean(stability["checkpoint_round_trip"], name="stability.checkpoint_round_trip")

    metrics = root["metrics"]
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be an object")
    required_metrics = _metric_names(stage)
    unknown_metrics = set(metrics).difference(required_metrics)
    if unknown_metrics:
        raise ValueError(f"metrics has unknown key: {sorted(unknown_metrics)[0]}")
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"metrics.{name} must be numeric")

    artifacts = root["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != REQUIRED_ARTIFACTS:
        raise ValueError("artifacts must contain exactly the required artifact references")
    for name, reference in artifacts.items():
        parsed = _mapping(
            reference,
            name=f"artifacts.{name}",
            keys=frozenset({"path", "sha256"}),
        )
        _nonempty_string(parsed["path"], name=f"artifacts.{name}.path")
        if not is_sha256(parsed["sha256"]):
            raise ValueError(f"artifacts.{name}.sha256 must be a SHA-256")
        if not Path(str(parsed["path"])).is_absolute():
            raise ValueError(f"artifacts.{name}.path must be absolute")
    environment = artifacts["environment"]
    assert isinstance(environment, Mapping)
    if Path(str(environment["path"])).name != f"environment.{environment['sha256']}.json":
        raise ValueError("environment evidence must use Task 6's content-addressed filename")
    return root


def verify_referenced_artifacts(evidence: Any, *, stage: str) -> dict[str, str]:
    """Hash every referenced regular file, returning concise observations."""

    validated = validate_training_evidence(evidence, stage=stage)
    observations: dict[str, str] = {}
    artifacts = validated["artifacts"]
    assert isinstance(artifacts, Mapping)
    for name in sorted(REQUIRED_ARTIFACTS):
        reference = artifacts[name]
        assert isinstance(reference, Mapping)
        path = Path(str(reference["path"]))
        if path.is_symlink() or not path.is_file():
            observations[name] = f"artifact missing or not a regular file: {path}"
            continue
        observed = sha256_file(path)
        expected = str(reference["sha256"]).lower()
        observations[name] = (
            f"verified: {path}; sha256={observed}"
            if observed == expected
            else f"artifact hash mismatch: {path}; expected={expected} observed={observed}"
        )
    return observations


def _artifacts_verified(observations: Mapping[str, str]) -> bool:
    return len(observations) == len(REQUIRED_ARTIFACTS) and all(
        observation.startswith("verified:") for observation in observations.values()
    )


def _metrics_are_finite(metrics: Mapping[str, Any], *, stage: str) -> bool:
    return set(metrics) == set(_metric_names(stage)) and all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (isinstance(value, int) or math.isfinite(value))
        for value in metrics.values()
    )


def score_training_stage(evidence: Any, *, stage: str) -> StageReport:
    """Score one stage; every safety failure is a non-averageable blocker."""

    if (
        not isinstance(evidence, AuthoritativeTrainingEvidence)
        or evidence._token is not _AUTHORITY_TOKEN
    ):
        raise TypeError(
            "score_training_stage requires verified AuthoritativeTrainingEvidence"
        )
    evidence = evidence.evidence

    try:
        validated = validate_training_evidence(evidence, stage=stage)
    except ValueError as error:
        # A malformed schema has no trustworthy sub-score, so fail before a report.
        raise ValueError(f"invalid {stage} training evidence: {error}") from error

    provenance = validated["provenance"]
    split = validated["split"]
    initialization = validated["initialization"]
    stability = validated["stability"]
    metrics = validated["metrics"]
    assert all(isinstance(value, Mapping) for value in (provenance, split, initialization, stability, metrics))
    artifact_observations = verify_referenced_artifacts(validated, stage=stage)
    artifacts_ok = _artifacts_verified(artifact_observations)
    metrics_ok = _metrics_are_finite(metrics, stage=stage)
    finite_ok = bool(stability["finite"]) and metrics_ok
    validation_observed = (
        "T0 intentionally records only its one-update training loss"
        if stage == "T0"
        else f"all {len(_metric_names(stage))} required {stage} validation metrics recorded"
    )

    checks = (
        CheckResult(
            "provenance.pinned_upstream", "provenance", "Pinned, clean upstream",
            15, bool(provenance["upstream_clean"]),
            blocker=True, expected="authoritatively audited profile commit and clean tree",
            observed=f"clean={provenance['upstream_clean']}; commit={provenance['upstream_commit']}",
        ),
        CheckResult(
            "provenance.no_official_model_checkpoint", "provenance", "ImageNet-backbone-only initialization",
            10, not bool(initialization["model_checkpoint_loaded"]), blocker=True,
            expected="official model checkpoint not loaded",
            observed=str(initialization["model_checkpoint_loaded"]),
        ),
        CheckResult(
            "split.no_overlap", "split", "Train and validation do not overlap",
            10, int(split["overlap"]) == 0, blocker=True, expected="overlap=0",
            observed=f"overlap={split['overlap']}",
        ),
        CheckResult(
            "split.complete", "split", "Approved split is complete",
            5, bool(split["complete"])
            and (int(split["dataset_population"]), int(split["train_count"]), int(split["validation_count"]))
            == (1201, 961, 240), blocker=True, expected="complete 1201/961/240 split",
            observed=(f"complete={split['complete']}; population={split['dataset_population']}; "
                      f"train={split['train_count']}; validation={split['validation_count']}"),
        ),
        CheckResult(
            "split.no_official_test_access", "split", "Official Test remains sealed",
            5, not bool(split["official_test_accessed"]), blocker=True,
            expected="official_test_accessed=false", observed=str(split["official_test_accessed"]),
        ),
        CheckResult(
            "stability.finite_metrics", "stability", "All required values are finite",
            12, finite_ok, blocker=True, expected="finite evidence and metrics",
            observed=f"finite={stability['finite']}; metrics_finite={metrics_ok}",
        ),
        CheckResult(
            "stability.resume_round_trip", "stability", "Checkpoint round trip succeeds",
            8, bool(stability["checkpoint_round_trip"]), blocker=True,
            expected="checkpoint_round_trip=true", observed=str(stability["checkpoint_round_trip"]),
        ),
        CheckResult(
            "validation.stage_evidence", "validation", "Required stage validation evidence",
            20, metrics_ok, expected="all required metrics present and finite",
            observed=validation_observed if metrics_ok else "missing or non-finite metric",
        ),
        CheckResult(
            "artifacts.hashes", "artifacts", "All referenced artifacts hash-verify from disk",
            15, artifacts_ok, blocker=True, expected="every artifact SHA-256 matches",
            observed="; ".join(artifact_observations[name] for name in sorted(artifact_observations)),
        ),
    )
    return score_stage(
        stage_id=stage,
        stage_name=f"STEERER UCF-QNRF {stage} training",
        threshold=STAGE_THRESHOLDS[stage],
        checks=checks,
        success_status=SUCCESS_STATUS,
    )


def weights_by_category(report: StageReport) -> dict[str, float]:
    """Return category totals for review tests and human scorecards."""

    totals: defaultdict[str, float] = defaultdict(float)
    for check in report.checks:
        totals[check.category] += check.weight
    return dict(totals)


__all__ = [
    "APPROVED_UPSTREAM_COMMIT",
    "AuthoritativeTrainingEvidence",
    "AuthoritativeTrainingInputs",
    "REQUIRED_ARTIFACTS",
    "REQUIRED_T1_METRICS",
    "STAGE_THRESHOLDS",
    "score_training_stage",
    "validate_training_evidence",
    "verify_authoritative_training_evidence",
    "verify_referenced_artifacts",
    "weights_by_category",
]
