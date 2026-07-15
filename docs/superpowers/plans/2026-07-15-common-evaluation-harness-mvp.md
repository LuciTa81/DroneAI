# Common Evaluation Harness MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the model-neutral, scored evaluation core that can inspect one adapter, evaluate complete labeled samples, calculate count and spatial metrics, select 12 reproducible review cases, render CSRNet-style four-column panels, and persist a small hash-verified report without saving every native output.

**Architecture:** Keep the repository's flat `src/droneai` pattern and existing `CheckResult`/`score_stage` gate. Introduce focused modules for contracts, model briefs, metrics, curation, panels, artifacts, gating, and orchestration. Prove the whole path with an in-memory fixture adapter; DM-Count/UCF-QNRF, remaining model adapters, drone dataset execution, and final fine-tuning receive separate follow-on plans so each review gate stays independently testable.

**Tech Stack:** Python 3.10+, dataclasses, NumPy, Pillow, SciPy, scikit-image, pytest, existing DroneAI scoring/integrity helpers, home5090 Docker profile.

## Global Constraints

- Preserve all Stage 0 through Stage 3C behavior and historical Colab provenance.
- Use `home5090_docker` as the default new runtime and `/workspace/data/results` for large artifacts.
- Do not download datasets, reuse ambiguous weights, run real test splits, or start training in this plan.
- The common technical gate totals 100 points, passes at 80, and cannot average away a blocker.
- Every expected sample must produce a prediction or explicit failure; failure is never converted to zero people.
- A full run requires an approved `model-brief.md` and reviewed adapter entry points.
- The selector renders at most 12 deterministic panels: best 3, strong 3, typical 3, and failure 3.
- The Git review bundle is limited to 25 MiB per run; datasets, checkpoints, full maps, and raw video stay outside Git.
- Technical score remains independent from the existing Stage 3C rights decision.
- No output is automatically deleted; cleanup remains an explicit user-approved operation.
- Final single-winner fine-tuning is outside this MVP and cannot begin from any artifact produced here.

## Program Decomposition

This is implementation plan 1 of 5:

1. common evaluation harness MVP and synthetic end-to-end proof;
2. DM-Count adapter, UCF-QNRF non-test smoke, and official-checkpoint evaluation;
3. STEERER, PET, and CSRNet adapter plans with per-model architecture review;
4. mandatory UP-COUNT cross-domain lane and conditional DroneCrowd lane;
5. single-winner selection and final fine-tuning gate.

Plan 5 must preserve the approved design's complete training contract: epochs, method, frozen/trainable layers, initialization, optimizer, scheduler, augmentations, loss terms and weights, metrics, seeds, checkpoint/resume, group-isolated train/validation/test roles, epoch history, wall time, GPU hours, VRAM, one-time sealed-test result, and pre/post fine-tuning comparison. Nothing in this MVP grants permission to skip or weaken those fields.

---

### Task 1: Add Evaluation Contracts and Adapter Protocol

**Files:**
- Create: `src/droneai/evaluation_contract.py`
- Create: `tests/test_evaluation_contract.py`

**Interfaces:**
- Consumes: `numpy.ndarray`, `pathlib.Path`.
- Produces: `ZoneBox`, `EvaluationSample`, `NativePrediction`, `ScalarEvaluation`, and `ModelAdapter`.

- [ ] **Step 1: Write failing validation and adapter-contract tests**

```python
from pathlib import Path

import numpy as np
import pytest

from droneai.evaluation_contract import EvaluationSample, NativePrediction, ZoneBox


def test_explicit_failure_cannot_be_serialized_as_zero_prediction(tmp_path: Path) -> None:
    prediction = NativePrediction(
        sample_id="sample-1",
        output_type="count",
        predicted_count=None,
        latency_ms=2.0,
        peak_vram_mb=4.0,
        failure_state="cuda_out_of_memory",
    )
    assert prediction.predicted_count is None
    with pytest.raises(ValueError, match="failed predictions cannot contain a count"):
        NativePrediction(
            sample_id="sample-1",
            output_type="count",
            predicted_count=0.0,
            latency_ms=2.0,
            peak_vram_mb=4.0,
            failure_state="cuda_out_of_memory",
        )


def test_sample_rejects_ground_truth_count_point_mismatch(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="point count"):
        EvaluationSample(
            sample_id="sample-1",
            dataset_id="fixture-v1",
            split_id="validation",
            image_path=image_path,
            source_sha256="a" * 64,
            width=32,
            height=24,
            ground_truth_count=2.0,
            ground_truth_points=((4.0, 5.0),),
            has_point_annotations=True,
        )


def test_density_prediction_requires_two_dimensional_finite_map() -> None:
    with pytest.raises(ValueError, match="finite 2D density"):
        NativePrediction(
            sample_id="sample-1",
            output_type="density",
            predicted_count=1.0,
            latency_ms=2.0,
            peak_vram_mb=4.0,
            density=np.asarray([[[float("nan")]]], dtype=np.float32),
        )


def test_sample_rejects_invalid_source_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source SHA-256"):
        EvaluationSample(
            sample_id="sample-1",
            dataset_id="fixture-v1",
            split_id="validation",
            image_path=tmp_path / "image.png",
            source_sha256="short",
            width=32,
            height=24,
            ground_truth_count=0.0,
        )


def test_sample_rejects_density_whose_sum_does_not_preserve_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="density sum"):
        EvaluationSample(
            sample_id="sample-1",
            dataset_id="fixture-v1",
            split_id="validation",
            image_path=tmp_path / "image.png",
            source_sha256="a" * 64,
            width=32,
            height=24,
            ground_truth_count=2.0,
            ground_truth_density=np.ones((3, 3), dtype=np.float32),
        )


def test_failed_prediction_still_rejects_invalid_runtime() -> None:
    with pytest.raises(ValueError, match="runtime values"):
        NativePrediction(
            sample_id="sample-1",
            output_type="count",
            predicted_count=None,
            latency_ms=-1.0,
            peak_vram_mb=0.0,
            failure_state="fixture_failure",
        )


def test_zone_box_contains_right_and_bottom_exclusively() -> None:
    zone = ZoneBox("zone-a", 0, 0, 10, 10)
    assert zone.contains(0, 0)
    assert zone.contains(9.9, 9.9)
    assert not zone.contains(10, 10)
```

- [ ] **Step 2: Run the contract tests and confirm the module is absent**

Run: `python -m pytest tests/test_evaluation_contract.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'droneai.evaluation_contract'`.

- [ ] **Step 3: Implement immutable contracts and validation**

```python
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from droneai.integrity import is_sha256

if TYPE_CHECKING:
    from droneai.model_brief import ModelBrief

OutputType = Literal["count", "density", "points", "hybrid"]
Point = tuple[float, float]


@dataclass(frozen=True)
class ZoneBox:
    zone_id: str
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not self.zone_id or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("zone requires an id and positive rectangle")

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    dataset_id: str
    split_id: str
    image_path: Path
    source_sha256: str
    width: int
    height: int
    ground_truth_count: float
    ground_truth_points: tuple[Point, ...] = ()
    ground_truth_density: np.ndarray | None = field(default=None, compare=False)
    condition_tags: dict[str, str] = field(default_factory=dict, compare=False)
    zones: tuple[ZoneBox, ...] = ()
    has_point_annotations: bool = False

    def __post_init__(self) -> None:
        if not self.sample_id or not self.dataset_id or not self.split_id:
            raise ValueError("sample identity is required")
        if not is_sha256(self.source_sha256):
            raise ValueError("sample source SHA-256 is required")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("sample dimensions must be positive")
        if any(zone.x0 < 0 or zone.y0 < 0 or zone.x1 > self.width or zone.y1 > self.height for zone in self.zones):
            raise ValueError("sample zones must stay inside image bounds")
        if not math.isfinite(self.ground_truth_count) or self.ground_truth_count < 0:
            raise ValueError("ground-truth count must be finite and non-negative")
        if self.ground_truth_points and not self.has_point_annotations:
            raise ValueError("point coordinates require point-annotation status")
        if any(not (0 <= x < self.width and 0 <= y < self.height) for x, y in self.ground_truth_points):
            raise ValueError("ground-truth points must stay inside image bounds")
        if self.has_point_annotations and (
            not math.isclose(self.ground_truth_count, round(self.ground_truth_count), abs_tol=1e-6)
            or len(self.ground_truth_points) != round(self.ground_truth_count)
        ):
            raise ValueError("point count must equal ground-truth count")
        if self.ground_truth_density is not None:
            density = np.asarray(self.ground_truth_density)
            if density.ndim != 2 or not np.isfinite(density).all():
                raise ValueError("ground-truth density must be a finite 2D array")
            tolerance = max(1e-4, 1e-5 * max(self.ground_truth_count, 1.0))
            if abs(float(density.sum()) - self.ground_truth_count) > tolerance:
                raise ValueError("ground-truth density sum must preserve ground-truth count")


@dataclass(frozen=True)
class NativePrediction:
    sample_id: str
    output_type: OutputType
    predicted_count: float | None
    latency_ms: float
    peak_vram_mb: float
    density: np.ndarray | None = field(default=None, compare=False)
    points: tuple[Point, ...] = ()
    point_confidences: tuple[float, ...] = ()
    confidence: float | None = None
    failure_state: str | None = None
    coordinate_space: str = "original_pixels"

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("prediction sample identity is required")
        if self.latency_ms <= 0 or self.peak_vram_mb < 0:
            raise ValueError("runtime values must be non-negative and latency positive")
        if self.failure_state:
            if self.predicted_count is not None or self.density is not None or self.points:
                raise ValueError("failed predictions cannot contain a count or native output")
            return
        if self.predicted_count is None or not math.isfinite(self.predicted_count) or self.predicted_count < 0:
            raise ValueError("successful predictions require a finite non-negative count")
        if self.density is not None:
            density = np.asarray(self.density)
            if density.ndim != 2 or not np.isfinite(density).all():
                raise ValueError("density output must be a finite 2D density map")
            tolerance = max(1e-4, 1e-5 * max(self.predicted_count, 1.0))
            if abs(float(density.sum()) - self.predicted_count) > tolerance:
                raise ValueError("predicted density sum must preserve predicted count")
        if self.point_confidences and len(self.point_confidences) != len(self.points):
            raise ValueError("point confidences must align with points")
        if self.output_type == "density" and self.density is None:
            raise ValueError("density output type requires a density map")
        if self.output_type == "points" and abs(self.predicted_count - len(self.points)) > 1e-6:
            raise ValueError("point output count must equal predicted point count")
        if self.output_type == "hybrid" and self.density is None and not self.points:
            raise ValueError("hybrid output requires density or points")


@dataclass(frozen=True)
class ScalarEvaluation:
    sample_id: str
    ground_truth_count: float
    predicted_count: float | None
    signed_error: float | None
    absolute_error: float | None
    normalized_error: float | None
    density_band: str
    latency_ms: float
    peak_vram_mb: float
    output_type: OutputType
    failure_state: str | None
    spatial_metric_name: str | None = None
    spatial_metric_value: float | None = None
    extra_metrics: dict[str, float] = field(default_factory=dict, compare=False)
    condition_values: dict[str, str] = field(default_factory=dict, compare=False)


class ModelAdapter(Protocol):
    def brief(self) -> "ModelBrief":
        raise NotImplementedError

    def predict(self, sample: EvaluationSample, *, retain_native: bool) -> NativePrediction:
        raise NotImplementedError
```

- [ ] **Step 4: Run contract tests**

Run: `python -m pytest tests/test_evaluation_contract.py -q`

Expected: `7 passed`.

- [ ] **Step 5: Commit contracts**

```bash
git add src/droneai/evaluation_contract.py tests/test_evaluation_contract.py
git commit -m "Add common evaluation contracts"
```

### Task 2: Require a Model Architecture Brief Before Full Runs

**Files:**
- Create: `src/droneai/model_brief.py`
- Create: `tests/test_model_brief.py`

**Interfaces:**
- Consumes: structured model metadata and review decision.
- Produces: `ModelBrief`, `render_model_brief(brief) -> str`, and `write_model_brief(path, brief) -> Path`.

- [ ] **Step 1: Write failing model-brief tests**

```python
from dataclasses import replace

import pytest

from droneai.model_brief import ModelBrief, render_model_brief


def _brief(review_status: str = "approved") -> ModelBrief:
    return ModelBrief(
        model_id="fixture-density",
        paper="Fixture Density Paper",
        role="density baseline",
        family="density",
        backbone="three-layer fixture CNN",
        parameter_count=128,
        blocks=("frontend", "density head"),
        feature_scales=("stride 4 density features",),
        input_contract="RGB NCHW float32 normalized to [0,1]",
        preprocessing_policy="resize to 32x32 without tiling",
        coordinate_transform="density stride 4 to original pixels",
        native_output="one-channel density map at stride 4",
        count_derivation="sum of native density map",
        zone_derivation="integrate density inside scaled zone boxes",
        original_losses=("pixel MSE weight=1.0",),
        official_protocol="fixture validation split",
        official_reported_metrics=("fixture MAE 0.0",),
        strengths=("deterministic",),
        failure_modes=("texture false positives",),
        runtime_risks=("none for fixture",),
        rights_status="PASS_RESEARCH_ONLY",
        code_rights_status="project fixture code",
        dataset_rights_status="project synthetic data",
        checkpoint_rights_status="project synthetic checkpoint",
        deployment_rights_status="not production evidence",
        upstream_commit="a" * 40,
        checkpoint_path="fixture-checkpoint.bin",
        checkpoint_sha256="b" * 64,
        reviewed_paths=("src/droneai/model_brief.py",),
        review_status=review_status,
    )


def test_approved_brief_renders_required_architecture_and_io() -> None:
    rendered = render_model_brief(_brief())
    assert "three-layer fixture CNN" in rendered
    assert "one-channel density map" in rendered
    assert "sum of native density map" in rendered
    assert "pixel MSE weight=1.0" in rendered
    assert "stride 4 density features" in rendered
    assert "integrate density inside scaled zone boxes" in rendered


def test_pending_brief_cannot_authorize_full_run() -> None:
    brief = _brief("pending")
    with pytest.raises(ValueError, match="approved model brief"):
        brief.require_full_run_approval()


def test_brief_rejects_missing_architecture_lists() -> None:
    with pytest.raises(ValueError, match="architecture and review lists"):
        replace(_brief(), blocks=())
```

- [ ] **Step 2: Run tests and confirm missing module failure**

Run: `python -m pytest tests/test_model_brief.py -q`

Expected: FAIL during collection for missing `droneai.model_brief`.

- [ ] **Step 3: Implement model brief validation and Markdown rendering**

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from droneai.integrity import is_sha256


@dataclass(frozen=True)
class ModelBrief:
    model_id: str
    paper: str
    role: str
    family: str
    backbone: str
    parameter_count: int | None
    blocks: tuple[str, ...]
    feature_scales: tuple[str, ...]
    input_contract: str
    preprocessing_policy: str
    coordinate_transform: str
    native_output: str
    count_derivation: str
    zone_derivation: str
    original_losses: tuple[str, ...]
    official_protocol: str
    official_reported_metrics: tuple[str, ...]
    strengths: tuple[str, ...]
    failure_modes: tuple[str, ...]
    runtime_risks: tuple[str, ...]
    rights_status: str
    code_rights_status: str
    dataset_rights_status: str
    checkpoint_rights_status: str
    deployment_rights_status: str
    upstream_commit: str
    checkpoint_path: str
    checkpoint_sha256: str
    reviewed_paths: tuple[str, ...]
    review_status: str

    def __post_init__(self) -> None:
        required_text = (
            self.model_id, self.paper, self.role, self.family, self.backbone,
            self.input_contract, self.preprocessing_policy, self.coordinate_transform,
            self.native_output, self.count_derivation, self.zone_derivation,
            self.official_protocol, self.rights_status, self.code_rights_status,
            self.dataset_rights_status, self.checkpoint_rights_status,
            self.deployment_rights_status, self.checkpoint_path,
        )
        if any(not value.strip() for value in required_text):
            raise ValueError("model brief fields cannot be empty")
        required_lists = (
            self.blocks,
            self.feature_scales,
            self.original_losses,
            self.official_reported_metrics,
            self.strengths,
            self.failure_modes,
            self.runtime_risks,
        )
        if any(not values or any(not value.strip() for value in values) for values in required_lists):
            raise ValueError("model brief architecture and review lists cannot be empty")
        if self.parameter_count is not None and self.parameter_count < 0:
            raise ValueError("parameter count must be non-negative when measured")
        if re.fullmatch(r"[0-9a-f]{40}", self.upstream_commit) is None or not is_sha256(self.checkpoint_sha256):
            raise ValueError("model brief requires commit and checkpoint SHA-256")

    def require_full_run_approval(self) -> None:
        if self.review_status != "approved" or not self.reviewed_paths:
            raise ValueError("full run requires an approved model brief and reviewed code paths")


def _bullets(values: tuple[str, ...]) -> list[str]:
    return [f"- {value}" for value in values]


def render_model_brief(brief: ModelBrief) -> str:
    lines = [
        f"# Model brief: {brief.model_id}", "",
        f"- Paper: {brief.paper}", f"- Role: {brief.role}",
        f"- Family: {brief.family}", f"- Backbone: {brief.backbone}",
        f"- Parameter count: {brief.parameter_count if brief.parameter_count is not None else 'not measured'}",
        f"- Input: {brief.input_contract}", f"- Preprocessing: {brief.preprocessing_policy}",
        f"- Coordinate transform: {brief.coordinate_transform}", f"- Native output: {brief.native_output}",
        f"- Count derivation: {brief.count_derivation}",
        f"- Zone derivation: {brief.zone_derivation}",
        f"- Official protocol: {brief.official_protocol}",
        f"- Rights status: {brief.rights_status}",
        f"- Code rights: {brief.code_rights_status}",
        f"- Dataset rights: {brief.dataset_rights_status}",
        f"- Checkpoint rights: {brief.checkpoint_rights_status}",
        f"- Deployment rights: {brief.deployment_rights_status}",
        f"- Upstream commit: `{brief.upstream_commit}`",
        f"- Checkpoint path: `{brief.checkpoint_path}`",
        f"- Checkpoint SHA-256: `{brief.checkpoint_sha256}`", "",
        "## Major blocks", "", *_bullets(brief.blocks), "",
        "## Feature scales", "", *_bullets(brief.feature_scales), "",
        "## Original losses", "", *_bullets(brief.original_losses), "",
        "## Official reported metrics", "", *_bullets(brief.official_reported_metrics), "",
        "## Strengths", "", *_bullets(brief.strengths), "",
        "## Failure modes", "", *_bullets(brief.failure_modes), "",
        "## Runtime risks", "", *_bullets(brief.runtime_risks), "",
        "## Reviewed code", "", *_bullets(brief.reviewed_paths), "",
        f"Review status: **{brief.review_status}**", "",
    ]
    return "\n".join(lines)


def write_model_brief(path: str | Path, brief: ModelBrief) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_model_brief(brief), encoding="utf-8")
    return target
```

- [ ] **Step 4: Run model-brief tests**

Run: `python -m pytest tests/test_model_brief.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit model brief**

```bash
git add src/droneai/model_brief.py tests/test_model_brief.py
git commit -m "Require reviewed model architecture briefs"
```

### Task 3: Calculate Count, Density, Localization, GAME, and Zone Metrics

**Files:**
- Modify: `pyproject.toml`
- Create: `src/droneai/evaluation_metrics.py`
- Create: `tests/test_evaluation_metrics.py`

**Interfaces:**
- Consumes: `EvaluationSample`, `NativePrediction`.
- Produces: `evaluate_sample(...) -> ScalarEvaluation` and `summarize_records(...) -> dict[str, object]`.

- [ ] **Step 1: Add failing metric tests**

```python
from pathlib import Path

import numpy as np
import pytest

from droneai.evaluation_contract import EvaluationSample, NativePrediction, ZoneBox
from droneai.evaluation_metrics import evaluate_sample, summarize_records


def _sample(tmp_path: Path) -> EvaluationSample:
    return EvaluationSample(
        sample_id="dense-1",
        dataset_id="fixture-v1",
        split_id="validation",
        image_path=tmp_path / "dense-1.png",
        source_sha256="a" * 64,
        width=8,
        height=8,
        ground_truth_count=2.0,
        ground_truth_points=((1.0, 1.0), (6.0, 6.0)),
        ground_truth_density=np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        condition_tags={"density_band": "high"},
        zones=(ZoneBox("left", 0, 0, 4, 8), ZoneBox("right", 4, 0, 8, 8)),
        has_point_annotations=True,
    )


def test_density_metrics_and_count_error_are_recorded(tmp_path: Path) -> None:
    prediction = NativePrediction(
        sample_id="dense-1", output_type="density", predicted_count=2.0,
        latency_ms=3.0, peak_vram_mb=100.0,
        density=np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    record = evaluate_sample(_sample(tmp_path), prediction, localization_radius=2.0)
    assert record.absolute_error == 0.0
    assert record.extra_metrics["density_psnr"] == pytest.approx(float("inf"))
    assert record.extra_metrics["density_ssim"] == pytest.approx(1.0)
    assert record.extra_metrics["predicted_mass"] == pytest.approx(2.0)
    assert record.extra_metrics["game_l1"] == pytest.approx(0.0)
    assert record.extra_metrics["zone_mae"] == pytest.approx(0.0)
    assert "localization_f1" not in record.extra_metrics


def test_point_matching_and_zone_error_are_recorded(tmp_path: Path) -> None:
    prediction = NativePrediction(
        sample_id="dense-1", output_type="points", predicted_count=2.0,
        latency_ms=3.0, peak_vram_mb=100.0,
        points=((1.5, 1.5), (6.5, 6.5)), point_confidences=(0.9, 0.8),
    )
    record = evaluate_sample(_sample(tmp_path), prediction, localization_radius=2.0)
    assert record.extra_metrics["localization_precision"] == 1.0
    assert record.extra_metrics["localization_recall"] == 1.0
    assert record.extra_metrics["localization_f1"] == 1.0
    assert record.extra_metrics["localization_radius"] == 2.0
    assert record.extra_metrics["zone_mae"] == 0.0
    assert "density_ssim" not in record.extra_metrics


def test_failed_prediction_reduces_coverage_without_becoming_zero(tmp_path: Path) -> None:
    failed = NativePrediction(
        sample_id="dense-1", output_type="count", predicted_count=None,
        latency_ms=1.0, peak_vram_mb=0.0, failure_state="fixture_failure",
    )
    record = evaluate_sample(_sample(tmp_path), failed, localization_radius=2.0)
    summary = summarize_records([record], expected_samples=1)
    assert record.predicted_count is None
    assert summary["coverage"] == 0.0
    assert summary["failures"] == 1
```

- [ ] **Step 2: Run metric tests and verify missing module**

Run: `python -m pytest tests/test_evaluation_metrics.py -q`

Expected: FAIL during collection for missing `droneai.evaluation_metrics`.

- [ ] **Step 3: Add isolated evaluation dependencies**

Add to `pyproject.toml`:

```toml
evaluation = [
  "scipy>=1.13,<2",
  "scikit-image>=0.24,<1",
]
```

Install for the local plan run:

Run: `python -m pip install -e ".[dev,evaluation]"`

Expected: exit 0 and importable `scipy` and `skimage`.

- [ ] **Step 4: Implement family-aware metrics**

```python
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from droneai.evaluation_contract import EvaluationSample, NativePrediction, ScalarEvaluation


def _point_metrics(gt: tuple[tuple[float, float], ...], pred: tuple[tuple[float, float], ...], radius: float) -> dict[str, float]:
    if not gt and not pred:
        return {"localization_precision": 1.0, "localization_recall": 1.0, "localization_f1": 1.0, "localization_mean_distance": 0.0}
    if not gt or not pred:
        return {"localization_precision": 0.0, "localization_recall": 0.0, "localization_f1": 0.0, "localization_mean_distance": float("inf")}
    gt_array = np.asarray(gt, dtype=np.float64)
    pred_array = np.asarray(pred, dtype=np.float64)
    distances = np.linalg.norm(gt_array[:, None, :] - pred_array[None, :, :], axis=2)
    gt_indices, pred_indices = linear_sum_assignment(distances)
    matched = [float(distances[g, p]) for g, p in zip(gt_indices, pred_indices) if distances[g, p] <= radius]
    true_positive = len(matched)
    precision = true_positive / len(pred)
    recall = true_positive / len(gt)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_distance = float(np.mean(matched)) if matched else float("inf")
    return {"localization_precision": precision, "localization_recall": recall, "localization_f1": f1, "localization_mean_distance": mean_distance}


def _point_zone_mae(sample: EvaluationSample, prediction: NativePrediction) -> float:
    if not sample.zones or not sample.has_point_annotations:
        return float("nan")
    errors = []
    for zone in sample.zones:
        ground_truth = sum(zone.contains(x, y) for x, y in sample.ground_truth_points)
        predicted = sum(zone.contains(x, y) for x, y in prediction.points)
        errors.append(abs(ground_truth - predicted))
    return float(np.mean(errors))


def _density_zone_mae(sample: EvaluationSample, ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    if not sample.zones:
        return float("nan")
    height, width = ground_truth.shape
    errors = []
    for zone in sample.zones:
        x0 = round(zone.x0 * width / sample.width)
        x1 = round(zone.x1 * width / sample.width)
        y0 = round(zone.y0 * height / sample.height)
        y1 = round(zone.y1 * height / sample.height)
        region = (slice(y0, y1), slice(x0, x1))
        errors.append(abs(float(ground_truth[region].sum()) - float(predicted[region].sum())))
    return float(np.mean(errors))


def _game_l1(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    if ground_truth.shape != predicted.shape:
        raise ValueError("density maps must share shape")
    height, width = ground_truth.shape
    y_mid, x_mid = height // 2, width // 2
    slices = ((slice(0, y_mid), slice(0, x_mid)), (slice(0, y_mid), slice(x_mid, width)), (slice(y_mid, height), slice(0, x_mid)), (slice(y_mid, height), slice(x_mid, width)))
    return float(sum(abs(float(ground_truth[s].sum()) - float(predicted[s].sum())) for s in slices))


def _density_ssim(ground_truth: np.ndarray, predicted: np.ndarray, *, data_range: float) -> float:
    minimum_side = min(ground_truth.shape)
    window = min(7, minimum_side)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        raise ValueError("density maps must be at least 3 by 3 for SSIM")
    return float(structural_similarity(ground_truth, predicted, data_range=data_range, win_size=window))


def evaluate_sample(sample: EvaluationSample, prediction: NativePrediction, *, localization_radius: float) -> ScalarEvaluation:
    if prediction.sample_id != sample.sample_id:
        raise ValueError("sample and prediction IDs differ")
    if any(not (0 <= x < sample.width and 0 <= y < sample.height) for x, y in prediction.points):
        raise ValueError("predicted points must use original image coordinates")
    density_band = sample.condition_tags.get("density_band", "unknown")
    if prediction.failure_state:
        return ScalarEvaluation(sample.sample_id, sample.ground_truth_count, None, None, None, None, density_band, prediction.latency_ms, prediction.peak_vram_mb, prediction.output_type, prediction.failure_state, condition_values=dict(sample.condition_tags))
    predicted = float(prediction.predicted_count)
    signed = predicted - sample.ground_truth_count
    extra: dict[str, float] = {}
    spatial_name = None
    spatial_value = None
    if sample.ground_truth_density is not None and prediction.density is not None:
        gt_density = np.asarray(sample.ground_truth_density, dtype=np.float64)
        pred_density = np.asarray(prediction.density, dtype=np.float64)
        data_range = max(float(gt_density.max() - gt_density.min()), 1.0)
        extra["density_psnr"] = float(peak_signal_noise_ratio(gt_density, pred_density, data_range=data_range))
        extra["density_ssim"] = _density_ssim(gt_density, pred_density, data_range=data_range)
        extra["predicted_mass"] = float(pred_density.sum())
        extra["game_l1"] = _game_l1(gt_density, pred_density)
        extra["zone_mae"] = _density_zone_mae(sample, gt_density, pred_density)
        spatial_name, spatial_value = "game_l1", extra["game_l1"]
    if sample.has_point_annotations and prediction.output_type in {"points", "hybrid"}:
        extra.update(_point_metrics(sample.ground_truth_points, prediction.points, localization_radius))
        extra["localization_radius"] = localization_radius
        extra["zone_mae"] = _point_zone_mae(sample, prediction)
        spatial_name, spatial_value = "localization_f1", extra["localization_f1"]
    return ScalarEvaluation(sample.sample_id, sample.ground_truth_count, predicted, signed, abs(signed), abs(signed) / max(sample.ground_truth_count, 1.0), density_band, prediction.latency_ms, prediction.peak_vram_mb, prediction.output_type, None, spatial_name, spatial_value, extra, dict(sample.condition_tags))


def summarize_records(records: Iterable[ScalarEvaluation], *, expected_samples: int) -> dict[str, object]:
    rows = list(records)
    valid = [row for row in rows if row.predicted_count is not None]
    errors = np.asarray([row.signed_error for row in valid], dtype=np.float64)
    by_band: dict[str, list[float]] = defaultdict(list)
    by_condition: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in valid:
        by_band[row.density_band].append(float(row.signed_error))
        for key, value in row.condition_values.items():
            by_condition[key][value].append(float(row.signed_error))
    median_latency = float(np.median([row.latency_ms for row in valid])) if valid else float("inf")
    return {
        "expected_samples": expected_samples,
        "recorded_samples": len(rows),
        "successful_samples": len(valid),
        "failures": len(rows) - len(valid),
        "coverage": len(valid) / expected_samples if expected_samples else 0.0,
        "mae": float(np.mean(np.abs(errors))) if len(errors) else float("inf"),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else float("inf"),
        "signed_bias": float(np.mean(errors)) if len(errors) else float("inf"),
        "median_latency_ms": median_latency,
        "throughput_fps_batch1": 1000.0 / median_latency if math.isfinite(median_latency) and median_latency > 0 else 0.0,
        "peak_vram_mb": max((row.peak_vram_mb for row in valid), default=float("inf")),
        "band_signed_bias": {band: float(np.mean(values)) for band, values in sorted(by_band.items())},
        "condition_signed_bias": {
            key: {value: float(np.mean(errors)) for value, errors in sorted(groups.items())}
            for key, groups in sorted(by_condition.items())
        },
    }
```

- [ ] **Step 5: Run metric tests**

Run: `python -m pytest tests/test_evaluation_metrics.py -q`

Expected: `3 passed`.

- [ ] **Step 6: Commit metrics**

```bash
git add pyproject.toml src/droneai/evaluation_metrics.py tests/test_evaluation_metrics.py
git commit -m "Add family-aware evaluation metrics"
```

### Task 4: Select 12 Review Samples Deterministically

**Files:**
- Create: `src/droneai/evaluation_curation.py`
- Create: `tests/test_evaluation_curation.py`

**Interfaces:**
- Consumes: complete `Sequence[ScalarEvaluation]` and predictions-table SHA-256.
- Produces: `Selection`, `select_review_samples(...) -> tuple[Selection, ...]`, and `selection_manifest(...) -> dict[str, object]`.

- [ ] **Step 1: Write failing curation tests for categories, ties, and failures**

```python
from droneai.evaluation_contract import ScalarEvaluation
from droneai.evaluation_curation import select_review_samples, selection_manifest


def _row(sample_id: str, band: str, error: float, spatial: float) -> ScalarEvaluation:
    return ScalarEvaluation(sample_id, 100.0, 100.0 + error, error, abs(error), abs(error) / 100.0, band, 1.0, 2.0, "count", None, "game_l1", spatial)


def test_selector_returns_twelve_unique_reasoned_samples() -> None:
    rows = []
    for band in ("low", "medium", "high"):
        rows.extend(_row(f"{band}-{index:02d}", band, float(index - 5), float(index)) for index in range(12))
    selected = select_review_samples(rows, spatial_direction="minimize")
    assert len(selected) == 12
    assert len({item.sample_id for item in selected}) == 12
    assert {item.category for item in selected} == {"best", "strong", "typical", "failure"}


def test_selector_ties_are_resolved_by_sample_id() -> None:
    rows = [_row("b", "low", 0.0, 0.0), _row("a", "low", 0.0, 0.0)]
    selected = select_review_samples(rows, spatial_direction="minimize")
    assert selected[0].sample_id == "a"


def test_manifest_records_predictions_hash_and_shortfall() -> None:
    rows = [_row("only", "low", 0.0, 0.0)]
    selected = select_review_samples(rows, spatial_direction="minimize")
    manifest = selection_manifest(selected, predictions_sha256="a" * 64, spatial_direction="minimize", density_band_rules=("low: count < 10",), expected_panels=12)
    assert manifest["predictions_sha256"] == "a" * 64
    assert manifest["selected_count"] == 1
    assert manifest["shortfall"] == 11
    assert manifest["shortfall_reason"] == "insufficient unique eligible samples"
    assert manifest["density_band_rules"] == ["low: count < 10"]


def test_maximize_spatial_metric_selects_lowest_value_as_worst() -> None:
    rows = [
        ScalarEvaluation("good", 1.0, 1.0, 0.0, 0.0, 0.0, "high", 1.0, 2.0, "points", None, "localization_f1", 0.9),
        ScalarEvaluation("bad", 1.0, 1.0, 0.0, 0.0, 0.0, "high", 1.0, 2.0, "points", None, "localization_f1", 0.1),
    ]
    selected = select_review_samples(rows, spatial_direction="maximize")
    assert any(item.sample_id == "bad" and item.reason == "worst spatial quality" for item in selected)
```

- [ ] **Step 2: Run curation tests and confirm missing module**

Run: `python -m pytest tests/test_evaluation_curation.py -q`

Expected: FAIL during collection for missing `droneai.evaluation_curation`.

- [ ] **Step 3: Implement unique deterministic category selection**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

from droneai.evaluation_contract import ScalarEvaluation
from droneai.integrity import is_sha256


@dataclass(frozen=True)
class Selection:
    sample_id: str
    category: str
    density_band: str
    reason: str


def _take_unique(candidates: Iterable[ScalarEvaluation], used: set[str]) -> ScalarEvaluation | None:
    for row in candidates:
        if row.sample_id not in used:
            used.add(row.sample_id)
            return row
    return None


def select_review_samples(
    rows: Sequence[ScalarEvaluation],
    *,
    spatial_direction: Literal["minimize", "maximize"],
) -> tuple[Selection, ...]:
    valid = [row for row in rows if row.absolute_error is not None]
    used: set[str] = set()
    selected: list[Selection] = []
    for band in ("low", "medium", "high"):
        band_rows = sorted((row for row in valid if row.density_band == band), key=lambda row: (float(row.absolute_error), row.sample_id))
        if not band_rows:
            continue
        targets = (("best", 0.0), ("strong", 0.25), ("typical", 0.5))
        errors = np.asarray([float(row.absolute_error) for row in band_rows])
        for category, quantile in targets:
            target = float(np.quantile(errors, quantile))
            candidates = sorted(band_rows, key=lambda row: (abs(float(row.absolute_error) - target), row.sample_id))
            chosen = _take_unique(candidates, used)
            if chosen:
                selected.append(Selection(chosen.sample_id, category, band, f"nearest error quantile {quantile:.2f}"))
    under = sorted(valid, key=lambda row: (float(row.signed_error), row.sample_id))
    over = sorted(valid, key=lambda row: (-float(row.signed_error), row.sample_id))
    spatial_rows = (row for row in valid if row.spatial_metric_value is not None)
    if spatial_direction == "minimize":
        spatial = sorted(spatial_rows, key=lambda row: (-float(row.spatial_metric_value), row.sample_id))
    else:
        spatial = sorted(spatial_rows, key=lambda row: (float(row.spatial_metric_value), row.sample_id))
    for label, candidates in (("severe undercount", under), ("severe overcount", over), ("worst spatial quality", spatial)):
        chosen = _take_unique(candidates, used)
        if chosen:
            selected.append(Selection(chosen.sample_id, "failure", chosen.density_band, label))
    return tuple(selected[:12])


def selection_manifest(
    selections: Sequence[Selection],
    *,
    predictions_sha256: str,
    spatial_direction: Literal["minimize", "maximize"],
    density_band_rules: Sequence[str],
    expected_panels: int = 12,
) -> dict[str, object]:
    if not is_sha256(predictions_sha256):
        raise ValueError("selection manifest requires predictions SHA-256")
    return {
        "schema_version": 1,
        "selector": "density-stratified-error-quantiles-v1",
        "expected_panels": expected_panels,
        "selected_count": len(selections),
        "shortfall": max(expected_panels - len(selections), 0),
        "shortfall_reason": "" if len(selections) >= expected_panels else "insufficient unique eligible samples",
        "tie_breaker": "lexical sample_id",
        "spatial_direction": spatial_direction,
        "density_band_rules": list(density_band_rules),
        "predictions_sha256": predictions_sha256,
        "selections": [asdict(item) for item in selections],
    }
```

- [ ] **Step 4: Run curation tests**

Run: `python -m pytest tests/test_evaluation_curation.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit curation**

```bash
git add src/droneai/evaluation_curation.py tests/test_evaluation_curation.py
git commit -m "Select reproducible evaluation examples"
```

### Task 5: Render CSRNet-Style Four-Column Review Panels

**Files:**
- Create: `src/droneai/evaluation_panels.py`
- Create: `tests/test_evaluation_panels.py`

**Interfaces:**
- Consumes: `EvaluationSample`, retained `NativePrediction`, `ScalarEvaluation`.
- Produces: `render_review_panel(...) -> Path` with input, ground truth, native prediction, and operator-zone columns.

- [ ] **Step 1: Write failing density and point panel tests**

```python
from pathlib import Path

import numpy as np
from PIL import Image

from droneai.evaluation_contract import EvaluationSample, NativePrediction, ScalarEvaluation, ZoneBox
from droneai.evaluation_panels import render_review_panel
from droneai.integrity import sha256_file


def _sample(tmp_path: Path) -> EvaluationSample:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (64, 48), "gray").save(image_path)
    return EvaluationSample(
        sample_id="s1",
        dataset_id="fixture",
        split_id="validation",
        image_path=image_path,
        source_sha256=sha256_file(image_path),
        width=64,
        height=48,
        ground_truth_count=2.0,
        ground_truth_points=((10.0, 10.0), (40.0, 30.0)),
        ground_truth_density=np.ones((12, 16), dtype=np.float32) / 96.0,
        condition_tags={"density_band": "medium"},
        zones=(ZoneBox("a", 0, 0, 32, 48), ZoneBox("b", 32, 0, 64, 48)),
        has_point_annotations=True,
    )


def test_density_panel_has_four_equal_columns(tmp_path: Path) -> None:
    prediction = NativePrediction("s1", "density", 2.0, 5.0, 100.0, density=np.ones((12, 16), dtype=np.float32) / 96.0)
    record = ScalarEvaluation("s1", 2.0, 2.0, 0.0, 0.0, 0.0, "medium", 5.0, 100.0, "density", None, "game_l1", 0.0, {"density_ssim": 1.0})
    path = render_review_panel(_sample(tmp_path), prediction, record, tmp_path / "panel.png", model_id="fixture", checkpoint_sha256="b" * 64, zone_warning_count=1.0, zone_critical_count=2.0)
    with Image.open(path) as rendered:
        assert rendered.width == 4 * 320
        assert rendered.height == 365


def test_point_panel_renders_without_derived_density_claim(tmp_path: Path) -> None:
    prediction = NativePrediction("s1", "points", 2.0, 5.0, 100.0, points=((11.0, 11.0), (41.0, 31.0)))
    record = ScalarEvaluation("s1", 2.0, 2.0, 0.0, 0.0, 0.0, "medium", 5.0, 100.0, "points", None, "localization_f1", 1.0, {"localization_f1": 1.0})
    assert render_review_panel(_sample(tmp_path), prediction, record, tmp_path / "points.png", model_id="fixture", checkpoint_sha256="b" * 64, zone_warning_count=1.0, zone_critical_count=2.0).is_file()
```

- [ ] **Step 2: Run panel tests and verify missing module**

Run: `python -m pytest tests/test_evaluation_panels.py -q`

Expected: FAIL during collection for missing `droneai.evaluation_panels`.

- [ ] **Step 3: Implement fixed-size four-column rendering with native-output labels**

Implement `src/droneai/evaluation_panels.py` with these exact public functions:

```python
def fit_source(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, float, float, float]:
    scale = min(size[0] / image.width, size[1] / image.height)
    fitted_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    fitted = image.resize(fitted_size, Image.Resampling.LANCZOS)
    offset_x = (size[0] - fitted_size[0]) / 2
    offset_y = (size[1] - fitted_size[1]) / 2
    canvas = Image.new("RGB", size, "black")
    canvas.paste(fitted, (round(offset_x), round(offset_y)))
    return canvas, scale, offset_x, offset_y


def render_heatmap(values: np.ndarray, size: tuple[int, int], *, content_box: tuple[int, int, int, int]) -> Image.Image:
    normalized = np.asarray(values, dtype=np.float32)
    scale = float(normalized.max()) or 1.0
    unit = np.clip(normalized / scale, 0.0, 1.0)
    rgb = np.stack((unit, np.sqrt(unit), 1.0 - unit), axis=2)
    left, top, right, bottom = content_box
    heatmap = Image.fromarray(np.uint8(rgb * 255), "RGB").resize((right - left, bottom - top), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", size, "black")
    canvas.paste(heatmap, (left, top))
    return canvas


def draw_points(base: Image.Image, points: tuple[tuple[float, float], ...], *, scale: float, offset_x: float, offset_y: float, color: str) -> Image.Image:
    output = base.copy()
    draw = ImageDraw.Draw(output)
    for x, y in points:
        screen_x, screen_y = offset_x + x * scale, offset_y + y * scale
        draw.ellipse((screen_x - 3, screen_y - 3, screen_x + 3, screen_y + 3), outline=color, width=2)
    return output


def zone_counts(sample: EvaluationSample, prediction: NativePrediction) -> dict[str, float]:
    counts: dict[str, float] = {}
    if prediction.density is not None:
        density = np.asarray(prediction.density)
        map_height, map_width = density.shape
        for zone in sample.zones:
            x0, x1 = round(zone.x0 * map_width / sample.width), round(zone.x1 * map_width / sample.width)
            y0, y1 = round(zone.y0 * map_height / sample.height), round(zone.y1 * map_height / sample.height)
            counts[zone.zone_id] = float(density[y0:y1, x0:x1].sum())
    else:
        for zone in sample.zones:
            counts[zone.zone_id] = float(sum(zone.contains(x, y) for x, y in prediction.points))
    return counts


def render_review_panel(
    sample: EvaluationSample,
    prediction: NativePrediction,
    record: ScalarEvaluation,
    output_path: str | Path,
    *,
    model_id: str,
    checkpoint_sha256: str,
    zone_warning_count: float,
    zone_critical_count: float,
) -> Path:
    column_size = (320, 240)
    with Image.open(sample.image_path) as source_handle:
        source, image_scale, offset_x, offset_y = fit_source(source_handle.convert("RGB"), column_size)
    content_box = (round(offset_x), round(offset_y), round(offset_x + sample.width * image_scale), round(offset_y + sample.height * image_scale))
    if sample.ground_truth_density is not None:
        ground_truth = render_heatmap(sample.ground_truth_density, column_size, content_box=content_box)
        ground_truth_label = "ground-truth density"
    else:
        ground_truth = draw_points(source, sample.ground_truth_points, scale=image_scale, offset_x=offset_x, offset_y=offset_y, color="lime")
        ground_truth_label = "ground-truth points"
    if prediction.density is not None:
        predicted = render_heatmap(prediction.density, column_size, content_box=content_box)
        prediction_label = "native predicted density"
    else:
        predicted = draw_points(source, prediction.points, scale=image_scale, offset_x=offset_x, offset_y=offset_y, color="red")
        prediction_label = "native predicted points"
    operator = source.copy()
    operator_draw = ImageDraw.Draw(operator)
    counts = zone_counts(sample, prediction)
    for zone in sample.zones:
        count = counts[zone.zone_id]
        risk, color = ("red", "red") if count >= zone_critical_count else (("yellow", "yellow") if count >= zone_warning_count else ("green", "lime"))
        box = (offset_x + zone.x0 * image_scale, offset_y + zone.y0 * image_scale, offset_x + zone.x1 * image_scale, offset_y + zone.y1 * image_scale)
        operator_draw.rectangle(box, outline=color, width=2)
        operator_draw.text((box[0] + 3, box[1] + 3), f"{zone.zone_id}: {count:.1f} {risk}", fill=color)
    canvas = Image.new("RGB", (4 * column_size[0], 365), "white")
    columns = ((source, "input"), (ground_truth, ground_truth_label), (predicted, prediction_label), (operator, "operator zones"))
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(columns):
        x = index * column_size[0]
        canvas.paste(image, (x, 30))
        draw.text((x + 4, 8), label, fill="black")
    spatial = "unavailable" if record.spatial_metric_value is None else f"{record.spatial_metric_name}={record.spatial_metric_value:.3f}"
    confidence = "unavailable" if prediction.confidence is None else f"{prediction.confidence:.3f}"
    draw.text((4, 280), f"model={model_id} dataset={sample.dataset_id} checkpoint={checkpoint_sha256[:12]} sample={sample.sample_id} input={sample.width}x{sample.height}", fill="black")
    draw.text((4, 305), f"gt={sample.ground_truth_count:.2f} pred={prediction.predicted_count:.2f} abs_error={record.absolute_error:.2f} {spatial} confidence={confidence}", fill="black")
    draw.text((4, 330), f"latency_ms={record.latency_ms:.2f} peak_vram_mb={record.peak_vram_mb:.1f}", fill="black")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)
    return target
```

Include imports for `Path`, `numpy as np`, `Image`, `ImageDraw`, and the three evaluation contracts.

- [ ] **Step 4: Run panel tests**

Run: `python -m pytest tests/test_evaluation_panels.py -q`

Expected: `2 passed` and valid PNGs.

- [ ] **Step 5: Commit panel renderer**

```bash
git add src/droneai/evaluation_panels.py tests/test_evaluation_panels.py
git commit -m "Render four-column evaluation panels"
```

### Task 6: Persist Hash-Verified Small Review Artifacts

**Files:**
- Create: `src/droneai/evaluation_artifacts.py`
- Create: `tests/test_evaluation_artifacts.py`

**Interfaces:**
- Consumes: scalar records, summary, selection manifest, model brief, panel paths.
- Produces: `write_predictions_csv`, `write_json`, `artifact_reference`, and `enforce_review_budget`.

- [ ] **Step 1: Write failing artifact and size-budget tests**

```python
import json
from pathlib import Path

import pytest

from droneai.evaluation_artifacts import artifact_reference, enforce_review_budget, write_json
from droneai.integrity import sha256_file, verify_artifact_reference


def test_json_artifact_reference_recalculates_sha256(tmp_path: Path) -> None:
    path = write_json(tmp_path / "metrics.json", {"mae": 1.25})
    reference = artifact_reference(path, base_dir=tmp_path)
    assert verify_artifact_reference(reference, base_dir=tmp_path)[0]
    assert json.loads(path.read_text())["mae"] == 1.25


def test_json_writer_converts_non_finite_metrics_to_null(tmp_path: Path) -> None:
    path = write_json(tmp_path / "metrics.json", {"mae": float("inf")})
    assert json.loads(path.read_text())["mae"] is None


def test_review_budget_rejects_more_than_twenty_five_mib(tmp_path: Path) -> None:
    (tmp_path / "large.bin").write_bytes(b"0" * (25 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="25 MiB"):
        enforce_review_budget(tmp_path)
```

- [ ] **Step 2: Run artifact tests and confirm missing module**

Run: `python -m pytest tests/test_evaluation_artifacts.py -q`

Expected: FAIL during collection for missing `droneai.evaluation_artifacts`.

- [ ] **Step 3: Implement deterministic JSON/CSV writing, relative hashes, and budget enforcement**

```python
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from droneai.evaluation_contract import ScalarEvaluation
from droneai.integrity import sha256_file

REVIEW_BUDGET_BYTES = 25 * 1024 * 1024


def _json_safe(payload: object) -> object:
    if isinstance(payload, float) and not math.isfinite(payload):
        return None
    if isinstance(payload, dict):
        return {str(key): _json_safe(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(value) for value in payload]
    return payload


def write_json(path: str | Path, payload: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding="utf-8")
    return target


def write_predictions_csv(path: str | Path, records: Iterable[ScalarEvaluation]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        row = asdict(record)
        extra = row.pop("extra_metrics")
        conditions = row.pop("condition_values")
        row.update(extra)
        row.update({f"condition_{key}": value for key, value in sorted(conditions.items())})
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def artifact_reference(path: str | Path, *, base_dir: str | Path) -> dict[str, str]:
    target = Path(path)
    return {"path": target.relative_to(base_dir).as_posix(), "sha256": sha256_file(target)}


def enforce_review_budget(root: str | Path, limit_bytes: int = REVIEW_BUDGET_BYTES) -> int:
    total = sum(path.stat().st_size for path in Path(root).rglob("*") if path.is_file())
    if total > limit_bytes:
        raise ValueError(f"review bundle exceeds 25 MiB: {total} bytes")
    return total
```

- [ ] **Step 4: Run artifact tests**

Run: `python -m pytest tests/test_evaluation_artifacts.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit artifact persistence**

```bash
git add src/droneai/evaluation_artifacts.py tests/test_evaluation_artifacts.py
git commit -m "Persist small hash-verified evaluation artifacts"
```

### Task 7: Add the Model-Dataset 100-Point Technical Gate

**Files:**
- Create: `src/droneai/evaluation_gate.py`
- Create: `tests/test_evaluation_gate.py`

**Interfaces:**
- Consumes: `EvaluationEvidence` with frozen thresholds and artifact references.
- Produces: `build_evaluation_checks(evidence) -> list[CheckResult]` and `run_evaluation_gate(...) -> StageReport`.

- [ ] **Step 1: Write failing weight, blocker, and rights-separation tests**

```python
from dataclasses import replace

from droneai.evaluation_gate import EvaluationEvidence, build_evaluation_checks, run_evaluation_gate


def _evidence() -> EvaluationEvidence:
    return EvaluationEvidence(
        run_id="fixture-run",
        expected_samples=12,
        recorded_samples=12,
        successful_samples=12,
        split_verified=True,
        leakage_free=True,
        mae=0.1,
        mae_max=1.0,
        rmse=0.2,
        rmse_max=2.0,
        absolute_bias=0.0,
        bias_max=1.0,
        spatial_pass=True,
        robustness_pass=True,
        median_latency_ms=2.0,
        latency_max_ms=10.0,
        peak_vram_mb=100.0,
        vram_max_mb=1000.0,
        provenance_verified=True,
        model_brief_approved=True,
        review_artifacts_verified=True,
        review_budget_ok=True,
        explicit_failures=0,
    )


def test_evaluation_checks_total_one_hundred_and_pass() -> None:
    evidence = _evidence()
    checks = build_evaluation_checks(evidence)
    assert sum(check.weight for check in checks) == 100
    assert run_evaluation_gate(evidence).status == "PASS_RESEARCH_ONLY"


def test_incomplete_sample_accounting_is_blocked() -> None:
    evidence = replace(_evidence(), recorded_samples=11)
    report = run_evaluation_gate(evidence)
    assert report.status == "BLOCKED"
    assert "coverage.accounting" in report.to_dict()["failed_blockers"]


def test_failure_count_must_reconcile_with_success_count() -> None:
    evidence = replace(_evidence(), successful_samples=11, explicit_failures=0)
    assert run_evaluation_gate(evidence).status == "BLOCKED"


def test_rights_status_is_not_in_technical_points() -> None:
    evidence = _evidence()
    assert not any("rights" in check.check_id for check in build_evaluation_checks(evidence))
```

- [ ] **Step 2: Run gate tests and confirm missing module**

Run: `python -m pytest tests/test_evaluation_gate.py -q`

Expected: FAIL during collection for missing `droneai.evaluation_gate`.

- [ ] **Step 3: Implement frozen evidence and eight score groups**

Create `EvaluationEvidence` as a frozen dataclass with the fields shown in the test. Implement `build_evaluation_checks` with these exact weights:

```python
return [
    CheckResult("data.split", "data and split integrity", "Frozen split and no leakage", 15, evidence.split_verified and evidence.leakage_free, True, observed=f"split={evidence.split_verified}; leakage_free={evidence.leakage_free}"),
    CheckResult("coverage.accounting", "evaluation coverage", "Every expected sample has a success or explicit failure record", 10, evidence.recorded_samples == evidence.expected_samples and evidence.successful_samples + evidence.explicit_failures == evidence.expected_samples, True, expected=f"records={evidence.expected_samples}; successes+failures={evidence.expected_samples}", observed=f"records={evidence.recorded_samples}; successes+failures={evidence.successful_samples + evidence.explicit_failures}"),
    CheckResult("count.quality", "count quality", "MAE, RMSE, and absolute bias meet frozen targets", 20, evidence.mae <= evidence.mae_max and evidence.rmse <= evidence.rmse_max and evidence.absolute_bias <= evidence.bias_max, False, observed=f"mae={evidence.mae}/{evidence.mae_max}; rmse={evidence.rmse}/{evidence.rmse_max}; bias={evidence.absolute_bias}/{evidence.bias_max}"),
    CheckResult("spatial.quality", "spatial output quality", "Family-specific spatial targets pass", 15, evidence.spatial_pass, False, observed=str(evidence.spatial_pass)),
    CheckResult("condition.robustness", "condition robustness", "Density and condition targets pass", 10, evidence.robustness_pass, False, observed=str(evidence.robustness_pass)),
    CheckResult("runtime.resources", "runtime and resources", "Latency and peak VRAM meet frozen targets", 10, evidence.median_latency_ms <= evidence.latency_max_ms and evidence.peak_vram_mb <= evidence.vram_max_mb, False, observed=f"latency={evidence.median_latency_ms}/{evidence.latency_max_ms}; vram={evidence.peak_vram_mb}/{evidence.vram_max_mb}"),
    CheckResult("provenance.bundle", "reproducibility and provenance", "Code, data, checkpoint, config, and environment are verified", 10, evidence.provenance_verified, True, observed=str(evidence.provenance_verified)),
    CheckResult("review.bundle", "review artifacts", "Approved model brief and verified budgeted review bundle exist", 10, evidence.model_brief_approved and evidence.review_artifacts_verified and evidence.review_budget_ok, True, observed=f"brief={evidence.model_brief_approved}; artifacts={evidence.review_artifacts_verified}; budget={evidence.review_budget_ok}"),
]
```

Implement `run_evaluation_gate` as:

```python
def run_evaluation_gate(evidence: EvaluationEvidence) -> StageReport:
    return score_stage(
        stage_id="model-evaluation",
        stage_name=f"Model-dataset evaluation: {evidence.run_id}",
        threshold=80,
        checks=build_evaluation_checks(evidence),
        success_status="PASS_RESEARCH_ONLY",
    )
```

- [ ] **Step 4: Run gate tests**

Run: `python -m pytest tests/test_evaluation_gate.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit the common gate**

```bash
git add src/droneai/evaluation_gate.py tests/test_evaluation_gate.py
git commit -m "Add scored model dataset evaluation gate"
```

### Task 8: Orchestrate Inference, Online Metrics, Curation, and Rerender

**Files:**
- Create: `src/droneai/evaluation_runner.py`
- Create: `tests/test_evaluation_runner.py`

**Interfaces:**
- Consumes: approved `ModelAdapter`, complete non-test samples, frozen `EvaluationProtocol`.
- Produces: `run_evaluation(...) -> StageReport` and a complete small run bundle.

- [ ] **Step 1: Write a failing end-to-end fixture-adapter test**

The test writes a small deterministic fixture checkpoint under `tmp_path`, hashes it, defines `FixtureAdapter` with `brief()` returning that approved and hash-matched `ModelBrief`, and makes `predict()` return deterministic density maps. It creates 36 small validation samples across low/medium/high bands, calls:

```python
rights_path = tmp_path / "fixture-rights.json"
rights_path.write_text('{"scope":"research_only","production_approved":false}', encoding="utf-8")
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
    rights_decision_path=str(rights_path),
    rights_decision_sha256=sha256_file(rights_path),
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
    density_band_rules=("low: count < 20", "medium: 20 <= count < 100", "high: count >= 100"),
    required_condition_keys=("source",),
    zone_warning_count=25.0,
    zone_critical_count=75.0,
)
report = run_evaluation(
    adapter=adapter,
    samples=samples,
    protocol=protocol,
    output_dir=tmp_path / "run",
)
```

Assert:

```python
assert report.status == "PASS_RESEARCH_ONLY"
assert (tmp_path / "run/model-brief.md").is_file()
assert (tmp_path / "run/summary.md").is_file()
assert (tmp_path / "run/sample-manifest.json").is_file()
assert (tmp_path / "run/rights-decision.json").is_file()
assert (tmp_path / "run/predictions.csv").is_file()
assert (tmp_path / "run/selection_manifest.json").is_file()
assert len(list((tmp_path / "run/figures").glob("*.png"))) == 12
assert not (tmp_path / "run/native-outputs").exists()
assert (tmp_path / "run/score.json").is_file()
assert (tmp_path / "run/score.md").is_file()

with pytest.raises(PermissionError, match="sealed test"):
    run_evaluation(
        adapter=adapter,
        samples=samples,
        protocol=replace(protocol, split_role="test"),
        output_dir=tmp_path / "unapproved-test",
    )
```

Import `replace` from `dataclasses` and `pytest` in this test module.

- [ ] **Step 2: Run the runner test and verify missing module**

Run: `python -m pytest tests/test_evaluation_runner.py -q`

Expected: FAIL during collection for missing `droneai.evaluation_runner`.

- [ ] **Step 3: Implement `EvaluationProtocol` and two-pass `run_evaluation`**

Implement this frozen protocol so metric direction and condition thresholds are fixed before inference:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from droneai.evaluation_artifacts import artifact_reference, enforce_review_budget, write_json, write_predictions_csv
from droneai.evaluation_contract import EvaluationSample, ModelAdapter, NativePrediction, ScalarEvaluation
from droneai.evaluation_curation import select_review_samples, selection_manifest
from droneai.evaluation_gate import EvaluationEvidence, run_evaluation_gate
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
        if not self.run_id or not self.protocol_id or not self.dataset_id or not self.split_id or not self.spatial_metric_name or self.expected_samples <= 0:
            raise ValueError("evaluation protocol requires run identity and positive sample count")
        if not self.rights_decision_path or not is_sha256(self.rights_decision_sha256):
            raise ValueError("evaluation protocol requires a rights decision and SHA-256")
        if self.split_role not in {"train", "validation", "test", "smoke"}:
            raise ValueError("evaluation protocol has an invalid split role")
        if self.spatial_direction not in {"minimize", "maximize"}:
            raise ValueError("evaluation protocol has an invalid spatial direction")
        if not self.required_density_bands or not self.density_band_rules or not self.required_condition_keys:
            raise ValueError("evaluation protocol requires density bands, rules, and condition keys")
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
        if any(not np.isfinite(value) or value < 0 for value in numeric_targets):
            raise ValueError("evaluation protocol targets must be finite and non-negative")
        if self.zone_critical_count < self.zone_warning_count:
            raise ValueError("critical zone threshold must be at least the warning threshold")


def _spatial_pass(records: Sequence[ScalarEvaluation], protocol: EvaluationProtocol) -> tuple[bool, float | None]:
    successful = [row for row in records if row.predicted_count is not None]
    values = [
        float(row.spatial_metric_value)
        for row in successful
        if row.spatial_metric_name == protocol.spatial_metric_name and row.spatial_metric_value is not None
    ]
    if len(values) != len(successful) or not values:
        return False, None
    mean_value = float(np.mean(values))
    if protocol.spatial_direction == "minimize":
        return mean_value <= protocol.spatial_target, mean_value
    return mean_value >= protocol.spatial_target, mean_value


def _robustness_pass(summary: dict[str, object], protocol: EvaluationProtocol) -> bool:
    band_bias = summary["band_signed_bias"]
    condition_bias = summary["condition_signed_bias"]
    if not isinstance(band_bias, dict) or not isinstance(condition_bias, dict):
        return False
    bands_pass = all(
        band in band_bias and abs(float(band_bias[band])) <= protocol.band_bias_max
        for band in protocol.required_density_bands
    )
    conditions_pass = all(
        key in condition_bias
        and isinstance(condition_bias[key], dict)
        and bool(condition_bias[key])
        and all(abs(float(value)) <= protocol.condition_bias_max for value in condition_bias[key].values())
        for key in protocol.required_condition_keys
    )
    return bands_pass and conditions_pass
```

Then implement `run_evaluation` with this exact dataflow. `retain_native=False` means “do not persist the native array,” not “omit the native value needed for online metrics.” The second pass recomputes only selected samples and verifies its count against the first pass before rendering.

```python
def run_evaluation(
    *,
    adapter: ModelAdapter,
    samples: Sequence[EvaluationSample],
    protocol: EvaluationProtocol,
    output_dir: str | Path,
) -> StageReport:
    if protocol.split_role == "test" and not protocol.sealed_test_access_approved:
        raise PermissionError("sealed test evaluation requires explicit approval")
    output = Path(output_dir)
    reserved_outputs = (
        "predictions.csv", "metrics.json", "selection_manifest.json", "model-brief.md",
        "summary.md", "sample-manifest.json", "rights-decision.json", "environment-summary.json", "evidence-manifest.json", "score.json", "score.md", "figures",
    )
    if any((output / name).exists() for name in reserved_outputs):
        raise FileExistsError(f"evaluation output already contains report artifacts: {output}")
    output.mkdir(parents=True, exist_ok=True)

    brief = adapter.brief()
    brief.require_full_run_approval()
    checkpoint_path = Path(brief.checkpoint_path)
    checkpoint_verified = checkpoint_path.is_file() and sha256_file(checkpoint_path) == brief.checkpoint_sha256
    reviewed_paths_verified = all(Path(path).is_file() for path in brief.reviewed_paths)
    rights_source = Path(protocol.rights_decision_path)
    rights_verified = rights_source.is_file() and sha256_file(rights_source) == protocol.rights_decision_sha256
    if not rights_verified:
        raise ValueError("rights decision hash verification failed before inference")
    rights_path = write_json(output / "rights-decision.json", json.loads(rights_source.read_text(encoding="utf-8")))
    if len(samples) != protocol.expected_samples:
        raise ValueError("sample count differs from frozen protocol")
    samples_by_id = {sample.sample_id: sample for sample in samples}
    if len(samples_by_id) != len(samples):
        raise ValueError("evaluation sample IDs must be unique")
    if any(sample.dataset_id != protocol.dataset_id or sample.split_id != protocol.split_id for sample in samples):
        raise ValueError("sample dataset or split differs from frozen protocol")
    if any(not sample.image_path.is_file() or sha256_file(sample.image_path) != sample.source_sha256 for sample in samples):
        raise ValueError("sample source hash verification failed before inference")
    sample_manifest_path = write_json(
        output / "sample-manifest.json",
        {
            "schema_version": 1,
            "protocol_id": protocol.protocol_id,
            "dataset_id": protocol.dataset_id,
            "split_id": protocol.split_id,
            "split_role": protocol.split_role,
            "expected_samples": protocol.expected_samples,
            "samples": [
                {"sample_id": sample.sample_id, "source_sha256": sample.source_sha256}
                for sample in samples
            ],
        },
    )

    records: list[ScalarEvaluation] = []
    for sample in samples:
        prediction = adapter.predict(sample, retain_native=False)
        records.append(evaluate_sample(sample, prediction, localization_radius=protocol.localization_radius))

    predictions_path = write_predictions_csv(output / "predictions.csv", records)
    predictions_ref = artifact_reference(predictions_path, base_dir=output)
    summary = summarize_records(records, expected_samples=protocol.expected_samples)
    spatial_pass, spatial_mean = _spatial_pass(records, protocol)
    summary["spatial_metric_name"] = protocol.spatial_metric_name
    summary["spatial_direction"] = protocol.spatial_direction
    summary["spatial_target"] = protocol.spatial_target
    summary["spatial_mean"] = spatial_mean
    metrics_path = write_json(output / "metrics.json", summary)

    selections = select_review_samples(records, spatial_direction=protocol.spatial_direction)
    panel_paths: list[Path] = []
    records_by_id = {record.sample_id: record for record in records}
    for panel_index, selection in enumerate(selections, start=1):
        sample = samples_by_id[selection.sample_id]
        retained = adapter.predict(sample, retain_native=True)
        first = records_by_id[selection.sample_id]
        if retained.failure_state or first.failure_state:
            raise RuntimeError(f"selected sample cannot be rendered: {selection.sample_id}")
        if retained.output_type != first.output_type or abs(float(retained.predicted_count) - float(first.predicted_count)) > 1e-6:
            raise RuntimeError(f"selected-sample rerun is not deterministic: {selection.sample_id}")
        panel_paths.append(
            render_review_panel(
                sample,
                retained,
                records_by_id[selection.sample_id],
                output / "figures" / f"selected-{panel_index:02d}-{selection.sample_id}.png",
                model_id=brief.model_id,
                checkpoint_sha256=brief.checkpoint_sha256,
                zone_warning_count=protocol.zone_warning_count,
                zone_critical_count=protocol.zone_critical_count,
            )
        )

    selection_payload = selection_manifest(
        selections,
        predictions_sha256=predictions_ref["sha256"],
        spatial_direction=protocol.spatial_direction,
        density_band_rules=protocol.density_band_rules,
        expected_panels=12,
    )
    selection_payload["panel_paths"] = [path.relative_to(output).as_posix() for path in panel_paths]
    selection_path = write_json(output / "selection_manifest.json", selection_payload)
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
                f"- Samples: {summary['successful_samples']} successful, {summary['failures']} explicit failures",
                f"- MAE/RMSE/bias: {summary['mae']:.4f} / {summary['rmse']:.4f} / {summary['signed_bias']:.4f}",
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

    referenced_paths = [sample_manifest_path, rights_path, predictions_path, metrics_path, selection_path, brief_path, summary_path, environment_path, *panel_paths]
    references = [artifact_reference(path, base_dir=output) for path in referenced_paths]
    references_verified = all(verify_artifact_reference(reference, base_dir=output)[0] for reference in references)
    evidence_path = write_json(
        output / "evidence-manifest.json",
        {
            "schema_version": 1,
            "protocol": asdict(protocol),
            "checkpoint": {"path": brief.checkpoint_path, "sha256": brief.checkpoint_sha256, "verified": checkpoint_verified},
            "reviewed_paths": {"paths": list(brief.reviewed_paths), "verified": reviewed_paths_verified},
            "rights_decision": {"source_path": protocol.rights_decision_path, "source_sha256": protocol.rights_decision_sha256, "verified": rights_verified},
            "artifacts": references,
        },
    )
    references.append(artifact_reference(evidence_path, base_dir=output))

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
        robustness_pass=_robustness_pass(summary, protocol),
        median_latency_ms=float(summary["median_latency_ms"]),
        latency_max_ms=protocol.latency_max_ms,
        peak_vram_mb=float(summary["peak_vram_mb"]),
        vram_max_mb=protocol.vram_max_mb,
        provenance_verified=(
            references_verified
            and all(verify_artifact_reference(reference, base_dir=output)[0] for reference in references)
            and checkpoint_verified
            and reviewed_paths_verified
            and bool(environment.get("git_commit"))
            and (not protocol.require_clean_git or environment.get("git_dirty") is False)
        ),
        model_brief_approved=True,
        review_artifacts_verified=len(panel_paths) == 12 and selection_payload["shortfall"] == 0,
        review_budget_ok=True,
        explicit_failures=int(summary["failures"]),
    )
    report = run_evaluation_gate(evidence)
    write_json(output / "score.json", report.to_dict())
    (output / "score.md").write_text(report.to_markdown(), encoding="utf-8")
    try:
        enforce_review_budget(output)
    except ValueError:
        report = run_evaluation_gate(replace(evidence, review_budget_ok=False))
        write_json(output / "score.json", report.to_dict())
        (output / "score.md").write_text(report.to_markdown(), encoding="utf-8")
    return report
```

Do not store `first_predictions` or retained native arrays on disk. Use the existing helper interfaces above rather than duplicating hashing, rendering, metric, or gate logic.

- [ ] **Step 4: Run the runner test**

Run: `python -m pytest tests/test_evaluation_runner.py -q`

Expected: `1 passed` with the embedded sealed-test assertion and exactly 12 panels.

- [ ] **Step 5: Run focused evaluation suite**

Run: `python -m pytest tests/test_evaluation_contract.py tests/test_model_brief.py tests/test_evaluation_metrics.py tests/test_evaluation_curation.py tests/test_evaluation_panels.py tests/test_evaluation_artifacts.py tests/test_evaluation_gate.py tests/test_evaluation_runner.py -q`

Expected: all focused tests pass with zero failures.

- [ ] **Step 6: Commit orchestration**

```bash
git add src/droneai/evaluation_runner.py tests/test_evaluation_runner.py
git commit -m "Orchestrate model evaluation reports"
```

### Task 9: Add a Reproducible Fixture CLI and Review Command

**Files:**
- Create: `scripts/run_evaluation_fixture.py`
- Create: `tests/test_run_evaluation_fixture.py`
- Create: `configs/evaluation/fixture_density.json`

**Interfaces:**
- Consumes: `--config`, `--output-dir`, and optional `--device`.
- Produces: the same bundle as `run_evaluation` and exits nonzero on `BLOCKED` or `REVIEW`.

- [ ] **Step 1: Write failing CLI integration test**

```python
import json
import subprocess
import sys
from pathlib import Path


def test_fixture_cli_writes_review_bundle(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_evaluation_fixture.py", "--config", "configs/evaluation/fixture_density.json", "--output-dir", str(tmp_path / "run"), "--device", "cpu"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    score = json.loads((tmp_path / "run/score.json").read_text())
    assert score["status"] == "PASS_RESEARCH_ONLY"
```

- [ ] **Step 2: Run CLI test and confirm missing script**

Run: `python -m pytest tests/test_run_evaluation_fixture.py -q`

Expected: FAIL because `scripts/run_evaluation_fixture.py` does not exist.

- [ ] **Step 3: Add frozen fixture configuration**

```json
{
  "schema_version": 1,
  "run_id": "fixture-density-validation",
  "protocol_id": "common-evaluation-fixture-v1",
  "model_id": "fixture-density",
  "dataset_id": "synthetic-fixture-v1",
  "split_id": "validation",
  "split_role": "validation",
  "expected_samples": 36,
  "seed": 2026,
  "split_verified": true,
  "leakage_free": true,
  "sealed_test_access_approved": false,
  "require_clean_git": false,
  "localization_radius": 4.0,
  "targets": {
    "mae_max": 0.01,
    "rmse_max": 0.01,
    "bias_max": 0.01,
    "band_bias_max": 0.01,
    "condition_bias_max": 0.01,
    "spatial_metric_name": "game_l1",
    "spatial_direction": "minimize",
    "spatial_target": 0.01,
    "required_density_bands": ["low", "medium", "high"],
    "density_band_rules": ["low: count < 20", "medium: 20 <= count < 100", "high: count >= 100"],
    "required_condition_keys": ["source"],
    "zone_warning_count": 25.0,
    "zone_critical_count": 75.0,
    "latency_max_ms": 100.0,
    "vram_max_mb": 10.0
  }
}
```

- [ ] **Step 4: Implement deterministic fixture samples, adapter, and CLI exit rule**

Implement `scripts/run_evaluation_fixture.py` with these definitions. CPU mode uses fixed synthetic runtime values so the integration test is stable. CUDA mode performs a real CUDA tensor copy timed with CUDA events after a warm-up; it fails rather than pretending CUDA ran when CUDA is unavailable.

```python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal, Sequence, cast

import numpy as np
from PIL import Image

from droneai.evaluation_contract import EvaluationSample, NativePrediction, ZoneBox
from droneai.evaluation_runner import EvaluationProtocol, run_evaluation
from droneai.integrity import sha256_file
from droneai.model_brief import ModelBrief


class FixtureAdapter:
    def __init__(self, brief: ModelBrief, device: str) -> None:
        self._brief = brief
        self._device = device
        self._torch = None
        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA fixture requested but CUDA is unavailable")
            self._torch = torch
            warmup = torch.ones((16, 16), device="cuda")
            warmup = warmup @ warmup
            torch.cuda.synchronize()

    def brief(self) -> ModelBrief:
        return self._brief

    def predict(self, sample: EvaluationSample, *, retain_native: bool) -> NativePrediction:
        density = np.asarray(sample.ground_truth_density, dtype=np.float32).copy()
        latency_ms = 1.0
        peak_vram_mb = 0.0
        if self._torch is not None:
            torch = self._torch
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            tensor = torch.as_tensor(density, device="cuda")
            output = tensor.clone()
            end.record()
            torch.cuda.synchronize()
            latency_ms = max(float(start.elapsed_time(end)), 1e-6)
            peak_vram_mb = float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
            density = output.cpu().numpy()
        return NativePrediction(
            sample_id=sample.sample_id,
            output_type="density",
            predicted_count=float(density.sum()),
            latency_ms=latency_ms,
            peak_vram_mb=peak_vram_mb,
            density=density,
        )


def build_fixture_samples(output_dir: Path, *, dataset_id: str, split_id: str, seed: int) -> tuple[EvaluationSample, ...]:
    input_dir = output_dir / "fixture-input"
    input_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    samples: list[EvaluationSample] = []
    band_bases = {"low": 2.0, "medium": 20.0, "high": 100.0}
    for band_index, (band, base_count) in enumerate(band_bases.items()):
        for index in range(12):
            sample_id = f"{band}-{index:02d}"
            image_path = input_dir / f"{sample_id}.png"
            color = tuple(int(value) for value in rng.integers(32, 224, size=3))
            Image.new("RGB", (64, 48), color).save(image_path)
            count = base_count + float(index)
            density = np.zeros((12, 16), dtype=np.float32)
            density[(band_index * 3 + index) % 12, (index * 5) % 16] = count
            samples.append(
                EvaluationSample(
                    sample_id=sample_id,
                    dataset_id=dataset_id,
                    split_id=split_id,
                    image_path=image_path,
                    source_sha256=sha256_file(image_path),
                    width=64,
                    height=48,
                    ground_truth_count=count,
                    ground_truth_density=density,
                    condition_tags={"density_band": band, "source": "synthetic"},
                    zones=(ZoneBox("left", 0, 0, 32, 48), ZoneBox("right", 32, 0, 64, 48)),
                )
            )
    return tuple(samples)


def build_protocol(config: dict[str, object], *, rights_decision_path: Path) -> EvaluationProtocol:
    targets = config["targets"]
    if not isinstance(targets, dict):
        raise ValueError("fixture targets must be an object")
    return EvaluationProtocol(
        run_id=str(config["run_id"]),
        protocol_id=str(config["protocol_id"]),
        dataset_id=str(config["dataset_id"]),
        split_id=str(config["split_id"]),
        split_role=cast(Literal["train", "validation", "test", "smoke"], str(config["split_role"])),
        expected_samples=int(config["expected_samples"]),
        split_verified=bool(config["split_verified"]),
        leakage_free=bool(config["leakage_free"]),
        sealed_test_access_approved=bool(config["sealed_test_access_approved"]),
        require_clean_git=bool(config["require_clean_git"]),
        rights_decision_path=str(rights_decision_path.resolve()),
        rights_decision_sha256=sha256_file(rights_decision_path),
        localization_radius=float(config["localization_radius"]),
        mae_max=float(targets["mae_max"]),
        rmse_max=float(targets["rmse_max"]),
        bias_max=float(targets["bias_max"]),
        band_bias_max=float(targets["band_bias_max"]),
        condition_bias_max=float(targets["condition_bias_max"]),
        latency_max_ms=float(targets["latency_max_ms"]),
        vram_max_mb=float(targets["vram_max_mb"]),
        spatial_metric_name=str(targets["spatial_metric_name"]),
        spatial_direction=cast(Literal["minimize", "maximize"], str(targets["spatial_direction"])),
        spatial_target=float(targets["spatial_target"]),
        required_density_bands=tuple(str(value) for value in targets["required_density_bands"]),
        density_band_rules=tuple(str(value) for value in targets["density_band_rules"]),
        required_condition_keys=tuple(str(value) for value in targets["required_condition_keys"]),
        zone_warning_count=float(targets["zone_warning_count"]),
        zone_critical_count=float(targets["zone_critical_count"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"fixture output must be new or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = build_fixture_samples(
        args.output_dir,
        dataset_id=str(config["dataset_id"]),
        split_id=str(config["split_id"]),
        seed=int(config["seed"]),
    )
    checkpoint_path = args.output_dir / "fixture-input" / "fixture-checkpoint.bin"
    checkpoint_path.write_bytes(b"fixture-no-learned-weights-v1\n")
    rights_decision_path = args.output_dir / "fixture-input" / "fixture-rights.json"
    rights_decision_path.write_text(
        json.dumps({"scope": "research_only", "production_approved": False}, sort_keys=True),
        encoding="utf-8",
    )
    upstream_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    brief = ModelBrief(
        model_id=str(config["model_id"]),
        paper="Synthetic harness fixture; no research claim",
        role="common harness integration proof",
        family="density",
        backbone="identity density fixture",
        parameter_count=0,
        blocks=("fixture input", "identity density output"),
        feature_scales=("native 12x16 annotation grid",),
        input_contract="RGB image plus synthetic density annotation",
        preprocessing_policy="no resize, crop, normalization, or tiling",
        coordinate_transform="density grid is scaled to original 64x48 image coordinates",
        native_output="one-channel density map",
        count_derivation="sum of native density map",
        zone_derivation="integrate density inside proportionally scaled zone boxes",
        original_losses=("none; no training",),
        official_protocol="36 synthetic validation samples across three density bands",
        official_reported_metrics=("none; synthetic harness proof",),
        strengths=("deterministic count preservation",),
        failure_modes=("does not measure real model quality",),
        runtime_risks=("CUDA proof covers tensor execution only",),
        rights_status="PASS_RESEARCH_ONLY",
        code_rights_status="project synthetic fixture",
        dataset_rights_status="project synthetic fixture",
        checkpoint_rights_status="project synthetic fixture",
        deployment_rights_status="not evaluated",
        upstream_commit=upstream_commit,
        checkpoint_path=str(checkpoint_path.resolve()),
        checkpoint_sha256=sha256_file(checkpoint_path),
        reviewed_paths=("scripts/run_evaluation_fixture.py", "src/droneai/evaluation_runner.py"),
        review_status="approved",
    )
    report = run_evaluation(
        adapter=FixtureAdapter(brief, args.device),
        samples=samples,
        protocol=build_protocol(config, rights_decision_path=rights_decision_path),
        output_dir=args.output_dir,
    )
    print(report.to_markdown())
    return 0 if report.is_success else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run CLI integration test**

Run: `python -m pytest tests/test_run_evaluation_fixture.py -q`

Expected: `1 passed`.

- [ ] **Step 6: Run the fixture command manually**

Run:

```powershell
$fixtureRun = Join-Path $env:TEMP ("droneai-fixture-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
python scripts/run_evaluation_fixture.py --config configs/evaluation/fixture_density.json --output-dir $fixtureRun --device cpu
```

Expected: exit 0, `PASS_RESEARCH_ONLY`, score at least 80, 36 prediction rows, and 12 PNG panels. The command writes only to a new temporary directory and performs no cleanup; generated fixture output is never staged as a research result.

- [ ] **Step 7: Commit CLI and configuration**

```bash
git add scripts/run_evaluation_fixture.py tests/test_run_evaluation_fixture.py configs/evaluation/fixture_density.json
git commit -m "Add common evaluation fixture smoke"
```

### Task 10: Document the Review Gate and Verify Both Local and home5090 Runtimes

**Files:**
- Modify: `README.md`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/HOME5090_RUNBOOK.md`
- Create: `docs/COMMON_EVALUATION_HARNESS.md`

**Interfaces:**
- Consumes: all interfaces and commands from Tasks 1 through 9.
- Produces: user-review instructions and evidence for the next DM-Count/UCF-QNRF plan.

- [ ] **Step 1: Write the common harness document**

Document these exact sections:

```markdown
# Common evaluation harness

## Required review order
1. Model architecture brief and reviewed code paths.
2. Rights decision for the requested action.
3. Synthetic or approved non-test smoke.
4. Frozen protocol and complete labeled evaluation.
5. 100-point score, blockers, and 12 selected panels.
6. Explicit user approval before the next model.

## Artifact split
- Git: model brief, sample and rights manifests, predictions table, score, and 12 panels.
- home5090 SSD: checkpoints, full logs, native outputs, and large evidence.

## Commands
python -m pip install -e ".[dev,evaluation]"
python scripts/run_evaluation_fixture.py --config configs/evaluation/fixture_density.json --output-dir /workspace/data/results/fixture-density/smoke --device cuda
python -m pytest -q
```

Explain that this fixture proves the harness only; it is not a model-quality or production claim.

- [ ] **Step 2: Link the common harness document from README and roadmap**

Add this paragraph after the `## Current experiment` narrative in `README.md`:

```markdown
Every new model-dataset run uses the [common evaluation harness](docs/COMMON_EVALUATION_HARNESS.md): architecture review first, complete labeled evaluation second, then a 100-point gate and 12 deterministic review panels.
```

Add this section before `## Model and data order` in `docs/ROADMAP.md`:

```markdown
## Common per-model review gate

Before advancing to the next model, freeze the model brief, dataset/split/checkpoint identity, metric directions, thresholds, and expected sample count. Evaluate every sample as a success or explicit failure, then review `score.md`, `predictions.csv`, `selection_manifest.json`, and exactly 12 selected four-column panels. Technical score and Stage 3C rights scope remain separate decisions.
```

Add a `## Common evaluation fixture` section after `## Harness environment and smoke` in `docs/HOME5090_RUNBOOK.md` with the CUDA command from Step 7 and these inspection paths:

```text
<run-dir>/score.md
<run-dir>/score.json
<run-dir>/summary.md
<run-dir>/predictions.csv
<run-dir>/sample-manifest.json
<run-dir>/rights-decision.json
<run-dir>/selection_manifest.json
<run-dir>/environment-summary.json
<run-dir>/figures/*.png
```

- [ ] **Step 3: Run the entire local test suite**

Run: `python -m pytest -q`

Expected: all existing and new tests pass with zero failures.

- [ ] **Step 4: Verify the local fixture output and Git budget**

Run:

```powershell
$fixtureRun = Join-Path $env:TEMP ("droneai-evaluation-fixture-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
python scripts/run_evaluation_fixture.py --config configs/evaluation/fixture_density.json --output-dir $fixtureRun --device cpu
Get-ChildItem -Recurse $fixtureRun | Measure-Object -Property Length -Sum
```

Expected: `PASS_RESEARCH_ONLY`, 12 panels, and total bytes below 26,214,400.

- [ ] **Step 5: Push the completed implementation commits before server verification**

Run: `git push origin agent/stage3-dm-count-reproduction`

Expected: remote branch advances to the local implementation commit without force.

- [ ] **Step 6: Pull safely and run the focused suite in home5090 Docker**

Run from the laptop:

```bash
ssh home5090-pop "cd /home/lucita/crowd-counting-lab && git status --porcelain=v1 && git status --short --branch"
```

Expected before continuing: the first command prints no porcelain lines and shows the expected tracking branch. If it prints any changed or untracked path, stop without pulling.

Then run:

```bash
ssh home5090-pop "cd /home/lucita/crowd-counting-lab && git pull --ff-only && docker exec crowd-jupyter bash -lc 'cd /workspace && python -m venv --system-site-packages .venvs/harness && . .venvs/harness/bin/activate && python -m pip install -e \".[dev,evaluation]\" && python -m pytest -q'"
```

Expected: fast-forward pull, dependency install exit 0, and all tests pass.

- [ ] **Step 7: Run the CUDA fixture in home5090 Docker**

Run:

```powershell
$commit = git rev-parse --short HEAD
$remoteRun = "/workspace/data/results/common-harness-fixture-$commit"
ssh home5090-pop "docker exec crowd-jupyter bash -lc 'cd /workspace && test ! -e $remoteRun && . .venvs/harness/bin/activate && python scripts/run_evaluation_fixture.py --config configs/evaluation/fixture_density.json --output-dir $remoteRun --device cuda'"
```

Expected: `PASS_RESEARCH_ONLY`, RTX 5090 environment evidence, 36 prediction rows, 12 panels, score artifacts, and no dataset or checkpoint download.

- [ ] **Step 8: Pull only the small fixture review bundle for inspection**

Run:

```powershell
$localRun = Join-Path $env:TEMP ("home5090-common-harness-fixture-" + $commit)
scp -r "home5090-pop:/mnt/crowd-data/CrowdCounting/results/common-harness-fixture-$commit" $localRun
```

Expected: score, predictions, sample and selection manifests, rights decision, model brief, environment summary, and 12 selected PNG panels are present locally.

- [ ] **Step 9: Commit documentation after verification**

```bash
git add README.md docs/ROADMAP.md docs/HOME5090_RUNBOOK.md docs/COMMON_EVALUATION_HARNESS.md
git commit -m "Document common model evaluation workflow"
git push origin agent/stage3-dm-count-reproduction
```

## Completion Gate for This Plan

Do not start the DM-Count/UCF-QNRF adapter plan until all of these are true:

- the full local suite passes;
- the home5090 Docker suite passes;
- the CUDA fixture produces a complete 100-point report and exactly 12 selected panels;
- the Git review bundle is below 25 MiB;
- no real dataset, checkpoint, or long-training artifact was downloaded or created;
- the user has reviewed the fixture model brief, representative panel layout, score, and blockers.
