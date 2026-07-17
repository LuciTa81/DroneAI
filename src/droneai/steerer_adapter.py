"""Official STEERER adapter for frozen UCF-QNRF checkpoint evaluation."""

from __future__ import annotations

import importlib
import math
import sys
import time
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from droneai.dm_count_adapter import _git_head, _git_status
from droneai.evaluation_contract import (
    EvaluationSample,
    ModelAdapter,
    NativePrediction,
    Point,
)
from droneai.integrity import is_sha256, sha256_file
from droneai.model_brief import ModelBrief

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
_DENSITY_FACTOR = 100.0
_REVIEWED_PATHS = (
    "configs/QNRF_final.py",
    "lib/models/build_counter.py",
    "lib/utils/points_from_den.py",
    "lib/core/cc_function.py",
    "lib/datasets/base_dataset.py",
    "lib/datasets/nwpu.py",
)
_UPSTREAM_NAMESPACE_ROOTS = ("lib", "mmcv_custom")


class STEERERBackend(Protocol):
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    def infer(
        self, normalized_chw: np.ndarray
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[Point, ...],
        float,
        float,
    ]:
        """Return x1/x4/x8 densities, processed points, latency, and peak VRAM."""


def _resized_dimensions(
    width: int, height: int, long_side_cap: int
) -> tuple[int, int, float]:
    if width <= 0 or height <= 0 or long_side_cap <= 0:
        raise ValueError("positive image dimensions and long-side cap are required")
    long_side = max(width, height)
    if long_side <= long_side_cap:
        return width, height, 1.0
    ratio = long_side_cap / long_side
    if width >= height:
        resized_width = long_side_cap
        resized_height = int(height * ratio + 0.5)
    else:
        resized_height = long_side_cap
        resized_width = int(width * ratio + 0.5)
    return resized_width, resized_height, float(ratio)


def calculate_steerer_size(
    width: int, height: int, long_side_cap: int = 3072
) -> tuple[int, int, float]:
    """Apply the official long-side cap, then bottom/right padding to 32."""

    resized_width, resized_height, ratio = _resized_dimensions(
        width, height, long_side_cap
    )
    padded_width = math.ceil(resized_width / 32) * 32
    padded_height = math.ceil(resized_height / 32) * 32
    return padded_width, padded_height, ratio


def _local_maximum_points_numpy(
    density: np.ndarray,
    *,
    gaussian_maximum: float,
    patch_size: int,
    density_scale: float,
    threshold: float,
) -> np.ndarray:
    source = np.asarray(density, dtype=np.float32)
    if source.ndim != 2 or not np.isfinite(source).all() or np.any(source < 0):
        raise ValueError("STEERER densities must be finite, non-negative 2D arrays")

    height, width = source.shape
    padded_height = math.ceil(height / patch_size) * patch_size
    padded_width = math.ceil(width / patch_size) * patch_size
    padded = np.zeros((padded_height, padded_width), dtype=np.float32)
    padded[:height, :width] = source

    patch_maximum = padded.reshape(
        padded_height // patch_size,
        patch_size,
        padded_width // patch_size,
        patch_size,
    ).max(axis=(1, 3))
    local_threshold = patch_maximum * threshold
    np.maximum(local_threshold, threshold * gaussian_maximum, out=local_threshold)
    np.minimum(local_threshold, 0.3 * gaussian_maximum, out=local_threshold)
    local_threshold = np.repeat(
        np.repeat(local_threshold, patch_size, axis=0), patch_size, axis=1
    )

    bordered = np.pad(padded, 1, mode="constant", constant_values=-np.inf)
    neighborhood_maximum = np.full_like(padded, -np.inf)
    for y_offset in range(3):
        for x_offset in range(3):
            np.maximum(
                neighborhood_maximum,
                bordered[
                    y_offset : y_offset + padded_height,
                    x_offset : x_offset + padded_width,
                ],
                out=neighborhood_maximum,
            )
    keep = (padded == neighborhood_maximum) & (padded >= local_threshold) & (padded > 0)
    ys, xs = np.nonzero(keep)
    points = np.column_stack((xs, ys)).astype(np.float32, copy=False)
    return points * density_scale


def _merge_multiscale_points(point_sets: Sequence[np.ndarray]) -> np.ndarray:
    if len(point_sets) != 3:
        raise ValueError("STEERER point merging requires x1, x4, and x8 outputs")
    merged = np.asarray(point_sets[0], dtype=np.float32).reshape(-1, 2).copy()
    for candidates, distance_threshold in zip(point_sets[1:], (32.0, 64.0)):
        additions = np.asarray(candidates, dtype=np.float32).reshape(-1, 2)
        if merged.size == 0:
            merged = additions.copy()
            continue
        if additions.size == 0:
            continue
        distances = np.linalg.norm(
            merged[:, np.newaxis, :] - additions[np.newaxis, :, :], axis=2
        )
        merged = np.concatenate(
            (merged, additions[distances.min(axis=0) > distance_threshold]), axis=0
        )
    return merged


def _points_to_original(
    points: Sequence[Point] | np.ndarray,
    *,
    resize_ratio: float,
    original_size: tuple[int, int],
) -> tuple[Point, ...]:
    width, height = original_size
    if (
        not math.isfinite(resize_ratio)
        or resize_ratio <= 0
        or width <= 0
        or height <= 0
    ):
        raise ValueError("valid resize ratio and original size are required")
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(array).all():
        raise ValueError("STEERER points must contain finite coordinates")
    array /= resize_ratio
    array[:, 0] = np.clip(array[:, 0], 0.0, np.nextafter(float(width), -np.inf))
    array[:, 1] = np.clip(array[:, 1], 0.0, np.nextafter(float(height), -np.inf))
    return tuple((float(x), float(y)) for x, y in array)


def _exclude_padding_points(
    points: Sequence[Point] | np.ndarray, *, valid_size: tuple[int, int]
) -> tuple[np.ndarray, int]:
    valid_width, valid_height = valid_size
    if valid_width <= 0 or valid_height <= 0:
        raise ValueError("valid resized point extent is required")
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(array).all():
        raise ValueError("STEERER points must contain finite coordinates")
    keep = (array[:, 0] < valid_width) & (array[:, 1] < valid_height)
    return array[keep], int((~keep).sum())


def _remap_density_axis(
    density: np.ndarray,
    *,
    axis: int,
    processed_extent: int,
    valid_extent: int,
    target_cells: int,
) -> np.ndarray:
    """Integrate uniform source-cell mass into target cells over a valid extent."""

    source_cells = density.shape[axis]
    boundary_positions = np.linspace(
        0.0,
        valid_extent * source_cells / processed_extent,
        target_cells + 1,
        dtype=np.float64,
    )
    np.clip(boundary_positions, 0.0, float(source_cells), out=boundary_positions)
    source_indices = np.floor(boundary_positions).astype(np.int64)
    fractions = boundary_positions - source_indices
    fractions[source_indices == source_cells] = 0.0

    prefix_shape = list(density.shape)
    prefix_shape[axis] = 1
    cumulative = np.concatenate(
        (
            np.zeros(prefix_shape, dtype=np.float64),
            np.cumsum(density, axis=axis, dtype=np.float64),
        ),
        axis=axis,
    )
    boundary_mass = np.take(cumulative, source_indices, axis=axis)
    cell_mass = np.take(
        density,
        np.minimum(source_indices, source_cells - 1),
        axis=axis,
    )
    fraction_shape = [1] * density.ndim
    fraction_shape[axis] = fractions.size
    boundary_mass += cell_mass * fractions.reshape(fraction_shape)
    return np.diff(boundary_mass, axis=axis)


def _remap_density_to_valid_extent(
    density: np.ndarray,
    *,
    processed_size: tuple[int, int],
    valid_size: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, float, float]:
    """Remap padded native-cell mass to valid image space and report discarded mass."""

    source = np.asarray(density, dtype=np.float64)
    if source.ndim != 2 or not np.isfinite(source).all() or np.any(source < 0):
        raise ValueError("STEERER density must be finite, non-negative, and two-dimensional")
    processed_width, processed_height = processed_size
    valid_width, valid_height = valid_size
    target_height, target_width = target_shape
    if (
        processed_width <= 0
        or processed_height <= 0
        or valid_width <= 0
        or valid_height <= 0
        or valid_width > processed_width
        or valid_height > processed_height
        or target_width <= 0
        or target_height <= 0
    ):
        raise ValueError("valid processed, resized, and density target sizes are required")

    remapped_x = _remap_density_axis(
        source,
        axis=1,
        processed_extent=processed_width,
        valid_extent=valid_width,
        target_cells=target_width,
    )
    remapped = _remap_density_axis(
        remapped_x,
        axis=0,
        processed_extent=processed_height,
        valid_extent=valid_height,
        target_cells=target_height,
    )
    np.maximum(remapped, 0.0, out=remapped)
    valid_mass = float(remapped.sum(dtype=np.float64))
    source_mass = float(source.sum(dtype=np.float64))
    discarded_mass = max(source_mass - valid_mass, 0.0)
    discarded_fraction = discarded_mass / source_mass if source_mass > 0.0 else 0.0

    output = np.asarray(remapped, dtype=np.float32)
    output_mass = float(output.sum(dtype=np.float64))
    if valid_mass == 0.0:
        output.fill(0.0)
    elif output_mass <= 0.0 or not math.isfinite(output_mass):
        raise ValueError("STEERER density remap lost positive valid-image mass")
    else:
        output *= valid_mass / output_mass
    return output, float(discarded_mass), float(discarded_fraction)


def extract_steerer_points(
    densities: Sequence[np.ndarray],
    gaussian_maximum: float,
    resize_ratio: float,
    original_size: tuple[int, int],
    threshold: float = 0.15,
) -> tuple[Point, ...]:
    """Reproduce official x1/x4/x8 local maxima and distance-based merging."""

    if len(densities) != 3:
        raise ValueError("STEERER localization requires x1, x4, and x8 densities")
    if (
        not math.isfinite(gaussian_maximum)
        or gaussian_maximum <= 0
        or not math.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("positive finite localization parameters are required")
    point_sets = tuple(
        _local_maximum_points_numpy(
            density,
            gaussian_maximum=gaussian_maximum,
            patch_size=patch_size,
            density_scale=density_scale,
            threshold=threshold,
        )
        for density, patch_size, density_scale in zip(
            densities, (32, 32, 16), (1.0, 4.0, 8.0)
        )
    )
    return _points_to_original(
        _merge_multiscale_points(point_sets),
        resize_ratio=resize_ratio,
        original_size=original_size,
    )


def _is_upstream_namespace(name: str) -> bool:
    return any(
        name == root or name.startswith(f"{root}.")
        for root in _UPSTREAM_NAMESPACE_ROOTS
    )


def _clear_upstream_namespaces() -> None:
    for name in tuple(sys.modules):
        if _is_upstream_namespace(name):
            del sys.modules[name]


def _validate_upstream_module_origins(upstream_dir: Path) -> None:
    upstream_root = upstream_dir.resolve()
    for name, module in tuple(sys.modules.items()):
        if not _is_upstream_namespace(name):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        if not Path(origin).resolve().is_relative_to(upstream_root):
            raise RuntimeError(
                f"official namespace module {name} originated outside the pinned "
                f"STEERER upstream: {origin}"
            )


def _load_official_components(upstream_dir: Path):
    """Import pinned components without leaking or reusing generic namespaces."""

    namespace_snapshot = {
        name: module
        for name, module in sys.modules.items()
        if _is_upstream_namespace(name)
    }
    previous_dont_write_bytecode = sys.dont_write_bytecode
    previous_sys_path = list(sys.path)
    _clear_upstream_namespaces()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream_dir))
    try:
        try:
            from mmcv import Config
        except ImportError as error:  # pragma: no cover - exercised on home5090
            raise RuntimeError("MMCV is required for the official STEERER backend") from error
        build_counter = importlib.import_module("lib.models.build_counter")
        points_from_den = importlib.import_module("lib.utils.points_from_den")
        _validate_upstream_module_origins(upstream_dir)
        return Config, build_counter.Baseline_Counter, points_from_den.local_maximum_points
    finally:
        _clear_upstream_namespaces()
        sys.modules.update(namespace_snapshot)
        sys.path[:] = previous_sys_path
        sys.dont_write_bytecode = previous_dont_write_bytecode


class TorchSTEERERBackend:
    """Lazy CUDA backend for the pinned official STEERER implementation."""

    def __init__(self, *, upstream_dir: Path, checkpoint_path: Path, device: str):
        try:
            import torch
        except ImportError as error:  # pragma: no cover - exercised on home5090
            raise RuntimeError("PyTorch is required for the official STEERER backend") from error

        Config, Baseline_Counter, local_maximum_points = _load_official_components(
            upstream_dir
        )
        self._torch = torch
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise ValueError("the official STEERER backend requires a CUDA device")
        config = Config.fromfile(str(upstream_dir / "configs" / "QNRF_final.py"))
        model = Baseline_Counter(
            config.network,
            config.dataset.den_factor,
            config.train.route_size,
            self._device,
        )
        try:
            state = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
        except TypeError:  # pragma: no cover - old PyTorch compatibility
            state = torch.load(checkpoint_path, map_location=self._device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        incompatible = model.load_state_dict(state, strict=False)
        self.missing_keys = tuple(incompatible.missing_keys)
        self.unexpected_keys = tuple(incompatible.unexpected_keys)
        self._model = model.to(self._device).eval()
        self._local_maximum_points = local_maximum_points

    def infer(
        self, normalized_chw: np.ndarray
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[Point, ...],
        float,
        float,
    ]:
        torch = self._torch
        batch = torch.from_numpy(np.ascontiguousarray(normalized_chw)).unsqueeze(0)
        batch = batch.to(self._device)

        torch.cuda.reset_peak_memory_stats(self._device)
        torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self._model(batch, labels=None)
        torch.cuda.synchronize(self._device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        peak_vram_mb = torch.cuda.max_memory_allocated(self._device) / (1024**2)

        density_tensors = (
            outputs[0] / _DENSITY_FACTOR,
            outputs[-2] / _DENSITY_FACTOR,
            outputs[-1] / _DENSITY_FACTOR,
        )
        point_sets = tuple(
            self._local_maximum_points(
                density.clone(),
                self._model.gaussian_maximum,
                patch_size=patch_size,
                den_scale=density_scale,
                threshold=0.15,
            )["points"]
            for density, patch_size, density_scale in zip(
                density_tensors, (32, 32, 16), (1.0, 4.0, 8.0)
            )
        )
        processed_points = _merge_multiscale_points(point_sets)
        densities = tuple(
            density[0, 0].detach().float().cpu().numpy() for density in density_tensors
        )
        return (
            densities,
            tuple((float(x), float(y)) for x, y in processed_points),
            max(float(latency_ms), 1e-9),
            float(peak_vram_mb),
        )


class STEERERAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        upstream_dir: str | Path,
        expected_upstream_commit: str,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        device: str,
        backend: STEERERBackend | None = None,
        long_side_cap: int = 3072,
    ):
        self.upstream_dir = Path(upstream_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.expected_upstream_commit = expected_upstream_commit.lower()
        self.checkpoint_sha256 = checkpoint_sha256.lower()
        self.device = device
        self.long_side_cap = long_side_cap
        if not self.upstream_dir.is_dir() or not (
            self.upstream_dir / "configs" / "QNRF_final.py"
        ).is_file():
            raise FileNotFoundError(
                "official STEERER upstream with configs/QNRF_final.py is required"
            )
        if _git_head(self.upstream_dir) != self.expected_upstream_commit:
            raise ValueError("STEERER upstream commit does not match the frozen commit")
        if _git_status(self.upstream_dir):
            raise ValueError("STEERER upstream working tree must be clean")
        if not self.checkpoint_path.is_file() or not is_sha256(self.checkpoint_sha256):
            raise FileNotFoundError("STEERER checkpoint and SHA-256 are required")
        observed_checkpoint_hash = sha256_file(self.checkpoint_path).lower()
        if observed_checkpoint_hash != self.checkpoint_sha256:
            raise ValueError(
                "checkpoint SHA-256 mismatch: "
                f"expected={self.checkpoint_sha256} observed={observed_checkpoint_hash}"
            )
        self._backend = backend or TorchSTEERERBackend(
            upstream_dir=self.upstream_dir,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

    def _normalized_image(
        self, sample: EvaluationSample
    ) -> tuple[np.ndarray, dict[str, str | int | float | bool | None]]:
        processed_width, processed_height, ratio = calculate_steerer_size(
            sample.width, sample.height, self.long_side_cap
        )
        resized_width, resized_height, _ = _resized_dimensions(
            sample.width, sample.height, self.long_side_cap
        )
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            if image.size != (sample.width, sample.height):
                raise ValueError("sample dimensions differ from the source image")
            if image.size != (resized_width, resized_height):
                image = image.resize(
                    (resized_width, resized_height), resample=Image.Resampling.BILINEAR
                )
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        padded = np.zeros((processed_height, processed_width, 3), dtype=np.float32)
        padded[:resized_height, :resized_width] = pixels
        normalized = np.ascontiguousarray(
            np.transpose((padded - IMAGENET_MEAN) / IMAGENET_STD, (2, 0, 1))
        )
        metadata: dict[str, str | int | float | bool | None] = {
            "original_width": sample.width,
            "original_height": sample.height,
            "resized_width": resized_width,
            "resized_height": resized_height,
            "processed_width": processed_width,
            "processed_height": processed_height,
            "resize_ratio": ratio,
            "coordinate_transform": (
                "processed localization coordinates divided by geometric resize ratio "
                "and clipped to original pixels; density mass-preserving resized"
            ),
        }
        return normalized, metadata

    def brief(self) -> ModelBrief:
        return ModelBrief(
            model_id="steerer-official-ucf-qnrf",
            paper="STEERER: Resolving Scale Variations for Counting and Localization via Selective Inheritance Learning",
            role="UCF-QNRF crowd-counting and localization research baseline",
            family="hybrid density and point localization",
            backbone="HRNet-W48",
            parameter_count=None,
            blocks=(
                "HRNet-W48 multi-resolution backbone",
                "multi-resolution counting heads",
                "selective inheritance/upsample module",
                "Gaussian density/local-maximum point decoder",
            ),
            feature_scales=("x1", "x4", "x8"),
            input_contract="one RGB UCF-QNRF image with frozen point annotations",
            preprocessing_policy=(
                "official long-side 3072 cap, bottom/right zero padding to multiples "
                "of 32, and ImageNet normalization"
            ),
            coordinate_transform=(
                "density mass-preserving resize and merged points divided by the "
                "geometric resize ratio then clipped to original pixels"
            ),
            native_output="multi-resolution density maps and merged localization points",
            count_derivation="sum of highest-resolution density-map mass",
            zone_derivation="fractional integration of count-preserving density over calibrated image zones",
            original_losses=(
                "multi-resolution MSE",
                "selective inheritance routing loss",
                "uncertainty weighting where configured",
            ),
            official_protocol=(
                "pinned QNRF_final.py, den_factor=100, batch size 1, frozen research checkpoint"
            ),
            official_reported_metrics=(
                "UCF-QNRF MAE 77.8",
                "UCF-QNRF RMSE 138.0",
                "UCF-QNRF localization F1 75.6",
            ),
            strengths=(
                "multi-scale density counting",
                "official local-maximum point localization",
            ),
            failure_modes=(
                "tiny heads",
                "extreme perspective",
                "occlusion",
                "domain shift",
                "padding artifacts",
                "threshold-sensitive localization",
            ),
            runtime_risks=(
                "large high-resolution CUDA activations",
                "legacy MMCV and upstream API compatibility",
            ),
            rights_status="RESEARCH_CHECKPOINT_EVALUATION_ONLY",
            code_rights_status="MIT verified at pinned upstream commit",
            dataset_rights_status="Apache-2.0 accepted by project owner with provenance risk recorded",
            checkpoint_rights_status="unverified; internal approval permits frozen research comparison only",
            deployment_rights_status="pending; research comparison is not deployment approval",
            upstream_commit=self.expected_upstream_commit,
            checkpoint_path=str(self.checkpoint_path),
            checkpoint_sha256=self.checkpoint_sha256,
            reviewed_paths=tuple(
                str((self.upstream_dir / relative).resolve())
                for relative in _REVIEWED_PATHS
            ),
            review_status="approved",
        )

    def predict(self, sample: EvaluationSample, *, retain_native: bool) -> NativePrediction:
        started = time.perf_counter()
        metadata: dict[str, str | int | float | bool | None] = {}
        try:
            normalized, metadata = self._normalized_image(sample)
            densities, processed_points, latency_ms, peak_vram_mb = self._backend.infer(
                normalized
            )
            if len(densities) != 3:
                raise ValueError("STEERER backend must return x1, x4, and x8 densities")
            native_density = np.asarray(densities[0], dtype=np.float32)
            target_shape = (
                tuple(sample.ground_truth_density.shape)
                if sample.ground_truth_density is not None
                else (sample.height, sample.width)
            )
            density, discarded_padding_mass, discarded_padding_fraction = (
                _remap_density_to_valid_extent(
                    native_density,
                    processed_size=(
                        int(metadata["processed_width"]),
                        int(metadata["processed_height"]),
                    ),
                    valid_size=(
                        int(metadata["resized_width"]),
                        int(metadata["resized_height"]),
                    ),
                    target_shape=target_shape,
                )
            )
            valid_processed_points, discarded_padding_point_count = (
                _exclude_padding_points(
                    processed_points,
                    valid_size=(
                        int(metadata["resized_width"]),
                        int(metadata["resized_height"]),
                    ),
                )
            )
            points = _points_to_original(
                valid_processed_points,
                resize_ratio=float(metadata["resize_ratio"]),
                original_size=(sample.width, sample.height),
            )
            missing_keys = tuple(getattr(self._backend, "missing_keys", ()))
            unexpected_keys = tuple(getattr(self._backend, "unexpected_keys", ()))
            metadata.update(
                {
                    "native_density_width": int(native_density.shape[1]),
                    "native_density_height": int(native_density.shape[0]),
                    "evaluation_density_width": int(density.shape[1]),
                    "evaluation_density_height": int(density.shape[0]),
                    "discarded_padding_mass": discarded_padding_mass,
                    "discarded_padding_mass_fraction": discarded_padding_fraction,
                    "discarded_padding_point_count": discarded_padding_point_count,
                    "localization_point_count": len(points),
                    "missing_checkpoint_key_count": len(missing_keys),
                    "unexpected_checkpoint_key_count": len(unexpected_keys),
                    "missing_checkpoint_keys": ",".join(missing_keys),
                    "unexpected_checkpoint_keys": ",".join(unexpected_keys),
                }
            )
            count = float(density.sum(dtype=np.float64))
            return NativePrediction(
                sample_id=sample.sample_id,
                output_type="hybrid",
                predicted_count=count,
                latency_ms=float(latency_ms),
                peak_vram_mb=float(peak_vram_mb),
                density=density,
                points=points,
                coordinate_space="original_pixels",
                metadata=metadata,
            )
        except Exception as error:
            elapsed_ms = max((time.perf_counter() - started) * 1000.0, 1e-9)
            return NativePrediction(
                sample_id=sample.sample_id,
                output_type="hybrid",
                predicted_count=None,
                latency_ms=elapsed_ms,
                peak_vram_mb=0.0,
                density=None,
                points=(),
                failure_state=f"{type(error).__name__}: {error}",
                metadata=metadata,
            )
