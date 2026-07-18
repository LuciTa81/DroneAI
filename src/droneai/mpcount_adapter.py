"""Official deterministic MPCount adapter for frozen UCF-QNRF evaluation."""

from __future__ import annotations

import importlib
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from droneai.dm_count_adapter import (
    _git_head,
    _git_status,
    resize_density_preserve_mass,
)
from droneai.evaluation_contract import EvaluationSample, NativePrediction
from droneai.integrity import is_sha256, sha256_file
from droneai.model_brief import ModelBrief


_UPSTREAM_ROOTS = ("models", "utils", "datasets", "trainers", "inference", "main")
_REVIEWED_PATHS = (
    "README.md",
    "LICENSE",
    "inference.py",
    "models/models.py",
    "utils/misc.py",
)


class MPCountBackend(Protocol):
    parameter_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    def infer(
        self, normalized_bchw: np.ndarray, patch_size: int
    ) -> tuple[np.ndarray, float, float]:
        """Return padded raw density, synchronized latency ms, and peak VRAM MB."""


def calculate_padding(
    width: int, height: int, unit_size: int
) -> tuple[int, int, int, int, int, int]:
    if width <= 0 or height <= 0 or unit_size <= 0:
        raise ValueError("positive image dimensions and unit size are required")
    padded_width = math.ceil(width / unit_size) * unit_size
    padded_height = math.ceil(height / unit_size) * unit_size
    horizontal = padded_width - width
    vertical = padded_height - height
    left = horizontal // 2
    right = horizontal - left
    top = vertical // 2
    bottom = vertical - top
    return left, top, right, bottom, padded_width, padded_height


def _is_upstream_name(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in _UPSTREAM_ROOTS)


def _clear_upstream_namespaces() -> None:
    for name in tuple(sys.modules):
        if _is_upstream_name(name):
            del sys.modules[name]


@contextmanager
def _official_namespace_scope(upstream_dir: Path):
    snapshot = {
        name: module for name, module in sys.modules.items() if _is_upstream_name(name)
    }
    previous_path = list(sys.path)
    previous_bytecode = sys.dont_write_bytecode
    _clear_upstream_namespaces()
    sys.path.insert(0, str(upstream_dir))
    sys.dont_write_bytecode = True
    try:
        yield
        root = upstream_dir.resolve()
        for name, module in tuple(sys.modules.items()):
            if not _is_upstream_name(name):
                continue
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(root):
                raise RuntimeError(
                    f"official namespace {name} originated outside pinned MPCount: "
                    f"{origin}"
                )
    finally:
        _clear_upstream_namespaces()
        sys.modules.update(snapshot)
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode


class TorchMPCountBackend:
    """CUDA backend for the exact official deterministic MPCount checkpoint."""

    def __init__(self, *, upstream_dir: Path, checkpoint_path: Path, device: str):
        try:
            import torch
        except ImportError as error:  # pragma: no cover - exercised on home5090
            raise RuntimeError("PyTorch is required for MPCount") from error

        self._torch = torch
        self._device = torch.device(device)
        self._upstream_dir = upstream_dir.resolve()
        if self._device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("official MPCount evaluation requires CUDA")

        with _official_namespace_scope(self._upstream_dir):
            official_models = importlib.import_module("models.models")
            model = official_models.DGModel_final(
                pretrained=False,
                deterministic=True,
            )
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                raise TypeError("MPCount checkpoint must be a state dictionary")
            incompatible = model.load_state_dict(state, strict=True)
            self.missing_keys = tuple(incompatible.missing_keys)
            self.unexpected_keys = tuple(incompatible.unexpected_keys)
            if self.missing_keys or self.unexpected_keys:
                raise RuntimeError("MPCount checkpoint did not load strictly")
            self.parameter_count = sum(
                parameter.numel() for parameter in model.parameters()
            )
            self._model = model.to(self._device).eval()

    def infer(
        self, normalized_bchw: np.ndarray, patch_size: int
    ) -> tuple[np.ndarray, float, float]:
        if patch_size <= 0:
            raise ValueError("MPCount patch size must be positive")
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(normalized_bchw)).to(
            self._device
        )
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise ValueError("MPCount backend requires one normalized BCHW image")
        height, width = (int(value) for value in tensor.shape[-2:])
        torch.cuda.reset_peak_memory_stats(self._device)
        torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        with torch.inference_mode(), _official_namespace_scope(self._upstream_dir):
            if height >= patch_size or width >= patch_size:
                density = torch.zeros(
                    (1, 1, height, width),
                    dtype=tensor.dtype,
                    device=self._device,
                )
                for top in range(0, height, patch_size):
                    bottom = min(top + patch_size, height)
                    for left in range(0, width, patch_size):
                        right = min(left + patch_size, width)
                        patch = tensor[:, :, top:bottom, left:right]
                        output = self._model(patch)[0]
                        if output.shape[-2:] != patch.shape[-2:]:
                            raise RuntimeError(
                                "MPCount patch density shape differs from input patch"
                            )
                        density[:, :, top:bottom, left:right] = output
            else:
                density = self._model(tensor)[0]
        torch.cuda.synchronize(self._device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        peak_vram_mb = torch.cuda.max_memory_allocated(self._device) / (1024.0**2)
        result = density[0, 0].detach().float().cpu().numpy()
        return result, max(float(latency_ms), 1e-9), float(peak_vram_mb)


class MPCountAdapter:
    def __init__(
        self,
        *,
        upstream_dir: str | Path,
        expected_upstream_commit: str,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        device: str,
        backend: MPCountBackend | None = None,
        unit_size: int = 16,
        patch_size: int = 3584,
        log_para: float = 1000.0,
    ) -> None:
        self.upstream_dir = Path(upstream_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.expected_upstream_commit = expected_upstream_commit.lower()
        self.checkpoint_sha256 = checkpoint_sha256.lower()
        self.device = device
        self.unit_size = unit_size
        self.patch_size = patch_size
        self.log_para = float(log_para)
        if self.unit_size <= 0 or self.patch_size <= 0 or self.log_para <= 0:
            raise ValueError("MPCount unit, patch, and log parameters must be positive")
        if not self.upstream_dir.is_dir() or not (
            self.upstream_dir / "models/models.py"
        ).is_file():
            raise FileNotFoundError(
                "official MPCount upstream with models/models.py is required"
            )
        if _git_head(self.upstream_dir) != self.expected_upstream_commit:
            raise ValueError("MPCount upstream commit does not match the frozen commit")
        if _git_status(self.upstream_dir):
            raise ValueError("MPCount upstream working tree must be clean")
        if not self.checkpoint_path.is_file() or not is_sha256(
            self.checkpoint_sha256
        ):
            raise FileNotFoundError("MPCount checkpoint and SHA-256 are required")
        observed = sha256_file(self.checkpoint_path).lower()
        if observed != self.checkpoint_sha256:
            raise ValueError(
                "checkpoint SHA-256 mismatch: "
                f"expected={self.checkpoint_sha256} observed={observed}"
            )
        self._backend = backend or TorchMPCountBackend(
            upstream_dir=self.upstream_dir,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

    def _normalized_image(
        self, sample: EvaluationSample
    ) -> tuple[np.ndarray, dict[str, str | int | float | bool | None]]:
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            if image.size != (sample.width, sample.height):
                raise ValueError("sample dimensions differ from the source image")
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        left, top, right, bottom, padded_width, padded_height = calculate_padding(
            sample.width,
            sample.height,
            self.unit_size,
        )
        padded = np.pad(
            pixels,
            ((top, bottom), (left, right), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
        normalized = np.transpose((padded - 0.5) / 0.5, (2, 0, 1))[None]
        metadata: dict[str, str | int | float | bool | None] = {
            "original_width": sample.width,
            "original_height": sample.height,
            "processed_width": padded_width,
            "processed_height": padded_height,
            "padding_left": left,
            "padding_top": top,
            "padding_right": right,
            "padding_bottom": bottom,
            "unit_size": self.unit_size,
            "patch_size": self.patch_size,
            "log_para": self.log_para,
            "normalization": "RGB tensor normalized with mean=0.5 and std=0.5",
            "coordinate_transform": (
                "crop symmetric padding, divide raw density by log_para, then "
                "mass-preserving resize to original-image /8 evaluation grid"
            ),
        }
        return np.asarray(normalized, dtype=np.float32), metadata

    def brief(self) -> ModelBrief:
        return ModelBrief(
            model_id="mpcount-official-sta-to-ucf-qnrf",
            paper="Single Domain Generalization for Crowd Counting (CVPR 2024)",
            role="domain-generalized density candidate for CCTV zone screening",
            family="density and single-domain generalization",
            backbone="VGG16-BN encoder with memory-enhanced density decoder",
            parameter_count=self._backend.parameter_count,
            blocks=(
                "VGG16-BN multi-stage convolutional encoder",
                "multi-scale decoder and density feature fusion",
                "1024-slot density memory bank",
                "patch-wise density classification mask",
                "deterministic learned upsampling and density head",
            ),
            feature_scales=("one-quarter", "one-eighth", "one-sixteenth", "full density"),
            input_contract="one RGB image padded symmetrically to a multiple of 16",
            preprocessing_policy=(
                "official RGB to tensor, mean/std 0.5 normalization, non-overlapping "
                f"patch inference at {self.patch_size} pixels"
            ),
            coordinate_transform=(
                "crop padding, divide density by 1000, preserve mass on the original-image /8 grid"
            ),
            native_output="non-negative full-resolution density map",
            count_derivation="sum of cropped density-map mass divided by 1000",
            zone_derivation="fractional density integration inside calibrated CCTV zones",
            original_losses=(
                "density regression loss",
                "memory attention consistency loss",
                "content error-mask loss",
                "patch-wise density classification loss",
            ),
            official_protocol=(
                "deterministic ShanghaiTech-A source checkpoint evaluated on UCF-QNRF, "
                "unit size 16 and log parameter 1000"
            ),
            official_reported_metrics=(
                "ShanghaiTech-A to UCF-QNRF MAE 112.8",
                "ShanghaiTech-A to UCF-QNRF RMSE 193.8",
            ),
            strengths=(
                "native density heatmap and zone aggregation",
                "explicit single-source domain-generalization design",
            ),
            failure_modes=(
                "extreme perspective or occlusion",
                "festival/CCTV shift outside source-domain coverage",
                "patch boundary artifacts",
                "small negative numerical outputs if runtime compatibility changes",
            ),
            runtime_risks=(
                "full-resolution density activations",
                "official Torch 2.0.1 code running on Torch 2.9",
            ),
            rights_status="PASS_COMMERCIAL_CANDIDATE",
            code_rights_status="Apache-2.0 verified at the pinned upstream commit",
            dataset_rights_status=(
                "Kaggle Apache-2.0 accepted by project owner; uploader provenance risk recorded"
            ),
            checkpoint_rights_status=(
                "Apache-2.0 accepted by project policy for the official deterministic checkpoint"
            ),
            deployment_rights_status="pending; evaluation approval is not deployment approval",
            upstream_commit=self.expected_upstream_commit,
            checkpoint_path=str(self.checkpoint_path),
            checkpoint_sha256=self.checkpoint_sha256,
            reviewed_paths=tuple(
                str((self.upstream_dir / relative).resolve())
                for relative in _REVIEWED_PATHS
            ),
            review_status="approved",
        )

    def predict(
        self, sample: EvaluationSample, *, retain_native: bool
    ) -> NativePrediction:
        started = time.perf_counter()
        metadata: dict[str, str | int | float | bool | None] = {}
        try:
            normalized, metadata = self._normalized_image(sample)
            raw_density, latency_ms, peak_vram_mb = self._backend.infer(
                normalized,
                self.patch_size,
            )
            raw = np.asarray(raw_density, dtype=np.float32)
            expected_shape = (int(normalized.shape[-2]), int(normalized.shape[-1]))
            if raw.ndim != 2 or raw.shape != expected_shape:
                raise ValueError(
                    "MPCount raw density must match the padded input dimensions"
                )
            if not np.isfinite(raw).all() or np.any(raw < 0):
                raise ValueError(
                    "MPCount density must be finite and non-negative"
                )
            top = int(metadata["padding_top"])
            left = int(metadata["padding_left"])
            cropped = raw[top : top + sample.height, left : left + sample.width]
            density_mass = np.asarray(cropped / self.log_para, dtype=np.float32)
            target_shape = (
                tuple(sample.ground_truth_density.shape)
                if sample.ground_truth_density is not None
                else (math.ceil(sample.height / 8), math.ceil(sample.width / 8))
            )
            density = resize_density_preserve_mass(
                density_mass,
                target_shape=target_shape,
            )
            count = float(density.sum(dtype=np.float64))
            metadata.update(
                {
                    "native_density_width": int(raw.shape[1]),
                    "native_density_height": int(raw.shape[0]),
                    "cropped_density_width": int(cropped.shape[1]),
                    "cropped_density_height": int(cropped.shape[0]),
                    "evaluation_density_width": int(density.shape[1]),
                    "evaluation_density_height": int(density.shape[0]),
                    "checkpoint_missing_key_count": len(self._backend.missing_keys),
                    "checkpoint_unexpected_key_count": len(
                        self._backend.unexpected_keys
                    ),
                }
            )
            return NativePrediction(
                sample_id=sample.sample_id,
                output_type="density",
                predicted_count=count,
                latency_ms=float(latency_ms),
                peak_vram_mb=float(peak_vram_mb),
                density=density,
                coordinate_space="original_pixels",
                metadata=metadata,
            )
        except Exception as error:
            elapsed_ms = max((time.perf_counter() - started) * 1000.0, 1e-9)
            return NativePrediction(
                sample_id=sample.sample_id,
                output_type="density",
                predicted_count=None,
                latency_ms=elapsed_ms,
                peak_vram_mb=0.0,
                failure_state=f"{type(error).__name__}: {error}",
                metadata=metadata,
            )


__all__ = ["MPCountAdapter", "TorchMPCountBackend", "calculate_padding"]
