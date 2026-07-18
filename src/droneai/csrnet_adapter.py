"""Project-owned CSRNet adapter for the rights-gated official checkpoint."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from droneai.dm_count_adapter import _git_head, _git_status, resize_density_preserve_mass
from droneai.evaluation_contract import EvaluationSample, NativePrediction
from droneai.integrity import is_sha256, sha256_file
from droneai.model_brief import ModelBrief


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REVIEWED_UPSTREAM_PATHS = ("README.md", "model.py")
_NEGATIVE_DENSITY_POLICIES = frozenset(
    {"fail", "clip_zero_preserve_raw_audit"}
)


class CSRNetBackend(Protocol):
    parameter_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    def infer(self, normalized_bchw: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Return native density, synchronized latency ms, and peak VRAM MB."""


class TorchCSRNetBackend:
    """CUDA backend using only the project-owned architecture implementation."""

    def __init__(self, *, checkpoint_path: Path, device: str) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - exercised on home5090
            raise RuntimeError("PyTorch is required for CSRNet") from error

        from droneai.csrnet import CSRNet

        self._torch = torch
        self._device = torch.device(device)
        if self._device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("CSRNet official-checkpoint evaluation requires CUDA")
        model = CSRNet(pretrained_frontend=False)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError("CSRNet safe checkpoint must be a state dictionary")
        incompatible = model.load_state_dict(state, strict=True)
        self.missing_keys = tuple(incompatible.missing_keys)
        self.unexpected_keys = tuple(incompatible.unexpected_keys)
        if self.missing_keys or self.unexpected_keys:
            raise RuntimeError("CSRNet checkpoint did not load strictly")
        self.parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self._model = model.to(self._device).eval()

    def infer(self, normalized_bchw: np.ndarray) -> tuple[np.ndarray, float, float]:
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(normalized_bchw)).to(self._device)
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
            raise ValueError("CSRNet backend requires one normalized BCHW image")
        torch.cuda.reset_peak_memory_stats(self._device)
        torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model(tensor)
        torch.cuda.synchronize(self._device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        peak_vram_mb = torch.cuda.max_memory_allocated(self._device) / (1024.0**2)
        if output.ndim != 4 or tuple(output.shape[:2]) != (1, 1):
            raise RuntimeError("CSRNet must return one BCHW density channel")
        density = output[0, 0].detach().float().cpu().numpy()
        return density, max(float(latency_ms), 1e-9), float(peak_vram_mb)


class CSRNetAdapter:
    def __init__(
        self,
        *,
        upstream_dir: str | Path,
        expected_upstream_commit: str,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        device: str,
        negative_density_policy: str = "fail",
        backend: CSRNetBackend | None = None,
    ) -> None:
        self.upstream_dir = Path(upstream_dir).resolve()
        self.expected_upstream_commit = expected_upstream_commit.lower()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.checkpoint_sha256 = checkpoint_sha256.lower()
        self.device = device
        self.negative_density_policy = negative_density_policy
        if self.negative_density_policy not in _NEGATIVE_DENSITY_POLICIES:
            raise ValueError("unsupported CSRNet negative-density policy")
        if not self.upstream_dir.is_dir() or any(
            not (self.upstream_dir / relative).is_file()
            for relative in _REVIEWED_UPSTREAM_PATHS
        ):
            raise FileNotFoundError("pinned CSRNet reference upstream is required")
        if _git_head(self.upstream_dir) != self.expected_upstream_commit:
            raise ValueError("CSRNet upstream commit does not match the frozen commit")
        if _git_status(self.upstream_dir):
            raise ValueError("CSRNet upstream working tree must be clean")
        if not self.checkpoint_path.is_file() or not is_sha256(self.checkpoint_sha256):
            raise FileNotFoundError("CSRNet safe checkpoint and SHA-256 are required")
        observed = sha256_file(self.checkpoint_path).lower()
        if observed != self.checkpoint_sha256:
            raise ValueError(
                "checkpoint SHA-256 mismatch: "
                f"expected={self.checkpoint_sha256} observed={observed}"
            )
        self._backend = backend or TorchCSRNetBackend(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        self._last_raw_density: np.ndarray | None = None

    def _normalized_image(self, sample: EvaluationSample) -> np.ndarray:
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            if image.size != (sample.width, sample.height):
                raise ValueError("sample dimensions differ from the source image")
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
        normalized = np.transpose((pixels - mean) / std, (2, 0, 1))[None]
        return np.asarray(normalized, dtype=np.float32)

    def brief(self) -> ModelBrief:
        return ModelBrief(
            model_id="csrnet-independent-shha-to-ucf-qnrf",
            paper=(
                "CSRNet: Dilated Convolutional Neural Networks for Understanding "
                "the Highly Congested Scenes (CVPR 2018)"
            ),
            role="classical density-map baseline for fixed-CCTV zone screening",
            family="density regression with dilated convolutions",
            backbone="VGG-16 frontend through conv4_3",
            parameter_count=self._backend.parameter_count,
            blocks=(
                "VGG-style frontend with three max-pooling stages",
                "six-layer dilation-2 convolutional backend",
                "one-channel 1x1 density output",
            ),
            feature_scales=("full-resolution RGB input", "one-eighth density output"),
            input_contract="one native-resolution RGB image",
            preprocessing_policy=(
                "official ImageNet normalization: mean 0.485/0.456/0.406 and "
                "standard deviation 0.229/0.224/0.225; no source resize"
            ),
            coordinate_transform=(
                "mass-preserving alignment from native stride-8 output to the "
                "original-image /8 evaluation grid"
            ),
            native_output=(
                "raw signed stride-8 density plus audited non-negative "
                "operational density"
            ),
            count_derivation=(
                "sum of max(raw_density, 0); raw signed sum retained for audit"
            ),
            zone_derivation="fractional density integration inside calibrated CCTV zones",
            original_losses=("Euclidean density-map regression loss",),
            official_protocol=(
                "ShanghaiTech Part A checkpoint evaluated cross-domain on one "
                "frozen UCF-QNRF validation sample"
            ),
            official_reported_metrics=(
                "ShanghaiTech Part A checkpoint best MAE 65.9016 in the saved checkpoint",
                "CSRNet paper ShanghaiTech Part A MAE 68.2 and MSE 115.0",
            ),
            strengths=(
                "direct heatmap and zone aggregation",
                "simple well-understood baseline architecture",
            ),
            failure_modes=(
                "cross-domain undercount or overcount",
                "negative raw density values because the published head has no ReLU",
                "perspective and scale shift in festival CCTV",
            ),
            runtime_risks=(
                "native 4K input increases early-layer activation memory",
                "2018 checkpoint evaluated under a modern Torch/CUDA runtime",
            ),
            rights_status="PASS_COMMERCIAL_CANDIDATE",
            code_rights_status="project-owned independent implementation for commercial R&D",
            dataset_rights_status=(
                "Kaggle Apache-2.0 accepted by project owner; uploader provenance risk recorded"
            ),
            checkpoint_rights_status=(
                "research comparison authorized; commercial checkpoint reuse unverified"
            ),
            deployment_rights_status="pending; evaluation approval is not deployment approval",
            upstream_commit=self.expected_upstream_commit,
            checkpoint_path=str(self.checkpoint_path),
            checkpoint_sha256=self.checkpoint_sha256,
            reviewed_paths=(
                str((_PROJECT_ROOT / "src/droneai/csrnet.py").resolve()),
                *(str((self.upstream_dir / relative).resolve()) for relative in _REVIEWED_UPSTREAM_PATHS),
            ),
            review_status="approved",
        )

    def raw_density_audit(self) -> np.ndarray | None:
        if self._last_raw_density is None:
            return None
        return self._last_raw_density.copy()

    def predict(self, sample: EvaluationSample, *, retain_native: bool) -> NativePrediction:
        started = time.perf_counter()
        self._last_raw_density = None
        backend_latency_ms: float | None = None
        backend_peak_vram_mb: float | None = None
        metadata: dict[str, str | int | float | bool | None] = {
            "normalization": "ImageNet RGB mean/std",
            "output_stride": 8,
            "retain_native_requested": retain_native,
            "negative_density_policy": self.negative_density_policy,
            "forward_completed": False,
            "checkpoint_missing_key_count": len(self._backend.missing_keys),
            "checkpoint_unexpected_key_count": len(self._backend.unexpected_keys),
        }
        try:
            normalized = self._normalized_image(sample)
            raw_density, backend_latency_ms, backend_peak_vram_mb = self._backend.infer(
                normalized
            )
            metadata["forward_completed"] = True
            metadata["backend_latency_ms"] = float(backend_latency_ms)
            metadata["backend_peak_vram_mb"] = float(backend_peak_vram_mb)
            raw = np.asarray(raw_density, dtype=np.float32)
            expected_shape = (sample.height // 8, sample.width // 8)
            metadata.update(
                {
                    "native_density_height": int(raw.shape[0]) if raw.ndim == 2 else -1,
                    "native_density_width": int(raw.shape[1]) if raw.ndim == 2 else -1,
                    "expected_native_density_height": expected_shape[0],
                    "expected_native_density_width": expected_shape[1],
                }
            )
            if raw.ndim != 2 or raw.shape != expected_shape:
                raise ValueError(
                    "CSRNet native density shape must equal floor(input / 8)"
                )
            if not np.isfinite(raw).all():
                raise ValueError("CSRNet density must be finite")
            if retain_native:
                self._last_raw_density = raw.copy()
            metadata.update(
                {
                    "native_density_min": float(raw.min()),
                    "native_density_max": float(raw.max()),
                    "native_negative_values": int(np.count_nonzero(raw < 0)),
                    "native_negative_fraction": float(np.mean(raw < 0)),
                    "native_negative_mass": float(-raw[raw < 0].sum(dtype=np.float64)),
                    "native_positive_mass": float(raw[raw > 0].sum(dtype=np.float64)),
                    "native_raw_sum": float(raw.sum(dtype=np.float64)),
                }
            )
            if np.any(raw < 0) and self.negative_density_policy == "fail":
                raise ValueError("CSRNet density must be non-negative")
            operational = (
                np.maximum(raw, 0.0).astype(np.float32, copy=False)
                if self.negative_density_policy == "clip_zero_preserve_raw_audit"
                else raw
            )
            metadata.update(
                {
                    "operational_clipped_values": int(np.count_nonzero(raw < 0)),
                    "operational_density_sum": float(
                        operational.sum(dtype=np.float64)
                    ),
                    "operational_count_method": "sum(max(raw_density, 0))",
                    "raw_density_audit_retained": retain_native,
                }
            )
            target_shape = (
                tuple(sample.ground_truth_density.shape)
                if sample.ground_truth_density is not None
                else (math.ceil(sample.height / 8), math.ceil(sample.width / 8))
            )
            density = resize_density_preserve_mass(
                operational,
                target_shape=target_shape,
            )
            count = float(density.sum(dtype=np.float64))
            metadata.update(
                {
                    "evaluation_density_height": int(density.shape[0]),
                    "evaluation_density_width": int(density.shape[1]),
                }
            )
            return NativePrediction(
                sample_id=sample.sample_id,
                output_type="density",
                predicted_count=count,
                latency_ms=float(backend_latency_ms),
                peak_vram_mb=float(backend_peak_vram_mb),
                density=density,
                coordinate_space="original_pixels",
                metadata=metadata,
            )
        except Exception as error:
            elapsed_ms = (
                max((time.perf_counter() - started) * 1000.0, 1e-9)
                if backend_latency_ms is None
                else float(backend_latency_ms)
            )
            peak_vram_mb = (
                0.0
                if backend_peak_vram_mb is None
                else float(backend_peak_vram_mb)
            )
            return NativePrediction(
                sample_id=sample.sample_id,
                output_type="density",
                predicted_count=None,
                latency_ms=elapsed_ms,
                peak_vram_mb=peak_vram_mb,
                failure_state=f"{type(error).__name__}: {error}",
                metadata=metadata,
            )


__all__ = ["CSRNetAdapter", "TorchCSRNetBackend"]
