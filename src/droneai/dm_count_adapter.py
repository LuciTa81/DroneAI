"""Official DM-Count adapter for frozen UCF-QNRF checkpoint evaluation."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from droneai.evaluation_contract import EvaluationSample, NativePrediction
from droneai.integrity import is_sha256, sha256_file
from droneai.model_brief import ModelBrief

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class DMCountBackend(Protocol):
    def infer(self, normalized_chw: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Return native density, synchronized latency milliseconds, and peak VRAM MB."""


def calculate_qnrf_size(
    *, width: int, height: int, min_size: int = 512, max_size: int = 2048
) -> tuple[int, int, float]:
    """Match the official short-side QNRF preprocessing policy."""

    if width <= 0 or height <= 0 or min_size <= 0 or max_size < min_size:
        raise ValueError("valid image dimensions and QNRF size limits are required")
    short_side = min(width, height)
    if short_side < min_size:
        ratio = min_size / short_side
    elif short_side > max_size:
        ratio = max_size / short_side
    else:
        ratio = 1.0
    return round(width * ratio), round(height * ratio), float(ratio)


def resize_density_preserve_mass(
    density: np.ndarray, *, target_shape: tuple[int, int]
) -> np.ndarray:
    source = np.asarray(density, dtype=np.float32)
    if source.ndim != 2 or not np.isfinite(source).all() or np.any(source < 0):
        raise ValueError("DM-Count density must be finite, non-negative, and two-dimensional")
    target_height, target_width = target_shape
    if target_height <= 0 or target_width <= 0:
        raise ValueError("density target shape must be positive")
    source_mass = float(source.sum(dtype=np.float64))
    if source.shape == target_shape:
        return source.copy()
    resized = np.asarray(
        Image.fromarray(source, mode="F").resize(
            (target_width, target_height), resample=Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ).copy()
    np.maximum(resized, 0.0, out=resized)
    resized_mass = float(resized.sum(dtype=np.float64))
    if source_mass == 0.0:
        resized.fill(0.0)
    elif resized_mass <= 0.0 or not math.isfinite(resized_mass):
        raise ValueError("density resize lost positive mass")
    else:
        resized *= source_mass / resized_mass
    return resized


def _git_head(upstream_dir: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=upstream_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().lower()


def _git_status(upstream_dir: Path) -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=upstream_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


class TorchDMCountBackend:
    """Lazy PyTorch backend that suppresses the upstream ImageNet download."""

    def __init__(self, *, upstream_dir: Path, checkpoint_path: Path, device: str):
        try:
            import torch
        except ImportError as error:  # pragma: no cover - exercised on home5090
            raise RuntimeError("PyTorch is required for the official DM-Count backend") from error

        models_path = upstream_dir / "models.py"
        spec = importlib.util.spec_from_file_location("droneai_dm_count_upstream_models", models_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import official DM-Count models.py: {models_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.model_zoo.load_url = lambda *args, **kwargs: {}
        model = module.vgg19()
        try:
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:  # pragma: no cover - old PyTorch compatibility
            state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        self._torch = torch
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()

    def infer(self, normalized_chw: np.ndarray) -> tuple[np.ndarray, float, float]:
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(normalized_chw)).unsqueeze(0)
        tensor = tensor.to(self._device)
        using_cuda = self._device.type == "cuda"
        if using_cuda:
            torch.cuda.synchronize(self._device)
            torch.cuda.reset_peak_memory_stats(self._device)
        started = time.perf_counter()
        with torch.inference_mode():
            output, _ = self._model(tensor)
        if using_cuda:
            torch.cuda.synchronize(self._device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(self._device) / (1024.0 * 1024.0)
            if using_cuda
            else 0.0
        )
        density = output[0, 0].detach().float().cpu().numpy()
        return density, max(latency_ms, 1e-9), float(peak_vram_mb)


class DMCountAdapter:
    def __init__(
        self,
        *,
        upstream_dir: str | Path,
        expected_upstream_commit: str,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        device: str,
        backend: DMCountBackend | None = None,
        min_size: int = 512,
        max_size: int = 2048,
    ):
        self.upstream_dir = Path(upstream_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.expected_upstream_commit = expected_upstream_commit.lower()
        self.checkpoint_sha256 = checkpoint_sha256.lower()
        self.device = device
        self.min_size = min_size
        self.max_size = max_size
        if not self.upstream_dir.is_dir() or not (self.upstream_dir / "models.py").is_file():
            raise FileNotFoundError("official DM-Count upstream with models.py is required")
        if _git_head(self.upstream_dir) != self.expected_upstream_commit:
            raise ValueError("DM-Count upstream commit does not match the frozen commit")
        if _git_status(self.upstream_dir):
            raise ValueError("DM-Count upstream working tree must be clean")
        if not self.checkpoint_path.is_file() or not is_sha256(self.checkpoint_sha256):
            raise FileNotFoundError("DM-Count checkpoint and SHA-256 are required")
        observed_checkpoint_hash = sha256_file(self.checkpoint_path).lower()
        if observed_checkpoint_hash != self.checkpoint_sha256:
            raise ValueError(
                "checkpoint SHA-256 mismatch: "
                f"expected={self.checkpoint_sha256} observed={observed_checkpoint_hash}"
            )
        self._backend = backend or TorchDMCountBackend(
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
            width, height, ratio = calculate_qnrf_size(
                width=sample.width,
                height=sample.height,
                min_size=self.min_size,
                max_size=self.max_size,
            )
            if image.size != (width, height):
                image = image.resize((width, height), resample=Image.Resampling.BICUBIC)
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        normalized = np.transpose(
            (pixels - IMAGENET_MEAN) / IMAGENET_STD,
            (2, 0, 1),
        )
        metadata: dict[str, str | int | float | bool | None] = {
            "original_width": sample.width,
            "original_height": sample.height,
            "processed_width": width,
            "processed_height": height,
            "resize_ratio": ratio,
            "coordinate_transform": (
                "mass-preserving bilinear resize to original-image /8 grid"
            ),
        }
        return normalized, metadata

    def brief(self) -> ModelBrief:
        return ModelBrief(
            model_id="dm-count-official-ucf-qnrf",
            paper="Distribution Matching for Crowd Counting (NeurIPS 2020)",
            role="first density-map baseline for common frozen-checkpoint screening",
            family="density",
            backbone="VGG-19 convolutional features",
            parameter_count=None,
            blocks=(
                "VGG-19 E feature extractor with four max-pooling stages",
                "bilinear 2x feature upsampling",
                "512-256-128 convolutional regression head",
                "1-channel convolution and ReLU density head",
            ),
            feature_scales=("processed RGB input", "approximately one-eighth density output"),
            input_contract="one RGB UCF-QNRF image with frozen point annotations",
            preprocessing_policy="official QNRF short-side 512-2048 resize and ImageNet normalization",
            coordinate_transform="native processed density is mass-preserving resized to the original-image /8 grid",
            native_output="non-negative density map",
            count_derivation="sum of density-map mass",
            zone_derivation="fractional integration of density mass over four original-image quadrants",
            original_losses=("counting L1", "optimal transport", "total variation"),
            official_protocol="official pinned QNRF validation list, batch size 1, frozen model_qnrf.pth",
            official_reported_metrics=("UCF-QNRF MAE 85.6", "UCF-QNRF RMSE 148.3"),
            strengths=("count-preserving density output", "usable heatmap and zone aggregation"),
            failure_modes=("domain shift", "extreme scale variation", "high-density undercount"),
            runtime_risks=("large high-resolution activations", "legacy upstream API compatibility"),
            rights_status="PASS_COMMERCIAL_CANDIDATE",
            code_rights_status="MIT verified at pinned upstream commit",
            dataset_rights_status="Apache-2.0 accepted by project owner with provenance risk recorded",
            checkpoint_rights_status="MIT repository scope accepted by project policy; SHA-256 required",
            deployment_rights_status="pending; frozen benchmark approval is not deployment approval",
            upstream_commit=self.expected_upstream_commit,
            checkpoint_path=str(self.checkpoint_path),
            checkpoint_sha256=self.checkpoint_sha256,
            reviewed_paths=tuple(
                str((self.upstream_dir / relative).resolve())
                for relative in (
                    "models.py",
                    "test.py",
                    "datasets/crowd.py",
                    "preprocess_dataset.py",
                    "preprocess/preprocess_dataset_qnrf.py",
                )
            ),
            review_status="approved",
        )

    def predict(self, sample: EvaluationSample, *, retain_native: bool) -> NativePrediction:
        started = time.perf_counter()
        metadata: dict[str, str | int | float | bool | None] = {}
        try:
            normalized, metadata = self._normalized_image(sample)
            native_density, latency_ms, peak_vram_mb = self._backend.infer(normalized)
            source = np.asarray(native_density, dtype=np.float32)
            target_shape = (
                tuple(sample.ground_truth_density.shape)
                if sample.ground_truth_density is not None
                else (math.ceil(sample.height / 8), math.ceil(sample.width / 8))
            )
            density = resize_density_preserve_mass(source, target_shape=target_shape)
            metadata.update(
                {
                    "native_density_width": int(source.shape[1]),
                    "native_density_height": int(source.shape[0]),
                    "evaluation_density_width": int(density.shape[1]),
                    "evaluation_density_height": int(density.shape[0]),
                }
            )
            count = float(density.sum(dtype=np.float64))
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
