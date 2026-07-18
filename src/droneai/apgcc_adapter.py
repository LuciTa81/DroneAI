"""Official APGCC adapter for frozen cross-domain checkpoint evaluation."""

from __future__ import annotations

import copy
import importlib
import math
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from droneai.dm_count_adapter import _git_head, _git_status
from droneai.evaluation_contract import EvaluationSample, ModelAdapter, NativePrediction, Point
from droneai.integrity import is_sha256, sha256_file
from droneai.model_brief import ModelBrief


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
POINT_THRESHOLD = 0.5
MAX_EVAL_SIDE = 2560
PAD_BLOCK = 128
_UPSTREAM_ROOTS = ("config", "models", "util", "datasets", "engine")
_REVIEWED_PATHS = (
    "README.md",
    "LICENSE",
    "apgcc/configs/SHHA_test.yml",
    "apgcc/models/APGCC.py",
    "apgcc/models/Decoder.py",
    "apgcc/models/Encoder.py",
    "apgcc/models/backbones/vgg.py",
    "apgcc/engine.py",
)


class APGCCBackend(Protocol):
    parameter_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    def infer(
        self, normalized_chw: np.ndarray
    ) -> tuple[
        tuple[tuple[float, float], ...],
        tuple[float, ...],
        float,
        float,
        dict[str, int | float],
    ]:
        """Return points, confidence, runtime, VRAM, and preprocessing metadata."""


def official_eval_dimensions(*, width: int, height: int) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("APGCC image dimensions must be positive")
    scale = min(1.0, MAX_EVAL_SIDE / max(width, height))
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    padded_width = ((resized_width - 1) // PAD_BLOCK + 1) * PAD_BLOCK
    padded_height = ((resized_height - 1) // PAD_BLOCK + 1) * PAD_BLOCK
    return resized_width, resized_height, padded_width, padded_height


def _is_upstream_name(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in _UPSTREAM_ROOTS)


def _clear_upstream_namespaces() -> None:
    for name in tuple(sys.modules):
        if _is_upstream_name(name):
            del sys.modules[name]


@contextmanager
def _official_namespace_scope(source_root: Path):
    snapshot = {
        name: module for name, module in sys.modules.items() if _is_upstream_name(name)
    }
    previous_path = list(sys.path)
    previous_bytecode = sys.dont_write_bytecode
    _clear_upstream_namespaces()
    sys.path.insert(0, str(source_root))
    sys.dont_write_bytecode = True
    try:
        yield
        root = source_root.resolve()
        for name, module in tuple(sys.modules.items()):
            if not _is_upstream_name(name):
                continue
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(root):
                raise RuntimeError(
                    f"official namespace {name} originated outside pinned APGCC: {origin}"
                )
    finally:
        _clear_upstream_namespaces()
        sys.modules.update(snapshot)
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode


class TorchAPGCCBackend:
    """CUDA backend for the exact official APGCC source and SHHA checkpoint."""

    def __init__(self, *, upstream_dir: Path, checkpoint_path: Path, device: str):
        try:
            import torch
        except ImportError as error:  # pragma: no cover - exercised on home5090
            raise RuntimeError("PyTorch is required for APGCC") from error

        self._torch = torch
        self._source_root = upstream_dir.resolve() / "apgcc"
        self._device = torch.device(device)
        if self._device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("official APGCC evaluation requires CUDA")

        with _official_namespace_scope(self._source_root):
            config_module = importlib.import_module("config")
            cfg = copy.deepcopy(config_module.cfg)
            cfg = config_module.merge_from_file(
                cfg, self._source_root / "configs" / "SHHA_test.yml"
            )
            official_models = importlib.import_module("models")
            vgg_module = importlib.import_module("models.backbones.vgg")
            original_builder = vgg_module.vgg16_bn
            vgg_module.vgg16_bn = lambda pretrained=True: original_builder(
                pretrained=False
            )
            try:
                model = official_models.build_model(cfg=cfg, training=False)
            finally:
                vgg_module.vgg16_bn = original_builder

            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            if not isinstance(checkpoint, dict) or not checkpoint or not all(
                isinstance(key, str) for key in checkpoint
            ):
                raise TypeError("APGCC checkpoint must be a raw model state dictionary")
            incompatible = model.load_state_dict(checkpoint, strict=True)
            self.missing_keys = tuple(incompatible.missing_keys)
            self.unexpected_keys = tuple(incompatible.unexpected_keys)
            if self.missing_keys or self.unexpected_keys:
                raise RuntimeError("APGCC checkpoint did not load strictly")
            self.parameter_count = sum(parameter.numel() for parameter in model.parameters())
            self._model = model.to(self._device).eval()

    def infer(self, normalized_chw: np.ndarray):
        torch = self._torch
        with _official_namespace_scope(self._source_root):
            tensor = torch.from_numpy(np.ascontiguousarray(normalized_chw)).unsqueeze(0)
            tensor = tensor.to(self._device)
            original_height, original_width = normalized_chw.shape[1:]
            (
                resized_width,
                resized_height,
                padded_width,
                padded_height,
            ) = official_eval_dimensions(width=original_width, height=original_height)
            if (resized_width, resized_height) != (original_width, original_height):
                tensor = torch.nn.functional.interpolate(
                    tensor,
                    size=(resized_height, resized_width),
                    mode="bilinear",
                    align_corners=True,
                )
            if (padded_width, padded_height) != (resized_width, resized_height):
                padded = tensor.new_zeros(
                    (tensor.shape[0], tensor.shape[1], padded_height, padded_width)
                )
                padded[:, :, :resized_height, :resized_width].copy_(tensor)
                tensor = padded
            torch.cuda.reset_peak_memory_stats(self._device)
            torch.cuda.synchronize(self._device)
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = self._model(tensor)
            torch.cuda.synchronize(self._device)
            latency_ms = (time.perf_counter() - started) * 1000.0
            peak_vram_mb = torch.cuda.max_memory_allocated(self._device) / (1024**2)

            scores = torch.softmax(outputs["pred_logits"], -1)[0, :, 1]
            selected = scores > POINT_THRESHOLD
            points = outputs["pred_points"][0, selected].detach().float().cpu().numpy()
            confidences = scores[selected].detach().float().cpu().numpy()
            return (
                tuple((float(x), float(y)) for x, y in points),
                tuple(float(value) for value in confidences),
                max(float(latency_ms), 1e-9),
                float(peak_vram_mb),
                {
                    "resized_width": resized_width,
                    "resized_height": resized_height,
                    "padded_width": padded_width,
                    "padded_height": padded_height,
                    "scale_x": resized_width / original_width,
                    "scale_y": resized_height / original_height,
                },
            )


class APGCCAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        upstream_dir: str | Path,
        expected_upstream_commit: str,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        device: str,
        backend: APGCCBackend | None = None,
    ):
        self.upstream_dir = Path(upstream_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.expected_upstream_commit = expected_upstream_commit.lower()
        self.checkpoint_sha256 = checkpoint_sha256.lower()
        self.device = device
        if not (
            self.upstream_dir / "apgcc" / "models" / "APGCC.py"
        ).is_file():
            raise FileNotFoundError("official APGCC upstream is required")
        if _git_head(self.upstream_dir) != self.expected_upstream_commit:
            raise ValueError("APGCC upstream commit does not match the frozen commit")
        if _git_status(self.upstream_dir):
            raise ValueError("APGCC upstream working tree must be clean")
        if not self.checkpoint_path.is_file() or not is_sha256(self.checkpoint_sha256):
            raise FileNotFoundError("APGCC checkpoint and SHA-256 are required")
        observed = sha256_file(self.checkpoint_path).lower()
        if observed != self.checkpoint_sha256:
            raise ValueError(
                "checkpoint SHA-256 mismatch: "
                f"expected={self.checkpoint_sha256} observed={observed}"
            )
        self._backend = backend or TorchAPGCCBackend(
            upstream_dir=self.upstream_dir,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

    def _normalized_image(self, sample: EvaluationSample) -> np.ndarray:
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
            if image.size != (sample.width, sample.height):
                raise ValueError("sample dimensions differ from the source image")
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        return np.ascontiguousarray(
            np.transpose((pixels - IMAGENET_MEAN) / IMAGENET_STD, (2, 0, 1))
        )

    def brief(self) -> ModelBrief:
        return ModelBrief(
            model_id="apgcc-official-shha-to-ucf-qnrf",
            paper="APGCC: Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance",
            role="ShanghaiTech-A to UCF-QNRF point compatibility comparison",
            family="point-based APGCC",
            backbone="VGG16-BN",
            parameter_count=int(self._backend.parameter_count),
            blocks=(
                "VGG16-BN convolutional backbone",
                "ASPP feature aggregation",
                "Implicit Feature Interpolation decoder",
                "confidence and point-offset heads",
            ),
            feature_scales=("stride-8 anchor grid", "stride-16 VGG features"),
            input_contract="one RGB validation image",
            preprocessing_policy=(
                "official evaluation resize to maximum side 2560, ImageNet "
                "normalization, then pad right/bottom to a multiple of 128"
            ),
            coordinate_transform=(
                "discard padded predictions and map APGCC (x,y) pixels back to "
                "the original image"
            ),
            native_output="pred_logits, pred_points, and offsets",
            count_derivation="number of confidence-thresholded in-bounds point predictions",
            zone_derivation="count accepted points inside each calibrated image zone",
            original_losses=(
                "person/background classification loss",
                "L2 point regression loss",
                "auxiliary positive and negative point guidance losses",
            ),
            official_protocol="SHHA checkpoint, IFI decoder, confidence threshold 0.5",
            official_reported_metrics=(
                "SHHA count MAE/RMSE reported by the paper",
                "SHHA localization precision/recall/F1 reported by the paper",
            ),
            strengths=("native point localization", "auxiliary point guidance", "IFI decoder"),
            failure_modes=(
                "ShanghaiTech-to-UCF domain shift",
                "tiny or occluded people",
                "confidence-threshold sensitivity",
                "out-of-frame point offsets",
            ),
            runtime_risks=(
                "native high-resolution proposal count",
                "legacy Python 3.8 upstream assumptions",
            ),
            rights_status="PASS_COMMERCIAL_CANDIDATE",
            code_rights_status="official code is MIT",
            dataset_rights_status="Kaggle Apache-2.0 accepted with provenance caveat",
            checkpoint_rights_status=(
                "research comparison authorized; commercial checkpoint reuse remains unverified"
            ),
            deployment_rights_status="derived weights and deployment remain pending",
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
            normalized = self._normalized_image(sample)
            candidates, confidences, latency, vram, transform = self._backend.infer(
                normalized
            )
            if len(candidates) != len(confidences):
                raise ValueError("APGCC point confidences must align with points")
            required_transform = {
                "resized_width",
                "resized_height",
                "padded_width",
                "padded_height",
                "scale_x",
                "scale_y",
            }
            if set(transform) != required_transform:
                raise ValueError("APGCC preprocessing metadata is incomplete")
            resized_width = int(transform["resized_width"])
            resized_height = int(transform["resized_height"])
            padded_width = int(transform["padded_width"])
            padded_height = int(transform["padded_height"])
            scale_x = float(transform["scale_x"])
            scale_y = float(transform["scale_y"])
            if not (
                0 < resized_width <= padded_width
                and 0 < resized_height <= padded_height
                and math.isfinite(scale_x)
                and math.isfinite(scale_y)
                and scale_x > 0
                and scale_y > 0
            ):
                raise ValueError("APGCC preprocessing metadata is invalid")
            points: list[Point] = []
            accepted_confidences: list[float] = []
            discarded = 0
            discarded_padding = 0
            for (x, y), confidence in zip(candidates, confidences):
                if not all(math.isfinite(value) for value in (x, y, confidence)):
                    raise ValueError("APGCC points and confidences must be finite")
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError("APGCC point confidences must be probabilities")
                if 0 <= x < resized_width and 0 <= y < resized_height:
                    original_x = x / scale_x
                    original_y = y / scale_y
                    if not (
                        0 <= original_x < sample.width
                        and 0 <= original_y < sample.height
                    ):
                        discarded += 1
                        continue
                    points.append((float(original_x), float(original_y)))
                    accepted_confidences.append(float(confidence))
                else:
                    discarded += 1
                    if 0 <= x < padded_width and 0 <= y < padded_height:
                        discarded_padding += 1
            metadata.update(
                {
                    "original_width": sample.width,
                    "original_height": sample.height,
                    "point_probability_threshold": POINT_THRESHOLD,
                    "native_density_available": False,
                    "raw_thresholded_point_count": len(candidates),
                    "discarded_out_of_bounds_point_count": discarded,
                    "discarded_padding_point_count": discarded_padding,
                    "resized_width": resized_width,
                    "resized_height": resized_height,
                    "padded_width": padded_width,
                    "padded_height": padded_height,
                    "padding_right": padded_width - resized_width,
                    "padding_bottom": padded_height - resized_height,
                    "scale_x": scale_x,
                    "scale_y": scale_y,
                    "missing_checkpoint_key_count": len(self._backend.missing_keys),
                    "unexpected_checkpoint_key_count": len(self._backend.unexpected_keys),
                }
            )
            return NativePrediction(
                sample_id=sample.sample_id,
                output_type="points",
                predicted_count=float(len(points)),
                latency_ms=float(latency),
                peak_vram_mb=float(vram),
                points=tuple(points),
                point_confidences=tuple(accepted_confidences),
                coordinate_space="original_pixels",
                metadata=metadata,
            )
        except Exception as error:
            metadata["traceback"] = traceback.format_exc()
            return NativePrediction(
                sample_id=sample.sample_id,
                output_type="points",
                predicted_count=None,
                latency_ms=max((time.perf_counter() - started) * 1000.0, 1e-9),
                peak_vram_mb=0.0,
                points=(),
                failure_state=f"{type(error).__name__}: {error}",
                metadata=metadata,
            )
