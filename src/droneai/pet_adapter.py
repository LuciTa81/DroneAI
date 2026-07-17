"""Official PET adapter for frozen UCF-QNRF checkpoint evaluation."""

from __future__ import annotations

import importlib
import math
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Protocol

import numpy as np
from PIL import Image

from droneai.dm_count_adapter import _git_head, _git_status
from droneai.evaluation_contract import EvaluationSample, ModelAdapter, NativePrediction, Point
from droneai.integrity import is_sha256, sha256_file
from droneai.model_brief import ModelBrief


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
_UPSTREAM_ROOTS = ("models", "util", "datasets", "engine", "eval", "test_single_image")
_REVIEWED_PATHS = (
    "README.md",
    "LICENSE",
    "models/pet.py",
    "models/backbones/backbone_vgg.py",
    "models/backbones/vgg.py",
    "models/transformer/prog_win_transformer.py",
    "util/misc.py",
    "preprocess_dataset.py",
    "engine.py",
)


class PETBackend(Protocol):
    parameter_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    def infer(
        self, normalized_chw: np.ndarray
    ) -> tuple[
        tuple[tuple[float, float], ...],
        tuple[float, ...],
        tuple[int, int],
        float,
        float,
    ]:
        """Return normalized (y,x) points, confidences, split shape, latency, VRAM."""


def calculate_pet_size(
    width: int, height: int, long_side_cap: int = 1536
) -> tuple[int, int, float]:
    """Match PET's UCF-QNRF long-side downsampling and integer flooring."""

    if width <= 0 or height <= 0 or long_side_cap <= 0:
        raise ValueError("positive image dimensions and long-side cap are required")
    factor = max(width / long_side_cap, height / long_side_cap, 1.0)
    resized_width = int(width / factor)
    resized_height = int(height / factor)
    ratio = 1.0 / factor
    return resized_width, resized_height, float(ratio)


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
                    f"official namespace {name} originated outside pinned PET: {origin}"
                )
    finally:
        _clear_upstream_namespaces()
        sys.modules.update(snapshot)
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode


def _official_args(device: str) -> SimpleNamespace:
    return SimpleNamespace(
        backbone="vgg16_bn",
        position_embedding="sine",
        dec_layers=2,
        dim_feedforward=512,
        hidden_dim=256,
        dropout=0.0,
        nheads=8,
        set_cost_class=1.0,
        set_cost_point=0.05,
        ce_loss_coef=1.0,
        point_loss_coef=5.0,
        eos_coef=0.5,
        device=device,
    )


def _pet_test_forward_current_torch(model, samples, features, pos, **kwargs):
    """Preserve official thresholding while keeping boolean indices on CUDA."""

    torch = importlib.import_module("torch")
    outputs = model.pet_forward(samples, features, pos, **kwargs)
    out_dense, out_sparse = outputs["dense"], outputs["sparse"]
    threshold = 0.5

    index_sparse = None
    if out_sparse is not None:
        sparse_scores = torch.nn.functional.softmax(
            out_sparse["pred_logits"], -1
        )[..., 1]
        index_sparse = sparse_scores > threshold

    index_dense = None
    if out_dense is not None:
        dense_scores = torch.nn.functional.softmax(
            out_dense["pred_logits"], -1
        )[..., 1]
        index_dense = dense_scores > threshold

    combined: dict[str, object] = {}
    source = out_sparse if out_sparse is not None else out_dense
    if source is None:
        raise RuntimeError("PET produced neither sparse nor dense query outputs")
    for name in source:
        if "pred" not in name:
            combined[name] = source[name]
        elif index_dense is None:
            combined[name] = out_sparse[name][index_sparse].unsqueeze(0)
        elif index_sparse is None:
            combined[name] = out_dense[name][index_dense].unsqueeze(0)
        else:
            combined[name] = torch.cat(
                [
                    out_sparse[name][index_sparse].unsqueeze(0),
                    out_dense[name][index_dense].unsqueeze(0),
                ],
                dim=1,
            )
    combined["split_map_raw"] = outputs["split_map_raw"]
    return combined


def _pet_query_embed_inference_current_torch(
    branch, samples, stride=8, src=None, **kwargs
):
    """Use a CPU copy of the selection mask only for CPU query coordinates."""

    torch = importlib.import_module("torch")
    window_partition = importlib.import_module(
        "models.transformer.utils"
    ).window_partition
    dense_input_embed = kwargs["dense_input_embed"]
    batch_size, channels = dense_input_embed.shape[:2]
    image_shape = torch.tensor(samples.tensors.shape[2:])
    shape = (image_shape + stride // 2 - 1) // stride

    shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
    shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
    shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
    points_queries = torch.vstack(
        [shift_y.flatten(), shift_x.flatten()]
    ).permute(1, 0)
    height, width = shift_x.shape

    query_embed = dense_input_embed[
        :, :, points_queries[:, 0], points_queries[:, 1]
    ]
    shift_y_down = points_queries[:, 0] // stride
    shift_x_down = points_queries[:, 1] // stride
    query_feats = src[:, :, shift_y_down, shift_x_down]

    query_embed = query_embed.reshape(batch_size, channels, height, width)
    points_queries = points_queries.reshape(height, width, 2).permute(
        2, 0, 1
    ).unsqueeze(0)
    query_feats = query_feats.reshape(batch_size, channels, height, width)

    dec_win_width, dec_win_height = kwargs["dec_win_size"]
    query_embed_win = window_partition(
        query_embed,
        window_size_h=dec_win_height,
        window_size_w=dec_win_width,
    )
    points_queries_win = window_partition(
        points_queries,
        window_size_h=dec_win_height,
        window_size_w=dec_win_width,
    )
    query_feats_win = window_partition(
        query_feats,
        window_size_h=dec_win_height,
        window_size_w=dec_win_width,
    )

    div_win = window_partition(
        kwargs["div"].unsqueeze(1),
        window_size_h=dec_win_height,
        window_size_w=dec_win_width,
    )
    valid_div = (div_win > 0.5).sum(dim=0)[:, 0]
    v_idx = valid_div > 0
    query_embed_win = query_embed_win[:, v_idx]
    query_feats_win = query_feats_win[:, v_idx]
    points_queries_win = points_queries_win[:, v_idx.cpu()].reshape(-1, 2)
    return query_embed_win, points_queries_win, query_feats_win, v_idx


class TorchPETBackend:
    """CUDA backend for the exact official PET source and checkpoint."""

    def __init__(self, *, upstream_dir: Path, checkpoint_path: Path, device: str):
        try:
            import torch
        except ImportError as error:  # pragma: no cover - exercised on home5090
            raise RuntimeError("PyTorch is required for PET") from error

        self._torch = torch
        self._upstream_dir = upstream_dir.resolve()
        self._device = torch.device(device)
        if self._device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("official PET evaluation requires CUDA")

        with _official_namespace_scope(self._upstream_dir):
            official_models = importlib.import_module("models")
            backbone_module = importlib.import_module("models.backbones.backbone_vgg")
            vgg_module = importlib.import_module("models.backbones.vgg")
            original_builder = backbone_module.vgg16_bn
            backbone_module.vgg16_bn = lambda pretrained=True: vgg_module.vgg16_bn(
                pretrained=False
            )
            try:
                model, _criterion = official_models.build_model(
                    _official_args(str(self._device))
                )
            finally:
                backbone_module.vgg16_bn = original_builder

            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
            except (TypeError, RuntimeError):
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
            if not isinstance(checkpoint, dict) or not isinstance(
                checkpoint.get("model"), dict
            ):
                raise TypeError("PET checkpoint must contain a model state dictionary")
            incompatible = model.load_state_dict(checkpoint["model"], strict=True)
            self.missing_keys = tuple(incompatible.missing_keys)
            self.unexpected_keys = tuple(incompatible.unexpected_keys)
            if self.missing_keys or self.unexpected_keys:
                raise RuntimeError("PET checkpoint did not load strictly")
            self.parameter_count = sum(parameter.numel() for parameter in model.parameters())
            model.test_forward = MethodType(_pet_test_forward_current_torch, model)
            for branch in (model.quadtree_sparse, model.quadtree_dense):
                branch.points_queris_embed_inference = MethodType(
                    _pet_query_embed_inference_current_torch, branch
                )
            self.compatibility_patch = (
                "boolean masks remain on prediction tensors; only the CPU query-coordinate "
                "array uses the same mask copied to CPU; threshold and selection are unchanged"
            )
            self._model = model.to(self._device).eval()

    def infer(self, normalized_chw: np.ndarray):
        torch = self._torch
        with _official_namespace_scope(self._upstream_dir):
            tensor = torch.from_numpy(np.ascontiguousarray(normalized_chw)).to(
                self._device
            )
            torch.cuda.reset_peak_memory_stats(self._device)
            torch.cuda.synchronize(self._device)
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = self._model([tensor], test=True)
            torch.cuda.synchronize(self._device)
            latency_ms = (time.perf_counter() - started) * 1000.0
            peak_vram_mb = torch.cuda.max_memory_allocated(self._device) / (1024**2)

            points = outputs["pred_points"][0].detach().float().cpu().numpy()
            scores = torch.softmax(outputs["pred_logits"], -1)[0, :, 1]
            confidences = scores.detach().float().cpu().numpy()
            split = outputs["split_map_raw"]
            split_shape = (int(split.shape[-2]), int(split.shape[-1]))
            return (
                tuple((float(y), float(x)) for y, x in points),
                tuple(float(value) for value in confidences),
                split_shape,
                max(float(latency_ms), 1e-9),
                float(peak_vram_mb),
            )


class PETAdapter(ModelAdapter):
    def __init__(
        self,
        *,
        upstream_dir: str | Path,
        expected_upstream_commit: str,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        device: str,
        backend: PETBackend | None = None,
        long_side_cap: int = 1536,
    ):
        self.upstream_dir = Path(upstream_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.expected_upstream_commit = expected_upstream_commit.lower()
        self.checkpoint_sha256 = checkpoint_sha256.lower()
        self.device = device
        self.long_side_cap = long_side_cap
        if not self.upstream_dir.is_dir() or not (
            self.upstream_dir / "models" / "pet.py"
        ).is_file():
            raise FileNotFoundError("official PET upstream with models/pet.py is required")
        if _git_head(self.upstream_dir) != self.expected_upstream_commit:
            raise ValueError("PET upstream commit does not match the frozen commit")
        if _git_status(self.upstream_dir):
            raise ValueError("PET upstream working tree must be clean")
        if not self.checkpoint_path.is_file() or not is_sha256(self.checkpoint_sha256):
            raise FileNotFoundError("PET checkpoint and SHA-256 are required")
        observed = sha256_file(self.checkpoint_path).lower()
        if observed != self.checkpoint_sha256:
            raise ValueError(
                "checkpoint SHA-256 mismatch: "
                f"expected={self.checkpoint_sha256} observed={observed}"
            )
        self._backend = backend or TorchPETBackend(
            upstream_dir=self.upstream_dir,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

    def _normalized_image(self, sample: EvaluationSample):
        width, height, ratio = calculate_pet_size(
            sample.width, sample.height, self.long_side_cap
        )
        with Image.open(sample.image_path) as source:
            image = source.convert("RGB")
            if image.size != (sample.width, sample.height):
                raise ValueError("sample dimensions differ from the source image")
            if image.size != (width, height):
                image = image.resize((width, height))
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        normalized = np.ascontiguousarray(
            np.transpose((pixels - IMAGENET_MEAN) / IMAGENET_STD, (2, 0, 1))
        )
        return normalized, width, height, ratio

    def brief(self) -> ModelBrief:
        return ModelBrief(
            model_id="pet-official-ucf-qnrf",
            paper="Point-Query Quadtree for Crowd Counting, Localization, and More",
            role="UCF-QNRF point-counting and localization research comparison",
            family="point-query quadtree",
            backbone="VGG16-BN",
            parameter_count=int(self._backend.parameter_count),
            blocks=(
                "VGG16-BN convolutional backbone",
                "progressive rectangular-window encoder",
                "point-query quadtree decoder",
                "point/non-point classification and coordinate heads",
            ),
            feature_scales=("stride-8 sparse queries", "stride-4 dense queries"),
            input_contract="one RGB UCF-QNRF image with frozen point annotations",
            preprocessing_policy=(
                "official 1536 long-side cap with integer flooring and ImageNet normalization"
            ),
            coordinate_transform=(
                "normalized PET (y,x) coordinates mapped to original (x,y) pixels; "
                "out-of-frame predictions clipped with count recorded"
            ),
            native_output="thresholded point coordinates, confidences, and quadtree split map",
            count_derivation="number of test-time point queries retained above probability 0.5",
            zone_derivation="count predicted points inside each calibrated image zone",
            original_losses=(
                "point/non-point cross entropy",
                "smooth L1 point-coordinate loss weight 5",
                "quadtree split loss weight 0.1",
            ),
            official_protocol=(
                "UCF-QNRF long side 1536, test-time probability threshold 0.5"
            ),
            official_reported_metrics=(
                "UCF-QNRF MAE 79.53",
                "UCF-QNRF RMSE 144.32",
            ),
            strengths=("native localization", "adaptive computation in dense regions"),
            failure_modes=(
                "tiny or occluded people",
                "viewpoint domain shift",
                "probability-threshold sensitivity",
                "zone-boundary coordinate errors",
            ),
            runtime_risks=(
                "legacy official PyTorch API assumptions",
                "high-resolution transformer memory and latency",
            ),
            rights_status="PASS_RESEARCH_ONLY",
            code_rights_status="official README says academic purposes only",
            dataset_rights_status="Kaggle Apache-2.0 accepted with provenance caveat",
            checkpoint_rights_status="official checkpoint treated as academic research only",
            deployment_rights_status="commercial fine-tuning and deployment blocked",
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
            normalized, resized_width, resized_height, ratio = self._normalized_image(
                sample
            )
            normalized_yx, confidences, split_shape, latency, vram = (
                self._backend.infer(normalized)
            )
            if len(normalized_yx) != len(confidences):
                raise ValueError("PET point confidences must align with points")
            points: list[Point] = []
            clipped = 0
            max_x = np.nextafter(float(sample.width), -np.inf)
            max_y = np.nextafter(float(sample.height), -np.inf)
            for y_norm, x_norm in normalized_yx:
                if not math.isfinite(y_norm) or not math.isfinite(x_norm):
                    raise ValueError("PET points must be finite")
                raw_x = x_norm * resized_width / ratio
                raw_y = y_norm * resized_height / ratio
                x = float(np.clip(raw_x, 0.0, max_x))
                y = float(np.clip(raw_y, 0.0, max_y))
                clipped += int(x != raw_x or y != raw_y)
                points.append((x, y))
            metadata.update(
                {
                    "original_width": sample.width,
                    "original_height": sample.height,
                    "resized_width": resized_width,
                    "resized_height": resized_height,
                    "resize_ratio": ratio,
                    "point_probability_threshold": 0.5,
                    "native_density_available": False,
                    "split_map_height": split_shape[0],
                    "split_map_width": split_shape[1],
                    "clipped_point_count": clipped,
                    "missing_checkpoint_key_count": len(self._backend.missing_keys),
                    "unexpected_checkpoint_key_count": len(
                        self._backend.unexpected_keys
                    ),
                    "compatibility_patch": getattr(
                        self._backend, "compatibility_patch", "fixture backend"
                    ),
                }
            )
            return NativePrediction(
                sample_id=sample.sample_id,
                output_type="points",
                predicted_count=float(len(points)),
                latency_ms=float(latency),
                peak_vram_mb=float(vram),
                points=tuple(points),
                point_confidences=tuple(float(value) for value in confidences),
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
