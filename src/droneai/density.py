"""Count-preserving density-map utilities."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def points_to_density(
    points: Iterable[tuple[float, float] | list[float]],
    *,
    width: int,
    height: int,
    output_stride: int = 8,
    sigma: float = 1.0,
) -> np.ndarray:
    """Create an output-resolution density map whose sum equals point count."""

    if width <= 0 or height <= 0 or output_stride <= 0 or sigma <= 0:
        raise ValueError("width, height, output_stride and sigma must be positive")
    if width % output_stride or height % output_stride:
        raise ValueError("width and height must be divisible by output_stride")

    density = np.zeros((height // output_stride, width // output_stride), dtype=np.float32)
    radius = max(1, int(math.ceil(3 * sigma)))
    for point in points:
        if len(point) != 2:
            raise ValueError("each point must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError(f"point {(x, y)} lies outside {width}x{height}")
        cx, cy = x / output_stride, y / output_stride
        x0 = max(0, int(math.floor(cx)) - radius)
        x1 = min(density.shape[1], int(math.floor(cx)) + radius + 1)
        y0 = max(0, int(math.floor(cy)) - radius)
        y1 = min(density.shape[0], int(math.floor(cy)) + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        kernel = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
        kernel_sum = float(kernel.sum())
        if kernel_sum <= 0:
            raise ValueError("Gaussian kernel has zero mass")
        density[y0:y1, x0:x1] += (kernel / kernel_sum).astype(np.float32)
    return density


def block_sum_density(density: np.ndarray, stride: int) -> np.ndarray:
    """Downsample a density map by summing non-overlapping blocks."""

    density = np.asarray(density, dtype=np.float32)
    if density.ndim != 2 or stride <= 0:
        raise ValueError("density must be 2D and stride must be positive")
    height, width = density.shape
    if height % stride or width % stride:
        raise ValueError("density dimensions must be divisible by stride")
    return density.reshape(height // stride, stride, width // stride, stride).sum(axis=(1, 3))
