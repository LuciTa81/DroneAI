import numpy as np
import pytest

from droneai.density import block_sum_density, points_to_density


def test_points_to_density_preserves_count_at_edges() -> None:
    points = [(0, 0), (63, 63), (7, 55), (32, 32)]
    density = points_to_density(points, width=64, height=64, output_stride=8, sigma=1.0)
    assert density.shape == (8, 8)
    assert float(density.sum()) == pytest.approx(4.0, abs=1e-5)


def test_block_sum_density_preserves_mass() -> None:
    source = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    reduced = block_sum_density(source, 8)
    assert reduced.shape == (8, 8)
    assert float(reduced.sum()) == pytest.approx(float(source.sum()), rel=1e-6)


def test_density_rejects_out_of_bounds_point() -> None:
    with pytest.raises(ValueError, match="outside"):
        points_to_density([(64, 0)], width=64, height=64)
