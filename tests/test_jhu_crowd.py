from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from droneai.jhu_crowd import index_jhu_validation


def _write_jhu_sample(
    root: Path,
    sample_id: str,
    *,
    points: tuple[tuple[int, int], ...],
    scene: str = "stadium",
    weather: int = 0,
    distractor: int = 0,
) -> None:
    image_root = root / "images"
    gt_root = root / "gt"
    image_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(
        image_root / f"{sample_id}.jpg"
    )
    (gt_root / f"{sample_id}.txt").write_text(
        "".join(f"{x} {y} 8 10 1 0\n" for x, y in points),
        encoding="utf-8",
    )
    label_path = root / "image_labels.txt"
    with label_path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"{sample_id},{len(points)},{scene},{weather},{distractor}\n"
        )


def test_index_jhu_validation_reads_official_points_and_labels(
    tmp_path: Path,
) -> None:
    val = tmp_path / "val"
    _write_jhu_sample(val, "0001", points=((10, 11), (20, 21)))

    rows = index_jhu_validation(val, expected_samples=1)

    assert rows[0].count == 2
    assert rows[0].points == ((10.0, 11.0), (20.0, 21.0))
    assert rows[0].condition_tags == {
        "scene": "stadium",
        "weather": "none",
        "distractor": "false",
        "density_band": "low",
    }


def test_index_jhu_validation_rejects_declared_count_mismatch(
    tmp_path: Path,
) -> None:
    val = tmp_path / "val"
    _write_jhu_sample(val, "0001", points=((10, 11),))
    (val / "image_labels.txt").write_text(
        "0001,2,stadium,0,0\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="declared count"):
        index_jhu_validation(val, expected_samples=1)


def test_index_jhu_validation_requires_exact_frozen_count(tmp_path: Path) -> None:
    val = tmp_path / "val"
    _write_jhu_sample(val, "0001", points=((10, 11),))

    with pytest.raises(ValueError, match="expected 500"):
        index_jhu_validation(val, expected_samples=500)
