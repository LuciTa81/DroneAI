"""Storage-bounded retention for raw upstream STEERER best snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import time


_SNAPSHOT_PATTERN = re.compile(
    r"^Ep_(?P<epoch>\d+)_mae_(?P<mae>[-+0-9.eE]+)_mse_(?P<mse>[-+0-9.eE]+)\.pth$"
)


@dataclass(frozen=True)
class OfficialBestSnapshot:
    path: Path
    epoch: int
    mae: float
    mse: float


@dataclass(frozen=True)
class RetentionResult:
    retained: tuple[Path, ...]
    recent: tuple[Path, ...]
    deleted: tuple[Path, ...]
    ignored: tuple[Path, ...]
    dry_run: bool


def parse_official_best_snapshot(path: str | Path) -> OfficialBestSnapshot | None:
    candidate = Path(path)
    matched = _SNAPSHOT_PATTERN.fullmatch(candidate.name)
    if matched is None:
        return None
    epoch = int(matched.group("epoch"))
    mae = float(matched.group("mae"))
    mse = float(matched.group("mse"))
    if epoch < 0 or not math.isfinite(mae) or not math.isfinite(mse):
        return None
    return OfficialBestSnapshot(path=candidate, epoch=epoch, mae=mae, mse=mse)


def _selected_paths(snapshots: list[OfficialBestSnapshot]) -> set[Path]:
    if not snapshots:
        return set()
    best_mae = min(snapshots, key=lambda item: (item.mae, -item.epoch, item.path.name))
    best_mse = min(snapshots, key=lambda item: (item.mse, -item.epoch, item.path.name))
    return {best_mae.path, best_mse.path}


def prune_official_best_snapshots(
    checkpoint_root: str | Path,
    *,
    min_age_seconds: float = 180.0,
    now_timestamp: float | None = None,
    dry_run: bool = False,
) -> RetentionResult:
    """Keep per-run best-MAE/best-MSE snapshots and prune only completed losers."""

    if min_age_seconds < 0:
        raise ValueError("min_age_seconds must be non-negative")
    root = Path(checkpoint_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("checkpoint_root must be a directory")
    now = time.time() if now_timestamp is None else float(now_timestamp)
    grouped: dict[Path, list[OfficialBestSnapshot]] = {}
    ignored: list[Path] = []
    for candidate in sorted(root.rglob("Ep_*.pth")):
        if candidate.is_symlink() or not candidate.is_file():
            ignored.append(candidate)
            continue
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root):
            ignored.append(candidate)
            continue
        snapshot = parse_official_best_snapshot(candidate)
        if snapshot is None:
            ignored.append(candidate)
            continue
        grouped.setdefault(candidate.parent, []).append(snapshot)

    retained: set[Path] = set()
    recent: list[Path] = []
    deleted: list[Path] = []
    for snapshots in grouped.values():
        selected = _selected_paths(snapshots)
        retained.update(selected)
        for snapshot in snapshots:
            if snapshot.path in selected:
                continue
            age = now - snapshot.path.stat().st_mtime
            if age < min_age_seconds:
                recent.append(snapshot.path)
                continue
            deleted.append(snapshot.path)
            if not dry_run:
                snapshot.path.unlink()

    return RetentionResult(
        retained=tuple(sorted(retained)),
        recent=tuple(sorted(recent)),
        deleted=tuple(sorted(deleted)),
        ignored=tuple(sorted(ignored)),
        dry_run=dry_run,
    )


__all__ = [
    "OfficialBestSnapshot",
    "RetentionResult",
    "parse_official_best_snapshot",
    "prune_official_best_snapshots",
]
