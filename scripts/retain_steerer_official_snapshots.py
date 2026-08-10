#!/usr/bin/env python3
"""Watch an A800 run and retain only its best-MAE and best-MSE snapshots."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from droneai.steerer_official_retention import prune_official_best_snapshots


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--launcher-status", type=Path, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--grace-seconds", type=float, default=180.0)
    return parser


def _append_event(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _event(result, state: str) -> dict[str, object]:
    payload = asdict(result)
    for field in ("retained", "recent", "deleted", "ignored"):
        payload[field] = [str(path) for path in payload[field]]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "launcher_state": state,
        **payload,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.interval_seconds <= 0 or args.grace_seconds < 0:
        raise ValueError("retention intervals are invalid")
    checkpoint_root = args.checkpoint_root.resolve(strict=True)
    launcher_status = args.launcher_status.resolve(strict=True)
    while True:
        status = json.loads(launcher_status.read_text(encoding="utf-8"))
        state = str(status.get("state"))
        terminal = state in {"completed", "failed"}
        result = prune_official_best_snapshots(
            checkpoint_root,
            min_age_seconds=0 if terminal else args.grace_seconds,
        )
        _append_event(args.audit_log, _event(result, state))
        if terminal:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
