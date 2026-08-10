"""Prepare and independently verify the STEERER A full Train/Test data lane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from droneai.steerer_official_data import (
    load_official_training_data_profile,
    prepare_official_train_test_lane,
    score_g2_prepared_data,
    verify_prepared_official_train_test_lane,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or verify STEERER A Train1201/Test334 data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--raw-train-root", required=True, type=Path)
    prepare.add_argument("--raw-test-root", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--config", required=True, type=Path)
    verify.add_argument("--output-dir", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_and_score(*, profile, output: Path, manifest: Path) -> dict[str, object]:
    verification = verify_prepared_official_train_test_lane(
        output,
        manifest_path=manifest,
        expected_train_samples=profile.train_samples,
        expected_test_samples=profile.test_samples,
    )
    score = score_g2_prepared_data(
        profile,
        verification,
        evidence_path=str(manifest),
    )
    evidence_dir = output / "manifests"
    verification_path = evidence_dir / "g2-verification.json"
    score_path = evidence_dir / "g2-score.json"
    score_markdown_path = evidence_dir / "g2-score.md"
    _write_json(verification_path, verification)
    _write_json(score_path, score.to_dict())
    score_markdown_path.write_text(score.to_markdown(), encoding="utf-8")
    return {
        "status": score.status,
        "score": score.score,
        "output_root": str(output),
        "manifest": str(manifest),
        "verification": str(verification_path),
        "score_path": str(score_path),
        "score_markdown": str(score_markdown_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profile = load_official_training_data_profile(args.config)
    output = args.output_dir.resolve()
    if output != Path(profile.processed_root).resolve():
        raise ValueError("output differs from the isolated A training profile")

    if args.command == "prepare":
        manifest = prepare_official_train_test_lane(
            args.raw_train_root,
            args.raw_test_root,
            output_root=output,
            expected_train_samples=profile.train_samples,
            expected_test_samples=profile.test_samples,
        )
    else:
        manifest = args.manifest.resolve()
        expected_manifest = output / "manifests" / "dataset-preparation.json"
        if manifest != expected_manifest:
            raise ValueError("manifest differs from the canonical A prepared-data manifest")

    result = _verify_and_score(profile=profile, output=output, manifest=manifest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
