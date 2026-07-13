"""CLI wrapper for the packaged UP-COUNT normalizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from droneai.up_count import prepare_up_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UP-COUNT for the Stage 1 gate")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-image-subset",
        action="store_true",
        help="Normalize only labels with a matching image (for the official 202-sequence sample pack).",
    )
    args = parser.parse_args()
    count = prepare_up_count(
        dataset_root=args.dataset_root,
        image_root=args.image_root,
        label_root=args.label_root,
        split_dir=args.split_dir,
        allow_image_subset=args.allow_image_subset,
    )
    print(f"Prepared {count} UP-COUNT samples")


if __name__ == "__main__":
    main()
