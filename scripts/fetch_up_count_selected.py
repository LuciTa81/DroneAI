"""Fetch the frozen UP-COUNT reference subset with HTTP Range requests."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

from droneai.up_count_remote import (
    build_up_count_catalog,
    extract_selected_up_count,
    select_up_count_members,
)

REMOTEZIP_VERSION = "0.12.3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-url", required=True)
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=166)
    parser.add_argument("--namespace", default="round2-upcount-v1")
    parser.add_argument("--minimum-frame-gap", type=int, default=30)
    parser.add_argument("--archive-size", type=int, required=True)
    parser.add_argument("--archive-md5", required=True)
    args = parser.parse_args()

    if version("remotezip") != REMOTEZIP_VERSION:
        raise RuntimeError(f"remotezip {REMOTEZIP_VERSION} is required")
    from remotezip import RemoteZip

    catalog = build_up_count_catalog(
        labels_zip=args.labels_zip,
        split_dir=args.split_dir,
    )
    with RemoteZip(args.images_url, initial_buffer_size=4 * 1024 * 1024) as images:
        selected = select_up_count_members(
            catalog,
            image_infos=images.infolist(),
            sample_count=args.sample_count,
            namespace=args.namespace,
            minimum_frame_gap=args.minimum_frame_gap,
        )
        manifest = extract_selected_up_count(
            image_zip=images,
            labels_zip=args.labels_zip,
            split_dir=args.split_dir,
            selected=selected,
            dataset_root=args.dataset_root,
            archive_identity={
                "url": args.images_url,
                "official_size": args.archive_size,
                "official_md5": args.archive_md5,
                "remotezip_version": REMOTEZIP_VERSION,
            },
            selection_identity={
                "namespace": args.namespace,
                "minimum_frame_gap": args.minimum_frame_gap,
                "sample_count": args.sample_count,
            },
        )
    print(f"Verified {manifest['selected_samples']} selected UP-COUNT images")


if __name__ == "__main__":
    main()
