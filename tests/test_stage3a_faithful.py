from pathlib import Path

import pytest

from scripts.run_stage3a_faithful import (
    PINNED_COMMIT,
    build_parser,
    checkpoint_due,
    make_trainer_args,
)


def _args():
    return build_parser().parse_args(
        [
            "--data-dir",
            "/data/part_A",
            "--run-dir",
            "/drive/seed-2026",
            "--work-dir",
            "/content/seed-2026",
        ]
    )


def test_faithful_defaults_match_official_sha_contract() -> None:
    args = _args()
    trainer_args = make_trainer_args(args)
    assert args.seed == 2026
    assert args.max_epoch == 1000
    assert args.checkpoint_every == 10
    assert trainer_args.dataset == "sha"
    assert trainer_args.crop_size == 256
    assert trainer_args.batch_size == 10
    assert trainer_args.val_epoch == 5
    assert trainer_args.val_start == 50
    assert trainer_args.lr == 1e-5
    assert trainer_args.num_of_iter_in_ot == 100
    assert PINNED_COMMIT == "cc5f2132e0d1328909f31b6d665b8e0b15c30467"
    assert args.data_dir == Path("/data/part_A")


def test_checkpoint_schedule_includes_zero_interval_and_final() -> None:
    assert checkpoint_due(0, 1000, 10)
    assert checkpoint_due(10, 1000, 10)
    assert checkpoint_due(999, 999, 10)
    assert not checkpoint_due(9, 1000, 10)
    with pytest.raises(ValueError):
        checkpoint_due(0, 1000, 0)
