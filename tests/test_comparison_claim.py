import pytest

from droneai.comparison_claim import is_ranking_eligible, validate_comparison_claim


@pytest.mark.parametrize(
    "status",
    ["VERIFIED_DISJOINT", "VERIFIED_OVERLAP", "UNKNOWN"],
)
def test_compatibility_smoke_accepts_every_checkpoint_status(status: str) -> None:
    validate_comparison_claim(status, "compatibility_smoke", "pinned evidence")

    assert not is_ranking_eligible(status, "compatibility_smoke")


@pytest.mark.parametrize("status", ["VERIFIED_OVERLAP", "UNKNOWN"])
def test_held_out_performance_rejects_unproven_checkpoint(status: str) -> None:
    with pytest.raises(ValueError, match="held-out"):
        validate_comparison_claim(
            status,
            "held_out_performance",
            "pinned evidence",
        )


def test_verified_disjoint_held_out_is_ranking_eligible() -> None:
    validate_comparison_claim(
        "VERIFIED_DISJOINT",
        "held_out_performance",
        "official Train/Test boundary",
    )

    assert is_ranking_eligible("VERIFIED_DISJOINT", "held_out_performance")


@pytest.mark.parametrize(
    ("status", "scope", "evidence", "message"),
    [
        ("INVALID", "compatibility_smoke", "evidence", "status"),
        ("UNKNOWN", "invalid", "evidence", "scope"),
        ("UNKNOWN", "compatibility_smoke", "", "evidence"),
    ],
)
def test_comparison_claim_rejects_invalid_metadata(
    status: str,
    scope: str,
    evidence: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_comparison_claim(status, scope, evidence)
