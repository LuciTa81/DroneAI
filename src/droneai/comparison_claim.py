from __future__ import annotations


CHECKPOINT_TRAINING_SPLIT_STATUSES = {
    "VERIFIED_DISJOINT",
    "VERIFIED_OVERLAP",
    "UNKNOWN",
}
COMPARISON_SCOPES = {"compatibility_smoke", "held_out_performance"}


def validate_comparison_claim(status: str, scope: str, evidence: str) -> None:
    if status not in CHECKPOINT_TRAINING_SPLIT_STATUSES:
        raise ValueError("checkpoint_training_split_status is invalid")
    if scope not in COMPARISON_SCOPES:
        raise ValueError("comparison_scope is invalid")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("checkpoint_split_evidence must be a non-empty string")
    if scope == "held_out_performance" and status != "VERIFIED_DISJOINT":
        raise ValueError(
            "held-out performance requires VERIFIED_DISJOINT checkpoint evidence"
        )


def is_ranking_eligible(status: str, scope: str) -> bool:
    return scope == "held_out_performance" and status == "VERIFIED_DISJOINT"
