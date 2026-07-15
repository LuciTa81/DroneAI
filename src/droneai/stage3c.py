"""Stage 3C: classify code, data, weight, and deployment rights by scope."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from droneai.scoring import CheckResult, StageReport, score_stage

REQUIRED_COMPONENTS = frozenset(
    {"code", "dataset", "pretrained_weights", "derived_weights", "deployment"}
)
RIGHTS_STATUSES = {"verified", "unverified", "pending", "prohibited"}
RIGHTS_BASES = {"explicit_terms", "not_stated", "pending_review", "internal_approval"}

RESEARCH_ACTIONS = ("catalog", "research_reproduction")
CANDIDATE_ACTIONS = ("catalog", "license_due_diligence")
PRODUCTION_ACTIONS = (
    "catalog",
    "synthetic_compatibility_smoke",
    "asset_download",
    "frozen_checkpoint_evaluation",
    "commercial_training",
    "derived_weight_use",
    "weight_reuse",
    "deployment",
)


@dataclass(frozen=True)
class RightsDecision:
    status: str
    allowed_actions: tuple[str, ...]
    ambiguous_components: tuple[str, ...]
    restricted_components: tuple[str, ...]
    blockers: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# Stage 3C rights decision",
            "",
            f"- Status: **{self.status}**",
            f"- Allowed harness actions: {', '.join(self.allowed_actions) or 'none'}",
            f"- Ambiguous components: {', '.join(self.ambiguous_components) or 'none'}",
            f"- Restricted components: {', '.join(self.restricted_components) or 'none'}",
            f"- Blockers: {', '.join(self.blockers) or 'none'}",
            "",
            self.rationale,
            "",
            "> This engineering classification is not legal advice. Candidate status is not permission.",
            "",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class Stage3CResult:
    report: StageReport
    decision: RightsDecision


def manifest_semantic_sha256(manifest: dict[str, Any]) -> str:
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _component_map(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    mapped: dict[str, dict[str, Any]] = {}
    components = manifest.get("components")
    if not isinstance(components, list):
        return mapped, ["components must be a list"]
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            errors.append(f"component[{index}] must be an object")
            continue
        kind = str(component.get("component_type") or "")
        if kind in mapped:
            errors.append(f"duplicate component_type: {kind}")
        elif kind:
            mapped[kind] = component
        else:
            errors.append(f"component[{index}] missing component_type")
    return mapped, errors


def _inspect_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    components, errors = _component_map(manifest)
    identity_ok = (
        manifest.get("schema_version") == 1
        and bool(manifest.get("candidate_id"))
        and manifest.get("intended_use") == "commercial_product_rnd"
    )
    if not identity_ok:
        errors.append("manifest requires schema_version=1, candidate_id, commercial_product_rnd")

    present = set(components)
    complete_ok = present == REQUIRED_COMPONENTS
    if not complete_ok:
        errors.append(
            f"component set mismatch: missing={sorted(REQUIRED_COMPONENTS - present)}; "
            f"extra={sorted(present - REQUIRED_COMPONENTS)}"
        )

    metadata_errors: list[str] = []
    provenance_errors: list[str] = []
    prohibited: list[str] = []
    ambiguous: list[str] = []
    restricted: list[str] = []
    commercial_verified: dict[str, bool] = {}
    research_verified: dict[str, bool] = {}

    for kind in sorted(REQUIRED_COMPONENTS & present):
        component = components[kind]
        if not component.get("component_id") or not component.get("source_url"):
            provenance_errors.append(f"{kind}: component_id/source_url missing")
        rights = component.get("rights")
        if not isinstance(rights, dict):
            metadata_errors.append(f"{kind}: rights must be an object")
            commercial_verified[kind] = False
            research_verified[kind] = False
            continue

        status = rights.get("status")
        basis = rights.get("basis")
        commercial = rights.get("commercial_use")
        research = rights.get("research_use")
        if status not in RIGHTS_STATUSES:
            metadata_errors.append(f"{kind}: invalid rights status {status!r}")
        if basis not in RIGHTS_BASES:
            metadata_errors.append(f"{kind}: invalid rights basis {basis!r}")
        commercial_type_ok = commercial is None or type(commercial) is bool
        research_type_ok = research is None or type(research) is bool
        if "commercial_use" not in rights or not commercial_type_ok:
            metadata_errors.append(f"{kind}: commercial_use must be true, false, or null")
        if "research_use" not in rights or not research_type_ok:
            metadata_errors.append(f"{kind}: research_use must be true, false, or null")
        if commercial is not None and basis == "not_stated":
            metadata_errors.append(
                f"{kind}: commercial_use={commercial!r} contradicts basis=not_stated; use null"
            )
        if status == "verified" and not (rights.get("license_id") and rights.get("evidence_url")):
            metadata_errors.append(f"{kind}: verified status requires license_id and evidence_url")
        if status == "prohibited":
            prohibited.append(kind)
            if commercial is not False or research is not False:
                metadata_errors.append(
                    f"{kind}: prohibited status requires commercial_use=false and research_use=false"
                )

        commercial_ok = bool(
            status == "verified"
            and commercial is True
            and rights.get("license_id")
            and rights.get("evidence_url")
        )
        research_ok = bool(
            status == "verified"
            and research is True
            and rights.get("license_id")
            and rights.get("evidence_url")
        )
        commercial_verified[kind] = commercial_ok
        research_verified[kind] = research_ok
        if not commercial_ok and status != "prohibited":
            ambiguous.append(kind)
        if commercial is False and status != "prohibited":
            restricted.append(kind)

    all_errors = tuple(errors + metadata_errors + provenance_errors)
    all_commercial = complete_ok and all(commercial_verified.get(k, False) for k in REQUIRED_COMPONENTS)
    all_research = complete_ok and all(research_verified.get(k, False) for k in REQUIRED_COMPONENTS)

    if all_errors or prohibited:
        status = "BLOCKED"
        rationale = "Rights metadata or an explicit prohibition blocks this bundle."
    elif all_commercial:
        status = "PRODUCTION_APPROVED"
        rationale = "All five required components have verified commercial evidence."
    elif all_research and restricted:
        status = "PASS_RESEARCH_ONLY"
        rationale = "Every component permits research, while one or more explicitly exclude commercial use."
    elif not restricted:
        status = "PASS_COMMERCIAL_CANDIDATE"
        rationale = (
            "No explicit commercial prohibition is recorded, but at least one required permission "
            "is absent, silent, pending, or unverified."
        )
    else:
        status = "BLOCKED"
        rationale = (
            "The bundle mixes an explicit commercial restriction with components whose research "
            "permission is not verified."
        )

    return {
        "identity_ok": identity_ok,
        "complete_ok": complete_ok,
        "metadata_ok": not metadata_errors,
        "provenance_ok": not provenance_errors,
        "errors": all_errors,
        "status": status,
        "rationale": rationale,
        "ambiguous": tuple(sorted(ambiguous)),
        "restricted": tuple(sorted(restricted)),
        "prohibited": tuple(sorted(prohibited)),
        "commercial_verified": commercial_verified,
    }


def classify_rights(manifest: dict[str, Any]) -> RightsDecision:
    inspection = _inspect_manifest(manifest)
    status = inspection["status"]
    if status == "PASS_COMMERCIAL_CANDIDATE":
        actions = CANDIDATE_ACTIONS
        if inspection["commercial_verified"].get("code"):
            actions += ("synthetic_compatibility_smoke",)
        frozen_inputs = ("code", "dataset", "pretrained_weights")
        if all(inspection["commercial_verified"].get(kind) for kind in frozen_inputs):
            actions += ("asset_download", "frozen_checkpoint_evaluation")
    else:
        actions = {
            "PASS_RESEARCH_ONLY": RESEARCH_ACTIONS,
            "PRODUCTION_APPROVED": PRODUCTION_ACTIONS,
            "BLOCKED": (),
        }[status]
    blockers = tuple(inspection["errors"]) + tuple(
        f"{kind}: use prohibited" for kind in inspection["prohibited"]
    )
    if status == "BLOCKED" and not blockers:
        blockers = (inspection["rationale"],)
    return RightsDecision(
        status=status,
        allowed_actions=actions,
        ambiguous_components=inspection["ambiguous"],
        restricted_components=inspection["restricted"],
        blockers=blockers,
        rationale=inspection["rationale"],
    )


def build_stage3c_checks(
    manifest: dict[str, Any], decision: RightsDecision | None = None
) -> list[CheckResult]:
    inspection = _inspect_manifest(manifest)
    decision = decision or classify_rights(manifest)
    commercial = inspection["commercial_verified"]
    scope_ok = decision.status != "BLOCKED"
    return [
        CheckResult(
            "manifest.identity", "manifest", "Candidate identity and intended commercial scope are explicit",
            10, inspection["identity_ok"], True, expected="schema v1, candidate_id, commercial_product_rnd",
            observed=str({key: manifest.get(key) for key in ("schema_version", "candidate_id", "intended_use")}),
        ),
        CheckResult(
            "components.complete", "rights inventory", "All five rights components are present exactly once",
            15, inspection["complete_ok"], True, expected=", ".join(sorted(REQUIRED_COMPONENTS)),
            observed=", ".join(sorted((manifest.get("components") and [str(c.get("component_type")) for c in manifest["components"] if isinstance(c, dict)]) or [])),
        ),
        CheckResult(
            "rights.metadata", "rights inventory", "Rights values and bases are internally consistent",
            20, inspection["metadata_ok"], True, expected="valid status/basis and tri-state use flags",
            observed="consistent" if inspection["metadata_ok"] else "; ".join(inspection["errors"]),
        ),
        CheckResult(
            "rights.provenance", "traceability", "Every component has stable identity and source provenance",
            15, inspection["provenance_ok"], True, expected="component_id and source_url per component",
            observed="recorded" if inspection["provenance_ok"] else "; ".join(inspection["errors"]),
        ),
        CheckResult(
            "rights.scope_coherence", "decision", "The bundle has one safe, non-contradictory scope",
            20, scope_ok, True, expected="research-only, commercial candidate, or production approved",
            observed=f"{decision.status}: {decision.rationale}",
        ),
        *[
            CheckResult(
                f"commercial.{kind}", "production rights", f"{kind} has verified commercial evidence",
                4, bool(commercial.get(kind)), False, expected="verified, commercial_use=true, license_id, evidence_url",
                observed=str(commercial.get(kind, False)),
            )
            for kind in sorted(REQUIRED_COMPONENTS)
        ],
    ]


def run_stage3c(*, manifest_path: str | Path, output_dir: str | Path) -> Stage3CResult:
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = classify_rights(manifest)
    success_status = (
        decision.status
        if decision.status != "BLOCKED"
        else "PASS_COMMERCIAL_CANDIDATE"
    )
    report = score_stage(
        stage_id="stage-3c",
        stage_name="Commercial rights and deployment suitability",
        threshold=80,
        checks=build_stage3c_checks(manifest, decision),
        success_status=success_status,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    components, _ = _component_map(manifest)
    decision_payload = decision.to_dict()
    decision_payload.update(
        {
            "candidate_id": manifest.get("candidate_id"),
            "manifest_semantic_sha256": manifest_semantic_sha256(manifest),
            "component_ids": {
                kind: component.get("component_id")
                for kind, component in sorted(components.items())
            },
        }
    )
    (output_dir / "manifest.snapshot.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "rights-decision.json").write_text(
        json.dumps(decision_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "rights-decision.md").write_text(decision.to_markdown(), encoding="utf-8")
    (output_dir / "score.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "score.md").write_text(report.to_markdown(), encoding="utf-8")
    return Stage3CResult(report=report, decision=decision)
