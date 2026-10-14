"""Code checks on a PI decision. Prompts ask for good decisions; these make bad ones impossible."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from principal_investigator.contract import (
    ConsideredOption,
    ConsultationRequest,
    Escalation,
    EvidencePoint,
    Finding,
    NoteDraft,
    PIDecision,
)
from principal_investigator.workspace import Workspace

MAX_FINDINGS = 6
MAX_HANDOFF = 8
MAX_LINE = 480
MAX_NOTES = 4


class GuardError(ValueError):
    """The model's decision breaks a rule; the message tells the model what to fix."""


@dataclass
class Draft:
    """A checked decision with everything the result reports about it."""

    decision: PIDecision
    rationale: str
    confidence: float | None
    options_considered: list[ConsideredOption] = field(default_factory=list)
    evidence: list[EvidencePoint] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    would_change_my_mind: list[str] = field(default_factory=list)
    escalate: Escalation | None = None
    handoff: list[str] = field(default_factory=list)
    notebook: list[NoteDraft] = field(default_factory=list)
    brief: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def _strings(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip()[:MAX_LINE] for v in value if str(v).strip()][:limit]


def _models(value: Any, model, limit: int) -> list:
    out = []
    for item in value if isinstance(value, list) else []:
        try:
            out.append(model.model_validate(item))
        except ValidationError:
            continue
    return out[:limit]


def _findings(raw: list, req: ConsultationRequest) -> list[Finding]:
    under_review = list(req.gate.artifact_ids) if req.gate else []
    known_artifacts = {a.artifact_id for a in req.artifacts} | set(under_review)
    findings = []
    for i, item in enumerate(raw[:MAX_FINDINGS], 1):
        if not isinstance(item, dict):
            raise GuardError("each finding must be an object")
        artifact_id = item.get("artifact_id")
        if artifact_id not in known_artifacts:
            artifact_id = under_review[0] if under_review else None
        try:
            findings.append(
                Finding(
                    finding_id=f"PI-{req.consultation_id[:8]}-{i}",
                    anchor={"artifact_id": artifact_id, "locator": item.get("locator") or ""},
                    severity=item.get("severity", "major"),
                    kind=item.get("kind", "fix"),
                    category=item.get("category", "other"),
                    issue=item.get("issue") or "",
                    suggestion=item.get("suggestion"),
                    author="researcher",
                )
            )
        except ValidationError as exc:
            raise GuardError(f"finding {i} is invalid: {exc.errors()[0]['msg']}") from None
    return findings


async def check(raw: dict[str, Any], req: ConsultationRequest, ws: Workspace) -> Draft:
    decision_raw = raw.get("decision")
    if not isinstance(decision_raw, dict):
        raise GuardError("`decision` must be an object")
    action = decision_raw.get("action")
    allowed = req.allowed_for(action) if isinstance(action, str) else None
    if allowed is None:
        names = ", ".join(a.action for a in req.allowed)
        raise GuardError(f"action {action!r} is not allowed; choose one of: {names}")

    data: dict[str, Any] = {"action": action, "note": decision_raw.get("note")}
    if action == "answer":
        option = decision_raw.get("option_id")
        if option not in allowed.option_ids:
            raise GuardError(f"option_id must be one of {allowed.option_ids}")
        data["option_id"] = option
    elif action == "comment":
        findings = decision_raw.get("findings")
        if not isinstance(findings, list) or not findings:
            raise GuardError("comment needs at least one finding")
        data["findings"] = _findings(findings, req)
    elif action == "accept":
        selection = decision_raw.get("selection") or {}
        if allowed.selection_key:
            if not isinstance(selection, dict):
                raise GuardError("selection must be an object")
            chosen = selection.get(allowed.selection_key)
            if not isinstance(chosen, list) or not chosen:
                raise GuardError(f"accept needs a nonempty selection[{allowed.selection_key!r}]")
            chosen = list(dict.fromkeys(str(c) for c in chosen))
            known: set[str] = set()
            for ref in req.artifacts:
                if ref.role == "primary" and ref.stage_key == req.situation.stage_key:
                    known.update(await ws.ids_of(ref.artifact_id))
            unknown = [c for c in chosen if known and c not in known]
            if unknown:
                raise GuardError(f"unknown ids {unknown}; choose from {sorted(known)}")
            data["selection"] = {allowed.selection_key: chosen}
    elif action == "send_back":
        target = decision_raw.get("target_stage_key")
        if target not in allowed.targets:
            raise GuardError(f"target_stage_key must be one of {allowed.targets}")
        data["target_stage_key"] = target
        if not (data["note"] or "").strip():
            raise GuardError("send_back needs a note that says what must change")
    elif action == "charter":
        data["charter"] = decision_raw.get("charter")
    elif action == "report" and not (data["note"] or "").strip():
        raise GuardError("report needs the report text in note")

    try:
        decision = PIDecision.model_validate(data)
    except ValidationError as exc:
        error = exc.errors()[0]
        where = ".".join(str(p) for p in error["loc"])
        raise GuardError(f"decision is invalid at {where}: {error['msg']}") from None

    confidence = raw.get("confidence")
    confidence = (
        min(max(float(confidence), 0.0), 1.0) if isinstance(confidence, int | float) else None
    )
    escalate = None
    if isinstance(raw.get("escalate"), dict):
        try:
            escalate = Escalation.model_validate(raw["escalate"])
        except ValidationError:
            raise GuardError("escalate needs a valid reason_code and a message") from None

    if action == "approve" and await ws.latest_review_passed(req.situation.stage_key) is False:
        escalate = Escalation(
            reason_code="integrity",
            message=(
                "The latest independent review still has a blocking finding, so the PI may not "
                "approve; please decide."
            ),
        )

    rationale = str(raw.get("rationale") or "").strip()
    if not rationale:
        raise GuardError("rationale is required")
    handoff = (
        _strings(raw.get("handoff"), MAX_HANDOFF)
        if action in ("accept", "send_back", "charter")
        else []
    )
    return Draft(
        decision=decision,
        rationale=rationale,
        confidence=confidence,
        options_considered=_models(raw.get("options_considered"), ConsideredOption, 6),
        evidence=_models(raw.get("evidence"), EvidencePoint, 20),
        assumptions=_strings(raw.get("assumptions"), 8),
        would_change_my_mind=_strings(raw.get("would_change_my_mind"), 8),
        escalate=escalate,
        handoff=handoff,
        notebook=_models(raw.get("notebook"), NoteDraft, MAX_NOTES),
        brief=str(raw.get("brief") or "").strip()[:1500],
        raw=raw,
    )
