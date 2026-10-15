"""Code checks on a PI decision. Prompts ask for good decisions; these make bad ones impossible."""

import re
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

MAX_FINDINGS = 12
#: B17: the PI arbitrates the review; its own findings are only what the reviewer cannot see.
MAX_OWN_FINDINGS = 2
#: The Platform passes at most five hand-off lines to the next stage.
MAX_HANDOFF = 5
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
        anchor = item.get("anchor") if isinstance(item.get("anchor"), dict) else {}
        artifact_id = item.get("artifact_id") or anchor.get("artifact_id")
        if artifact_id not in known_artifacts:
            artifact_id = under_review[0] if under_review else None
        locator = item.get("locator") or anchor.get("locator") or ""
        try:
            findings.append(
                Finding(
                    finding_id=f"PI-{req.consultation_id[:8]}-{i}",
                    anchor={"artifact_id": artifact_id, "locator": locator},
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
        params = decision_raw.get("params") or {}
        if not isinstance(params, dict) or not all(
            isinstance(k, str)
            and isinstance(v, list)
            and len(v) <= 100
            and all(isinstance(i, str) and i for i in v)
            for k, v in params.items()
        ):
            raise GuardError("params must map a name to a list of ids, e.g. {'dropped': ['P3']}")
        data["params"] = params
    elif action == "comment":
        data["findings"] = await _comment(decision_raw, req, ws)
    elif action == "accept":
        selection = decision_raw.get("selection") or {}
        if allowed.selection_key:
            if not isinstance(selection, dict):
                raise GuardError("selection must be an object")
            chosen = selection.get(allowed.selection_key)
            if not isinstance(chosen, list) or not chosen:
                raise GuardError(f"accept needs a nonempty selection[{allowed.selection_key!r}]")
            chosen = list(dict.fromkeys(str(c) for c in chosen))
            known = set(await ws.current_ids(req.situation.stage_key))
            unknown = [c for c in chosen if known and c not in known]
            if unknown:
                raise GuardError(f"unknown ids {unknown}; choose from {sorted(known)}")
            blocked = await ws.blocking(req.situation.stage_key)
            if blocked is not None:
                whole, ids = blocked
                if whole:
                    raise GuardError(
                        "the latest review has a blocking finding on the whole artifact; "
                        "send it back or stop instead of accepting"
                    )
                risks = _strings(raw.get("handoff"), MAX_HANDOFF)
                if refused := [c for c in chosen if c in ids and not _named(c, ids[c], risks)]:
                    named = "; ".join(f"{c}: {', '.join(sorted(ids[c]))}" for c in refused)
                    raise GuardError(
                        f"{refused} still have blocking review findings ({named}). Select only "
                        "ids without one; or accept one as a known risk with a handoff line "
                        "that names the id and its finding id and says how the next stage "
                        "controls it; or send the stage back"
                    )
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

    if action == "approve" and not await _approvable(req, ws):
        escalate = Escalation(
            reason_code="integrity",
            message=(
                "The latest independent review has a blocking finding on the whole artifact or "
                "on every hypothesis, so the PI may not approve; please decide."
            ),
        )

    rationale = str(raw.get("rationale") or "").strip()
    if not rationale:
        raise GuardError(
            "`rationale` is required: an argument of decide next to `decision`. "
            f"You passed: {', '.join(sorted(raw))}"
        )
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


async def _comment(raw: dict[str, Any], req: ConsultationRequest, ws: Workspace) -> list[Finding]:
    """B17: adopt the reviewer's findings by id; own findings only for what serves the charter."""
    adopt, own = raw.get("adopt") or [], raw.get("findings") or []
    if not isinstance(adopt, list) or not isinstance(own, list):
        raise GuardError("`adopt` must be a list of reviewer finding ids, `findings` a list")
    if not adopt and not own:
        raise GuardError("comment needs reviewer finding ids in `adopt`, or a finding of your own")
    if len(adopt) + len(own) > MAX_FINDINGS:
        raise GuardError(f"at most {MAX_FINDINGS} findings; adopt the ones that matter most")
    if len(own) > MAX_OWN_FINDINGS:
        raise GuardError(
            f"at most {MAX_OWN_FINDINGS} findings of your own; adopt the reviewer's by id"
        )
    for item in own:
        if not isinstance(item, dict) or "charter:" not in str(item.get("issue") or ""):
            raise GuardError(
                "a finding of your own must cite the charter line it serves "
                "(charter:<field>[i]) in `issue`; reviewer issues go in `adopt` by id"
            )
    reviewer = {f["finding_id"]: f for f in await ws.review_findings(req.situation.stage_key)}
    adopt = list(dict.fromkeys(str(i) for i in adopt))
    if unknown := [i for i in adopt if i not in reviewer]:
        raise GuardError(f"unknown reviewer finding ids {unknown}; choose from {sorted(reviewer)}")
    copied = [
        reviewer[i]
        | {
            "issue": f"[{i}] {reviewer[i].get('issue') or ''}",
            "category": reviewer[i].get("category") or "other",
        }
        for i in adopt
    ]
    return _findings(copied + own, req)


def _named(hypothesis: str, finding_ids: set[str], lines: list[str]) -> bool:
    """B19: a handoff line names the hypothesis and one of its blocking findings."""

    def token(value: str):
        return re.compile(rf"(?<![\w-]){re.escape(value)}(?![\w-])")

    names, findings = token(hypothesis), [token(f) for f in finding_ids]
    return any(names.search(line) and any(f.search(line) for f in findings) for line in lines)


async def _approvable(req: ConsultationRequest, ws: Workspace) -> bool:
    """Approval publishes the artifact; blocked hypotheses are then refused at selection.

    Allowed when the review passed, or when every blocking finding is on a hypothesis and at
    least one hypothesis has none.
    """
    blocked = await ws.blocked(req.situation.stage_key)
    if blocked is None:
        return True
    whole, ids = blocked
    if whole:
        return False
    if not ids:
        return True
    return bool(set(await ws.current_ids(req.situation.stage_key)) - ids)
