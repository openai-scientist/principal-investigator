"""Code checks on a PI decision. Prompts ask for good decisions; these make bad ones impossible."""

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from principal_investigator.contract import (
    ConsideredOption,
    ConsultationRequest,
    CriticPoint,
    CriticReport,
    CriticResolution,
    CriticResponse,
    DirectiveApplied,
    Escalation,
    EvidencePoint,
    NoteDraft,
    ObligationCheck,
    PIDecision,
)
from principal_investigator.workspace import _HEADING, ArtifactError, Workspace

MAX_POINTS = 8
CHARTER_LISTS = (
    "objectives",
    "scope_in",
    "scope_out",
    "success_criteria",
    "kill_criteria",
    "constraints",
)
INTEGRITY_TAGS = ("post_hoc_criteria", "spin", "suppressed_result", "fabrication", "leakage")
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
    critic_response: list[CriticResponse] = field(default_factory=list)
    directives_applied: list[DirectiveApplied] = field(default_factory=list)
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
    if action == "accept":
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
    options = _models(raw.get("options_considered"), ConsideredOption, 6)
    critic_response = _models(raw.get("critic_response"), CriticResponse, 20)
    directives_applied = _models(raw.get("directives_applied"), DirectiveApplied, 20)

    _check_critic_response(req, critic_response)
    if escalate is None:
        await _check_actionable(req, ws, action, critic_response)
    if escalate is None:
        _check_directives(req, directives_applied)
    await _check_subset(req, ws, action, options)
    if req.kind == "close":
        _check_close(req, data["note"] or "")

    return Draft(
        decision=decision,
        rationale=rationale,
        confidence=confidence,
        options_considered=options,
        evidence=_models(raw.get("evidence"), EvidencePoint, 20),
        assumptions=_strings(raw.get("assumptions"), 8),
        would_change_my_mind=_strings(raw.get("would_change_my_mind"), 8),
        escalate=escalate,
        handoff=handoff,
        notebook=_models(raw.get("notebook"), NoteDraft, MAX_NOTES),
        critic_response=critic_response,
        directives_applied=directives_applied,
        brief=str(raw.get("brief") or "").strip()[:1500],
        raw=raw,
    )


def _token(value: str):
    return re.compile(rf"(?<![\w-]){re.escape(value)}(?![\w-])")


def _check_critic_response(req: ConsultationRequest, given: list[CriticResponse]) -> None:
    """G-critic: the PI answers every serious point of the Critic's report."""
    if req.kind != "stage_outcome" or req.critique is None:
        return
    answered = {r.point_id for r in given if r.reason.strip()}
    if missing := [
        p.point_id
        for p in req.critique.points
        if p.severity == "serious" and p.point_id not in answered
    ]:
        raise GuardError(
            f"the Critic raised serious points {missing}; answer each in `critic_response`, an "
            "argument of decide next to `decision`: {point_id, response: accepted|rejected, "
            "reason}. If you reject one, the researcher decides"
        )


async def _check_actionable(
    req: ConsultationRequest, ws: Workspace, action: str, given: list[CriticResponse]
) -> None:
    """G-actionable (R8): send_back, rerun or stop need a problem that revising can fix."""
    if req.kind != "stage_outcome" or action not in ("send_back", "rerun", "stop"):
        return
    points = req.critique.points if req.critique else []
    accepted = {r.point_id for r in given if r.response == "accepted"}
    # Any serious integrity point may stop the run; it rewinds a stage only when revising can
    # fix it (live: "no completed leakage report" cannot be fixed by revising a plan).
    if any(
        p.severity == "serious"
        and (
            (p.lens == "integrity" and (action == "stop" or p.fix_class == "fix"))
            or (p.fix_class == "fix" and p.point_id in accepted)
        )
        for p in points
    ):
        return
    try:
        findings = await ws.review_findings(req.situation.stage_key)
    except ArtifactError:
        return  # nothing can be verified; the agent already saw the read error
    # An engine finding without a kind is a fix (older engines).
    if any(f.get("severity") == "blocking" and f.get("kind", "fix") == "fix" for f in findings):
        return
    raise GuardError(
        "the open problems cannot be fixed by revising this stage; accept (a subset if needed) "
        "and carry them in `handoff` as known risks, or set `escalate` for the researcher"
    )


RESEARCHER_NOTE = "Researcher's note:"


def directive_entries(req: ConsultationRequest) -> list:
    """Researcher directives, and overrides that carry the researcher's note (a send-back's
    request is an instruction the next stage must follow)."""
    return [
        e
        for e in req.notebook
        if e.actor == "user"
        and e.entry_id
        and (e.kind == "directive" or (e.kind == "override" and RESEARCHER_NOTE in e.text))
    ]


def _directives(req: ConsultationRequest) -> list[str]:
    return [e.entry_id for e in directive_entries(req)]


def _check_directives(req: ConsultationRequest, given: list[DirectiveApplied]) -> None:
    """G-directive: every researcher directive is cited with how the decision follows it."""
    if req.kind not in ("charter", "stage_outcome"):
        return
    cited = {d.entry_id for d in given if d.how.strip()}
    if missing := [i for i in _directives(req) if i not in cited]:
        raise GuardError(
            f"researcher directives {missing} are not in `directives_applied`; add "
            "{entry_id, how} for each (how this decision follows it), or set `escalate` if "
            "following it would break integrity"
        )


async def _check_subset(
    req: ConsultationRequest, ws: Workspace, action: str, options: list[ConsideredOption]
) -> None:
    """G-subset: when part of the stage is sound, rejecting all of it must be argued."""
    accept = req.allowed_for("accept")
    if req.kind != "stage_outcome" or action == "accept" or not (accept and accept.selection_key):
        return
    try:
        free = await selectable(ws, req.situation.stage_key)
    except ArtifactError:
        return  # nothing can be verified; the agent already saw the read error
    if not free:
        return
    if not any(o.action == "accept" and (o.why_not or "").strip() for o in options):
        raise GuardError(
            "some items are not blocked by the review: `options_considered` must include an "
            "`accept` option (a subset) with `why_not` before you choose another action"
        )


def _check_close(req: ConsultationRequest, note: str) -> None:
    """G-close: the report names every obligation that is not controlled."""
    if missing := [
        o.obligation_id
        for o in req.obligations
        if o.status != "controlled" and not _token(o.obligation_id).search(note)
    ]:
        raise GuardError(
            f"the report must name every open or unresolved obligation by id; missing {missing}. "
            "Add a Limitations section that lists each with what it means for the conclusions. "
            "Name each id separately (R1, R2, R3), not as a range such as R1-R3"
        )


async def selectable(ws: Workspace, stage_key: str | None) -> list[str]:
    """Current items of the stage that the latest review does not block."""
    known = await ws.current_ids(stage_key)
    blocked = await ws.blocked(stage_key)
    if blocked is None:
        return known
    whole, ids = blocked
    return [] if whole else [i for i in known if i not in ids]


def _named(hypothesis: str, finding_ids: set[str], lines: list[str]) -> bool:
    """B19: a handoff line names the hypothesis and one of its blocking findings."""

    names, findings = _token(hypothesis), [_token(f) for f in finding_ids]
    return any(names.search(line) and any(f.search(line) for f in findings) for line in lines)


# -- the Critic's report -------------------------------------------------------------------

_CHARTER_REF = re.compile(r"charter:(\w+)\[(\d+)\]")


_JSON_KEY = re.compile(r'"([^"\\]+)"\s*:')

_LOCATOR_FORMS = (
    "hypothesis:<id> (a current item id), section:<heading> (a heading of the artifact), "
    "or document:<name> / an artifact id from the catalog"
)


async def _check_locator(
    obligation_id: str, locator: str, req: ConsultationRequest, ws: Workspace
) -> None:
    """A `controlled` claim must point at something that exists in the stage's artifact."""
    stage = req.situation.stage_key
    kind, _, value = locator.partition(":")
    value = value.strip()
    try:
        if kind == "hypothesis" and value:
            ok = value.split("/", 1)[0] in await ws.current_ids(stage)
        elif kind == "section" and len(wanted := value.lstrip("#").strip().lower()) >= 3:
            text = await ws.primary_text(stage) or ""
            # Sections are Markdown headings, or keys of a JSON artifact.
            names = [m.group(2) for m in _HEADING.finditer(text)]
            if text.lstrip().startswith(("{", "[")):
                names += _JSON_KEY.findall(text)
            ok = any(wanted in name.lower() for name in names)
        elif kind == "document" and value:
            ok = any(value in (r.artifact_id, r.schema_id.split("/")[0]) for r in ws.refs.values())
        else:
            ok = locator in ws.refs
    except ArtifactError:
        return  # the artifact cannot be read, so the locator cannot be verified
    if not ok:
        raise GuardError(
            f"{obligation_id}: locator {locator!r} does not resolve in the artifact; "
            f"use {_LOCATOR_FORMS}"
        )


def _check_ref(lens: str, ref: str, req: ConsultationRequest) -> None:
    if lens == "charter":
        match = _CHARTER_REF.fullmatch(ref)
        values = getattr(req.charter, match.group(1), None) if match and req.charter else None
        if match is None or match.group(1) not in CHARTER_LISTS or values is None:
            raise GuardError(f"ref {ref!r}: use charter:<field>[i] with a field of the charter")
        if int(match.group(2)) >= len(values):
            raise GuardError(f"ref {ref!r}: the charter has no such line")
    elif lens == "risk":
        if ref not in {f"risk:{o.obligation_id}" for o in req.obligations}:
            raise GuardError(f"ref {ref!r}: not an obligation of this request")
    elif lens == "directive":
        if ref not in {f"directive:{i}" for i in _directives(req)}:
            raise GuardError(f"ref {ref!r}: not a researcher directive in the notebook")
    elif ref not in {f"integrity:{t}" for t in INTEGRITY_TAGS}:
        raise GuardError(f"ref {ref!r}: use integrity:<tag>, tag one of {list(INTEGRITY_TAGS)}")


async def check_report(
    raw: dict[str, Any], req: ConsultationRequest, ws: Workspace
) -> CriticReport:
    """The Critic's report is valid only if every point is anchored to a lens and every
    obligation of the request has one status."""
    summary = str(raw.get("summary") or "").strip()
    if not summary:
        raise GuardError("`summary` is required: one or two sentences")
    raw_points = raw.get("points") or []
    if not isinstance(raw_points, list):
        raise GuardError("`points` must be a list")
    if len(raw_points) > MAX_POINTS:
        raise GuardError(f"at most {MAX_POINTS} points; keep the strongest")
    points = []
    for i, item in enumerate(raw_points, 1):
        if not isinstance(item, dict):
            raise GuardError("each point must be an object")
        try:
            point = CriticPoint.model_validate(item | {"point_id": f"C{i}"})
        except ValidationError as exc:
            error = exc.errors()[0]
            raise GuardError(f"point {i} is invalid at {error['loc']}: {error['msg']}") from None
        if not point.point.strip():
            raise GuardError(f"point {i} has no text")
        try:
            _check_ref(point.lens, point.ref, req)
        except GuardError as exc:
            raise GuardError(f"point {i}: {exc}") from None
        points.append(point)

    raw_checks = raw.get("obligations") or []
    if not isinstance(raw_checks, list):
        raise GuardError("`obligations` must be a list")
    checks = []
    for item in raw_checks:
        try:
            checks.append(ObligationCheck.model_validate(item))
        except ValidationError:
            raise GuardError(
                "each obligation check needs obligation_id and status controlled|unresolved"
            ) from None
    expected = [o.obligation_id for o in req.obligations]
    given = [c.obligation_id for c in checks]
    if sorted(given) != sorted(expected):
        raise GuardError(
            f"`obligations` must hold exactly one check for each of {expected}; you gave {given}"
        )
    for c in checks:
        if c.status == "controlled" and not (c.locator or "").strip():
            raise GuardError(f"{c.obligation_id}: controlled needs a locator in the artifact")
        if c.status == "controlled":
            await _check_locator(c.obligation_id, c.locator.strip(), req, ws)
        if c.status == "unresolved" and not (c.note or "").strip():
            raise GuardError(f"{c.obligation_id}: unresolved needs a note saying what is missing")

    prior = [
        p
        for p in (req.prior_critique.points if req.prior_critique else [])
        if p.severity == "serious"
    ]
    resolutions = _check_resolutions(raw.get("resolutions"), prior)
    maintained = [r for r in resolutions if r.status == "maintained"]
    # A maintained point stays serious even if the Critic relisted it as minor.
    kept = {(p.lens, p.ref) for p in points if p.severity == "serious"}
    by_id = {p.point_id: p for p in prior}
    maintained_keys = {(by_id[r.point_id].lens, by_id[r.point_id].ref) for r in maintained}
    added = []
    for r in maintained:
        earlier = by_id[r.point_id]
        if (earlier.lens, earlier.ref) in kept:
            continue
        added.append(
            CriticPoint(
                point_id=f"C{len(points) + len(added) + 1}",
                severity="serious",
                lens=earlier.lens,
                ref=earlier.ref,
                locator=earlier.locator,
                point=f"[maintains {r.point_id}] {r.reason}",
                fix_class=earlier.fix_class,
            )
        )

    expected = _verdict([*points, *added])
    if raw.get("verdict") != expected:
        raise GuardError(
            f"verdict must be {expected!r} for these points (serious iff any serious point or "
            "maintained resolution; concerns iff only minor points; sound iff none)"
        )
    if req.prior_critique is not None:
        # A repeat review keeps new serious points only for integrity and directive.
        for p in points:
            if (
                p.severity == "serious"
                and p.lens in ("charter", "risk")
                and (p.lens, p.ref) not in maintained_keys
            ):
                p.severity = "minor"
                p.point = f"{NEW_IN_REPEAT}{p.point}"
    final = [*points, *added]
    return CriticReport(
        verdict=_verdict(final),
        summary=summary[:600],
        points=final,
        obligations=checks,
        resolutions=resolutions,
    )


NEW_IN_REPEAT = "[new in repeat review] "


def _verdict(points: list[CriticPoint]) -> str:
    if any(p.severity == "serious" for p in points):
        return "serious"
    return "concerns" if points else "sound"


def _check_resolutions(raw: Any, prior: list[CriticPoint]) -> list[CriticResolution]:
    """In a repeat review, each earlier serious point gets exactly one resolution."""
    raw = raw or []
    if not isinstance(raw, list):
        raise GuardError("`resolutions` must be a list")
    out = []
    for item in raw:
        try:
            out.append(CriticResolution.model_validate(item))
        except ValidationError:
            raise GuardError(
                "each resolution needs point_id, status resolved|maintained and reason"
            ) from None
    expected = [p.point_id for p in prior]
    given = [r.point_id for r in out]
    if sorted(given) != sorted(expected):
        raise GuardError(
            f"`resolutions` must hold exactly one entry for each earlier serious point "
            f"{expected}; you gave {given}"
        )
    for r in out:
        if not r.reason.strip():
            raise GuardError(f"{r.point_id}: a resolution needs a reason")
    return out
