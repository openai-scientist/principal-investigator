"""One consultation: the PI agent investigates and decides, or the Critic agent reviews a stage."""

from datetime import UTC, date, datetime
from importlib import resources

from principal_investigator.agent import AgentLimit, Outcome, run_agent
from principal_investigator.contract import (
    ConsultationRequest,
    ConsultationResult,
    CriticReport,
    Obligation,
)
from principal_investigator.critic import run_critic
from principal_investigator.guards import Draft, selectable
from principal_investigator.llm import ChatLLM, LLMError
from principal_investigator.tools import Session, pi_tools
from principal_investigator.workspace import ArtifactError

PI_MAX_STEPS = 24
TOKEN_CAP = 600_000

AUTHORITY_TEXT = {
    "decide": "decide — your decision takes effect now; the researcher sees it in the log.",
    "advise": "advise — the researcher decides; they see your decision as a recommendation.",
}


class DeliberationError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def prompt(name: str) -> str:
    package = resources.files("principal_investigator.prompts")
    return package.joinpath(f"{name}.md").read_text(encoding="utf-8")


def pi_system(kind: str) -> str:
    return "\n\n".join([prompt("constitution"), prompt(kind), prompt("output")])


def _charter_lines(req: ConsultationRequest) -> list[str]:
    if req.charter is None:
        return ["(none yet: you are writing it now)"]
    c = req.charter
    lines = [f"Question: {c.question}"]
    if c.motivation:
        lines.append(f"Motivation: {c.motivation}")
    for label, values in (
        ("objectives", c.objectives),
        ("scope_in", c.scope_in),
        ("scope_out", c.scope_out),
        ("success_criteria", c.success_criteria),
        ("kill_criteria", c.kill_criteria),
        ("constraints", c.constraints),
    ):
        lines += [f"charter:{label}[{i}]: {value}" for i, value in enumerate(values)]
    lines.append(f"Risk posture: {c.risk_posture}")
    lines.append(f"Profile: domain={c.profile.domain}, research_type={c.profile.research_type}")
    return lines


def briefing(
    req: ConsultationRequest, catalog: str, today: date | None = None, review: str | None = None
) -> str:
    """What the agent knows when it starts. Artifacts are listed, not included: it reads them."""
    s = req.situation
    today = today or datetime.now(UTC).date()
    out = [
        "# Run",
        # Without it, models judge recent papers by their training cutoff and call them
        # "future-dated" (seen in a live run).
        f"Today is {today.isoformat()} (UTC). Work published up to today is not future-dated.",
        f"Interaction mode: {req.interaction_mode}."
        + (
            ""
            if req.kind == "stage_review"
            else f" Your authority here: {AUTHORITY_TEXT[req.authority]}"
        ),
        f"Topic given by the researcher: {req.topic}",
        f"Research profile: domain={req.profile.domain}, research_type={req.profile.research_type}",
        "",
        "# Charter",
        *_charter_lines(req),
        "",
        "# Situation",
        f"Current stage: {s.stage_key or '(before the first stage)'}; iteration {s.iteration} of "
        f"at most {s.max_iterations}.",
        "Pipeline:",
    ]
    for stage in s.pipeline:
        back = f", may send back to {stage.loop_back_to}" if stage.loop_back_to else ""
        out.append(
            f"- {stage.stage_key} (engine {stage.engine}): status={stage.status or 'not started'}"
            f"{', researcher checkpoint' if stage.checkpoint else ''}{back}"
        )
    b = s.budget
    out.append(
        f"Budget: spent {b.spent_usd} of {b.limit_usd} USD"
        + (" (some costs unknown)" if b.cost_unknown else "")
    )
    out += ["", f"# Decision to make: {req.kind}"]
    if s.failure is not None:
        out.append(f"The stage failed: {s.failure.code}: {s.failure.message}")
    if review:
        out.append(review)
    if req.critique is not None:
        out += _critic_lines(req.critique)
    elif req.kind == "stage_outcome":
        out.append("Critic report unavailable: decide without it.")
    if req.obligations:
        out.append("Accepted risks (obligations):")
        out += [_obligation_line(o) for o in req.obligations]
    directives = [e for e in req.notebook if e.kind == "directive" and e.actor == "user"]
    if directives := [e for e in directives if e.entry_id]:
        out.append(
            "Researcher directives (use `directive:<entry_id>` as a ref):"
            if req.kind == "stage_review"
            else "Researcher directives (cite each in `directives_applied`):"
        )
        out += [f"- entry {e.entry_id}: {e.text}" for e in directives]
    if req.allowed:
        out.append("Allowed actions:")
    for a in req.allowed:
        extra = []
        if a.option_ids:
            extra.append(f"option_ids={a.option_ids}")
        if a.selection_key:
            extra.append(f"selection_key={a.selection_key}")
        if a.targets:
            extra.append(f"targets={a.targets}")
        out.append(f"- {a.action}" + (f" ({'; '.join(extra)})" if extra else ""))
    recent = req.notebook[-8:]
    out += ["", f"# Lab notebook: {len(req.notebook)} entries (most recent below; read_notebook)"]
    out += [
        f"- {e.actor}/{e.kind}" + (f" (entry {e.entry_id})" if e.entry_id else "") + f": {e.text}"
        for e in recent
    ] or ["(empty)"]
    out += ["", "# Artifacts you can read", catalog]
    return "\n".join(out)


def _obligation_line(o: Obligation) -> str:
    where = ", ".join(x for x in (o.item_id, o.finding_id) if x)
    return (
        f"- {o.obligation_id} [{o.status}] from {o.source_stage_key}"
        + (f" ({where})" if where else "")
        + f" for {o.target_stage_key}: {o.text}"
        + (f" (controlled at {o.locator})" if o.locator else "")
        + (f" (note: {o.note})" if o.note else "")
    )


def _critic_lines(report: CriticReport) -> list[str]:
    out = [f"Critic report (independent, own model): {report.verdict}. {report.summary}"]
    out += [
        f"- {p.point_id} [{p.severity}/{p.lens} {p.ref}]"
        + (f" @ {p.locator}" if p.locator else "")
        + f": {p.point}"
        + (f" Suggestion: {p.suggestion}" if p.suggestion else "")
        for p in report.points
    ]
    out += [
        f"- obligation {c.obligation_id}: {c.status}"
        + (f" at {c.locator}" if c.locator else "")
        + (f" ({c.note})" if c.note else "")
        for c in report.obligations
    ]
    return out


async def _stage_lines(req: ConsultationRequest, session: Session) -> str | None:
    """Review facts for the briefing: what the latest review blocks, and what may go on."""
    stage = req.situation.stage_key
    status = await session.ws.review_status(stage)
    accept = req.allowed_for("accept")
    if req.kind != "stage_outcome" or not (accept and accept.selection_key):
        return status
    try:
        free = await selectable(session.ws, stage)
    except ArtifactError:
        return status
    if free:
        line = f"Items you may select without a known risk: {', '.join(free)}."
    elif await session.ws.current_ids(stage):
        line = "No item can be selected without a known risk."
    else:
        return status
    return f"{status}\n{line}" if status else line


async def deliberate(
    req: ConsultationRequest, session: Session, *, lead: ChatLLM, critic: ChatLLM
) -> ConsultationResult:
    review = (
        await _stage_lines(req, session) if req.kind in ("stage_review", "stage_outcome") else None
    )
    start = briefing(req, session.ws.catalog(), review=review)
    if req.kind == "stage_review":
        return await _review(req, session, critic, start)

    outcome = Outcome()
    try:
        draft: Draft = await run_agent(
            lead,
            system=pi_system(req.kind),
            user=start,
            tools=pi_tools(session, outcome),
            outcome=outcome,
            actor="pi",
            trace=session.trace,
            meter=session.meter,
            max_steps=PI_MAX_STEPS,
            token_cap=TOKEN_CAP,
            terminal="decide",
        )
    except AgentLimit as exc:
        raise DeliberationError(exc.code, str(exc)) from exc
    except LLMError as exc:
        raise DeliberationError("llm_failed", str(exc)) from exc

    _keep_researcher_profile(draft, req)
    return ConsultationResult(
        consultation_id=req.consultation_id,
        status="decided",
        decision=draft.decision,
        rationale=draft.rationale,
        confidence=draft.confidence,
        options_considered=draft.options_considered,
        evidence=draft.evidence,
        assumptions=draft.assumptions,
        would_change_my_mind=draft.would_change_my_mind,
        critic_response=draft.critic_response,
        directives_applied=draft.directives_applied,
        escalate=draft.escalate,
        handoff=draft.handoff,
        notebook=[*session.notes, *draft.notebook],
        brief=draft.brief,
        usage=dict(session.meter.usage),
        trace=session.trace,
    )


async def _review(
    req: ConsultationRequest, session: Session, critic: ChatLLM, start: str
) -> ConsultationResult:
    try:
        report = await run_critic(
            req, session, critic, system=prompt("critic"), start=start, token_cap=TOKEN_CAP
        )
    except AgentLimit as exc:
        raise DeliberationError(exc.code, str(exc)) from exc
    except LLMError as exc:
        raise DeliberationError("llm_failed", str(exc)) from exc
    serious = sum(p.severity == "serious" for p in report.points)
    return ConsultationResult(
        consultation_id=req.consultation_id,
        status="decided",
        decision={"action": "report", "note": report.summary},
        rationale=report.summary,
        report=report,
        brief=(
            f"The Critic reviewed {req.situation.stage_key}: {report.verdict}. "
            f"It raised {serious} serious and {len(report.points) - serious} minor points and "
            f"checked {len(report.obligations)} accepted risks."
        ),
        usage=dict(session.meter.usage),
        trace=session.trace,
    )


def _keep_researcher_profile(draft: Draft, req: ConsultationRequest) -> None:
    """A profile value the researcher chose is not the PI's to change."""
    charter = draft.decision.charter
    if charter is None:
        return
    if req.profile.domain:
        charter.profile.domain = req.profile.domain
    if req.profile.research_type:
        charter.profile.research_type = req.profile.research_type
