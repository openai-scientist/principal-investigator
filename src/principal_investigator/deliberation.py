"""One consultation: the PI agent investigates, consults its Critic agent, and decides."""

import json
from importlib import resources
from typing import Any

from principal_investigator.agent import AgentLimit, Outcome, run_agent
from principal_investigator.contract import (
    ConsultationRequest,
    ConsultationResult,
    Critique,
    NoteDraft,
)
from principal_investigator.guards import Draft
from principal_investigator.llm import ChatLLM, LLMError
from principal_investigator.tools import Session, critic_tools, pi_tools

PI_MAX_STEPS = 24
CRITIC_MAX_STEPS = 10
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


def briefing(req: ConsultationRequest, catalog: str) -> str:
    """What the agent knows when it starts. Artifacts are listed, not included: it reads them."""
    s = req.situation
    out = [
        "# Run",
        f"Interaction mode: {req.interaction_mode}. Your authority here: "
        f"{AUTHORITY_TEXT[req.authority]}",
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
    if req.gate is not None:
        g = req.gate
        out.append(f"Gate {g.gate_id} ({g.kind}) at step {g.step_key}: {g.prompt}")
        for option in g.options:
            mark = " (engine default)" if option.option_id == g.default_option_id else ""
            out.append(f"- option {option.option_id}: {option.label}{mark}")
        if g.artifact_ids:
            out.append(f"Artifacts under review: {', '.join(g.artifact_ids)}")
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
    out += [f"- {e.actor}/{e.kind}: {e.text}" for e in recent] or ["(empty)"]
    out += ["", "# Artifacts you can read", catalog]
    return "\n".join(out)


async def deliberate(
    req: ConsultationRequest, session: Session, *, lead: ChatLLM, critic: ChatLLM
) -> ConsultationResult:
    start = briefing(req, session.ws.catalog())

    async def run_critic(proposal: dict[str, Any], questions: str) -> dict[str, Any]:
        outcome = Outcome()
        user = (
            f"{start}\n\n# The PI's proposal\n{json.dumps(proposal, ensure_ascii=False, indent=1)}"
            + (f"\n\n# The PI asks you\n{questions}" if questions else "")
        )
        return await run_agent(
            critic,
            system=prompt("critic"),
            user=user,
            tools=critic_tools(session, outcome),
            outcome=outcome,
            actor="critic",
            trace=session.trace,
            meter=session.meter,
            max_steps=CRITIC_MAX_STEPS,
            token_cap=TOKEN_CAP,
            terminal="verdict",
        )

    session.critic = run_critic
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
    points = [p for c in session.critiques for p in c["verdict"].get("points", [])]
    flags = [f for c in session.critiques for f in c["verdict"].get("integrity_flags", [])]
    notebook = [*session.notes, *draft.notebook]
    notebook += [NoteDraft(kind="risk", text=f"Critic integrity flag: {f}") for f in flags]
    critique = None
    if session.critiques:
        revised = any(c["verdict"].get("verdict") == "revise" for c in session.critiques)
        critique = Critique(
            verdict="revised" if revised else "upheld",
            points=[f"[{p['severity']}] {p['point']}" for p in points],
        )
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
        critique=critique,
        escalate=draft.escalate,
        handoff=draft.handoff,
        notebook=notebook,
        brief=draft.brief,
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
