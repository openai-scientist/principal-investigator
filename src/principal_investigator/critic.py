"""The Critic: an independent review of a completed stage, with a guarded report."""

from principal_investigator.agent import Outcome, run_agent
from principal_investigator.contract import ConsultationRequest, CriticReport
from principal_investigator.llm import ChatLLM
from principal_investigator.tools import Session, critic_tools

CRITIC_MAX_STEPS = 12


async def run_critic(
    req: ConsultationRequest,
    session: Session,
    llm: ChatLLM,
    *,
    system: str,
    start: str,
    token_cap: int,
) -> CriticReport:
    """Raises AgentLimit or LLMError; the caller turns them into a failed consultation."""
    outcome = Outcome()
    report: CriticReport = await run_agent(
        llm,
        system=system,
        user=start,
        tools=critic_tools(session, outcome),
        outcome=outcome,
        actor="critic",
        trace=session.trace,
        meter=session.meter,
        max_steps=CRITIC_MAX_STEPS,
        token_cap=token_cap,
        terminal="report",
    )
    lead, critic = req.models.lead, req.models.critic
    report.model = f"{critic.provider}/{critic.model}"
    if (critic.provider, critic.model) == (lead.provider, lead.model):
        report.model_shared_with = "lead"
    return report
