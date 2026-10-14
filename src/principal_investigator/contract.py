"""PI contract wire models, written for this service.

The Platform (management) owns the contract and exports JSON Schema to
`management/docs/contract/`. This service never imports Platform code; the copy in
`tests/contract/vendor/` keeps these models honest.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

CONTRACT_VERSION = "1.2.0"
SERVICE_KEY_HEADER = "X-Service-Key"

Kind = Literal["charter", "gate_question", "gate_review", "stage_outcome", "failure", "close"]
Action = Literal[
    "charter", "answer", "approve", "comment", "accept", "send_back", "stop", "rerun", "report"
]
EscalationReason = Literal[
    "integrity", "charter_conflict", "low_confidence", "out_of_scope", "cannot_decide"
]


class Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


def compatible(version: str) -> bool:
    return version.split(".")[0] == CONTRACT_VERSION.split(".")[0]


class ResearchProfile(Wire):
    domain: str | None = None
    research_type: str | None = None


class Charter(Wire):
    question: str = Field(min_length=1)
    motivation: str = ""
    objectives: list[str] = Field(min_length=1)
    scope_in: list[str] = Field(default_factory=list)
    scope_out: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(min_length=1)
    kill_criteria: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    risk_posture: Literal["exploratory", "confirmatory"] = "exploratory"
    profile: ResearchProfile = Field(default_factory=ResearchProfile)


class PipelineStageView(Wire):
    stage_key: str
    engine: str
    status: str | None = None
    checkpoint: bool = False
    loop_back_to: list[str] = Field(default_factory=list)


class BudgetView(Wire):
    limit_usd: Decimal
    spent_usd: Decimal = Decimal("0")
    cost_unknown: bool = False


class Situation(Wire):
    stage_key: str | None = None
    stage_run_id: str | None = None
    iteration: int = 1
    max_iterations: int
    pipeline: list[PipelineStageView]
    budget: BudgetView
    failure: "FailureInfo | None" = None


class GateOption(Wire):
    option_id: str
    label: str


class Gate(Wire):
    gate_id: str
    kind: Literal["question", "artifact_review"]
    step_key: str | None = None
    prompt: str
    options: list[GateOption] = Field(default_factory=list)
    default_option_id: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)


class AllowedAction(Wire):
    action: Action
    option_ids: list[str] = Field(default_factory=list)
    selection_key: str | None = None
    targets: list[str] = Field(default_factory=list)


class ConsultArtifact(Wire):
    artifact_id: str
    schema_id: str = Field(alias="schema")
    version: int
    sha256: str
    download_url: str
    primary: bool = False
    role: Literal["primary", "review", "recommendation", "input", "history", "other"]
    stage_key: str | None = None
    size_bytes: int | None = None


class NotebookEntry(Wire):
    kind: str
    text: str
    actor: Literal["pi", "user"]
    stage_key: str | None = None
    at: datetime


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    provider: Literal["openai_compatible", "bedrock"]
    base_url: str | None = None
    model: str = Field(min_length=1)
    api_key: SecretStr | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class PIModels(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    lead: ModelConfig
    critic: ModelConfig


class ConsultationRequest(Wire):
    contract_version: str
    consultation_id: str = Field(min_length=1)
    platform_run_id: str
    kind: Kind
    interaction_mode: Literal["auto", "light", "copilot", "full"]
    authority: Literal["decide", "advise"]
    topic: str
    charter: Charter | None = None
    profile: ResearchProfile = Field(default_factory=ResearchProfile)
    situation: Situation
    gate: Gate | None = None
    allowed: list[AllowedAction] = Field(default_factory=list)
    artifacts: list[ConsultArtifact] = Field(default_factory=list)
    notebook: list[NotebookEntry] = Field(default_factory=list)
    models: PIModels
    callback_url: str

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if not compatible(self.contract_version):
            raise ValueError("incompatible contract_version")
        if self.kind in ("gate_question", "gate_review") and self.gate is None:
            raise ValueError("gate kinds require gate")
        if self.kind != "charter" and self.charter is None:
            raise ValueError("every kind except charter requires the charter")
        return self

    def allowed_for(self, action: str) -> AllowedAction | None:
        return next((a for a in self.allowed if a.action == action), None)

    def redacted(self) -> dict[str, Any]:
        """The request as stored: credentials removed."""
        data = self.model_dump(mode="json", by_alias=True)
        for role in ("lead", "critic"):
            data["models"][role].pop("api_key", None)
        return data


class Anchor(Wire):
    artifact_id: str | None = None
    locator: str = Field(min_length=1)


class Finding(Wire):
    finding_id: str
    anchor: Anchor
    severity: Literal["blocking", "major", "minor"]
    kind: Literal["fix", "needs_evidence"] = "fix"
    category: Literal[
        "leakage", "baseline", "control", "statistics", "reproducibility", "evidence", "other"
    ] = "other"
    issue: str = Field(min_length=1)
    suggestion: str | None = None
    author: Literal["reviewer", "researcher"] = "researcher"
    evidence_refs: list[str] = Field(default_factory=list)


class PIDecision(Wire):
    action: Action
    option_id: str | None = None
    note: str | None = None
    findings: list[Finding] = Field(default_factory=list)
    selection: dict[str, list[str]] = Field(default_factory=dict)
    target_stage_key: str | None = None
    charter: Charter | None = None


class ConsideredOption(Wire):
    action: str
    summary: str
    why_not: str | None = None


class EvidencePoint(Wire):
    ref: str
    point: str


class Critique(Wire):
    verdict: Literal["upheld", "revised"]
    points: list[str] = Field(default_factory=list)


class Escalation(Wire):
    reason_code: EscalationReason
    message: str = Field(min_length=1)


class NoteDraft(Wire):
    kind: Literal["lesson", "risk", "observation", "question"]
    text: str = Field(min_length=1)


class TokenUsage(Wire):
    input_tokens: int = 0
    output_tokens: int = 0


class FailureInfo(Wire):
    code: str
    message: str


class TraceStep(Wire):
    actor: Literal["pi", "critic"]
    tool: str
    summary: str


class ConsultationResult(Wire):
    consultation_id: str
    status: Literal["decided", "failed"]
    failure: FailureInfo | None = None
    decision: PIDecision | None = None
    rationale: str = ""
    confidence: float | None = Field(None, ge=0, le=1)
    options_considered: list[ConsideredOption] = Field(default_factory=list)
    evidence: list[EvidencePoint] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    would_change_my_mind: list[str] = Field(default_factory=list)
    critique: Critique | None = None
    escalate: Escalation | None = None
    handoff: list[str] = Field(default_factory=list)
    notebook: list[NoteDraft] = Field(default_factory=list)
    brief: str = ""
    usage: dict[Literal["lead", "critic"], TokenUsage] = Field(default_factory=dict)
    trace: list[TraceStep] = Field(default_factory=list)


class ConsultationAccepted(Wire):
    consultation_id: str
    created: bool


class ConsultationState(Wire):
    consultation_id: str
    status: Literal["running", "decided", "failed"]
    result: ConsultationResult | None = None


class PICapabilities(Wire):
    service_key: Literal["pi"] = "pi"
    service_version: str
    contract_version: str = CONTRACT_VERSION
    kinds: list[Kind]
    providers: list[Literal["openai_compatible", "bedrock"]] = Field(
        default_factory=lambda: ["openai_compatible"]
    )


class Health(Wire):
    status: Literal["ok", "degraded"]
    service_version: str
    contract_version: str = CONTRACT_VERSION
    active_consultations: int = 0


Situation.model_rebuild()
ConsultationRequest.model_rebuild()
