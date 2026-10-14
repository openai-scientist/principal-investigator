"""The tools of the PI and of its Critic."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from principal_investigator.agent import Outcome, Tool, ToolError
from principal_investigator.contract import ConsultationRequest, NoteDraft, TraceStep
from principal_investigator.guards import Draft, GuardError, check
from principal_investigator.llm import Meter
from principal_investigator.workspace import ArtifactError, Workspace

OPENALEX = "https://api.openalex.org/works"
HF_DATASETS = "https://huggingface.co/api/datasets"
#: Kinds whose decision has consequences: the PI must hear the Critic before deciding.
NEEDS_CRITIC = {"charter", "gate_review", "stage_outcome"}
#: Decision fields the Critic must have seen exactly; notes and findings may be refined after.
SUBSTANCE = ("action", "option_id", "selection", "target_stage_key")
MAX_CRITIC_CALLS = 3
NOTE_KINDS = ["lesson", "risk", "observation", "question"]


@dataclass
class Session:
    """Everything one consultation's agents share."""

    req: ConsultationRequest
    ws: Workspace
    meter: Meter
    trace: list[TraceStep] = field(default_factory=list)
    notes: list[NoteDraft] = field(default_factory=list)
    critiques: list[dict[str, Any]] = field(default_factory=list)
    transport: httpx.AsyncBaseTransport | None = None
    #: Runs the Critic agent on a proposal; set by the deliberation module.
    critic: Callable[[dict[str, Any], str], Awaitable[dict[str, Any]]] | None = None


def _object(value: Any) -> Any:
    """Models sometimes pass a nested object as a JSON string; accept both."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


# -- shared read and verify tools ---------------------------------------------------------


def read_tools(s: Session) -> list[Tool]:
    async def list_artifacts(_: dict) -> str:
        return s.ws.catalog()

    async def read_artifact(args: dict) -> str:
        artifact_id = str(args.get("artifact_id") or "")
        offset = args.get("offset") or 0
        section = args.get("section")
        if section is not None and not isinstance(section, str):
            raise ToolError("`section` must be a string such as H2, a heading, or a JSON key")
        try:
            return await s.ws.read(
                artifact_id, args.get("section"), int(offset) if str(offset).isdigit() else 0
            )
        except ArtifactError as exc:
            raise ToolError(str(exc)) from exc

    async def read_notebook(args: dict) -> str:
        kind = args.get("kind")
        lines = [
            f"- {e.at:%Y-%m-%d %H:%M} {e.actor}/{e.kind}"
            + (f" [{e.stage_key}]" if e.stage_key else "")
            + f": {e.text}"
            for e in s.req.notebook
            if not kind or e.kind == kind
        ]
        lines += [f"- (this consultation) pi/{n.kind}: {n.text}" for n in s.notes]
        return "\n".join(lines) or "(no notebook entries)"

    async def search_literature(args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ToolError("query is required")
        limit = min(int(args.get("limit") or 6), 10)
        params = {
            "search": query,
            "per-page": limit,
            "select": "display_name,publication_year,cited_by_count,doi,primary_location",
        }
        data = await _get_json(s, OPENALEX, params)
        rows = []
        for w in data.get("results", []):
            venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
            rows.append(
                f"- {w.get('display_name')} ({w.get('publication_year')}; "
                f"cited {w.get('cited_by_count')}; {venue or 'venue unknown'}) {w.get('doi') or ''}"
            )
        return "\n".join(rows) or f"No works found for {query!r}."

    async def check_dataset(args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ToolError("query is required")
        data = await _get_json(s, HF_DATASETS, {"search": query, "limit": 8, "full": "true"})
        rows = []
        for d in data if isinstance(data, list) else []:
            tags = d.get("tags") or []
            license_ = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), "?")
            rows.append(
                f"- {d.get('id')}: downloads={d.get('downloads')}, likes={d.get('likes')}, "
                f"license={license_}, gated={d.get('gated')}, private={d.get('private')}"
            )
        return "\n".join(rows) or f"No HuggingFace dataset matches {query!r}."

    return [
        Tool(
            "list_artifacts",
            "List the run's artifacts you can read: id, schema, version, role, stage, size.",
            _schema({}),
            list_artifacts,
            lambda a, o: f"listed {o.count(chr(10)) + 1 if o else 0} artifacts",
        ),
        Tool(
            "read_artifact",
            "Read an artifact. Without `section` you get an overview (hypothesis list, table of "
            "contents, or the first page). `section` is a hypothesis id (H2), a Markdown section "
            "title, or a JSON key. Long text is paged; pass `offset` to continue.",
            _schema(
                {
                    "artifact_id": {"type": "string"},
                    "section": {"type": "string"},
                    "offset": {"type": "integer"},
                },
                ["artifact_id"],
            ),
            read_artifact,
            lambda a, o: (
                f"read {a.get('artifact_id')}"
                + (f" section {a.get('section')!r}" if a.get("section") else "")
                + (" (error)" if o.startswith("error") else "")
            ),
        ),
        Tool(
            "read_notebook",
            "Read the run's lab notebook: earlier decisions, lessons, risks, researcher "
            "directives and overrides. Optional filter `kind`.",
            _schema({"kind": {"type": "string"}}),
            read_notebook,
            lambda a, o: "read the notebook" + (f" ({a.get('kind')})" if a.get("kind") else ""),
        ),
        Tool(
            "search_literature",
            "Search scholarly literature (OpenAlex) to check novelty, a cited claim, or a known "
            "baseline. Returns titles, years, citation counts and DOIs.",
            _schema({"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"]),
            search_literature,
            lambda a, o: f"searched literature for {a.get('query')!r}",
        ),
        Tool(
            "check_dataset",
            "Check on the HuggingFace Hub that a dataset exists and is usable: id, downloads, "
            "license, gated.",
            _schema({"query": {"type": "string"}}, ["query"]),
            check_dataset,
            lambda a, o: f"checked datasets for {a.get('query')!r}",
        ),
    ]


async def _get_json(s: Session, url: str, params: dict[str, Any]) -> Any:
    try:
        async with httpx.AsyncClient(timeout=20, transport=s.transport) as client:
            response = await client.get(url, params=params)
    except httpx.HTTPError as exc:
        raise ToolError(f"{url} unreachable ({type(exc).__name__})") from exc
    if response.status_code != 200:
        raise ToolError(f"{url} answered HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise ToolError(f"{url} returned invalid JSON") from exc


# -- PI-only tools -----------------------------------------------------------------------

DECIDE_PARAMETERS = _schema(
    {
        "decision": {
            "type": "object",
            "description": "The decision; its format is given in your instructions for this kind.",
        },
        "rationale": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "options_considered": {"type": "array", "items": {"type": "object"}},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"ref": {"type": "string"}, "point": {"type": "string"}},
            },
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "would_change_my_mind": {"type": "array", "items": {"type": "string"}},
        "escalate": {
            "type": ["object", "null"],
            "properties": {"reason_code": {"type": "string"}, "message": {"type": "string"}},
        },
        "handoff": {"type": "array", "items": {"type": "string"}},
        "brief": {"type": "string"},
        "critique_response": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "response": {"type": "string", "enum": ["accepted", "rejected"]},
                    "reason": {"type": "string"},
                },
            },
        },
    },
    ["decision", "rationale", "confidence", "brief"],
)


def pi_tools(s: Session, outcome: Outcome) -> list[Tool]:
    async def write_note(args: dict) -> str:
        kind, text = args.get("kind"), str(args.get("text") or "").strip()
        if kind not in NOTE_KINDS or not text:
            raise ToolError(f"kind must be one of {NOTE_KINDS} and text nonempty")
        if len(s.notes) >= 6:
            raise ToolError("at most 6 notes per consultation; keep only what helps later")
        s.notes.append(NoteDraft(kind=kind, text=text[:600]))
        return "noted"

    async def consult_critic(args: dict) -> str:
        proposal = args["proposal"] = _object(args.get("proposal"))
        if isinstance(proposal, dict):
            proposal["decision"] = _object(proposal.get("decision"))
        if not isinstance(proposal, dict) or not isinstance(proposal.get("decision"), dict):
            raise ToolError("proposal must be an object with a `decision` and your `rationale`")
        if len(s.critiques) >= MAX_CRITIC_CALLS:
            raise ToolError("you already consulted the Critic the maximum number of times")
        assert s.critic is not None
        verdict = await s.critic(proposal, str(args.get("questions") or ""))
        s.critiques.append({"proposal": proposal, "verdict": verdict})
        return json.dumps(verdict, ensure_ascii=False, indent=1)

    async def decide(args: dict) -> str:
        decision = args["decision"] = _object(args.get("decision"))
        if not isinstance(decision, dict):
            raise ToolError("`decision` must be an object with an `action`")
        if s.req.kind in NEEDS_CRITIC:
            if not s.critiques:
                raise ToolError("consult the Critic with your draft decision before deciding")
            last = s.critiques[-1]
            reviewed = last["proposal"]["decision"]
            changed = [k for k in SUBSTANCE if reviewed.get(k) != decision.get(k)]
            if changed:
                raise ToolError(
                    "your decision differs from the draft the Critic reviewed "
                    f"({', '.join(changed)}); consult the Critic on this decision first"
                )
            majors = [p for p in last["verdict"].get("points", []) if p.get("severity") == "major"]
            if majors and not args.get("critique_response"):
                raise ToolError(
                    "the Critic raised major points; answer each in `critique_response` "
                    "(accepted or rejected, with a reason)"
                )
        try:
            draft: Draft = await check(args, s.req, s.ws)
        except GuardError as exc:
            raise ToolError(f"decision rejected: {exc}") from exc
        except ArtifactError as exc:
            raise ToolError(f"decision could not be checked: {exc}") from exc
        outcome.done, outcome.value = True, draft
        return "decision recorded"

    return [
        *read_tools(s),
        Tool(
            "write_note",
            "Add an entry to the lab notebook for later decisions (lesson, risk, observation, "
            "question). One or two sentences.",
            _schema(
                {"kind": {"type": "string", "enum": NOTE_KINDS}, "text": {"type": "string"}},
                ["kind", "text"],
            ),
            write_note,
            lambda a, o: f"noted a {a.get('kind')}",
        ),
        Tool(
            "consult_critic",
            "Ask the independent Scientific Critic to challenge a draft decision. Pass the full "
            "draft as `proposal` (the same object you would pass to decide) and optional "
            "`questions`. The Critic investigates with its own tools and returns a verdict.",
            _schema(
                {"proposal": {"type": "object"}, "questions": {"type": "string"}}, ["proposal"]
            ),
            consult_critic,
            lambda a, o: (
                "consulted the Critic on "
                + str(((a.get("proposal") or {}).get("decision") or {}).get("action"))
            ),
        ),
        Tool(
            "decide",
            "Finish with your decision. It is checked by code; if rejected, fix it and call "
            "decide again.",
            DECIDE_PARAMETERS,
            decide,
            lambda a, o: f"decided {(a.get('decision') or {}).get('action')}",
        ),
    ]


# -- Critic-only terminal tool -------------------------------------------------------------


def critic_tools(s: Session, outcome: Outcome) -> list[Tool]:
    async def verdict(args: dict) -> str:
        value = args.get("verdict")
        if value not in ("uphold", "revise"):
            raise ToolError("verdict must be uphold or revise")
        points = []
        for p in args.get("points") or []:
            if isinstance(p, dict) and str(p.get("point") or "").strip():
                severity = "major" if p.get("severity") == "major" else "minor"
                points.append({"severity": severity, "point": str(p["point"]).strip()[:800]})
        if value == "revise" and not any(p["severity"] == "major" for p in points):
            raise ToolError("revise needs at least one major point")
        flags = [str(f).strip()[:400] for f in args.get("integrity_flags") or [] if str(f).strip()]
        outcome.done = True
        outcome.value = {"verdict": value, "points": points, "integrity_flags": flags}
        return "verdict recorded"

    return [
        *read_tools(s),
        Tool(
            "verdict",
            "Finish with your verdict on the PI's proposal.",
            _schema(
                {
                    "verdict": {"type": "string", "enum": ["uphold", "revise"]},
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "severity": {"type": "string", "enum": ["major", "minor"]},
                                "point": {"type": "string"},
                            },
                        },
                    },
                    "integrity_flags": {"type": "array", "items": {"type": "string"}},
                },
                ["verdict", "points"],
            ),
            verdict,
            lambda a, o: (
                f"verdict {a.get('verdict')} "
                f"({sum(p.get('severity') == 'major' for p in a.get('points') or [])} major)"
            ),
        ),
    ]
