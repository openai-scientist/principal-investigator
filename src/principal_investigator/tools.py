"""The tools of the PI and of its Critic."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from principal_investigator.agent import Outcome, Tool, ToolError
from principal_investigator.contract import ConsultationRequest, NoteDraft, TraceStep
from principal_investigator.guards import Draft, GuardError, check, check_report
from principal_investigator.llm import Meter
from principal_investigator.workspace import ArtifactError, Workspace

OPENALEX = "https://api.openalex.org/works"
HF_DATASETS = "https://huggingface.co/api/datasets"
NOTE_KINDS = ["lesson", "risk", "observation", "question"]
#: Words a model adds to a dataset query that make the Hub's name search match nothing.
_QUERY_NOISE = re.compile(r"\b(hugging\s*face|hf|hub|datasets?|benchmarks?)\b", re.IGNORECASE)
#: Seconds before the one retry of a lookup that hit a rate limit or a transient failure.
RETRY_DELAY = 2.0


@dataclass
class Session:
    """Everything one consultation's agents share."""

    req: ConsultationRequest
    ws: Workspace
    meter: Meter
    trace: list[TraceStep] = field(default_factory=list)
    notes: list[NoteDraft] = field(default_factory=list)
    transport: httpx.AsyncBaseTransport | None = None
    contact_email: str | None = None
    openalex_api_key: str | None = None


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
            + (f" (entry {e.entry_id})" if e.entry_id else "")
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
        if s.contact_email:
            params["mailto"] = s.contact_email
        if s.openalex_api_key:
            params["api_key"] = s.openalex_api_key
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
        if "/" not in query:  # an owner/name id is exact; only a free name gets cleaned
            query = " ".join(_QUERY_NOISE.sub(" ", query).split())
        if not query:
            raise ToolError("query is required: the dataset's name, e.g. 'cifar10'")
        params = {"search": query, "limit": 8, "full": "true", "sort": "downloads", "direction": -1}
        data = await _get_json(s, HF_DATASETS, params)
        rows = []
        for d in data if isinstance(data, list) else []:
            tags = d.get("tags") or []
            license_ = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), "?")
            rows.append(
                f"- {d.get('id')}: downloads={d.get('downloads')}, likes={d.get('likes')}, "
                f"license={license_}, gated={d.get('gated')}, private={d.get('private')}"
            )
        if not rows:
            return (
                f"No HuggingFace dataset matches {query!r}. The Hub searches dataset ids; try a "
                "shorter name. Absence here does not show that the dataset is unavailable: "
                "many originals live on Zenodo, GitHub or a project page."
            )
        return "\n".join(rows) + (
            "\nMost downloaded first. A community mirror may carry another license than the "
            "original; license=? means the card declares none."
        )

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
            "license, gated. `query` is the dataset's name only (e.g. 'cifar10', 'defects4j').",
            _schema({"query": {"type": "string"}}, ["query"]),
            check_dataset,
            lambda a, o: f"checked datasets for {a.get('query')!r}",
        ),
    ]


async def _get_json(s: Session, url: str, params: dict[str, Any]) -> Any:
    """GET a public API; a rate limit or a transient failure is retried once."""
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=20, transport=s.transport) as client:
                response = await client.get(url, params=params)
        except httpx.HTTPError as exc:
            if attempt == 1:
                await asyncio.sleep(RETRY_DELAY)
                continue
            raise ToolError(f"{url} unreachable ({type(exc).__name__})") from exc
        if attempt == 1 and (response.status_code == 429 or response.status_code >= 500):
            await asyncio.sleep(RETRY_DELAY)
            continue
        break
    if response.status_code != 200:
        raise ToolError(f"{url} answered HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise ToolError(f"{url} returned invalid JSON") from exc


# -- PI-only tools -----------------------------------------------------------------------

#: Arguments of `decide` besides `decision`.
TOP_LEVEL = (
    "rationale",
    "confidence",
    "options_considered",
    "evidence",
    "assumptions",
    "would_change_my_mind",
    "escalate",
    "handoff",
    "brief",
    "critic_response",
    "directives_applied",
    "notebook",
)

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
        "critic_response": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point_id": {"type": "string"},
                    "response": {"type": "string", "enum": ["accepted", "rejected"]},
                    "reason": {"type": "string"},
                },
            },
        },
        "directives_applied": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"entry_id": {"type": "string"}, "how": {"type": "string"}},
            },
        },
    },
    ["decision", "rationale", "confidence", "brief"],
)


EXTERNAL_TOOLS = ("search_literature", "check_dataset")


def pi_tools(s: Session, outcome: Outcome) -> list[Tool]:
    async def write_note(args: dict) -> str:
        kind, text = args.get("kind"), str(args.get("text") or "").strip()
        if kind not in NOTE_KINDS or not text:
            raise ToolError(f"kind must be one of {NOTE_KINDS} and text nonempty")
        if len(s.notes) >= 6:
            raise ToolError("at most 6 notes per consultation; keep only what helps later")
        s.notes.append(NoteDraft(kind=kind, text=text[:600]))
        return "noted"

    async def decide(args: dict) -> str:
        decision = args["decision"] = _object(args.get("decision"))
        if not isinstance(decision, dict):
            raise ToolError("`decision` must be an object with an `action`")
        # Seen live: models nest the other arguments inside `decision`; lift them out.
        for key in TOP_LEVEL:
            if key in decision and key not in args:
                args[key] = _object(decision.pop(key))
        try:
            draft: Draft = await check(args, s.req, s.ws)
        except GuardError as exc:
            raise ToolError(f"decision rejected: {exc}") from exc
        except ArtifactError as exc:
            raise ToolError(f"decision could not be checked: {exc}") from exc
        outcome.done, outcome.value = True, draft
        return "decision recorded"

    # After the charter the engines have verified sources and datasets; live v2 runs showed the
    # PI repeating exd's dataset validator at stage outcomes.
    external = () if s.req.kind == "charter" else EXTERNAL_TOOLS
    tools = [
        *(t for t in read_tools(s) if t.name not in external),
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
            "decide",
            "Finish with your decision. It is checked by code; if rejected, fix it and call "
            "decide again.",
            DECIDE_PARAMETERS,
            decide,
            lambda a, o: f"decided {(a.get('decision') or {}).get('action')}",
        ),
    ]
    return tools


# -- Critic tools --------------------------------------------------------------------------

CRITIC_READ_TOOLS = ("list_artifacts", "read_artifact", "read_notebook")

REPORT_PARAMETERS = _schema(
    {
        "verdict": {"type": "string", "enum": ["sound", "concerns", "serious"]},
        "summary": {"type": "string"},
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["serious", "minor"]},
                    "lens": {
                        "type": "string",
                        "enum": ["charter", "risk", "directive", "integrity"],
                    },
                    "ref": {"type": "string"},
                    "locator": {"type": "string"},
                    "point": {"type": "string"},
                    "suggestion": {"type": "string"},
                    "fix_class": {"type": "string", "enum": ["fix", "needs_evidence", "later"]},
                },
            },
        },
        "obligations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "obligation_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["controlled", "unresolved"]},
                    "locator": {"type": "string"},
                    "note": {"type": "string"},
                },
            },
        },
    },
    ["verdict", "summary", "points", "obligations"],
)
REPORT_PARAMETERS["properties"]["resolutions"] = {
    "type": "array",
    "description": "Only in a repeat review: one entry for each earlier serious point.",
    "items": {
        "type": "object",
        "properties": {
            "point_id": {"type": "string"},
            "status": {"type": "string", "enum": ["resolved", "maintained"]},
            "reason": {"type": "string"},
        },
    },
}


def critic_tools(s: Session, outcome: Outcome) -> list[Tool]:
    async def report(args: dict) -> str:
        for key in ("points", "obligations", "resolutions"):
            args[key] = _object(args.get(key))
        try:
            outcome.value = await check_report(args, s.req, s.ws)
        except GuardError as exc:
            raise ToolError(f"report rejected: {exc}") from exc
        outcome.done = True
        return "report recorded"

    return [
        *(t for t in read_tools(s) if t.name in CRITIC_READ_TOOLS),
        Tool(
            "report",
            "Finish with your report. It is checked by code; if rejected, fix it and call "
            "report again.",
            REPORT_PARAMETERS,
            report,
            lambda a, o: (
                f"report {a.get('verdict')} ({len(a.get('points') or [])} points, "
                f"{len(a.get('obligations') or [])} obligations)"
            ),
        ),
    ]
