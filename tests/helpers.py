"""Test doubles: a scripted LLM and a fake Platform (artifact downloads and the result callback)."""

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx

from principal_investigator.contract import ConsultationRequest
from principal_investigator.llm import Meter, Reply, ToolCall

KEY = "k" * 40
PLATFORM = "http://platform.test"

HYPOTHESES = {
    "hypotheses": [
        {
            "id": "H1",
            "statement": "Retrieval of misleading passages raises error by 5 points.",
            "falsification_criteria": "Increase below 5 points.",
            "risk": "medium",
        },
        {"id": "H2", "statement": "Reranking reduces the effect.", "risk": "high"},
        {"id": "H3", "statement": "Larger models are less affected.", "risk": "low"},
    ]
}
FINDINGS_PASSED = {
    "round": 1,
    "reviewed_artifact_id": "a1",
    "reviewer_model": "r",
    "passed": True,
    "findings": [
        {
            "finding_id": "F1",
            "anchor": {"locator": "H2"},
            "severity": "minor",
            "issue": "Vague",
            "author": "reviewer",
        }
    ],
}
FINDINGS_BLOCKED = FINDINGS_PASSED | {
    "passed": False,
    "findings": [
        {
            "finding_id": "F2",
            "anchor": {"locator": "H1"},
            "severity": "blocking",
            "issue": "Leakage",
            "author": "reviewer",
        }
    ],
}


def blob(doc: Any) -> bytes:
    return json.dumps(doc).encode()


class FakePlatform:
    def __init__(self, artifacts: dict[str, bytes]):
        self.artifacts = artifacts
        self.results: list[dict] = []
        self.seen_keys: list[str | None] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.seen_keys.append(request.headers.get("X-Service-Key"))
        path = request.url.path
        if path.startswith("/artifacts/"):
            body = self.artifacts.get(path.rsplit("/", 1)[1])
            return httpx.Response(200, content=body) if body is not None else httpx.Response(404)
        if path.startswith("/callback/"):
            self.results.append(json.loads(request.content))
            return httpx.Response(200, json={})
        return httpx.Response(404)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def ref(artifact_id: str, content: bytes, *, schema: str, role: str, version: int = 1) -> dict:
    return {
        "artifact_id": artifact_id,
        "schema": schema,
        "version": version,
        "sha256": hashlib.sha256(content).hexdigest(),
        "download_url": f"{PLATFORM}/artifacts/{artifact_id}",
        "role": role,
        "stage_key": "hypothesis",
    }


CHARTER = {
    "question": "Do misleading retrieved passages raise QA error in small models?",
    "objectives": ["Falsifiable hypotheses", "A reviewed experiment plan"],
    "success_criteria": ["Each hypothesis names a dataset, baseline and metric"],
    "kill_criteria": ["No public dataset measures the outcome"],
}


def request(kind: str = "stage_outcome", **overrides) -> ConsultationRequest:
    model = {
        "provider": "openai_compatible",
        "base_url": "http://llm.test/v1",
        "model": "m",
        "api_key": "secret-key-value",
    }
    data = {
        "contract_version": "1.3.0",
        "consultation_id": "c-0001",
        "platform_run_id": "run-1",
        "kind": kind,
        "interaction_mode": "auto",
        "authority": "decide",
        "topic": "Misleading retrieval in small QA models",
        "charter": CHARTER,
        "situation": {
            "stage_key": "hypothesis",
            "stage_run_id": "s1",
            "iteration": 1,
            "max_iterations": 3,
            "pipeline": [
                {
                    "stage_key": "hypothesis",
                    "engine": "i2h",
                    "status": "COMPLETED",
                    "checkpoint": True,
                },
                {
                    "stage_key": "experiment_plan",
                    "engine": "exd",
                    "checkpoint": True,
                    "loop_back_to": ["hypothesis"],
                },
            ],
            "budget": {"limit_usd": "10", "spent_usd": "1.2"},
        },
        "allowed": [
            {"action": "accept", "selection_key": "hypothesis_ids"},
            {"action": "send_back", "targets": ["hypothesis"]},
            {"action": "stop"},
        ],
        "artifacts": [],
        "notebook": [],
        "models": {"lead": model, "critic": model | {"model": "c"}},
        "callback_url": f"{PLATFORM}/callback/c-0001",
    }
    data.update(overrides)
    return ConsultationRequest.model_validate(data)


def decision(action: str, **fields) -> dict:
    return {
        "decision": {"action": action, **fields},
        "rationale": "H1 and H3 fit charter:success_criteria[0].",
        "confidence": 0.8,
        "options_considered": [{"action": "stop", "summary": "End", "why_not": "Too early"}],
        "evidence": [{"ref": "H1", "point": "falsifiable"}],
        "assumptions": ["Data exists"],
        "would_change_my_mind": ["No dataset"],
        "escalate": None,
        "handoff": ["Prioritise H1"],
        "notebook": [{"kind": "lesson", "text": "H2 too risky"}],
        "brief": "Accepted H1 and H3.",
    }


def call(name: str, **args) -> Reply:
    """A model reply that calls one tool."""
    call.counter += 1
    return Reply(content=None, tool_calls=[ToolCall(f"t{call.counter}", name, json.dumps(args))])


call.counter = 0


def say(text: str) -> Reply:
    return Reply(content=text, tool_calls=[])


class ScriptedLLM:
    """Returns queued replies in order and records every message list it saw."""

    def __init__(self, replies: list[Reply], *, role: str, meter: Meter):
        self.replies = list(replies)
        self.role = role
        self.meter = meter
        self.calls: list[list[dict]] = []

    async def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        self.calls.append([dict(m) for m in messages])
        self.meter.add(self.role, 100, 10)
        if not self.replies:
            raise AssertionError(f"{self.role} called more often than scripted")
        return self.replies.pop(0)

    def observations(self) -> list[str]:
        """Tool results the agent saw, in order, from the last call."""
        return [m["content"] for m in self.calls[-1] if m["role"] == "tool"]


class FakeWeb:
    """OpenAlex and the HuggingFace Hub."""

    def __init__(self):
        self.queries: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.queries.append(str(request.url))
        if "openalex" in request.url.host:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "display_name": "Misleading retrieval hurts small QA models",
                            "publication_year": 2024,
                            "cited_by_count": 12,
                            "doi": "10.1/x",
                            "primary_location": {"source": {"display_name": "ACL"}},
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json=[
                {
                    "id": "google-research-datasets/natural_questions",
                    "downloads": 9000,
                    "likes": 50,
                    "gated": False,
                    "private": False,
                    "tags": ["license:cc-by-sa-3.0"],
                }
            ],
        )

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def tmp_store(tmp_path: Path):
    from principal_investigator.store import Store

    return Store(tmp_path / "store")
