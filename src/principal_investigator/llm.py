"""OpenAI-compatible chat client with tool calling. Never logs messages or the key."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from principal_investigator.contract import ModelConfig, TokenUsage


class LLMError(RuntimeError):
    pass


@dataclass
class Meter:
    """Token usage per role for one consultation."""

    usage: dict[str, TokenUsage] = field(default_factory=dict)

    def add(self, role: str, tokens_in: int, tokens_out: int) -> None:
        current = self.usage.setdefault(role, TokenUsage())
        current.input_tokens += tokens_in
        current.output_tokens += tokens_out

    @property
    def total(self) -> int:
        return sum(u.input_tokens + u.output_tokens for u in self.usage.values())


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Reply:
    content: str | None
    tool_calls: list[ToolCall]


class ChatLLM(Protocol):
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply: ...


class OpenAICompatibleLLM:
    """One role (lead or critic) of one consultation."""

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        role: str,
        meter: Meter,
        timeout: float = 180.0,
        max_attempts: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if cfg.provider != "openai_compatible":
            raise LLMError("only openai_compatible models are supported")
        if not cfg.base_url or cfg.api_key is None:
            raise LLMError("model base_url and api_key are required")
        self.cfg = cfg
        self.role = role
        self.meter = meter
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.transport = transport

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": 0.2,
            **({"tools": tools, "tool_choice": "auto"} if tools else {}),
            **self.cfg.extra,
        }
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg.api_key.get_secret_value()}"}
        last_error = "no attempt"
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            for attempt in range(self.max_attempts):
                try:
                    response = await client.post(url, json=body, headers=headers)
                except httpx.RequestError as exc:
                    last_error = type(exc).__name__
                    await asyncio.sleep(2**attempt)
                    continue
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                    await asyncio.sleep(2**attempt)
                    continue
                if response.status_code >= 400:
                    raise LLMError(f"model request rejected: HTTP {response.status_code}")
                try:
                    data = response.json()
                    message = data["choices"][0]["message"]
                    usage = data.get("usage") or {}
                    calls = [
                        ToolCall(
                            id=c["id"],
                            name=c["function"]["name"],
                            arguments=c["function"].get("arguments") or "{}",
                        )
                        for c in message.get("tool_calls") or []
                    ]
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise LLMError("malformed model response") from exc
                self.meter.add(
                    self.role,
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                )
                return Reply(content=message.get("content"), tool_calls=calls)
        raise LLMError(f"model call failed after {self.max_attempts} attempts: {last_error}")
