"""A small, generic tool-using agent loop shared by the PI and its Critic."""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from principal_investigator.contract import TraceStep
from principal_investigator.llm import ChatLLM, Meter

OBSERVATION_LIMIT = 12_000


class ToolError(RuntimeError):
    """A tool could not do what was asked; the agent sees the message and may try again."""


class AgentLimit(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class Outcome:
    """Set by the terminal tool when it accepts the agent's final answer."""

    done: bool = False
    value: Any = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any]], Awaitable[str]]
    #: One trace line from the arguments and observation; never full content.
    summarize: Callable[[dict[str, Any], str], str] = field(
        default=lambda args, obs: obs.splitlines()[0][:160] if obs else ""
    )

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


async def run_agent(
    llm: ChatLLM,
    *,
    system: str,
    user: str,
    tools: list[Tool],
    outcome: Outcome,
    actor: str,
    trace: list[TraceStep],
    meter: Meter,
    max_steps: int,
    token_cap: int,
    terminal: str,
) -> Any:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    specs = [t.spec() for t in tools]
    by_name = {t.name: t for t in tools}
    for _ in range(max_steps):
        if meter.total > token_cap:
            raise AgentLimit("token_limit", f"{actor} exceeded {token_cap} tokens")
        reply = await llm.chat(messages, specs)
        if not reply.tool_calls:
            messages.append({"role": "assistant", "content": reply.content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": f"Continue by calling a tool. When ready, call `{terminal}`.",
                }
            )
            continue
        messages.append(
            {
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    }
                    for c in reply.tool_calls
                ],
            }
        )
        for call in reply.tool_calls:
            tool = by_name.get(call.name)
            args: dict[str, Any] = {}
            if tool is None:
                observation = f"error: unknown tool {call.name!r}"
            else:
                try:
                    parsed = json.loads(call.arguments or "{}")
                    args = parsed if isinstance(parsed, dict) else {}
                    observation = await tool.run(args)
                except json.JSONDecodeError as exc:
                    observation = f"error: arguments are not valid JSON ({exc})"
                except ToolError as exc:
                    observation = f"error: {exc}"
                except Exception as exc:  # a bad argument must not end the consultation
                    observation = f"error: {call.name} failed ({type(exc).__name__}: {exc})"
            if tool is None or observation.startswith("error"):
                summary = f"{call.name} failed: {observation.removeprefix('error: ')}"
            else:
                summary = tool.summarize(args, observation)
            trace.append(TraceStep(actor=actor, tool=call.name, summary=summary[:300]))
            if len(observation) > OBSERVATION_LIMIT:
                observation = observation[:OBSERVATION_LIMIT] + "\n[... observation truncated]"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
            if outcome.done:
                return outcome.value
    raise AgentLimit("step_limit", f"{actor} did not call {terminal} within {max_steps} steps")
