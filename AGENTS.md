# AGENTS.md

How to change this repo safely. `README.md` says what the service does.

## Verify

| Check | Command |
|---|---|
| Tests (offline) | `uv run pytest -q` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format --check .` |

## Rules

- The PI is an agent. Add capability as a **tool** (`tools.py`) or a **guard** (`guards.py`), not
  as more text pushed into the prompt. Prompts (`prompts/*.md`) state principles and decision
  formats only.
- Every rule that must never be broken (allowed actions, ids, no approval over a blocking
  review, critic before a consequential decision) is enforced in code, and a violation returns to
  the agent as an observation.
- Never store, log or return model keys. `ConsultationRequest.redacted()` is what is stored.
  Trace summaries are one line and never contain artifact content.
- Wire models live in `contract.py`. The Platform owns the contract
  (`management/docs/contract/`). Never import Platform code. A wire change starts in management.
- Tests never touch the network: use `tests/helpers.py` (`ScriptedLLM`, `FakePlatform`,
  `FakeWeb`).
- Do not commit or push unless the user asks.
