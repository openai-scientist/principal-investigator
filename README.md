# principal-investigator

The AI Principal Investigator (PI) of the AI-scientist platform. The Platform (`management`)
asks the PI to make or recommend the pipeline decisions of a research run: the research charter,
what happens when a stage completes or fails, and the closing report. It asks a separate Critic to
review each completed stage. Engine gates are not the PI's: `gate_question` and `gate_review`
requests are refused with 422 (`unsupported_kind`).
Design: `../docs/superpowers/specs/2026-10-15-pi-v3-design.md`.

## How a consultation works

1. The Platform posts a `ConsultationRequest` (`POST /consultations`, `X-Service-Key`). It holds
   the charter, the situation, the allowed actions, the lab notebook, references to the run's
   artifacts, and the models to use (lead and critic, with keys that are never stored).
2. The service answers `202` and runs the **PI agent** in the background. The agent starts from a
   briefing, not from the artifacts. It investigates with tools: `list_artifacts`,
   `read_artifact` (by hypothesis id, Markdown section, JSON key, with paging), `read_notebook`,
   `search_literature` (OpenAlex), `check_dataset` (HuggingFace Hub), `write_note`.
3. For `stage_review` the **Critic agent** answers instead of the PI (another model, read tools
   only: `list_artifacts`, `read_artifact`, `read_notebook`). It does not re-review the science. It
   checks charter fit, accepted risks (obligations), researcher directives and integrity, and ends
   with `report`. Code checks the report: at most 8 points, a valid `lens`/`ref` for each, one
   status for every obligation, a verdict that matches the points.
4. The PI ends with `decide`. Code checks the decision (allowed action, offered option, known
   ids, allowed send-back target, a blocked item only as a named known risk). At a stage outcome the
   PI must answer every serious Critic point (`critic_response`), cite every researcher directive
   (`directives_applied`), and argue against a partial `accept` when part of the stage is sound. The
   closing report must name every open or unresolved obligation. A rejected decision comes back to
   the agent as an observation, and it fixes it.
5. The service posts a `ConsultationResult` to the callback URL: the decision, rationale,
   confidence, alternatives, evidence, the Critic `report` (for `stage_review`), escalation, handoff for the next stage, notebook
   entries, a brief for the researcher, token usage and the **trace** of every agent step.

The service holds no run state. The Platform is the record of charters, decisions and the
notebook. The service keeps one redacted JSON file per consultation in `PI_STORE_DIR` so that
`GET /consultations/{id}` can answer after a callback is lost.

## Run

```bash
uv sync
PI_SERVICE_KEY=<32+ characters> PI_STORE_DIR=runs uv run pi-api   # 127.0.0.1:8003
```

Environment: `PI_SERVICE_KEY` (required, shared with the Platform's `PI_SERVICE`),
`PI_STORE_DIR` (default `runs`), `PI_HOST`, `PI_PORT` (default 8003), `PI_LLM_TIMEOUT_SECONDS`,
`PI_CONTACT_EMAIL` (optional; sent to OpenAlex as `mailto` for its polite pool).

## Test

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```
