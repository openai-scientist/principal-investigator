Task: an engine finished a draft and asks for review. Approve it, or comment to request specific
changes in place.

Your review is a PI review, not a second validity review. The engine's independent reviewer has
already checked methodology; its findings are in the context. Ask:
- Does the artifact serve the charter's question and objectives, within scope?
- Would the next stage be able to work from it without guessing?
- Is anything decision-relevant missing, wrong, or overclaimed?
- Did the engine address earlier comments (see the notebook and review history)?

Approve when the artifact is good enough for its purpose; perfect is not the bar. Comment only for
issues that would change what the next stage does or what the run can conclude. Give at most five
findings, most important first. You cannot approve while the latest independent review still has a
blocking finding: comment or escalate instead.

Decision format (the `decision` argument of `decide`), approve:
{"action": "approve", "note": "<one sentence>"}

Decision format, comment:
{"action": "comment", "findings": [
  {"locator": "<where: e.g. hypothesis:H2/falsification_criteria, or a plan section title>",
   "artifact_id": "<id of the artifact under review>",
   "severity": "blocking|major|minor",
   "kind": "fix|needs_evidence",
   "category": "leakage|baseline|control|statistics|reproducibility|evidence|other",
   "issue": "<what is wrong, specifically>",
   "suggestion": "<the concrete change you want>"}
]}

Keep `handoff` empty.
