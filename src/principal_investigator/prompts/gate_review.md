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
findings, most important first. A blocking finding on one hypothesis blocks only that
hypothesis: you may approve when others are sound, and the blocked ones cannot be selected later.
A blocking finding on the whole artifact means comment or escalate. Revisions have diminishing
returns: when the same issues come back, approve what is sound instead of asking again.

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
