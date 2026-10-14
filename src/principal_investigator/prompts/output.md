Arguments of the `decide` tool:

{
  "decision": { ...the decision format for this kind... },
  "rationale": "Why this decision, in 3-8 sentences, citing ids you actually read or checked.",
  "confidence": 0.0,
  "options_considered": [
    {"action": "<another allowed action or variant>", "summary": "<what it would do>", "why_not": "<why not>"}
  ],
  "evidence": [{"ref": "<id from what you read, e.g. H2, F1-3, charter:success_criteria[0], a DOI or dataset id you checked>", "point": "<what it shows>"}],
  "assumptions": ["<assumption your decision depends on>"],
  "would_change_my_mind": ["<observation that would make you decide differently>"],
  "escalate": null,
  "handoff": ["<instruction for the team of the next stage; empty unless the format asks for it>"],
  "brief": "<2-4 sentences for the researcher>",
  "critique_response": [{"point": "<Critic point>", "response": "accepted|rejected", "reason": "<why>"}]
}

`escalate`, when set, is {"reason_code": "integrity|charter_conflict|low_confidence|out_of_scope|cannot_decide", "message": "<one sentence>"}.
Consider at least one real alternative in `options_considered`.
