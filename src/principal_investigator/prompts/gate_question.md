Task: an engine paused and asked a question. Choose one of its options.

Read the gate prompt and options carefully. The engine marks one option as its default; a default
is the engine's convenience, not evidence. Choose the option that best serves the charter. If two
options are close, prefer the one that keeps more future choices open at lower cost.

Decision format (the `decision` argument of `decide`):
{"action": "answer", "option_id": "<one of the allowed option ids>", "note": "<one or two sentences
 the engine team will read: why, and anything they should watch>", "params": {}}

Some options act on items named in `params`, for example dropping papers from a shortlist:
`"params": {"dropped": ["<paper id>", ...]}` with ids read from the gate's artifact.

Keep `handoff` empty. Keep the rationale short: a gate is a small decision.
