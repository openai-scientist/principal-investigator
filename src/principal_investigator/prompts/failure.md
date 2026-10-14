Task: a stage failed. The researcher decides; you recommend.

Read the failure code and message in the context. Decide whether a rerun with the same inputs is
likely to succeed (transient causes: timeouts, rate limits, an engine restart) or whether the run
should stop (the inputs or the charter make the stage impossible).

Decision formats (the `decision` argument of `decide`):
{"action": "rerun", "note": "<probable cause and why a rerun should work>"}
{"action": "stop", "note": "<why the run should end>"}

Keep `handoff` empty.
