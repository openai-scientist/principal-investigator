Task: a stage is complete. Decide what the run does next.

Allowed actions are listed in the context. Typical choices:
- `accept`: the stage's result is the input of the next stage. When a `selection_key` is given (for
  example `hypothesis_ids`), choose which items go forward. This is a portfolio decision. For each
  candidate weigh: fit with the charter's question and success criteria; falsifiability (a concrete
  failing observation); feasibility with public data and the remaining budget; information value
  (would either outcome teach us something?); risk spread across the portfolio. Usually choose 1-3.
  Do not choose an item only because the engine ranked it first. Explain each chosen and each
  rejected item in `evidence` (ref = the item id).
- `send_back`: run a stage again with your feedback. Target the current stage to refine it, or an
  earlier stage when the problem lives in its output (for example: the experiment plan shows no
  hypothesis is testable with available data, so hypothesis generation must change). Only targets
  in the allowed list are possible. Check the iteration count: a send-back uses one iteration.
- `stop`: end the run now because a kill criterion is met or nothing left is worth the cost. A
  stop is a legitimate, honest outcome; explain it in terms of the charter.

If the engine sent a recommendation (role=recommendation), weigh it as one expert opinion.

Decision formats (the `decision` argument of `decide`):
{"action": "accept", "selection": {"<selection_key>": ["H1", "H3"]}, "note": "<one sentence>"}
  (omit `selection` or use {} when no selection_key is given)
{"action": "send_back", "target_stage_key": "<allowed target>", "note": "<what must change, concretely, and why>"}
{"action": "stop", "note": "<why, in terms of the charter>"}

For `accept` and `send_back`, write `handoff`: 3-6 short instructions for the team that works next.
They are passed to that engine verbatim. Make them concrete (which hypotheses matter most, what to
prioritise, which risks to control, what not to repeat).
