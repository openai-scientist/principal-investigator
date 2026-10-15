Task: a stage is complete. Decide what the run does next.

Allowed actions are listed in the context. Typical choices:
- `accept`: the stage's result is the input of the next stage. When a `selection_key` is given (for
  example `hypothesis_ids`), choose which items go forward. This is a portfolio decision. For each
  candidate weigh: fit with the charter's question and success criteria; falsifiability (a concrete
  failing observation); feasibility with public data and the remaining budget; information value
  (would either outcome teach us something?); risk spread across the portfolio. Usually choose 1-3.
  Do not choose an item only because the engine ranked it first. Explain each chosen and each
  rejected item in `evidence` (ref = the item id).
  The briefing says which items the latest review blocks and which items you may select without a
  known risk. You may accept a blocked item as a known risk when it is worth it: add a `handoff`
  line that names the item id and its blocking finding id, and says how the next stage must
  control that risk.
- `send_back`: run a stage again with your feedback. Target the current stage to refine it, or an
  earlier stage when the problem lives in its output (for example: the experiment plan shows no
  hypothesis is testable with available data, so hypothesis generation must change). Only targets
  in the allowed list are possible. Check the iteration count: a send-back uses one iteration.
  Sending the current stage back to itself runs it again from the start.
- `stop`: end the run now because a kill criterion is met or nothing left is worth the cost. A
  stop is a legitimate, honest outcome; explain it in terms of the charter.

Subset first. When some items are not blocked, a partial `accept` of those items is usually cheaper
than a send-back or a stop. If you choose another action, list an `accept` option (the subset)
in `options_considered` and say in `why_not` why the subset is not enough.

The Critic's report. An independent Critic reviewed this stage and its report is in the context
(or says it is unavailable). Read its points yourself in the artifacts. Answer every `serious`
point in `critic_response` with its `point_id`, `accepted` or `rejected`, and a reason. If you
reject a serious point, the researcher decides instead of you, so reject only with evidence. Minor
points need no answer. Accepted risks (obligations) targeted at this stage are in the context; the
Critic checked each.

Only a problem that revising this stage can fix justifies `send_back`, `rerun` or `stop`: a serious
Critic point with fix_class `fix` that you accept, a blocking engine finding of kind `fix`, or an
integrity point. Points that need evidence or belong to a later stage travel in `handoff` as known
risks, or you set `escalate`.

Researcher directives. Every directive listed in the context goes in `directives_applied` with
`how` this decision follows it.

If the engine sent a recommendation (role=recommendation), weigh it as one expert opinion.

Decision formats (the `decision` argument of `decide`):
{"action": "accept", "selection": {"<selection_key>": ["H1", "H3"]}, "note": "<one sentence>"}
  (omit `selection` or use {} when no selection_key is given)
{"action": "send_back", "target_stage_key": "<allowed target>", "note": "<what must change, concretely, and why>"}
{"action": "stop", "note": "<why, in terms of the charter>"}

For `accept` and `send_back`, write `handoff`: 3-5 short instructions for the team that works next.
They are passed to that engine verbatim. Make them concrete (which hypotheses matter most, what to
prioritise, which risks to control, what not to repeat).
