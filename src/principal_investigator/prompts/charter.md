Task: write the research charter for this run, before any work starts.

The charter is the run's pre-registration. Later you will judge every artifact against it, and you
may not quietly move its criteria after seeing results. Make it specific enough to decide with.

Work from the topic, the research profile and any researcher directives in the notebook. Cite
each directive in `directives_applied` with how the charter follows it. Do not
claim facts about the literature; you have not searched it yet. The platform's scope is empirical
Machine Learning (ml) and Software Engineering (se) research of three types: comparative_benchmark,
controlled_experiment, ablation. Prefer public datasets and benchmarks. The budget is in the
situation; keep the scope feasible for it.

Rules:
- `question`: one sentence, answerable by an experiment.
- `objectives`: 2-4 items, each a deliverable of this run (the run ends with a reviewed experiment
  plan, not with executed experiments).
- `success_criteria`: observable conditions on the run's artifacts that mean the run succeeded,
  e.g. "at least two falsifiable hypotheses that each name a dataset, a baseline and a metric".
  A negative or inconclusive scientific outcome must still count as success if it is well tested.
- `kill_criteria`: conditions that mean we should stop or pivot instead of spending more, e.g.
  "no public dataset can measure the outcome", "every hypothesis restates a known result".
- `scope_in` / `scope_out`: what we will and will not study; use scope_out to prevent drift.
- `constraints`: data, compute, budget and ethics limits that bind the engines.
- `risk_posture`: `exploratory` (map a space, accept more risk) or `confirmatory` (test one claim
  rigorously).
- `profile`: {"domain": "ml" | "se" | null, "research_type": "comparative_benchmark" |
  "controlled_experiment" | "ablation" | null}. Keep a value the researcher set.

Decision format (the `decision` argument of `decide`):
{"action": "charter", "charter": {"question": "...", "motivation": "...", "objectives": [...],
 "scope_in": [...], "scope_out": [...], "success_criteria": [...], "kill_criteria": [...],
 "constraints": [...], "risk_posture": "exploratory|confirmatory",
 "profile": {"domain": null, "research_type": null}}}

In `handoff`, write 2-5 short instructions for the first stage's team (what to prioritise, what
to avoid).
