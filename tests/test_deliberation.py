import json

import pytest
from pydantic import ValidationError

from principal_investigator.deliberation import DeliberationError, briefing, deliberate
from principal_investigator.llm import Meter
from principal_investigator.tools import Session
from principal_investigator.workspace import Workspace
from tests.helpers import (
    FINDINGS_BLOCKED,
    HYPOTHESES,
    KEY,
    FakePlatform,
    FakeWeb,
    ScriptedLLM,
    blob,
    call,
    decision,
    ref,
    request,
    say,
)


async def _run(req, pi_replies, artifacts=None, web=None):
    platform, web = FakePlatform(artifacts or {}), web or FakeWeb()
    meter = Meter()
    session = Session(
        req=req,
        ws=Workspace(req.artifacts, KEY, transport=platform.transport),
        meter=meter,
        transport=web.transport,
    )
    lead = ScriptedLLM(pi_replies, role="lead", meter=meter)
    critic = ScriptedLLM([], role="critic", meter=meter)
    result = await deliberate(req, session, lead=lead, critic=critic)
    return result, lead


def _hypothesis_request(**overrides):
    content = blob(HYPOTHESES)
    req = request(
        artifacts=[ref("a1", content, schema="i2h.hypotheses/v1", role="primary")], **overrides
    )
    return req, {"a1": content}


ACCEPT_13 = decision("accept", selection={"hypothesis_ids": ["H1", "H3"]})

ON_H1 = FINDINGS_BLOCKED | {
    "findings": [
        FINDINGS_BLOCKED["findings"][0] | {"anchor": {"locator": "hypothesis:H1/statement"}}
    ]
}


def _blocked_request(findings=ON_H1, **overrides):
    content, review = blob(HYPOTHESES), blob(findings)
    req = request(
        artifacts=[
            ref("a1", content, schema="i2h.hypotheses/v1", role="primary"),
            ref("r1", review, schema="review.findings/v1", role="review"),
        ],
        **overrides,
    )
    return req, {"a1": content, "r1": review}


def _directive(entry_id="12", text="Use only public data."):
    return {
        "entry_id": entry_id,
        "kind": "directive",
        "text": text,
        "actor": "user",
        "at": "2026-10-10T00:00:00Z",
    }


async def test_pi_investigates_verifies_and_decides_without_a_critic_tool():
    req, artifacts = _hypothesis_request()
    web = FakeWeb()
    pi = [
        call("list_artifacts"),
        call("read_artifact", artifact_id="a1"),
        call("read_artifact", artifact_id="a1", section="H2"),
        call("check_dataset", query="natural questions"),
        call("search_literature", query="misleading retrieval small models"),
        call("write_note", kind="risk", text="H2 needs a reranker baseline."),
        call("consult_critic", proposal=ACCEPT_13),
        call("decide", **ACCEPT_13),
    ]
    result, lead = await _run(req, pi, artifacts, web)

    assert result.status == "decided"
    assert result.decision.selection == {"hypothesis_ids": ["H1", "H3"]}
    assert result.handoff == ["Prioritise H1"]
    assert result.critique is None
    assert set(result.usage) == {"lead"}
    seen = lead.observations()
    assert "3 hypotheses" in seen[1] and '"risk": "high"' in seen[2]
    assert "natural_questions" in seen[3] and "license=cc-by-sa-3.0" in seen[3]
    assert "ACL" in seen[4]
    assert seen[6] == "error: unknown tool 'consult_critic'"
    assert "Retrieval of misleading" not in lead.calls[0][1]["content"]  # not pre-stuffed
    assert [(t.actor, t.tool) for t in result.trace if t.actor != "pi"] == []
    assert result.trace[2].summary == "read a1 section 'H2'"
    assert any(n.kind == "risk" and "reranker" in n.text for n in result.notebook)


async def test_guard_rejections_come_back_as_observations_the_agent_fixes():
    req, artifacts = _hypothesis_request()
    bad = decision("accept", selection={"hypothesis_ids": ["H9"]})
    good = decision("accept", selection={"hypothesis_ids": ["H1"]})
    result, lead = await _run(req, [call("decide", **bad), call("decide", **good)], artifacts)
    assert result.decision.selection["hypothesis_ids"] == ["H1"]
    assert "unknown ids ['H9']" in lead.observations()[0]


async def test_a_malformed_decision_is_an_observation_not_a_crash():
    req, artifacts = _hypothesis_request()
    pi = [call("decide", decision="not json at all"), call("decide", **ACCEPT_13)]
    result, lead = await _run(req, pi, artifacts)
    assert result.status == "decided"
    assert "`decision` must be an object" in lead.observations()[0]


async def test_text_without_a_tool_call_is_nudged_back_to_tools():
    req, artifacts = _hypothesis_request()
    result, lead = await _run(req, [say("Let me think."), call("decide", **ACCEPT_13)], artifacts)
    assert result.status == "decided"
    assert "Continue by calling a tool" in lead.calls[1][-1]["content"]


async def test_step_limit_fails_the_consultation():
    req, artifacts = _hypothesis_request()
    with pytest.raises(DeliberationError) as error:
        await _run(req, [call("list_artifacts")] * 24, artifacts)
    assert error.value.code == "step_limit"


async def test_unreadable_artifact_is_an_observation_not_a_crash():
    req = request(artifacts=[ref("a1", b"other", schema="i2h.hypotheses/v1", role="primary")])
    pi = [
        call("read_artifact", artifact_id="a1"),
        call(
            "decide",
            **decision("stop", note="Inputs unreadable")
            | {"escalate": {"reason_code": "cannot_decide", "message": "Inputs unreadable."}},
        ),
    ]
    result, lead = await _run(req, pi, {"a1": blob(HYPOTHESES)})
    assert "sha256" in lead.observations()[0]
    assert result.decision.action == "stop"


async def test_accept_refuses_hypotheses_with_blocking_findings():
    req, artifacts = _blocked_request()
    without = decision("accept", selection={"hypothesis_ids": ["H3"]})
    result, lead = await _run(
        req, [call("decide", **ACCEPT_13), call("decide", **without)], artifacts
    )
    assert result.decision.selection == {"hypothesis_ids": ["H3"]}
    assert "['H1'] still have blocking review findings" in lead.observations()[0]


async def test_accept_takes_a_blocked_hypothesis_named_as_a_known_risk():
    req, artifacts = _blocked_request()
    risk = "Known risk H1 (F2 leakage): the plan must split by source before tuning."
    result, _ = await _run(req, [call("decide", **(ACCEPT_13 | {"handoff": [risk]}))], artifacts)
    assert result.decision.selection == {"hypothesis_ids": ["H1", "H3"]}
    assert result.handoff == [risk]


async def test_accept_is_refused_when_the_whole_artifact_is_blocked():
    req, artifacts = _blocked_request(FINDINGS_BLOCKED)
    stop = decision("stop", note="Whole artifact blocked.")
    result, lead = await _run(req, [call("decide", **ACCEPT_13), call("decide", **stop)], artifacts)
    assert "blocking finding on the whole artifact" in lead.observations()[0]
    assert result.decision.action == "stop"


# -- G-critic ------------------------------------------------------------------------------

CRITIQUE = {
    "verdict": "serious",
    "summary": "A success criterion was weakened.",
    "points": [
        {
            "point_id": "C1",
            "severity": "serious",
            "lens": "charter",
            "ref": "charter:success_criteria[0]",
            "point": "Hypotheses name no baseline.",
        },
        {
            "point_id": "C2",
            "severity": "minor",
            "lens": "integrity",
            "ref": "integrity:spin",
            "point": "Wording is optimistic.",
        },
    ],
    "obligations": [],
}


async def test_every_serious_critic_point_needs_a_response():
    req, artifacts = _hypothesis_request(critique=CRITIQUE)
    answered = ACCEPT_13 | {
        "critic_response": [{"point_id": "C1", "response": "rejected", "reason": "H1 names BM25."}]
    }
    result, lead = await _run(
        req, [call("decide", **ACCEPT_13), call("decide", **answered)], artifacts
    )
    assert "serious points ['C1']" in lead.observations()[0]
    (response,) = result.critic_response
    assert (response.point_id, response.response) == ("C1", "rejected")
    assert "Critic report (independent, own model): serious." in lead.calls[0][1]["content"]
    assert "C1 [serious/charter charter:success_criteria[0]]" in lead.calls[0][1]["content"]


async def test_a_response_without_a_reason_does_not_count():
    req, artifacts = _hypothesis_request(critique=CRITIQUE)
    empty = ACCEPT_13 | {
        "critic_response": [{"point_id": "C1", "response": "accepted", "reason": " "}]
    }
    full = ACCEPT_13 | {
        "critic_response": [{"point_id": "C1", "response": "accepted", "reason": "Verified."}]
    }
    result, lead = await _run(req, [call("decide", **empty), call("decide", **full)], artifacts)
    assert "serious points ['C1']" in lead.observations()[0]
    assert result.critic_response[0].reason == "Verified."


async def test_a_minor_only_report_needs_no_response():
    minor = CRITIQUE | {"verdict": "concerns", "points": CRITIQUE["points"][1:]}
    req, artifacts = _hypothesis_request(critique=minor)
    result, _ = await _run(req, [call("decide", **ACCEPT_13)], artifacts)
    assert result.status == "decided" and result.critic_response == []


async def test_briefing_says_when_the_critic_report_is_unavailable():
    req, _ = _hypothesis_request()
    assert "Critic report unavailable" in briefing(req, "(no artifacts)")


# -- G-actionable (R8) ---------------------------------------------------------------------


def _point(
    point_id, fix_class, severity="serious", lens="charter", ref="charter:success_criteria[0]"
):
    return {
        "point_id": point_id,
        "severity": severity,
        "lens": lens,
        "ref": ref,
        "point": "A problem.",
        "fix_class": fix_class,
    }


def _critique(*points):
    return CRITIQUE | {"points": list(points)}


def _answer(base, *ids, response="accepted"):
    return base | {
        "critic_response": [
            {"point_id": i, "response": response, "reason": "Checked."} for i in ids
        ]
    }


SUBSET = {"options_considered": [{"action": "accept", "summary": "All", "why_not": "Not enough"}]}
STOP = decision("stop", note="Cannot proceed.") | SUBSET
SEND_BACK = decision("send_back", target_stage_key="hypothesis", note="Name the baseline.") | SUBSET
NOT_FIXABLE = "cannot be fixed by revising this stage"


@pytest.mark.parametrize("action", [STOP, SEND_BACK])
async def test_leaving_the_stage_is_refused_when_no_open_problem_is_fixable(action):
    critique = _critique(_point("C1", "needs_evidence"), _point("C2", "later"))
    req, artifacts = _hypothesis_request(critique=critique)
    final = ACCEPT_13 | {
        "critic_response": [
            {"point_id": i, "response": "accepted", "reason": "Verified."} for i in ("C1", "C2")
        ]
    }
    result, lead = await _run(
        req, [call("decide", **_answer(action, "C1", "C2")), call("decide", **final)], artifacts
    )
    assert NOT_FIXABLE in lead.observations()[0]
    assert result.decision.action == "accept"


async def test_send_back_is_allowed_with_an_accepted_fix_point():
    req, artifacts = _hypothesis_request(critique=_critique(_point("C1", "fix")))
    result, _ = await _run(req, [call("decide", **_answer(SEND_BACK, "C1"))], artifacts)
    assert result.decision.action == "send_back"


async def test_a_rejected_fix_point_does_not_make_the_stage_fixable():
    req, artifacts = _hypothesis_request(critique=_critique(_point("C1", "fix")))
    rejected = _answer(STOP, "C1", response="rejected")
    final = _answer(ACCEPT_13, "C1", response="rejected")
    _, lead = await _run(req, [call("decide", **rejected), call("decide", **final)], artifacts)
    assert NOT_FIXABLE in lead.observations()[0]


async def test_a_serious_integrity_point_makes_the_stage_fixable():
    point = _point("C1", "later", lens="integrity", ref="integrity:spin")
    req, artifacts = _hypothesis_request(critique=_critique(point))
    result, _ = await _run(
        req, [call("decide", **_answer(STOP, "C1", response="rejected"))], artifacts
    )
    assert result.decision.action == "stop"


async def test_an_integrity_point_that_needs_evidence_does_not_justify_a_send_back():
    point = _point("C1", "needs_evidence", lens="integrity", ref="integrity:leakage")
    req, artifacts = _hypothesis_request(critique=_critique(point))
    final = _answer(ACCEPT_13, "C1")
    _, lead = await _run(
        req, [call("decide", **_answer(SEND_BACK, "C1")), call("decide", **final)], artifacts
    )
    assert NOT_FIXABLE in lead.observations()[0]


async def test_a_blocking_review_finding_of_kind_fix_makes_the_stage_fixable():
    findings = ON_H1 | {"findings": [ON_H1["findings"][0] | {"kind": "fix"}]}
    req, artifacts = _blocked_request(findings)
    argued = SEND_BACK | {
        "options_considered": [{"action": "accept", "summary": "H2", "why_not": "Too thin"}]
    }
    result, _ = await _run(req, [call("decide", **argued)], artifacts)
    assert result.decision.action == "send_back"


async def test_a_blocking_finding_of_kind_needs_evidence_does_not():
    findings = ON_H1 | {"findings": [ON_H1["findings"][0] | {"kind": "needs_evidence"}]}
    req, artifacts = _blocked_request(findings)
    argued = STOP | {
        "options_considered": [{"action": "accept", "summary": "H2", "why_not": "Too thin"}]
    }
    final = decision("accept", selection={"hypothesis_ids": ["H2", "H3"]})
    _, lead = await _run(req, [call("decide", **argued), call("decide", **final)], artifacts)
    assert NOT_FIXABLE in lead.observations()[0]


async def test_escalation_always_allows_leaving_the_stage():
    req, artifacts = _hypothesis_request(critique=_critique(_point("C1", "needs_evidence")))
    escalated = _answer(STOP, "C1") | {
        "escalate": {"reason_code": "cannot_decide", "message": "A license is unverified."}
    }
    result, _ = await _run(req, [call("decide", **escalated)], artifacts)
    assert result.decision.action == "stop" and result.escalate is not None


async def test_the_briefing_shows_each_points_fix_class():
    req, _ = _hypothesis_request(critique=_critique(_point("C1", "needs_evidence")))
    assert "C1 [serious/charter charter:success_criteria[0]] (fix_class: needs_evidence)" in (
        briefing(req, "(no artifacts)")
    )


# -- G-subset ------------------------------------------------------------------------------


async def test_choosing_another_action_while_a_subset_is_sound_needs_an_accept_option():
    req, artifacts = _blocked_request()
    send_back = decision("send_back", target_stage_key="hypothesis", note="Redo it all")
    argued = send_back | {
        "options_considered": [
            {
                "action": "accept",
                "summary": "H2 and H3 only",
                "why_not": "H2 and H3 miss the question",
            }
        ]
    }
    result, lead = await _run(
        req, [call("decide", **send_back), call("decide", **argued)], artifacts
    )
    assert "must include an `accept` option (a subset) with `why_not`" in lead.observations()[0]
    assert result.decision.action == "send_back"
    assert "you may select without a known risk: H2, H3" in lead.calls[0][1]["content"]


async def test_no_subset_argument_is_needed_when_nothing_can_be_selected():
    req, artifacts = _blocked_request(FINDINGS_BLOCKED)
    stop = decision("stop", note="Whole artifact blocked.")
    result, lead = await _run(req, [call("decide", **stop)], artifacts)
    assert result.decision.action == "stop"
    assert "No item can be selected without a known risk" in lead.calls[0][1]["content"]


async def test_nothing_is_said_about_selection_when_the_stage_has_no_item_ids():
    content, review = blob({"summary": "no items"}), blob(FINDINGS_BLOCKED)
    req = request(
        artifacts=[
            ref("a1", content, schema="other.doc/v1", role="primary"),
            ref("r1", review, schema="review.findings/v1", role="review"),
        ]
    )
    stop = decision("stop", note="Stopping.")
    result, lead = await _run(req, [call("decide", **stop)], {"a1": content, "r1": review})
    assert result.decision.action == "stop"
    assert "can be selected" not in lead.calls[0][1]["content"]


# -- G-directive ---------------------------------------------------------------------------


async def test_every_researcher_directive_must_be_cited():
    req, artifacts = _hypothesis_request(notebook=[_directive()])
    cited = ACCEPT_13 | {
        "directives_applied": [{"entry_id": "12", "how": "Both datasets are public."}]
    }
    result, lead = await _run(
        req, [call("decide", **ACCEPT_13), call("decide", **cited)], artifacts
    )
    assert "directives ['12'] are not in `directives_applied`" in lead.observations()[0]
    assert result.directives_applied[0].how == "Both datasets are public."
    assert "entry 12: Use only public data." in lead.calls[0][1]["content"]


async def test_escalation_waives_the_directive_citation():
    req, artifacts = _hypothesis_request(notebook=[_directive()])
    escalated = ACCEPT_13 | {
        "escalate": {"reason_code": "integrity", "message": "The directive asks to hide a result."}
    }
    result, _ = await _run(req, [call("decide", **escalated)], artifacts)
    assert result.escalate.reason_code == "integrity"


async def test_directives_also_bind_the_charter_and_a_pi_note_is_not_a_directive():
    note = _directive("13", "PI said so") | {"actor": "pi"}
    req = request(
        kind="charter",
        charter=None,
        allowed=[{"action": "charter"}],
        notebook=[_directive(), note],
    )
    charter = decision(
        "charter", charter={"question": "Q?", "objectives": ["O"], "success_criteria": ["S"]}
    )
    cited = charter | {"directives_applied": [{"entry_id": "12", "how": "Only public data."}]}
    result, lead = await _run(req, [call("decide", **charter), call("decide", **cited)])
    assert "directives ['12']" in lead.observations()[0]
    assert result.decision.charter.question == "Q?"


# -- G-close -------------------------------------------------------------------------------


def _obligation(obligation_id, status):
    return {
        "obligation_id": obligation_id,
        "source_stage_key": "hypothesis",
        "item_id": "H1",
        "finding_id": "F2",
        "text": "Leakage",
        "target_stage_key": "experiment_plan",
        "status": status,
    }


async def test_the_closing_report_names_every_open_or_unresolved_obligation():
    req = request(
        kind="close",
        allowed=[{"action": "report"}],
        obligations=[
            _obligation("R1", "unresolved"),
            _obligation("R2", "controlled"),
            _obligation("R3", "open"),
        ],
    )
    partial = decision("report", note="Limitations: R10 is unclear and R1 stays unresolved.")
    full = decision("report", note="Limitations: R1 is unresolved and R3 was never checked.")
    result, lead = await _run(req, [call("decide", **partial), call("decide", **full)])
    assert "missing ['R3']" in lead.observations()[0]
    assert "not as a range" in lead.observations()[0]
    assert (
        "R1 [unresolved] from hypothesis (H1, F2) for experiment_plan"
        in lead.calls[0][1]["content"]
    )
    assert result.decision.note.startswith("Limitations: R1 is unresolved")


async def test_a_closing_report_without_obligations_needs_no_limitations_list():
    req = request(kind="close", allowed=[{"action": "report"}])
    result, _ = await _run(req, [call("decide", **decision("report", note="Done."))])
    assert result.decision.action == "report"


# -- kinds ---------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["gate_question", "gate_review"])
def test_gate_kinds_are_refused(kind):
    with pytest.raises(ValidationError, match="unsupported_kind"):
        request(kind=kind)


async def test_charter_keeps_the_profile_the_researcher_chose():
    req = request(
        kind="charter", charter=None, allowed=[{"action": "charter"}], profile={"domain": "se"}
    )
    charter = decision(
        "charter",
        charter={
            "question": "Q?",
            "objectives": ["O"],
            "success_criteria": ["S"],
            "profile": {"domain": "ml", "research_type": "ablation"},
        },
    )
    result, _ = await _run(req, [call("decide", **charter)])
    assert result.decision.charter.profile.domain == "se"
    assert result.decision.charter.profile.research_type == "ablation"
    assert result.handoff == ["Prioritise H1"]


async def test_failure_kind_sees_the_failure():
    situation = request().situation.model_dump(mode="json") | {
        "failure": {"code": "engine_restarted", "message": "The engine restarted."}
    }
    req = request(
        kind="failure", situation=situation, allowed=[{"action": "rerun"}, {"action": "stop"}]
    )
    result, lead = await _run(req, [call("decide", **decision("rerun", note="Transient restart."))])
    assert result.decision.action == "rerun"
    assert "engine_restarted: The engine restarted." in lead.calls[0][1]["content"]


async def test_nested_objects_passed_as_json_strings_are_accepted():
    req, artifacts = _hypothesis_request()
    as_text = ACCEPT_13 | {"decision": json.dumps(ACCEPT_13["decision"])}
    result, _ = await _run(req, [call("decide", **as_text)], artifacts)
    assert result.decision.selection == {"hypothesis_ids": ["H1", "H3"]}


async def test_decide_lifts_arguments_nested_in_the_decision():
    req, artifacts = _hypothesis_request()
    nested = {
        "decision": ACCEPT_13["decision"] | {k: v for k, v in ACCEPT_13.items() if k != "decision"}
    }
    result, _ = await _run(req, [call("decide", **nested)], artifacts)
    assert result.decision.action == "accept"
    assert result.rationale == ACCEPT_13["rationale"]


# -- workspace and briefing ----------------------------------------------------------------


async def test_catalogue_marks_the_current_attempt():
    content = blob(HYPOTHESES)
    current = ref("a1", content, schema="i2h.hypotheses/v1", role="primary") | {
        "stage_run_id": "s2"
    }
    earlier = ref("a0", content, schema="i2h.hypotheses/v1", role="primary") | {
        "stage_run_id": "s1"
    }
    req = request(artifacts=[current, earlier])
    ws = Workspace(req.artifacts, KEY, current_stage_run="s2")
    catalog = ws.catalog()
    assert "a1: i2h.hypotheses/v1 v1, role=primary, stage=hypothesis, current attempt" in catalog
    assert "a0: i2h.hypotheses/v1 v1, role=primary, stage=hypothesis, earlier attempt" in catalog


def test_briefing_states_today():
    from datetime import date

    req, _ = _hypothesis_request()
    assert "Today is 2026-10-15 (UTC)" in briefing(req, "(no artifacts)", date(2026, 10, 15))


async def test_the_current_attempts_review_wins_over_an_earlier_higher_version():
    content = blob(HYPOTHESES)
    old, new = blob(FINDINGS_BLOCKED), blob(ON_H1)
    req = request(
        artifacts=[
            ref("a1", content, schema="i2h.hypotheses/v1", role="primary"),
            ref("r0", old, schema="review.findings/v1", role="review")
            | {"version": 4, "stage_run_id": "s1"},
            ref("r1", new, schema="review.findings/v1", role="review") | {"stage_run_id": "s2"},
        ]
    )
    ws = Workspace(
        req.artifacts,
        KEY,
        current_stage_run="s2",
        transport=FakePlatform({"a1": content, "r0": old, "r1": new}).transport,
    )
    assert await ws.blocked("hypothesis") == (False, {"H1"})
    status = await ws.review_status("hypothesis")
    assert status.startswith("Latest independent review: blocking findings on H1; without one:")
    assert "H1," not in status.split("without one:")[1]


async def test_no_fallback_to_an_earlier_attempts_review():
    content, old = blob(HYPOTHESES), blob(FINDINGS_BLOCKED)
    req = request(
        artifacts=[
            ref("a1", content, schema="i2h.hypotheses/v1", role="primary") | {"stage_run_id": "s2"},
            ref("a0", content, schema="i2h.hypotheses/v1", role="primary")
            | {"stage_run_id": "s1", "version": 3},
            ref("r0", old, schema="review.findings/v1", role="review")
            | {"version": 4, "stage_run_id": "s1"},
        ]
    )
    ws = Workspace(
        req.artifacts,
        KEY,
        current_stage_run="s2",
        transport=FakePlatform({"a1": content, "a0": content, "r0": old}).transport,
    )
    assert await ws.blocked("hypothesis") is None
    assert await ws.current_ids("hypothesis") == ["H1", "H2", "H3"]


async def test_a_risk_line_must_name_the_exact_finding_and_hypothesis():
    from principal_investigator.guards import _named

    assert _named("H1", {"F2"}, ["H1 (F2): split by source"])
    assert not _named("H1", {"F2"}, ["H1 (F21): split by source"])
    assert not _named("H1", {"F2"}, ["H10 (F2): split by source"])
    assert not _named("H1", {"F2"}, ["H1: split by source"])


async def test_briefing_survives_an_unreadable_review():
    content = blob(HYPOTHESES)
    req = request(
        artifacts=[
            ref("a1", content, schema="i2h.hypotheses/v1", role="primary"),
            ref("r1", b"{}", schema="review.findings/v1", role="review"),
        ]
    )
    ws = Workspace(req.artifacts, KEY, transport=FakePlatform({"a1": content}).transport)
    assert await ws.review_status("hypothesis") is None
