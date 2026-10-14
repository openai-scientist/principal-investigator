import pytest

from principal_investigator.deliberation import DeliberationError, deliberate
from principal_investigator.llm import Meter
from principal_investigator.tools import Session
from principal_investigator.workspace import Workspace
from tests.helpers import (
    FINDINGS_BLOCKED,
    FINDINGS_PASSED,
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

UPHOLD = call("verdict", verdict="uphold", points=[])


async def _run(req, pi_replies, critic_replies, artifacts=None, web=None):
    platform, web = FakePlatform(artifacts or {}), web or FakeWeb()
    meter = Meter()
    session = Session(
        req=req,
        ws=Workspace(req.artifacts, KEY, transport=platform.transport),
        meter=meter,
        transport=web.transport,
    )
    lead = ScriptedLLM(pi_replies, role="lead", meter=meter)
    critic = ScriptedLLM(critic_replies, role="critic", meter=meter)
    result = await deliberate(req, session, lead=lead, critic=critic)
    return result, lead, critic


def _hypothesis_request(**overrides):
    content = blob(HYPOTHESES)
    req = request(
        artifacts=[ref("a1", content, schema="i2h.hypotheses/v1", role="primary")], **overrides
    )
    return req, {"a1": content}


ACCEPT_13 = decision("accept", selection={"hypothesis_ids": ["H1", "H3"]})


async def test_pi_investigates_verifies_consults_critic_and_decides():
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
    critic = [call("read_artifact", artifact_id="a1", section="H3"), UPHOLD]
    result, lead, crit = await _run(req, pi, critic, artifacts, web)

    assert result.status == "decided"
    assert result.decision.selection == {"hypothesis_ids": ["H1", "H3"]}
    assert result.handoff == ["Prioritise H1"]
    assert result.critique.verdict == "upheld"
    assert set(result.usage) == {"lead", "critic"}
    # The agent saw what it asked for, and only that.
    seen = lead.observations()
    assert "3 hypotheses" in seen[1] and '"risk": "high"' in seen[2]
    assert "natural_questions" in seen[3] and "license=cc-by-sa-3.0" in seen[3]
    assert "ACL" in seen[4]
    assert "Retrieval of misleading" not in lead.calls[0][1]["content"]  # not pre-stuffed
    assert any("openalex" in q for q in web.queries)
    # The trace tells the researcher what the PI and the Critic did, in order.
    assert [(t.actor, t.tool) for t in result.trace] == [
        ("pi", "list_artifacts"),
        ("pi", "read_artifact"),
        ("pi", "read_artifact"),
        ("pi", "check_dataset"),
        ("pi", "search_literature"),
        ("pi", "write_note"),
        ("critic", "read_artifact"),
        ("critic", "verdict"),
        ("pi", "consult_critic"),
        ("pi", "decide"),
    ]
    assert result.trace[2].summary == "read a1 section 'H2'"
    assert any(n.kind == "risk" and "reranker" in n.text for n in result.notebook)
    # The Critic got the proposal.
    assert '"H3"' in crit.calls[0][1]["content"]


async def test_consequential_decision_requires_the_critic_first():
    req, artifacts = _hypothesis_request()
    pi = [
        call("decide", **ACCEPT_13),
        call("consult_critic", proposal=ACCEPT_13),
        call("decide", **ACCEPT_13),
    ]
    result, lead, _ = await _run(req, pi, [UPHOLD], artifacts)
    assert result.trace[0].summary.startswith("decide failed: consult the Critic")


async def test_major_critique_must_be_answered_and_a_changed_action_reviewed_again():
    req, artifacts = _hypothesis_request()
    revise = call(
        "verdict",
        verdict="revise",
        points=[{"severity": "major", "point": "H3 lacks data; send back"}],
    )
    send_back = decision("send_back", target_stage_key="hypothesis", note="Need datasets")
    answered = send_back | {
        "critique_response": [
            {"point": "H3 lacks data", "response": "accepted", "reason": "Verified"}
        ]
    }
    pi = [
        call("consult_critic", proposal=ACCEPT_13),
        call("decide", **ACCEPT_13),  # major point not answered
        call("decide", **send_back),  # action changed without review
        call("consult_critic", proposal=send_back),
        call("decide", **answered),
    ]
    result, lead, _ = await _run(req, pi, [revise, UPHOLD], artifacts)
    errors = [o for o in lead.observations() if o.startswith("error")]
    assert "answer each in `critique_response`" in errors[0]
    assert "differs from the draft the Critic reviewed" in errors[1]
    assert result.decision.target_stage_key == "hypothesis"
    assert result.critique.verdict == "revised"


async def test_guard_rejections_come_back_as_observations_the_agent_fixes():
    req, artifacts = _hypothesis_request()
    bad = decision("accept", selection={"hypothesis_ids": ["H9"]})
    good = decision("accept", selection={"hypothesis_ids": ["H1"]})
    pi = [
        call("consult_critic", proposal=bad),
        call("decide", **bad),
        call("consult_critic", proposal=good),
        call("decide", **good),
    ]
    result, lead, _ = await _run(req, pi, [UPHOLD, UPHOLD], artifacts)
    assert result.decision.selection["hypothesis_ids"] == ["H1"]
    assert "unknown ids ['H9']" in lead.observations()[1]


async def test_changing_the_selection_after_the_critic_needs_a_new_review():
    req, artifacts = _hypothesis_request()
    reviewed = decision("accept", selection={"hypothesis_ids": ["H1"]})
    swapped = decision("accept", selection={"hypothesis_ids": ["H3"]})
    pi = [
        call("consult_critic", proposal=reviewed),
        call("decide", **swapped),
        call("decide", **reviewed),
    ]
    result, lead, _ = await _run(req, pi, [UPHOLD], artifacts)
    assert result.decision.selection["hypothesis_ids"] == ["H1"]
    assert "differs from the draft the Critic reviewed (selection)" in lead.observations()[1]


async def test_a_malformed_decision_is_an_observation_not_a_crash():
    req, artifacts = _hypothesis_request()
    pi = [
        call("consult_critic", proposal=ACCEPT_13),
        call("decide", decision="not json at all"),
        call("decide", **ACCEPT_13),
    ]
    result, lead, _ = await _run(req, pi, [UPHOLD], artifacts)
    assert result.status == "decided"
    assert "`decision` must be an object" in lead.observations()[1]


async def test_text_without_a_tool_call_is_nudged_back_to_tools():
    req, artifacts = _hypothesis_request()
    pi = [
        say("Let me think."),
        call("consult_critic", proposal=ACCEPT_13),
        call("decide", **ACCEPT_13),
    ]
    result, lead, _ = await _run(req, pi, [UPHOLD], artifacts)
    assert result.status == "decided"
    assert "Continue by calling a tool" in lead.calls[1][-1]["content"]


async def test_step_limit_fails_the_consultation():
    req, artifacts = _hypothesis_request()
    with pytest.raises(DeliberationError) as error:
        await _run(req, [call("list_artifacts")] * 24, [], artifacts)
    assert error.value.code == "step_limit"


async def test_unreadable_artifact_is_an_observation_not_a_crash():
    req = request(artifacts=[ref("a1", b"other", schema="i2h.hypotheses/v1", role="primary")])
    pi = [
        call("read_artifact", artifact_id="a1"),
        call("consult_critic", proposal=decision("stop", note="x")),
        call("decide", **decision("stop", note="Inputs unreadable")),
    ]
    result, lead, _ = await _run(req, pi, [UPHOLD], {"a1": blob(HYPOTHESES)})
    assert "sha256" in lead.observations()[0]
    assert result.decision.action == "stop"


GATE_REVIEW = {
    "gate_id": "g1",
    "kind": "artifact_review",
    "step_key": "review",
    "prompt": "Review",
    "artifact_ids": ["a1"],
}


def _review_request(findings):
    content, review = blob(HYPOTHESES), blob(findings)
    req = request(
        kind="gate_review",
        gate=GATE_REVIEW,
        allowed=[{"action": "approve"}, {"action": "comment"}],
        artifacts=[
            ref("a1", content, schema="i2h.hypotheses/v1", role="primary"),
            ref("r1", review, schema="review.findings/v1", role="review"),
        ],
    )
    return req, {"a1": content, "r1": review}


async def test_approve_while_review_blocks_becomes_an_integrity_escalation():
    req, artifacts = _review_request(FINDINGS_BLOCKED)
    approve = decision("approve")
    pi = [call("consult_critic", proposal=approve), call("decide", **approve)]
    result, _, _ = await _run(req, pi, [UPHOLD], artifacts)
    assert result.escalate.reason_code == "integrity"


ON_H1 = FINDINGS_BLOCKED | {
    "findings": [
        FINDINGS_BLOCKED["findings"][0] | {"anchor": {"locator": "hypothesis:H1/statement"}}
    ]
}


async def test_approve_is_allowed_when_blocking_findings_are_on_some_hypotheses_only():
    req, artifacts = _review_request(ON_H1)
    approve = decision("approve")
    pi = [call("consult_critic", proposal=approve), call("decide", **approve)]
    result, _, _ = await _run(req, pi, [UPHOLD], artifacts)
    assert result.decision.action == "approve" and result.escalate is None


async def test_accept_refuses_hypotheses_with_blocking_findings():
    content, review = blob(HYPOTHESES), blob(ON_H1)
    req = request(
        artifacts=[
            ref("a1", content, schema="i2h.hypotheses/v1", role="primary"),
            ref("r1", review, schema="review.findings/v1", role="review"),
        ]
    )
    with_h1 = ACCEPT_13
    without = decision("accept", selection={"hypothesis_ids": ["H3"]})
    pi = [
        call("consult_critic", proposal=with_h1),
        call("decide", **with_h1),
        call("consult_critic", proposal=without),
        call("decide", **without),
    ]
    result, lead, _ = await _run(req, pi, [UPHOLD, UPHOLD], {"a1": content, "r1": review})
    assert result.decision.selection == {"hypothesis_ids": ["H3"]}
    assert "['H1'] still have blocking review findings" in lead.observations()[1]


async def test_approve_after_passed_review_has_no_escalation_and_no_handoff():
    req, artifacts = _review_request(FINDINGS_PASSED)
    approve = decision("approve")
    pi = [
        call("read_artifact", artifact_id="r1"),
        call("consult_critic", proposal=approve),
        call("decide", **approve),
    ]
    result, lead, _ = await _run(req, pi, [UPHOLD], artifacts)
    assert "passed=True" in lead.observations()[0]
    assert result.escalate is None and result.handoff == []


async def test_comment_findings_are_anchored_and_authored_as_the_research_lead():
    req, artifacts = _review_request(FINDINGS_PASSED)
    comment = decision(
        "comment",
        findings=[
            {
                "locator": "hypothesis:H2/statement",
                "artifact_id": "not-known",
                "severity": "major",
                "category": "baseline",
                "issue": "No baseline",
                "suggestion": "Name BM25",
            }
        ],
    )
    pi = [call("consult_critic", proposal=comment), call("decide", **comment)]
    result, _, _ = await _run(req, pi, [UPHOLD], artifacts)
    finding = result.decision.findings[0]
    assert finding.author == "researcher"
    assert finding.anchor.artifact_id == "a1"


async def test_small_gate_needs_no_critic_but_a_valid_option():
    gate = {
        "gate_id": "g2",
        "kind": "question",
        "prompt": "Which?",
        "default_option_id": "o1",
        "options": [{"option_id": "o1", "label": "A"}, {"option_id": "o2", "label": "B"}],
    }
    req = request(
        kind="gate_question", gate=gate, allowed=[{"action": "answer", "option_ids": ["o1", "o2"]}]
    )
    pi = [
        call("decide", **decision("answer", option_id="o3")),
        call("decide", **decision("answer", option_id="o2")),
    ]
    result, lead, _ = await _run(req, pi, [])
    assert result.decision.option_id == "o2"
    assert result.critique is None
    assert "option o1: A (engine default)" in lead.calls[0][1]["content"]


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
    pi = [call("consult_critic", proposal=charter), call("decide", **charter)]
    result, _, _ = await _run(req, pi, [UPHOLD])
    assert result.decision.charter.profile.domain == "se"
    assert result.decision.charter.profile.research_type == "ablation"
    assert result.handoff == ["Prioritise H1"]


async def test_failure_kind_sees_the_failure_and_needs_no_critic():
    situation = request().situation.model_dump(mode="json") | {
        "failure": {"code": "engine_restarted", "message": "The engine restarted."}
    }
    req = request(
        kind="failure", situation=situation, allowed=[{"action": "rerun"}, {"action": "stop"}]
    )
    pi = [call("decide", **decision("rerun", note="Transient restart."))]
    result, lead, _ = await _run(req, pi, [])
    assert result.decision.action == "rerun"
    assert "engine_restarted: The engine restarted." in lead.calls[0][1]["content"]


async def test_nested_objects_passed_as_json_strings_are_accepted():
    import json

    req, artifacts = _hypothesis_request()
    as_text = ACCEPT_13 | {"decision": json.dumps(ACCEPT_13["decision"])}
    pi = [call("consult_critic", proposal=json.dumps(ACCEPT_13)), call("decide", **as_text)]
    result, _, _ = await _run(req, pi, [UPHOLD], artifacts)
    assert result.decision.selection == {"hypothesis_ids": ["H1", "H3"]}
