import copy

import pytest

from principal_investigator.deliberation import DeliberationError, deliberate
from principal_investigator.llm import Meter
from principal_investigator.tools import Session
from principal_investigator.workspace import Workspace
from tests.helpers import (
    HYPOTHESES,
    KEY,
    FakePlatform,
    FakeWeb,
    ScriptedLLM,
    blob,
    call,
    ref,
    request,
)

OBLIGATIONS = [
    {
        "obligation_id": "R1",
        "source_stage_key": "hypothesis",
        "item_id": "H1",
        "finding_id": "F4-1",
        "text": "H1 may leak test data",
        "target_stage_key": "hypothesis",
    },
    {
        "obligation_id": "R2",
        "source_stage_key": "hypothesis",
        "item_id": "H2",
        "text": "H2 has no baseline",
        "target_stage_key": "hypothesis",
    },
]
DIRECTIVE = {
    "entry_id": "12",
    "kind": "directive",
    "text": "Use only public data.",
    "actor": "user",
    "at": "2026-10-10T00:00:00Z",
}
PI_DECISION = {
    "entry_id": "9",
    "kind": "decision",
    "text": "Accepted H1.",
    "actor": "pi",
    "at": "2026-10-10T00:00:00Z",
}
CHECKS = [
    {"obligation_id": "R1", "status": "controlled", "locator": "hypothesis:H1/falsification"},
    {"obligation_id": "R2", "status": "unresolved", "note": "No baseline named."},
]
SERIOUS = {
    "severity": "serious",
    "lens": "charter",
    "ref": "charter:success_criteria[0]",
    "locator": "hypotheses/H2",
    "point": "H2 names no baseline.",
    "suggestion": "Name BM25.",
}
MINOR = {
    "severity": "minor",
    "lens": "integrity",
    "ref": "integrity:spin",
    "point": "The summary is optimistic.",
}


def report(verdict="serious", points=None, obligations=None, summary="One serious gap."):
    return {
        "verdict": verdict,
        "summary": summary,
        "points": [SERIOUS] if points is None else points,
        "obligations": copy.deepcopy(CHECKS) if obligations is None else obligations,
    }


def _stage_review(**overrides):
    content = blob(HYPOTHESES)
    base = {
        "kind": "stage_review",
        "authority": "advise",
        "allowed": [],
        "artifacts": [ref("a1", content, schema="i2h.hypotheses/v1", role="primary")],
        "obligations": OBLIGATIONS,
        "notebook": [DIRECTIVE, PI_DECISION],
    }
    return request(**(base | overrides)), {"a1": content}


async def _review(req, replies, artifacts, *, pi_replies=()):
    platform, web = FakePlatform(artifacts), FakeWeb()
    meter = Meter()
    session = Session(
        req=req,
        ws=Workspace(req.artifacts, KEY, transport=platform.transport),
        meter=meter,
        transport=web.transport,
    )
    lead = ScriptedLLM(list(pi_replies), role="lead", meter=meter)
    critic = ScriptedLLM(replies, role="critic", meter=meter)
    result = await deliberate(req, session, lead=lead, critic=critic)
    return result, critic, lead


async def test_stage_review_returns_a_guarded_report_from_the_critic_model():
    req, artifacts = _stage_review()
    replies = [
        call("list_artifacts"),
        call("read_artifact", artifact_id="a1", section="H2"),
        call("read_notebook", kind="directive"),
        call("report", **report()),
    ]
    result, critic, lead = await _review(req, replies, artifacts)

    assert result.status == "decided" and lead.calls == []
    assert result.decision.action == "report" and result.decision.note == "One serious gap."
    assert result.rationale == "One serious gap."
    r = result.report
    assert r.verdict == "serious" and r.model == "openai_compatible/c"
    assert r.model_shared_with is None
    assert [p.point_id for p in r.points] == ["C1"]
    assert [(c.obligation_id, c.status) for c in r.obligations] == [
        ("R1", "controlled"),
        ("R2", "unresolved"),
    ]
    assert "1 serious and 0 minor" in result.brief
    assert set(result.usage) == {"critic"}
    assert [(t.actor, t.tool) for t in result.trace] == [
        ("critic", "list_artifacts"),
        ("critic", "read_artifact"),
        ("critic", "read_notebook"),
        ("critic", "report"),
    ]
    assert "entry 12" in critic.observations()[2]
    start = critic.calls[0][1]["content"]
    assert "R1 [open] from hypothesis (H1, F4-1) for hypothesis: H1 may leak test data" in start
    assert "charter:success_criteria[0]" in start and "entry 12: Use only public data." in start
    assert "use `directive:<entry_id>` as a ref" in start and "directives_applied" not in start
    assert "Your authority" not in start and "Allowed actions" not in start


async def test_the_report_carries_fix_class_per_point():
    req, artifacts = _stage_review()
    points = [SERIOUS | {"fix_class": "needs_evidence"}, MINOR]
    result, _, _ = await _review(req, [call("report", **report("serious", points))], artifacts)
    assert [p.fix_class for p in result.report.points] == ["needs_evidence", "fix"]


async def test_an_unknown_fix_class_returns_to_the_critic_as_a_tool_error():
    req, artifacts = _stage_review()
    bad = report("serious", [SERIOUS | {"fix_class": "never"}])
    replies = [call("report", **bad), call("report", **report())]
    _, critic, _ = await _review(req, replies, artifacts)
    assert "fix_class" in critic.observations()[0]


async def test_the_critic_has_read_tools_only():
    req, artifacts = _stage_review(obligations=[])
    replies = [
        call("search_literature", query="x"),
        call("write_note", kind="risk", text="x"),
        call("report", **report("sound", points=[], obligations=[], summary="Nothing to add.")),
    ]
    result, critic, _ = await _review(req, replies, artifacts)
    assert critic.observations()[:2] == [
        "error: unknown tool 'search_literature'",
        "error: unknown tool 'write_note'",
    ]
    assert result.report.verdict == "sound" and result.report.points == []


async def test_a_critic_sharing_the_lead_model_is_marked():
    model = request().models.lead.model_dump()
    req, artifacts = _stage_review(models={"lead": model, "critic": model})
    replies = [call("report", **report())]
    result, _, _ = await _review(req, replies, artifacts)
    assert result.report.model_shared_with == "lead"


async def test_minor_points_give_concerns():
    req, artifacts = _stage_review()
    replies = [call("report", **report("concerns", points=[MINOR]))]
    result, _, _ = await _review(req, replies, artifacts)
    assert result.report.verdict == "concerns"


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (report(points=[SERIOUS | {"ref": "charter:success_criteria[7]"}]), "has no such line"),
        (report(points=[SERIOUS | {"ref": "charter:question"}]), "use charter:<field>[i]"),
        (report(points=[SERIOUS | {"ref": "charter:nonsense[0]"}]), "use charter:<field>[i]"),
        (
            report(points=[SERIOUS | {"lens": "risk", "ref": "risk:R9"}]),
            "not an obligation of this request",
        ),
        (
            report(points=[SERIOUS | {"lens": "directive", "ref": "directive:9"}]),
            "not a researcher directive",
        ),
        (
            report(points=[SERIOUS | {"lens": "integrity", "ref": "integrity:vibes"}]),
            "use integrity:<tag>",
        ),
        (
            report(points=[SERIOUS | {"lens": "integrity", "ref": "charter:objectives[0]"}]),
            "use integrity:<tag>",
        ),
        (report(points=[SERIOUS | {"lens": "science"}]), "point 1 is invalid"),
        (report(points=[SERIOUS | {"point": " "}]), "point 1 has no text"),
        (report(points=[SERIOUS] * 9), "at most 8 points"),
        (report(summary=" "), "`summary` is required"),
        (report(obligations=CHECKS[:1]), "exactly one check for each of ['R1', 'R2']"),
        (report(obligations=CHECKS + CHECKS[:1]), "exactly one check"),
        (
            report(obligations=[CHECKS[0] | {"obligation_id": "R9"}, CHECKS[1]]),
            "exactly one check",
        ),
        (
            report(obligations=[{"obligation_id": "R1", "status": "controlled"}, CHECKS[1]]),
            "R1: controlled needs a locator",
        ),
        (
            report(obligations=[CHECKS[0], {"obligation_id": "R2", "status": "unresolved"}]),
            "R2: unresolved needs a note",
        ),
        (report(obligations=[{"obligation_id": "R1", "status": "maybe"}, CHECKS[1]]), "status"),
        (report("sound"), "verdict must be 'serious'"),
        (report("serious", points=[MINOR]), "verdict must be 'concerns'"),
        (report("concerns", points=[]), "verdict must be 'sound'"),
    ],
)
async def test_report_guards_return_to_the_critic_as_tool_errors(bad, message):
    req, artifacts = _stage_review()
    result, critic, _ = await _review(
        req, [call("report", **bad), call("report", **report())], artifacts
    )
    first = critic.observations()[0]
    assert first.startswith("error: report rejected") and message in first
    assert result.report.verdict == "serious"


@pytest.mark.parametrize(
    ("locator", "ok"),
    [
        ("hypothesis:H1/falsification", True),
        ("hypothesis:H9", False),
        ("section:hypotheses", True),
        ("section:No Such Heading", False),
        ("section:a", False),  # a substring of the text, not a heading
        ("document:a1", True),
        ("a1", True),
        ("document:zzz", False),
        ("hypotheses/H1", False),
    ],
)
async def test_a_controlled_locator_must_resolve_in_the_artifact(locator, ok):
    req, artifacts = _stage_review()
    checks = [CHECKS[0] | {"locator": locator}, CHECKS[1]]
    replies = [call("report", **report(obligations=checks)), call("report", **report())]
    result, critic, _ = await _review(req, replies, artifacts)
    if ok:
        assert result.report.obligations[0].locator == locator
        assert critic.observations() == []
    else:
        assert "does not resolve" in critic.observations()[0]
        assert "hypothesis:<id>" in critic.observations()[0]


async def test_a_directive_ref_must_be_a_user_directive():
    req, artifacts = _stage_review()
    point = SERIOUS | {"lens": "directive", "ref": "directive:12"}
    result, _, _ = await _review(req, [call("report", **report(points=[point]))], artifacts)
    assert result.report.points[0].ref == "directive:12"


async def test_a_researcher_note_on_an_override_is_a_directive_the_critic_can_cite():
    # Live: the note "Make H1's prediction a number" was followed by a reversed H1 nobody caught.
    note = DIRECTIVE | {
        "entry_id": "30",
        "kind": "override",
        "text": "The researcher overrode the PI at stage_review: sent the stage back to "
        "hypothesis. Researcher's note: Make H1's prediction a number.",
    }
    plain = DIRECTIVE | {"entry_id": "31", "kind": "override", "text": "Overrode the PI."}
    req, artifacts = _stage_review(notebook=[note, plain])
    point = SERIOUS | {"lens": "directive", "ref": "directive:30"}
    bad = SERIOUS | {"lens": "directive", "ref": "directive:31"}
    replies = [call("report", **report(points=[bad])), call("report", **report(points=[point]))]
    result, critic, _ = await _review(req, replies, artifacts)
    assert "not a researcher directive" in critic.observations()[0]
    assert result.report.points[0].ref == "directive:30"
    assert "entry 30: The researcher overrode" in critic.calls[0][1]["content"]


async def test_the_critic_step_limit_fails_the_consultation():
    req, artifacts = _stage_review()
    with pytest.raises(DeliberationError) as error:
        await _review(req, [call("list_artifacts")] * 12, artifacts)
    assert error.value.code == "step_limit"


# -- R3: convergence in a repeat review ----------------------------------------------------

PRIOR = {
    "verdict": "serious",
    "summary": "Two serious gaps.",
    "points": [
        {
            "point_id": "C1",
            "severity": "serious",
            "lens": "charter",
            "ref": "charter:success_criteria[0]",
            "locator": "hypotheses/H2",
            "point": "H2 names no baseline.",
        },
        {
            "point_id": "C2",
            "severity": "serious",
            "lens": "integrity",
            "ref": "integrity:spin",
            "point": "The summary is optimistic.",
        },
        {
            "point_id": "C3",
            "severity": "minor",
            "lens": "integrity",
            "ref": "integrity:leakage",
            "point": "Small leak.",
        },
    ],
    "obligations": [],
}
RESOLVED = {"point_id": "C1", "status": "resolved", "reason": "H2 now names BM25."}
MAINTAINED = {"point_id": "C2", "status": "maintained", "reason": "Still optimistic."}
BOTH_RESOLVED = [RESOLVED, MAINTAINED | {"status": "resolved"}]


def _repeat():
    return _stage_review(prior_critique=PRIOR, obligations=[])


def _repeat_report(verdict, points, resolutions):
    return report(verdict, points=points, obligations=[]) | {"resolutions": resolutions}


async def test_a_repeat_review_briefing_lists_the_earlier_serious_points():
    req, artifacts = _repeat()
    replies = [call("report", **_repeat_report("sound", [], BOTH_RESOLVED))]
    _, critic, _ = await _review(req, replies, artifacts)
    start = critic.calls[0][1]["content"]
    assert "repeat review" in start and "- C1 [charter charter:success_criteria[0]]" in start
    assert "- C2 [integrity integrity:spin]" in start and "Small leak" not in start
    assert "only for an integrity or directive problem" in start


async def test_maintained_points_stay_serious_and_resolved_ones_close():
    req, artifacts = _repeat()
    replies = [call("report", **_repeat_report("serious", [], [RESOLVED, MAINTAINED]))]
    result, _, _ = await _review(req, replies, artifacts)
    r = result.report
    assert r.verdict == "serious"
    assert [(p.point_id, p.severity, p.lens, p.ref) for p in r.points] == [
        ("C1", "serious", "integrity", "integrity:spin")
    ]
    assert r.points[0].point == "[maintains C2] Still optimistic."
    assert [(x.point_id, x.status) for x in r.resolutions] == [
        ("C1", "resolved"),
        ("C2", "maintained"),
    ]


async def test_a_maintained_point_is_not_added_when_the_critic_listed_it():
    req, artifacts = _repeat()
    listed = MINOR | {"severity": "serious", "point": "Still optimistic, restated."}
    replies = [call("report", **_repeat_report("serious", [listed], [RESOLVED, MAINTAINED]))]
    result, _, _ = await _review(req, replies, artifacts)
    assert [p.point for p in result.report.points] == ["Still optimistic, restated."]
    assert result.report.points[0].severity == "serious"


async def test_a_maintained_point_relisted_as_minor_stays_serious():
    req, artifacts = _repeat()
    replies = [
        call("report", **_repeat_report("concerns", [MINOR], [RESOLVED, MAINTAINED])),
        call("report", **_repeat_report("serious", [MINOR], [RESOLVED, MAINTAINED])),
    ]
    result, critic, _ = await _review(req, replies, artifacts)
    assert "maintained resolution" in critic.observations()[0]
    assert result.report.verdict == "serious"
    assert [(p.severity, p.ref) for p in result.report.points] == [
        ("minor", "integrity:spin"),
        ("serious", "integrity:spin"),
    ]


async def test_a_new_charter_point_in_a_repeat_review_is_minor_but_integrity_stays():
    req, artifacts = _repeat()
    new_charter = SERIOUS | {"ref": "charter:success_criteria[0]", "point": "Second gap."}
    new_integrity = MINOR | {
        "severity": "serious",
        "ref": "integrity:fabrication",
        "point": "Invented.",
    }
    points = [new_charter, new_integrity]
    replies = [call("report", **_repeat_report("serious", points, BOTH_RESOLVED))]
    result, _, _ = await _review(req, replies, artifacts)
    first, second = result.report.points
    assert (first.severity, first.point) == ("minor", "[new in repeat review] Second gap.")
    assert second.severity == "serious" and result.report.verdict == "serious"


@pytest.mark.parametrize("verdict", ["serious", "concerns"])
async def test_a_new_risk_point_is_downgraded_and_the_verdict_is_recomputed(verdict):
    obligations = [OBLIGATIONS[0]]
    req, artifacts = _stage_review(prior_critique=PRIOR, obligations=obligations)
    new_risk = SERIOUS | {"lens": "risk", "ref": "risk:R1", "point": "R1 not handled."}
    bad = _repeat_report(verdict, [new_risk], BOTH_RESOLVED)
    bad["obligations"] = [CHECKS[0]]
    result, _, _ = await _review(req, [call("report", **bad)], artifacts)
    assert result.report.verdict == "concerns"
    assert result.report.points[0].severity == "minor"


@pytest.mark.parametrize(
    ("resolutions", "message"),
    [
        ([], "exactly one entry for each earlier serious point"),
        ([RESOLVED], "exactly one entry"),
        ([RESOLVED, RESOLVED, MAINTAINED], "exactly one entry"),
        ([RESOLVED, MAINTAINED | {"point_id": "C3"}], "exactly one entry"),
        ([RESOLVED, MAINTAINED | {"status": "done"}], "each resolution needs"),
        ([RESOLVED, MAINTAINED | {"reason": " "}], "needs a reason"),
    ],
)
async def test_a_repeat_review_needs_one_resolution_per_earlier_serious_point(resolutions, message):
    req, artifacts = _repeat()
    good = _repeat_report("serious", [], [RESOLVED, MAINTAINED])
    replies = [call("report", **_repeat_report("sound", [], resolutions)), call("report", **good)]
    result, critic, _ = await _review(req, replies, artifacts)
    assert message in critic.observations()[0]
    assert result.report.verdict == "serious"


async def test_resolutions_are_rejected_when_there_is_no_earlier_report():
    req, artifacts = _stage_review(obligations=[])
    bad = report("sound", points=[], obligations=[]) | {"resolutions": [RESOLVED]}
    replies = [call("report", **bad), call("report", **report("sound", points=[], obligations=[]))]
    result, critic, _ = await _review(req, replies, artifacts)
    assert "exactly one entry" in critic.observations()[0]
    assert result.report.resolutions == []
