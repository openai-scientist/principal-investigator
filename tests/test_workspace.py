from principal_investigator.contract import ConsultArtifact
from principal_investigator.workspace import PAGE, Workspace
from tests.helpers import KEY, FakePlatform, ref

PLAN = (
    "# Experiment plan\n\nIntro.\n\n## Datasets\nNatural Questions.\n\n### Splits\nDev only.\n\n"
    "## Analysis plan\nPaired bootstrap.\n" + "x" * (PAGE + 100)
)


def _workspace(content: bytes, schema="exd.exp_plan/v1"):
    r = ConsultArtifact.model_validate(ref("p1", content, schema=schema, role="primary"))
    return Workspace([r], KEY, transport=FakePlatform({"p1": content}).transport)


async def test_long_markdown_overview_lists_sections_then_pages():
    ws = _workspace(PLAN.encode())
    overview = await ws.read("p1")
    assert overview.startswith("Sections (read one with section=<title>):")
    assert "## Analysis plan" in overview and "offset=8000" in overview


async def test_markdown_section_includes_subsections_and_stops_at_a_peer():
    ws = _workspace(PLAN.encode())
    section = await ws.read("p1", "datasets")
    assert "Natural Questions" in section and "Dev only." in section
    assert "Paired bootstrap" not in section
    missing = await ws.read("p1", "Budget")
    assert missing.startswith("No section matching 'Budget'")


async def test_unknown_artifact_is_an_error_the_agent_can_read():
    ws = _workspace(b"{}")
    try:
        await ws.read("nope")
    except Exception as exc:  # noqa: BLE001
        assert "list_artifacts" in str(exc)
    else:
        raise AssertionError("expected an error")
