"""The literature and dataset lookups: what they send, and how they survive a flaky public API."""

import httpx
import pytest

from principal_investigator import tools
from principal_investigator.agent import ToolError
from principal_investigator.llm import Meter
from principal_investigator.tools import Session, read_tools


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    monkeypatch.setattr(tools, "RETRY_DELAY", 0)


def _tool(name: str, handler, **session) -> object:
    s = Session(req=None, ws=None, meter=Meter(), transport=httpx.MockTransport(handler), **session)
    return next(t for t in read_tools(s) if t.name == name).run


async def test_a_dataset_query_drops_the_words_that_make_the_hub_match_nothing():
    seen = []

    def handler(request):
        seen.append(request.url.params)
        return httpx.Response(200, json=[{"id": "uoft-cs/cifar10", "tags": ["license:unknown"]}])

    out = await _tool("check_dataset", handler)({"query": "CIFAR-10 Hugging Face dataset"})
    assert seen[0]["search"] == "CIFAR-10"
    assert (seen[0]["sort"], seen[0]["direction"]) == ("downloads", "-1")
    assert "uoft-cs/cifar10" in out and "license=unknown" in out and "mirror" in out


async def test_no_hub_match_is_not_reported_as_an_unavailable_dataset():
    out = await _tool("check_dataset", lambda r: httpx.Response(200, json=[]))({"query": "x"})
    assert "does not show that the dataset is unavailable" in out


async def test_a_query_of_only_noise_words_is_refused():
    with pytest.raises(ToolError, match="name"):
        await _tool("check_dataset", lambda r: httpx.Response(200, json=[]))(
            {"query": "HF dataset"}
        )


async def test_literature_search_sends_the_openalex_key_when_set():
    seen = []

    def handler(request):
        seen.append(request.url.params)
        return httpx.Response(200, json={"results": []})

    await _tool("search_literature", handler, openalex_api_key="k-123")(
        {"query": "label smoothing"}
    )
    await _tool("search_literature", handler)({"query": "label smoothing"})
    assert seen[0]["api_key"] == "k-123"
    assert "api_key" not in seen[1]


@pytest.mark.parametrize("first", ["network", 429, 503])
async def test_one_rate_limit_or_transient_failure_is_retried(first):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            if first == "network":
                raise httpx.ConnectError("dropped", request=request)
            return httpx.Response(first)
        return httpx.Response(200, json={"results": []})

    out = await _tool("search_literature", handler)({"query": "q"})
    assert len(calls) == 2 and "No works found" in out


async def test_a_lasting_failure_is_reported_after_one_retry():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429)

    with pytest.raises(ToolError, match="429"):
        await _tool("search_literature", handler)({"query": "q"})
    assert len(calls) == 2


async def test_an_exact_hub_id_is_searched_as_given():
    seen = []

    def handler(request):
        seen.append(request.url.params["search"])
        return httpx.Response(200, json=[])

    await _tool("check_dataset", handler)({"query": "hf-internal-testing/fixtures-datasets"})
    assert seen == ["hf-internal-testing/fixtures-datasets"]
