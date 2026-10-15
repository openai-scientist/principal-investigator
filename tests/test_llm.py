import httpx
import pytest

from principal_investigator import llm as llm_module
from principal_investigator.contract import ModelConfig
from principal_investigator.llm import RETRY_DELAYS, LLMError, Meter, OpenAICompatibleLLM

OK = {
    "choices": [{"message": {"content": "hi"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
}


def _client(statuses, sleeps, monkeypatch):
    seen = iter(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(seen)
        if isinstance(status, Exception):
            raise status
        return httpx.Response(status, json=OK if status == 200 else {})

    async def no_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(llm_module.asyncio, "sleep", no_sleep)
    cfg = ModelConfig(
        provider="openai_compatible", base_url="http://llm.test/v1", model="m", api_key="k"
    )
    return OpenAICompatibleLLM(
        cfg, role="lead", meter=Meter(), transport=httpx.MockTransport(handler)
    )


def test_the_delays_total_about_a_minute():
    assert RETRY_DELAYS == (2, 5, 15, 30)


@pytest.mark.parametrize("status", [429, 500, 503, 524])
async def test_transient_statuses_are_retried_with_backoff(status, monkeypatch):
    sleeps = []
    client = _client([status, status, status, status, 200], sleeps, monkeypatch)
    reply = await client.chat([], [])
    assert reply.content == "hi" and sleeps == [2, 5, 15, 30]


async def test_connection_errors_are_retried(monkeypatch):
    sleeps = []
    client = _client([httpx.ConnectTimeout("x"), 200], sleeps, monkeypatch)
    assert (await client.chat([], [])).content == "hi" and sleeps == [2]


async def test_it_gives_up_after_five_attempts(monkeypatch):
    sleeps = []
    client = _client([503] * 5, sleeps, monkeypatch)
    with pytest.raises(LLMError, match="after 5 attempts: HTTP 503"):
        await client.chat([], [])
    assert sleeps == [2, 5, 15, 30]


@pytest.mark.parametrize("status", [400, 401, 404])
async def test_other_client_errors_fail_at_once(status, monkeypatch):
    sleeps = []
    client = _client([status], sleeps, monkeypatch)
    with pytest.raises(LLMError, match=f"HTTP {status}"):
        await client.chat([], [])
    assert sleeps == []
