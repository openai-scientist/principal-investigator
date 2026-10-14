import asyncio
import json

import httpx

from principal_investigator.api.app import create_app
from principal_investigator.jobs import Jobs
from principal_investigator.settings import Settings
from tests.helpers import (
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
    tmp_store,
)

UPHOLD = call("verdict", verdict="uphold", points=[])


def _setup(tmp_path, artifacts, lead_replies, critic_replies):
    platform = FakePlatform(artifacts)
    replies = {"lead": list(lead_replies), "critic": list(critic_replies)}

    def factory(cfg, *, role, meter, timeout):
        return ScriptedLLM(replies[role], role=role, meter=meter)

    settings = Settings(service_key=KEY, store_dir=tmp_path / "store")
    jobs = Jobs(
        settings,
        tmp_store(tmp_path),
        llm_factory=factory,
        transport=platform.transport,
        web_transport=FakeWeb().transport,
    )
    app = create_app(settings, jobs=jobs)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://pi")
    return platform, jobs, client


async def _drain(jobs):
    await asyncio.gather(*jobs.tasks.values())


async def test_consultation_runs_in_background_and_calls_back(tmp_path):
    content = blob(HYPOTHESES)
    req = request(artifacts=[ref("a1", content, schema="i2h.hypotheses/v1", role="primary")])
    accept = decision("accept", selection={"hypothesis_ids": ["H1"]})
    pi = [call("consult_critic", proposal=accept), call("decide", **accept)]
    platform, jobs, client = _setup(tmp_path, {"a1": content}, pi, [UPHOLD])
    body = json.loads(req.model_dump_json(by_alias=True))
    body["models"]["lead"]["api_key"] = body["models"]["critic"]["api_key"] = "secret-key-value"

    async with client:
        assert (await client.post("/consultations", json=body)).status_code == 401
        headers = {"X-Service-Key": KEY}
        first = await client.post("/consultations", json=body, headers=headers)
        assert first.status_code == 202 and first.json()["created"] is True
        again = await client.post("/consultations", json=body, headers=headers)
        assert again.status_code == 200 and again.json()["created"] is False
        await _drain(jobs)
        state = await client.get("/consultations/c-0001", headers=headers)

    assert state.json()["status"] == "decided"
    assert platform.results[0]["decision"]["selection"] == {"hypothesis_ids": ["H1"]}
    assert [t["tool"] for t in platform.results[0]["trace"]] == [
        "verdict",
        "consult_critic",
        "decide",
    ]
    assert set(platform.seen_keys) == {KEY}
    stored = (tmp_path / "store" / "c-0001.json").read_text(encoding="utf-8")
    assert "secret-key-value" not in stored


async def test_agent_failure_is_reported_with_its_trace(tmp_path):
    req = request()
    platform, jobs, client = _setup(tmp_path, {}, [call("list_artifacts")] * 24, [])
    async with client:
        jobs.submit(req)
        await _drain(jobs)
    result = platform.results[0]
    assert result["status"] == "failed" and result["failure"]["code"] == "step_limit"
    assert len(result["trace"]) == 24


async def test_restart_marks_running_consultations_failed(tmp_path):
    store = tmp_store(tmp_path)
    store.create("c-9", {"callback_url": "http://platform.test/cb/c-9"})
    delivered = []

    def handler(request):
        delivered.append(json.loads(request.content))
        return httpx.Response(200)

    jobs = Jobs(
        Settings(service_key=KEY, store_dir=tmp_path / "store"),
        store,
        transport=httpx.MockTransport(handler),
    )
    jobs.fail_interrupted()
    assert store.read("c-9")["result"]["failure"]["code"] == "pi_restarted"
    await _drain(jobs)
    assert delivered[0]["failure"]["code"] == "pi_restarted"


async def test_health_is_public(tmp_path):
    _, _, client = _setup(tmp_path, {}, [], [])
    async with client:
        response = await client.get("/health")
    assert response.json()["status"] == "ok"
