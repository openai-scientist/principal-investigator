"""Run consultations in the background and deliver results to the Platform."""

import asyncio
import logging
from collections.abc import Callable

import httpx

from principal_investigator.contract import (
    SERVICE_KEY_HEADER,
    ConsultationRequest,
    ConsultationResult,
    FailureInfo,
)
from principal_investigator.deliberation import DeliberationError, deliberate
from principal_investigator.llm import ChatLLM, Meter, OpenAICompatibleLLM
from principal_investigator.settings import Settings
from principal_investigator.store import Store
from principal_investigator.tools import Session
from principal_investigator.workspace import Workspace

logger = logging.getLogger(__name__)

#: Builds the client of one role for one consultation. Tests replace it with fakes.
LLMFactory = Callable[..., ChatLLM]

CALLBACK_DELAYS = (0, 2, 5, 15, 30, 60)


def default_llm_factory(cfg, *, role: str, meter: Meter, timeout: float) -> ChatLLM:
    return OpenAICompatibleLLM(cfg, role=role, meter=meter, timeout=timeout)


class Jobs:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        *,
        llm_factory: LLMFactory = default_llm_factory,
        transport: httpx.AsyncBaseTransport | None = None,
        web_transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.settings = settings
        self.store = store
        self.llm_factory = llm_factory
        self.transport = transport
        #: Transport for literature and dataset lookups; tests replace it.
        self.web_transport = web_transport
        self.tasks: dict[str, asyncio.Task] = {}

    @property
    def active(self) -> int:
        return sum(not t.done() for t in self.tasks.values())

    def submit(self, req: ConsultationRequest) -> bool:
        """Start the consultation; False when it already exists (the request is idempotent)."""
        if self.store.read(req.consultation_id) is not None:
            return False
        self.store.create(req.consultation_id, req.redacted())
        self.tasks[req.consultation_id] = asyncio.create_task(self._run(req))
        return True

    async def _run(self, req: ConsultationRequest) -> None:
        meter = Meter()
        ws = Workspace(req.artifacts, self.settings.service_key, transport=self.transport)
        session = Session(req=req, ws=ws, meter=meter, transport=self.web_transport)
        try:
            timeout = self.settings.llm_timeout_seconds
            lead = self.llm_factory(req.models.lead, role="lead", meter=meter, timeout=timeout)
            critic = self.llm_factory(
                req.models.critic, role="critic", meter=meter, timeout=timeout
            )
            result = await deliberate(req, session, lead=lead, critic=critic)
        except DeliberationError as exc:
            result = self._failed(req, exc.code, str(exc), session)
        except Exception as exc:  # noqa: BLE001 - every failure must reach the Platform
            logger.exception("consultation %s failed", req.consultation_id)
            result = self._failed(req, "internal_error", type(exc).__name__, session)
        self.store.finish(req.consultation_id, result)
        await self.deliver(req.callback_url, result)

    @staticmethod
    def _failed(req, code: str, message: str, session: Session) -> ConsultationResult:
        return ConsultationResult(
            consultation_id=req.consultation_id,
            status="failed",
            failure=FailureInfo(code=code, message=message[:500]),
            usage=dict(session.meter.usage),
            trace=session.trace,
        )

    async def deliver(self, url: str, result: ConsultationResult) -> bool:
        body = result.model_dump(mode="json")
        headers = {SERVICE_KEY_HEADER: self.settings.service_key}
        async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
            for delay in CALLBACK_DELAYS:
                await asyncio.sleep(delay)
                try:
                    response = await client.post(url, json=body, headers=headers)
                except httpx.HTTPError:
                    continue
                if response.status_code < 300:
                    return True
                if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
                    logger.warning(
                        "Platform rejected result of %s: HTTP %s",
                        result.consultation_id,
                        response.status_code,
                    )
                    return False
        logger.warning("could not deliver result of %s", result.consultation_id)
        return False

    def fail_interrupted(self) -> None:
        """At startup, consultations a previous process left running end as failed, and the
        Platform is told so in the background (otherwise it would wait for them forever)."""
        for consultation_id in self.store.all_ids():
            record = self.store.read(consultation_id)
            if not record or record.get("status") != "running":
                continue
            result = ConsultationResult(
                consultation_id=consultation_id,
                status="failed",
                failure=FailureInfo(code="pi_restarted", message="The PI service restarted."),
            )
            self.store.finish(consultation_id, result)
            url = (record.get("request") or {}).get("callback_url")
            if url:
                self.tasks[consultation_id] = asyncio.create_task(self.deliver(url, result))
