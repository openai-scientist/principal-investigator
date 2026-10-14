"""HTTP API of the PI service (PI contract, owned by the Platform)."""

import hmac
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from principal_investigator.contract import (
    ConsultationAccepted,
    ConsultationRequest,
    ConsultationResult,
    ConsultationState,
    Health,
    PICapabilities,
)
from principal_investigator.jobs import Jobs
from principal_investigator.settings import VERSION, Settings
from principal_investigator.store import Store

KINDS = ["charter", "gate_question", "gate_review", "stage_outcome", "failure", "close"]


def create_app(settings: Settings, *, jobs: Jobs | None = None) -> FastAPI:
    jobs = jobs or Jobs(settings, Store(settings.store_dir))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        jobs.fail_interrupted()
        yield

    app = FastAPI(title="Principal Investigator", version=VERSION, lifespan=lifespan)
    app.state.jobs = jobs

    def require_key(x_service_key: str | None = Header(default=None)) -> None:
        if not x_service_key or not hmac.compare_digest(
            x_service_key.encode(), settings.service_key.encode()
        ):
            raise HTTPException(
                status_code=401, detail={"code": "UNAUTHORIZED", "message": "bad key"}
            )

    @app.get("/health", response_model=Health)
    def health():
        return Health(status="ok", service_version=VERSION, active_consultations=jobs.active)

    @app.get("/capabilities", response_model=PICapabilities, dependencies=[Depends(require_key)])
    def capabilities():
        return PICapabilities(service_version=VERSION, kinds=KINDS)

    @app.post(
        "/consultations",
        response_model=ConsultationAccepted,
        status_code=202,
        dependencies=[Depends(require_key)],
    )
    async def consult(request: ConsultationRequest):
        created = jobs.submit(request)
        body = ConsultationAccepted(consultation_id=request.consultation_id, created=created)
        return JSONResponse(status_code=202 if created else 200, content=body.model_dump())

    @app.get(
        "/consultations/{consultation_id}",
        response_model=ConsultationState,
        dependencies=[Depends(require_key)],
    )
    def get_consultation(consultation_id: str):
        try:
            record = jobs.store.read(consultation_id)
        except ValueError:
            record = None
        if record is None:
            raise HTTPException(
                status_code=404, detail={"code": "NOT_FOUND", "message": "unknown consultation"}
            )
        result = record.get("result")
        return ConsultationState(
            consultation_id=consultation_id,
            status=record["status"],
            result=ConsultationResult.model_validate(result) if result else None,
        )

    return app
