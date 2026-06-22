"""FastAPI application: the imperative shell over the pure domain core.

Run:   uvicorn delegation_lab.api.main:app --reload
Docs:  http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from delegation_lab.api.config import get_settings
from delegation_lab.api.router import api_router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.exception_handler(ValueError)
    async def domain_value_error(request: Request, exc: ValueError) -> JSONResponse:
        # Domain contracts raise ValueError (e.g. MSO infeasible). Translate the
        # domain's vocabulary into a clean HTTP 422 at the adapter boundary.
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    return app


app = create_app()
