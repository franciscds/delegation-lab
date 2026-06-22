"""Health route."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from delegation_lab.api.config import Settings, get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment}
