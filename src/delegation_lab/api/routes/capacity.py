"""Capacity and autonomy endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from delegation_lab.api.schemas import (
    AutonomyRequest,
    AutonomyResponse,
    CapacityRequest,
    CapacityResponse,
)
from delegation_lab.domain.capacity import (
    autonomy_buffer,
    autonomy_time,
    delegation_capacity,
    quality_target_feasible,
)

router = APIRouter(tags=["capacity"])


@router.post("/capacity", response_model=CapacityResponse)
def capacity(req: CapacityRequest) -> CapacityResponse:
    cap = delegation_capacity(req.skill, req.catch_rate, req.budget)
    feasible = quality_target_feasible(req.p_min, cap) if req.p_min is not None else None
    return CapacityResponse(capacity_bits=cap, feasible=feasible)


@router.post("/autonomy", response_model=AutonomyResponse)
def autonomy(req: AutonomyRequest) -> AutonomyResponse:
    buffer = autonomy_buffer(
        req.operational_ceiling, req.p_min, req.gap_coefficient, req.process_entropy
    )
    t = autonomy_time(
        req.operational_ceiling,
        req.p_min,
        req.gap_coefficient,
        req.process_entropy,
        req.drift,
    )
    return AutonomyResponse(buffer=buffer, autonomy_time=t, feasible=buffer > 0.0)
