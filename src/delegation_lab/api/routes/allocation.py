"""Allocation endpoints: water-filling and corrector threshold."""

from __future__ import annotations

from fastapi import APIRouter

from delegation_lab.api.schemas import (
    AllocationRequest,
    AllocationResponse,
    ThresholdRequest,
    ThresholdResponse,
)
from delegation_lab.domain.allocation import (
    corrector_threshold,
    governed_allocation,
    solve_water_level,
)

router = APIRouter(prefix="/allocation", tags=["allocation"])


@router.post("/water-filling", response_model=AllocationResponse)
def water_filling(req: AllocationRequest) -> AllocationResponse:
    water_level = solve_water_level(req.sigmas, req.p_min, alpha_max=req.alpha_max)
    alphas = [governed_allocation(s, water_level, alpha_max=req.alpha_max) for s in req.sigmas]
    delivered = sum(a * s for a, s in zip(alphas, req.sigmas, strict=True)) / len(req.sigmas)
    return AllocationResponse(water_level=water_level, alphas=alphas, delivered=delivered)


@router.post("/threshold", response_model=ThresholdResponse)
def threshold(req: ThresholdRequest) -> ThresholdResponse:
    coverage = corrector_threshold(req.p_min, req.raw, req.catch_rate)
    return ThresholdResponse(coverage_required=coverage, feasible=coverage <= 1.0)
