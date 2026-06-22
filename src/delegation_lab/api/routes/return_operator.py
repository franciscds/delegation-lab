"""Return Operator endpoint. Thin adapter: validate -> call domain -> respond."""

from __future__ import annotations

from fastapi import APIRouter

from delegation_lab.api.schemas import ReturnOperatorRequest, ReturnOperatorResponse
from delegation_lab.domain.return_operator import (
    corrected_quality,
    masking_index,
    raw_fixed_point,
)

router = APIRouter(prefix="/return-operator", tags=["return-operator"])


@router.post("", response_model=ReturnOperatorResponse)
def compute(req: ReturnOperatorRequest) -> ReturnOperatorResponse:
    raw = raw_fixed_point(
        req.skill,
        observation_rate=req.observation_rate,
        forget_rate=req.forget_rate,
        prior_support=req.prior_support,
    )
    corrected = corrected_quality(raw, catch_rate=req.catch_rate, coverage=req.coverage)
    return ReturnOperatorResponse(
        raw=raw, corrected=corrected, masking=masking_index(raw, corrected)
    )
