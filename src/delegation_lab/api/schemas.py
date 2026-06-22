"""Pydantic request/response models: the JSON <-> domain boundary.

These models do input validation (ranges, shapes) *before* any domain function
is called. The domain still enforces its own contracts; the schemas catch
malformed requests early and produce clean 422s with field-level messages.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from delegation_lab.domain.dag import Aggregation


# ---- return operator ------------------------------------------------------
class ReturnOperatorRequest(BaseModel):
    skill: float = Field(..., ge=0.0, le=1.0)
    observation_rate: float = Field(10.0, gt=0.0)
    forget_rate: float = Field(2.0, ge=0.0)
    prior_support: float = Field(0.0, ge=0.0, le=1.0)
    catch_rate: float = Field(..., ge=0.0, le=1.0)
    coverage: float = Field(1.0, ge=0.0, le=1.0)


class ReturnOperatorResponse(BaseModel):
    raw: float
    corrected: float
    masking: float


# ---- allocation -----------------------------------------------------------
class AllocationRequest(BaseModel):
    sigmas: list[float] = Field(..., min_length=1)
    p_min: float = Field(..., ge=0.0, le=1.0)
    alpha_max: float = Field(1.0, ge=0.0, le=1.0)


class AllocationResponse(BaseModel):
    water_level: float
    alphas: list[float]
    delivered: float


class ThresholdRequest(BaseModel):
    p_min: float = Field(..., ge=0.0, le=1.0)
    raw: float = Field(..., ge=0.0, lt=1.0)
    catch_rate: float = Field(..., gt=0.0, le=1.0)


class ThresholdResponse(BaseModel):
    coverage_required: float
    feasible: bool


# ---- dag ------------------------------------------------------------------
class NodeSpec(BaseModel):
    name: str
    skill: float = Field(..., ge=0.0, le=1.0)
    catch_rate: float = Field(..., ge=0.0, le=1.0)
    parents: list[str] = Field(default_factory=list)
    aggregation: Aggregation = Aggregation.PRODUCT
    weights: list[float] | None = None


class DagRequest(BaseModel):
    nodes: list[NodeSpec] = Field(..., min_length=1)
    observation_rate: float = Field(10.0, gt=0.0)
    forget_rate: float = Field(2.0, ge=0.0)
    prior_support: float = Field(0.0, ge=0.0, le=1.0)
    coverage: float = Field(1.0, ge=0.0, le=1.0)


class NodeResultModel(BaseModel):
    raw: float
    corrected: float
    masking: float


class DagResponse(BaseModel):
    results: dict[str, NodeResultModel]


# ---- capacity & autonomy --------------------------------------------------
class CapacityRequest(BaseModel):
    skill: float = Field(..., ge=0.0, le=1.0)
    catch_rate: float = Field(..., ge=0.0, le=1.0)
    budget: float = Field(..., ge=0.0, le=1.0)
    p_min: float | None = Field(None, ge=0.0, le=1.0)


class CapacityResponse(BaseModel):
    capacity_bits: float
    feasible: bool | None = None


class AutonomyRequest(BaseModel):
    operational_ceiling: float = Field(..., ge=0.0, le=1.0)
    p_min: float = Field(..., ge=0.0, le=1.0)
    gap_coefficient: float = Field(..., gt=0.0)
    process_entropy: float = Field(0.0, ge=0.0)
    drift: float = Field(..., gt=0.0)


class AutonomyResponse(BaseModel):
    buffer: float
    autonomy_time: float
    feasible: bool
