"""DAG propagation endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from delegation_lab.api.schemas import DagRequest, DagResponse, NodeResultModel
from delegation_lab.domain.dag import DelegationGraph, Node

router = APIRouter(prefix="/dag", tags=["dag"])


@router.post("/propagate", response_model=DagResponse)
def propagate(req: DagRequest) -> DagResponse:
    graph = DelegationGraph()
    for spec in req.nodes:
        graph.add(
            Node(
                name=spec.name,
                skill=spec.skill,
                catch_rate=spec.catch_rate,
                parents=tuple(spec.parents),
                aggregation=spec.aggregation,
                weights=tuple(spec.weights) if spec.weights is not None else None,
            )
        )
    results = graph.propagate(
        observation_rate=req.observation_rate,
        forget_rate=req.forget_rate,
        prior_support=req.prior_support,
        coverage=req.coverage,
    )
    return DagResponse(
        results={
            name: NodeResultModel(raw=r.raw, corrected=r.corrected, masking=r.masking)
            for name, r in results.items()
        }
    )
