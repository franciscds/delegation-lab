"""Aggregate all API route modules under one router."""

from __future__ import annotations

from fastapi import APIRouter

from delegation_lab.api.routes import allocation, capacity, dag, health, return_operator

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(return_operator.router)
api_router.include_router(allocation.router)
api_router.include_router(dag.router)
api_router.include_router(capacity.router)
