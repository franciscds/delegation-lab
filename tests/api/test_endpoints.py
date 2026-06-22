"""API tests: each endpoint is a thin adapter; assert it wires to the domain.

The values must match the same paper worked-examples the domain tests use -
proving the adapter adds no logic, only translation.
"""

import math


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_return_operator_matches_paper(client):
    # eta=10, delta=2, skill=0.8, prior=0, c=0.7 -> raw=0.667, corr=0.90, M*=1.35
    r = client.post(
        "/api/v1/return-operator",
        json={"skill": 0.8, "catch_rate": 0.7},
    )
    assert r.status_code == 200
    body = r.json()
    assert math.isclose(body["raw"], 2 / 3, abs_tol=1e-3)
    assert math.isclose(body["corrected"], 0.90, abs_tol=1e-3)
    assert math.isclose(body["masking"], 1.35, abs_tol=1e-3)


def test_allocation_water_filling_meets_target(client):
    r = client.post(
        "/api/v1/allocation/water-filling",
        json={"sigmas": [0.4, 0.55, 0.6, 0.7, 0.8], "p_min": 0.3},
    )
    assert r.status_code == 200
    assert math.isclose(r.json()["delivered"], 0.3, abs_tol=1e-3)


def test_allocation_infeasible_returns_422(client):
    # MSO infeasible -> domain raises ValueError -> handler maps to 422
    r = client.post(
        "/api/v1/allocation/water-filling",
        json={"sigmas": [0.4, 0.5, 0.6], "p_min": 0.95},
    )
    assert r.status_code == 422
    assert "infeasible" in r.json()["detail"].lower()


def test_threshold_worked_example(client):
    r = client.post(
        "/api/v1/allocation/threshold",
        json={"p_min": 0.80, "raw": 0.55, "catch_rate": 0.65},
    )
    assert r.status_code == 200
    assert math.isclose(r.json()["coverage_required"], 0.855, abs_tol=2e-3)


def test_dag_propagate_diamond(client):
    r = client.post(
        "/api/v1/dag/propagate",
        json={
            "nodes": [
                {"name": "A", "skill": 0.8, "catch_rate": 0.6},
                {"name": "B", "skill": 0.7, "catch_rate": 0.6, "parents": ["A"]},
                {"name": "C", "skill": 0.7, "catch_rate": 0.6, "parents": ["A"]},
                {"name": "D", "skill": 0.7, "catch_rate": 0.6, "parents": ["B", "C"]},
            ]
        },
    )
    assert r.status_code == 200
    results = r.json()["results"]
    assert set(results) == {"A", "B", "C", "D"}
    assert results["D"]["raw"] < results["A"]["raw"]


def test_dag_cycle_returns_422(client):
    r = client.post(
        "/api/v1/dag/propagate",
        json={
            "nodes": [
                {"name": "X", "skill": 0.7, "catch_rate": 0.6, "parents": ["Y"]},
                {"name": "Y", "skill": 0.7, "catch_rate": 0.6, "parents": ["X"]},
            ]
        },
    )
    assert r.status_code == 422
    assert "cycle" in r.json()["detail"].lower()


def test_capacity_and_feasibility(client):
    r = client.post(
        "/api/v1/capacity",
        json={"skill": 0.9, "catch_rate": 0.9, "budget": 1.0, "p_min": 0.6},
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["capacity_bits"] <= 1.0
    assert body["feasible"] is True


def test_autonomy_semi_real_workflow(client):
    # paper: C=0.86, p=0.75, lambda=0.02, H=2.3, mu=0.012 -> Beff~0.064, T~5.3
    r = client.post(
        "/api/v1/autonomy",
        json={
            "operational_ceiling": 0.86,
            "p_min": 0.75,
            "gap_coefficient": 0.02,
            "process_entropy": 2.3,
            "drift": 0.012,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert math.isclose(body["buffer"], 0.064, abs_tol=1e-3)
    assert math.isclose(body["autonomy_time"], 5.33, abs_tol=2e-2)
    assert body["feasible"] is True


def test_validation_error_returns_422(client):
    # skill out of [0,1] -> pydantic rejects before domain is touched
    r = client.post("/api/v1/return-operator", json={"skill": 1.5, "catch_rate": 0.7})
    assert r.status_code == 422
