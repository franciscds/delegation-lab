"""Module 2 tests: water-filling, Fisher metric, corrector threshold.

Reproduces closed-form facts from Azevedo (2026):
  - Fisher metric g(0.5) = 4 (eq. 3).
  - Allocation density peaks exactly at sigma = 0.75 (Euler-Lagrange, eq. 8).
  - Corrector threshold for p_min=0.80, raw=0.55, c=0.65 is ~0.855.
  - Delivery constraint is met after solving for the water level.
  - Infeasible targets raise (MSO "no solution").
"""

import math

import pytest

from delegation_lab.domain.allocation import (
    allocate,
    allocation_density,
    corrector_threshold,
    fisher_metric,
    governed_allocation,
    solve_water_level,
)

TOL = 1e-3


def test_fisher_metric_at_half():
    # g(0.5) = 1 / (0.5 * 0.5) = 4
    assert math.isclose(fisher_metric(0.5), 4.0, abs_tol=1e-12)


def test_fisher_metric_symmetry_and_domain():
    assert math.isclose(fisher_metric(0.2), fisher_metric(0.8), abs_tol=1e-12)
    with pytest.raises(ValueError):
        fisher_metric(0.0)
    with pytest.raises(ValueError):
        fisher_metric(1.0)


def test_allocation_density_peaks_at_three_quarters():
    # Scan the unit interval; argmax must land on 0.75 (the Euler-Lagrange peak).
    grid = [i / 1000 for i in range(1, 1000)]
    best = max(grid, key=allocation_density)
    assert math.isclose(best, 0.75, abs_tol=2e-3)
    # density vanishes at the extremes
    assert allocation_density(0.0) == 0.0
    assert allocation_density(1.0) == 0.0


def test_governed_allocation_respects_cap():
    # A huge water level must be clipped to alpha_max.
    assert governed_allocation(0.75, water_level=1e6, alpha_max=0.4) == 0.4


def test_delivery_constraint_is_satisfied():
    sigmas = [0.4, 0.55, 0.6, 0.7, 0.8]
    p_min = 0.30
    lam = solve_water_level(sigmas, p_min)
    delivered = sum(governed_allocation(s, lam) * s for s in sigmas) / len(sigmas)
    assert math.isclose(delivered, p_min, abs_tol=TOL)


def test_allocate_returns_per_point_vector_within_bounds():
    sigmas = [0.4, 0.55, 0.6, 0.7, 0.8]
    alphas = allocate(sigmas, 0.30, alpha_max=0.9)
    assert len(alphas) == len(sigmas)
    assert all(0.0 <= a <= 0.9 for a in alphas)


def test_infeasible_target_raises():
    # Average sigma ~0.5, but ask for p_min = 0.95 -> impossible even at alpha=1.
    with pytest.raises(ValueError):
        solve_water_level([0.4, 0.5, 0.6], 0.95)


def test_corrector_threshold_worked_example():
    # paper: (0.80 - 0.55) / ((1 - 0.55) * 0.65) = 0.25 / 0.2925 = 0.855
    kn = corrector_threshold(p_min=0.80, raw=0.55, catch_rate=0.65)
    assert math.isclose(kn, 0.855, abs_tol=2e-3)


def test_corrector_threshold_zero_when_raw_meets_target():
    # If raw already >= p_min, no correction is needed.
    assert corrector_threshold(p_min=0.50, raw=0.60, catch_rate=0.65) == 0.0
