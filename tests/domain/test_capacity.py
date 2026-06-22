"""Module 4 tests: capacity, entropy, autonomy buffer/time.

Reproduces closed-form facts and worked examples from Azevedo (2026):
  - H_b(0.5) = 1 bit; H_b(0) = H_b(1) = 0.
  - Uniform routing over k options has entropy log2(k).
  - Delegation capacity is monotone non-decreasing in the review budget.
  - Process-complexity worked example: C=0.80, p=0.50, lambda=0.02 -> H_max=15.
  - Semi-real workflow: B_eff ~ 0.064 and T_auto ~ 5.3.
"""

import math

import pytest

from delegation_lab.domain.capacity import (
    autonomy_buffer,
    autonomy_time,
    binary_entropy,
    delegation_capacity,
    max_process_entropy,
    process_entropy,
    quality_target_feasible,
    shannon_entropy,
)

TOL = 1e-3


def test_binary_entropy_extremes_and_peak():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert math.isclose(binary_entropy(0.5), 1.0, abs_tol=1e-12)
    with pytest.raises(ValueError):
        binary_entropy(1.5)


def test_shannon_entropy_uniform_and_deterministic():
    # uniform over 4 -> 2 bits; deterministic -> 0
    assert math.isclose(shannon_entropy([0.25] * 4), 2.0, abs_tol=1e-12)
    assert shannon_entropy([1.0]) == 0.0
    assert shannon_entropy([1.0, 0.0, 0.0]) == 0.0


def test_delegation_capacity_monotone_in_budget():
    caps = [delegation_capacity(0.55, 0.65, b) for b in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(caps[i + 1] >= caps[i] - 1e-12 for i in range(len(caps) - 1))
    # full budget capacity equals the reviewed BSC capacity
    assert math.isclose(
        delegation_capacity(0.55, 0.65, 1.0),
        1.0 - binary_entropy((1 - 0.55) * (1 - 0.65)),
        abs_tol=1e-12,
    )


def test_quality_target_feasibility():
    cap = delegation_capacity(0.9, 0.9, 1.0)  # strong node, high capacity
    assert quality_target_feasible(0.6, cap) is True
    # an almost-zero capacity cannot support a demanding target
    assert quality_target_feasible(0.95, 0.01) is False


def test_process_entropy_is_additive():
    assert math.isclose(process_entropy(1.0, 0.5, 0.3), 1.8, abs_tol=1e-12)
    assert process_entropy() == 0.0


def test_max_process_entropy_worked_example():
    # C=0.80, p=0.50, lambda=0.02 -> (0.80-0.50)/0.02 = 15 bits
    assert math.isclose(max_process_entropy(0.80, 0.50, 0.02), 15.0, abs_tol=TOL)
    with pytest.raises(ValueError):
        max_process_entropy(0.8, 0.5, 0.0)


def test_semi_real_workflow_buffer_and_autonomy_time():
    # paper: C_op=0.86, p_min=0.75, lambda=0.02, H(W)=2.3, mu_eff=0.012
    beff = autonomy_buffer(0.86, 0.75, 0.02, 2.3)
    assert math.isclose(beff, 0.064, abs_tol=TOL)  # ~0.064
    tauto = autonomy_time(0.86, 0.75, 0.02, 2.3, drift=0.012)
    assert math.isclose(tauto, 5.33, abs_tol=2e-2)  # ~5.3 time units


def test_autonomy_time_zero_when_buffer_nonpositive():
    # buffer < 0 (infeasible) -> autonomy time clamped to 0
    assert autonomy_time(0.70, 0.80, 0.02, 2.3, drift=0.01) == 0.0
    with pytest.raises(ValueError):
        autonomy_time(0.86, 0.75, 0.02, 2.3, drift=0.0)
