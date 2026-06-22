"""Module 5 tests: simulator validates the closed-form domain (Section 3).

- Simulated steady state converges to the closed-form fixed point (eq. 5).
- Drift-dominated autonomy time scales as 1/mu: log-log slope ~ -1 (paper: -0.99).
- Mean first-passage time decreases monotonically with drift.
"""

import pytest

from delegation_lab.domain.return_operator import raw_fixed_point
from delegation_lab.simulation import (
    autonomy_time_scaling,
    fixed_point_gap,
    mean_first_passage,
    simulate_return_operator,
    steady_state_estimate,
)


def test_simulation_converges_to_closed_form_fixed_point():
    # Over a long run the simulated tail mean matches the theory to the noise floor.
    for skill in (0.55, 0.70, 0.80):
        gap = fixed_point_gap(skill, steps=4000, seed=1)
        assert gap < 0.02


def test_trajectory_shape_and_steady_state():
    traj = simulate_return_operator(0.8, steps=400, seed=3)
    assert len(traj) == 401  # steps + 1 (includes the start)
    theory = raw_fixed_point(0.8, observation_rate=10, forget_rate=2)
    assert abs(steady_state_estimate(traj) - theory) < 0.05


def test_autonomy_time_loglog_slope_is_minus_one():
    # Drift over two orders of magnitude -> 1/mu scaling -> slope ~ -1.
    drifts = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    slope, times = autonomy_time_scaling(
        buffer=0.1, drifts=drifts, noise=0.005, n_seeds=20, dt=0.01, max_steps=200_000
    )
    assert -1.1 < slope < -0.9  # paper reports -0.99
    # times strictly decrease as drift grows
    assert all(times[i + 1] < times[i] for i in range(len(times) - 1))


def test_mean_first_passage_decreases_with_drift():
    slow = mean_first_passage(0.1, 0.01, 0.005, n_seeds=20, max_steps=200_000)
    fast = mean_first_passage(0.1, 0.1, 0.005, n_seeds=20, max_steps=200_000)
    assert fast < slow


def test_first_passage_requires_positive_drift():
    with pytest.raises(ValueError):
        mean_first_passage(0.1, 0.0, 0.005)
