"""Mean-field simulator: validates the closed-form domain against simulation.

This is an application-layer module (not pure domain): it uses randomness, but
is deterministic given a seed. Stdlib only (`random` + `statistics`). It
reproduces the internal validation checks of Azevedo (2026), Section 3:

  * the Return Operator trajectory converges to the closed-form fixed point;
  * the drift-dominated autonomy time scales as 1/mu (log-log slope ~ -1).

Conventions kept: full type annotations + a `Complexity:` block per function.
"""

from __future__ import annotations

import math
import random
import statistics

from delegation_lab.domain.return_operator import raw_fixed_point


def simulate_return_operator(
    skill: float,
    *,
    observation_rate: float = 10.0,
    forget_rate: float = 2.0,
    prior_support: float = 0.0,
    steps: int = 400,
    dt: float = 0.01,
    seed: int = 0,
    start: float = 0.0,
) -> list[float]:
    """Simulate the leaky-integrator dynamics (eq. 4) with Bernoulli outcome noise.

    Each step the agent emits an outcome ~ Bernoulli(skill); the estimate moves
    toward that observed 0/1 (learning, rate eta) and relaxes toward the prior
    (forgetting, rate delta):

        sigma <- sigma + dt*eta*(outcome - sigma) - dt*delta*(sigma - sigma0)

    The expected drift equals eq. 4, so the trajectory fluctuates around the
    closed-form fixed point sigma* = (eta*skill + delta*sigma0)/(eta+delta).

    Returns:
        list[float]: the sigma_raw trajectory of length `steps + 1`.

    Complexity:
        Time  O(steps).  Space O(steps).
    """
    rng = random.Random(seed)
    sigma = start
    trajectory = [sigma]
    for _ in range(steps):
        outcome = 1.0 if rng.random() < skill else 0.0
        sigma += dt * observation_rate * (outcome - sigma)
        sigma -= dt * forget_rate * (sigma - prior_support)
        trajectory.append(sigma)
    return trajectory


def steady_state_estimate(trajectory: list[float], *, tail_fraction: float = 0.25) -> float:
    """Mean of the trajectory tail: an estimate of the simulated steady state.

    Complexity:
        Time  O(steps).  Space O(steps_tail).
    """
    if not trajectory:
        raise ValueError("trajectory must be non-empty")
    cut = max(1, int(len(trajectory) * (1.0 - tail_fraction)))
    return statistics.fmean(trajectory[cut:])


def first_passage_time(
    buffer: float,
    drift: float,
    noise: float,
    *,
    dt: float = 0.01,
    max_steps: int = 1_000_000,
    seed: int = 0,
) -> float | None:
    """First-passage time of drifted Brownian motion below the threshold.

        d sigma = -drift*dt + noise*sqrt(dt)*Z,   Z ~ N(0, 1)

    Starts a distance `buffer` above an absorbing barrier at 0 and returns the
    time to first hit it, or None if the barrier is not reached in `max_steps`.

    Complexity:
        Time  O(max_steps) worst case.  Space O(1).
    """
    if drift <= 0.0:
        raise ValueError("drift must be > 0")
    rng = random.Random(seed)
    sigma = buffer
    sqrt_dt = math.sqrt(dt)
    for step in range(1, max_steps + 1):
        sigma += -drift * dt + noise * sqrt_dt * rng.gauss(0.0, 1.0)
        if sigma <= 0.0:
            return step * dt
    return None


def mean_first_passage(
    buffer: float,
    drift: float,
    noise: float,
    *,
    n_seeds: int = 30,
    dt: float = 0.01,
    max_steps: int = 1_000_000,
) -> float:
    """Mean first-passage time over independent seeds (Monte Carlo estimate).

    Runs that never reach the barrier are dropped; raises if none do.

    Complexity:
        Time  O(n_seeds * max_steps) worst case.  Space O(n_seeds).
    """
    times = [
        t
        for s in range(n_seeds)
        if (t := first_passage_time(buffer, drift, noise, dt=dt, max_steps=max_steps, seed=s))
        is not None
    ]
    if not times:
        raise ValueError("no run reached the barrier; increase max_steps or noise")
    return statistics.fmean(times)


def autonomy_time_scaling(
    buffer: float,
    drifts: list[float],
    noise: float,
    *,
    n_seeds: int = 30,
    dt: float = 0.01,
    max_steps: int = 1_000_000,
) -> tuple[float, list[float]]:
    """Fit the log-log slope of mean autonomy time vs drift (should be ~ -1).

    Tests the drift-dominated scaling law T_auto ~ buffer / mu (eq. 17): a 1/mu
    relationship is a straight line of slope -1 on log-log axes.

    Returns:
        tuple[float, list[float]]: (slope, mean_times_per_drift).

    Complexity:
        Time  O(len(drifts) * n_seeds * max_steps) worst case.
    """
    times = [
        mean_first_passage(buffer, mu, noise, n_seeds=n_seeds, dt=dt, max_steps=max_steps)
        for mu in drifts
    ]
    log_mu = [math.log(mu) for mu in drifts]
    log_t = [math.log(t) for t in times]
    slope = statistics.linear_regression(log_mu, log_t).slope
    return slope, times


def fixed_point_gap(
    skill: float,
    *,
    observation_rate: float = 10.0,
    forget_rate: float = 2.0,
    prior_support: float = 0.0,
    steps: int = 4000,
    dt: float = 0.01,
    seed: int = 0,
) -> float:
    """Absolute gap between the simulated steady state and the closed-form fixed point.

    A small gap is the validation that the simulator and the theory agree.

    Complexity:
        Time  O(steps).  Space O(steps).
    """
    traj = simulate_return_operator(
        skill,
        observation_rate=observation_rate,
        forget_rate=forget_rate,
        prior_support=prior_support,
        steps=steps,
        dt=dt,
        seed=seed,
    )
    theory = raw_fixed_point(
        skill,
        observation_rate=observation_rate,
        forget_rate=forget_rate,
        prior_support=prior_support,
    )
    return abs(steady_state_estimate(traj) - theory)
