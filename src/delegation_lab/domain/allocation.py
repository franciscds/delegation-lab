"""Governed-delegation allocation: the Euler-Lagrange / water-filling solution.

Pure functions, stdlib only. Implements the MSO optimal allocation (eq. 8), the
Fisher metric it rests on (eq. 3), and the corrector capacity threshold. Same
conventions as `return_operator`: full type annotations + a `Complexity:` block.

The MSO (eq. 2) minimizes governance burden on the Fisher manifold subject to a
delivery constraint. Its solution allocates governed delegation proportionally
to  sigma * sqrt(sigma * (1 - sigma)), which peaks at sigma = 0.75. The water
level lambda is set so the scope meets the delivery target p_min.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def fisher_metric(sigma: float) -> float:
    """Fisher information metric for a Bernoulli outcome, g(sigma) (eq. 3).

        g(sigma) = 1 / (sigma * (1 - sigma))

    This is the local cost of calibration: it blows up near sigma = 0 or 1
    (certainty is statistically expensive to move) and is minimal at 0.5.

    Args:
        sigma: success probability in the open interval (0, 1).

    Returns:
        float: the metric g(sigma) > 0.

    Raises:
        ValueError: if sigma is not strictly inside (0, 1).

    Complexity:
        Time  O(1).  Space O(1).
    """
    if not 0.0 < sigma < 1.0:
        raise ValueError("Fisher metric is defined only for sigma in (0, 1)")
    return 1.0 / (sigma * (1.0 - sigma))


def allocation_density(sigma: float) -> float:
    """Lambda-independent shape of the optimal allocation: sigma * sqrt(sigma(1-sigma)).

    This is the per-point marginal return of governed delegation. It is the part
    of eq. 8 that does not depend on the water level lambda. It is 0 at the
    extremes (sigma in {0, 1}) and maximal at sigma = 0.75 (proved by setting the
    derivative of sigma^1.5 * (1-sigma)^0.5 to zero).

    Args:
        sigma: raw competence in [0, 1].

    Returns:
        float: the allocation density >= 0.

    Complexity:
        Time  O(1).  Space O(1).
    """
    if sigma <= 0.0 or sigma >= 1.0:
        return 0.0
    return sigma * math.sqrt(sigma * (1.0 - sigma))


def governed_allocation(sigma: float, water_level: float, *, alpha_max: float = 1.0) -> float:
    """Optimal governed-delegation intensity alpha*(x) at one point (eq. 8).

        alpha*(x) = min( alpha_max,  (lambda / 2) * sigma * sqrt(sigma(1-sigma)) )

    Args:
        sigma: raw competence sigma_raw(x) in [0, 1].
        water_level: the Lagrange multiplier lambda (>= 0), the "water level".
        alpha_max: trust ceiling alpha_max(x) = G(sigma_raw) in [0, 1].

    Returns:
        float: governed-delegation intensity alpha*(x) in [0, alpha_max].

    Complexity:
        Time  O(1).  Space O(1).
    """
    uncapped: float = 0.5 * water_level * allocation_density(sigma)
    return min(alpha_max, uncapped)


def _delivered(sigmas: Sequence[float], water_level: float, alpha_max: float) -> float:
    """Total delivered quality sum_i alpha*(sigma_i) * sigma_i at a given water level.

    Monotone non-decreasing in `water_level` (each alpha only grows until capped),
    which is what makes the bisection in `solve_water_level` valid.

    Complexity:
        Time  O(N).  Space O(1).   (N = len(sigmas))
    """
    return sum(governed_allocation(s, water_level, alpha_max=alpha_max) * s for s in sigmas)


def solve_water_level(
    sigmas: Sequence[float],
    p_min: float,
    *,
    alpha_max: float = 1.0,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> float:
    """Find the water level lambda meeting the delivery constraint (eq. 2 constraint).

    Solves  sum_i alpha*(sigma_i) * sigma_i = p_min * N  for lambda by bisection,
    exploiting monotonicity of the delivered quality in lambda.

    Args:
        sigmas: raw competence per scope point.
        p_min: target average delivered quality in [0, 1].
        alpha_max: per-point trust ceiling (scalar).
        tol: absolute tolerance on the delivery residual.
        max_iter: bisection iteration cap.

    Returns:
        float: the water level lambda >= 0 satisfying the constraint.

    Raises:
        ValueError: if the target is infeasible, i.e. even at full allocation
            (alpha = alpha_max everywhere) the scope cannot deliver p_min. This is
            the MSO "no solution" case: do not delegate under the current design.

    Complexity:
        Time  O(N * log(1/tol)).  Space O(1).
    """
    n: int = len(sigmas)
    if n == 0:
        raise ValueError("scope must contain at least one point")
    target: float = p_min * n

    # Feasibility: the most the scope can ever deliver is alpha_max * sum(sigma).
    max_deliverable: float = alpha_max * sum(sigmas)
    if target > max_deliverable + tol:
        raise ValueError(
            "MSO infeasible: delivery target exceeds the operational ceiling; "
            "do not expand autonomy under the current design"
        )

    # Bracket lambda. lo delivers <= target; hi must deliver >= target.
    lo: float = 0.0
    hi: float = 1.0
    while _delivered(sigmas, hi, alpha_max) < target:
        hi *= 2.0
        if hi > 1e18:  # everything is capped; constraint met by feasibility check
            return hi

    for _ in range(max_iter):
        mid: float = 0.5 * (lo + hi)
        if _delivered(sigmas, mid, alpha_max) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def allocate(sigmas: Sequence[float], p_min: float, *, alpha_max: float = 1.0) -> list[float]:
    """Full water-filling allocation over a scope: the alpha* vector.

    Convenience wrapper: solves for the water level, then applies eq. 8 per point.

    Returns:
        list[float]: alpha*(sigma_i) for each scope point.

    Complexity:
        Time  O(N * log(1/tol)).  Space O(N).
    """
    lam: float = solve_water_level(sigmas, p_min, alpha_max=alpha_max)
    return [governed_allocation(s, lam, alpha_max=alpha_max) for s in sigmas]


def corrector_threshold(p_min: float, raw: float, catch_rate: float) -> float:
    """Minimum corrector coverage K/N to sustain the quality target.

        K/N >= max( 0,  (p_min - sigma_raw) / ((1 - sigma_raw) * catch_rate) )

    Below this coverage the delegation cannot maintain quality regardless of how
    the governed workload is allocated. A return value > 1 means the target is
    unreachable even at full review (infeasible under the current catch rate).

    Args:
        p_min: target delivered quality in [0, 1].
        raw: raw competence sigma_raw in [0, 1).
        catch_rate: corrector catch rate c in (0, 1].

    Returns:
        float: required review coverage K/N (>= 0; > 1 signals infeasibility).

    Raises:
        ValueError: if catch_rate <= 0 or raw >= 1 (threshold undefined).

    Complexity:
        Time  O(1).  Space O(1).
    """
    if catch_rate <= 0.0:
        raise ValueError("catch_rate must be > 0")
    if raw >= 1.0:
        raise ValueError("raw must be < 1 for the threshold to be defined")
    return max(0.0, (p_min - raw) / ((1.0 - raw) * catch_rate))
