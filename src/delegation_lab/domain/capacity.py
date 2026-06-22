"""Delegation capacity and autonomy feasibility.

Pure functions, stdlib only. Implements the information-theoretic side of the
MSO framework: binary entropy, the governed-delegation channel capacity
(eq. 13), the quality-target feasibility corollary, process entropy (eq. 14),
the effective autonomy buffer (eq. 16), and the autonomy time (eq. 17).

Same conventions: full type annotations + a `Complexity:` block per function.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def binary_entropy(p: float) -> float:
    """Binary (Shannon) entropy H_b(p) in bits.

        H_b(p) = -p*log2(p) - (1-p)*log2(1-p),   with H_b(0) = H_b(1) = 0.

    Measures the uncertainty of a Bernoulli(p) outcome: 0 at the extremes
    (certain), maximal 1 bit at p = 0.5 (maximally uncertain).

    Args:
        p: probability in [0, 1].

    Returns:
        float: entropy in bits, in [0, 1].

    Raises:
        ValueError: if p is outside [0, 1].

    Complexity:
        Time  O(1).  Space O(1).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if p in (0.0, 1.0):
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def shannon_entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy H = -sum p_i log2 p_i of a discrete distribution (bits).

    Zero-probability outcomes contribute 0 (the limit p*log p -> 0). A uniform
    distribution over k outcomes has entropy log2(k); a deterministic one has 0.

    Args:
        probabilities: a distribution; should sum to ~1.

    Returns:
        float: entropy in bits.

    Raises:
        ValueError: if any probability is negative.

    Complexity:
        Time  O(k).  Space O(1).   (k = number of outcomes)
    """
    total: float = 0.0
    for p in probabilities:
        if p < 0.0:
            raise ValueError("probabilities must be non-negative")
        if p > 0.0:
            total -= p * math.log2(p)
    return total


def bsc_capacity(skill: float, catch_rate: float) -> float:
    """Single-node BSC capacity C_v = 1 - H_b(eps_eff) in bits.

        eps_eff = (1 - skill) * (1 - catch_rate)

    The node is modelled as a Binary Symmetric Channel with effective error
    rate eps_eff; its capacity is what remains after that noise.

    Complexity:
        Time  O(1).  Space O(1).
    """
    eps_eff: float = (1.0 - skill) * (1.0 - catch_rate)
    return 1.0 - binary_entropy(eps_eff)


def delegation_capacity(skill: float, catch_rate: float, budget: float) -> float:
    """Revealed-action governed-delegation channel capacity C_del(B) (eq. 13).

        C_del(B) = (1 - B) * (1 - H_b(eps0)) + B * (1 - H_b(eps1))
        eps0 = 1 - skill                     (unreviewed crossover)
        eps1 = (1 - skill) * (1 - catch_rate) (reviewed crossover)

    The optimum reviews as much as the budget allows (q* = B), so capacity is a
    budget-weighted blend of the unreviewed and reviewed node capacities. It is
    monotone non-decreasing in `budget`.

    Args:
        skill: agent skill sigma_skill in [0, 1].
        catch_rate: corrector catch rate c in [0, 1].
        budget: average review budget B (review fraction) in [0, 1].

    Returns:
        float: delegation capacity in bits, in [0, 1].

    Raises:
        ValueError: if budget is outside [0, 1].

    Complexity:
        Time  O(1).  Space O(1).
    """
    if not 0.0 <= budget <= 1.0:
        raise ValueError("budget must be in [0, 1]")
    eps0: float = 1.0 - skill
    eps1: float = (1.0 - skill) * (1.0 - catch_rate)
    unreviewed: float = 1.0 - binary_entropy(eps0)
    reviewed: float = 1.0 - binary_entropy(eps1)
    return (1.0 - budget) * unreviewed + budget * reviewed


def quality_target_feasible(p_min: float, capacity: float) -> bool:
    """Whether a quality target is achievable given a channel capacity (corollary).

    A uniform binary source at correctness p_min carries I = 1 - H_b(1 - p_min)
    bits, achievable iff it is below capacity:

        1 - H_b(1 - p_min) < capacity

    Complexity:
        Time  O(1).  Space O(1).
    """
    required_bits: float = 1.0 - binary_entropy(1.0 - p_min)
    return required_bits < capacity


def process_entropy(routing: float = 0.0, tool_calls: float = 0.0, timing: float = 0.0) -> float:
    """Total workflow process entropy H(W) (eq. 14).

        H(W) = H(routing) + H(tool calls) + H(timing)

    Each component is itself a Shannon entropy (use `shannon_entropy` to compute
    them from the corresponding decision distributions). Deterministic decisions
    contribute 0; the additive form assumes conditional independence across them.

    Complexity:
        Time  O(1).  Space O(1).
    """
    return routing + tool_calls + timing


def autonomy_buffer(
    operational_ceiling: float, p_min: float, gap_coefficient: float, process_entropy: float
) -> float:
    """Effective autonomy buffer B_eff (eq. 16).

        B_eff = C_op - p_min - lambda * H(W)

    The central geometric quantity: > 0 means delegated autonomy is feasible,
    = 0 is the autonomy cliff, < 0 means no governance policy sustains the target.

    Complexity:
        Time  O(1).  Space O(1).
    """
    return operational_ceiling - p_min - gap_coefficient * process_entropy


def max_process_entropy(operational_ceiling: float, p_min: float, gap_coefficient: float) -> float:
    """Delegation process capacity H_max: the most workflow complexity tolerable.

        H_max(p_min) = (C_op - p_min) / lambda

    Above this, output quality drops below p_min.

    Raises:
        ValueError: if gap_coefficient <= 0.

    Complexity:
        Time  O(1).  Space O(1).
    """
    if gap_coefficient <= 0.0:
        raise ValueError("gap_coefficient (lambda) must be > 0")
    return (operational_ceiling - p_min) / gap_coefficient


def autonomy_time(
    operational_ceiling: float,
    p_min: float,
    gap_coefficient: float,
    process_entropy: float,
    drift: float,
) -> float:
    """Drift-dominated autonomy time T*_auto (eq. 17).

        T*_auto = B_eff / mu_eff = (C_op - p_min - lambda*H(W)) / mu_eff

    Expected time the pipeline can run before quality drifts below p_min. When
    the buffer is non-positive the system is already at/over the cliff, so the
    autonomy time is 0 (intervene immediately).

    Args:
        operational_ceiling: C_op.
        p_min: quality target.
        gap_coefficient: governance gap lambda.
        process_entropy: H(W).
        drift: effective drift rate mu_eff > 0.

    Returns:
        float: autonomy time (>= 0) in the model's time units.

    Raises:
        ValueError: if drift <= 0 (mean first-passage time is undefined).

    Complexity:
        Time  O(1).  Space O(1).
    """
    if drift <= 0.0:
        raise ValueError("drift (mu_eff) must be > 0 for a finite autonomy time")
    buffer: float = autonomy_buffer(operational_ceiling, p_min, gap_coefficient, process_entropy)
    return max(0.0, buffer) / drift
