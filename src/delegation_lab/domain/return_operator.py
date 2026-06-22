"""Return Operator: the core competence/quality dynamics of the MSO framework.

Pure functions, no I/O, no framework dependencies. Each function maps directly
to an equation in Azevedo (2026), "Minimal Oversight". Equation numbers below
refer to that paper.

Convention for this library:
  * Every public function carries full type annotations, including an explicit
    return type (enforced by ruff's ANN rules + mypy in CI).
  * Every function documents its asymptotic cost in a `Complexity:` block.

The Return Operator R is the cyclic process:
    B operates -> C corrects -> records accumulate -> sigma updates -> alpha updates.
At steady state it collapses to two closed-form fixed points (eq. 5 and 6),
from which the masking index follows by a single division.
"""

from __future__ import annotations


def raw_fixed_point(
    skill: float,
    *,
    observation_rate: float,
    forget_rate: float,
    prior_support: float = 0.0,
) -> float:
    """Equilibrium raw competence sigma*_raw (eq. 5).

    Steady state of the leaky-integrator dynamics (eq. 4): the agent "charges
    up" toward its true skill via observations (rate eta) and relaxes toward a
    prior via decay (rate delta). The fixed point is their weighted average:

        sigma*_raw = (eta * skill + delta * prior) / (eta + delta)

    Args:
        skill: agent's true Bernoulli competence sigma_skill in [0, 1].
        observation_rate: eta > 0, how fast evidence accumulates.
        forget_rate: delta >= 0, how fast stale evidence decays.
        prior_support: sigma_0, the support level decay relaxes toward
            (0.0 = conservative "no demonstrated support"; 0.5 = uninformative).

    Returns:
        float: equilibrium raw competence in [0, 1].

    Complexity:
        Time  O(1) - fixed number of float arithmetic ops.
        Space O(1) - no allocation.
    """
    total_rate: float = observation_rate + forget_rate
    if total_rate <= 0.0:
        raise ValueError("observation_rate + forget_rate must be > 0")
    return (observation_rate * skill + forget_rate * prior_support) / total_rate


def corrected_quality(raw: float, *, catch_rate: float, coverage: float = 1.0) -> float:
    """Delivered quality sigma*_corr after the corrector acts (eq. 6).

    The corrector catches a fraction `catch_rate` of the agent's errors. The
    error mass is (1 - raw); a fraction (catch_rate * coverage) of it is
    rescued and promoted to correct output:

        sigma*_corr = raw + (1 - raw) * catch_rate * coverage

    With coverage = 1.0 this is eq. 6 (full review). With coverage = K/N it is
    the simulator's expected-value update (Section 3), where only a fraction
    K/N of outputs is reviewed per cycle.

    Args:
        raw: raw competence sigma*_raw in [0, 1].
        catch_rate: corrector catch rate c in [0, 1].
        coverage: review coverage K/N in [0, 1] (default 1.0 = full review).

    Returns:
        float: delivered (corrected) quality in [0, 1].

    Complexity:
        Time  O(1) - fixed number of float arithmetic ops.
        Space O(1) - no allocation.
    """
    return raw + (1.0 - raw) * catch_rate * coverage


def masking_index(raw: float, corrected: float) -> float:
    """Masking index M* = sigma*_corr / sigma*_raw.

    M* > 1 means the corrector is hiding agent weakness from the authorization
    mechanism: the system looks more competent than the producing agent is.
    M* = 1 means no masking. Authorization must key off `raw`, not `corrected`.

    Args:
        raw: raw competence sigma*_raw in (0, 1].
        corrected: delivered quality sigma*_corr in [0, 1].

    Returns:
        float: masking index M* >= 1 under a non-degrading corrector.

    Raises:
        ValueError: if raw <= 0 (the index is undefined / degenerate).

    Complexity:
        Time  O(1) - one comparison and one division.
        Space O(1) - no allocation.
    """
    if raw <= 0.0:
        raise ValueError("raw competence must be > 0 to compute a masking index")
    return corrected / raw
