"""Delegation DAGs: uncertainty propagation and chain masking.

Pure functions + small immutable data types, stdlib only. Implements the
effective-skill propagation (eq. 7), the three aggregation modes, topological
propagation over a delegation DAG (O(V+E)), and the recursive chain ceiling
(eq. 11). Same conventions: full type annotations + a `Complexity:` block.

An agent's effective skill depends on the quality of its inputs:
    sigma_skill_eff(v) = sigma_skill(v) * AGG( sigma_corr(parents of v) )
The choice of AGG is set by the merge semantics: product (independent
dimensions, highest masking), min (weakest link), or weighted mean (dilution).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from delegation_lab.domain.return_operator import corrected_quality, raw_fixed_point


class Aggregation(str, Enum):
    """How a node combines its parents' delivered qualities (eq. 7)."""

    PRODUCT = "product"  # independent dimensions; errors compound multiplicatively
    MIN = "min"  # weakest link; all inputs must be correct
    WEIGHTED_MEAN = "weighted_mean"  # importance-weighted blend; dilutes errors


def aggregate(
    parent_qualities: Sequence[float],
    mode: Aggregation = Aggregation.PRODUCT,
    *,
    weights: Sequence[float] | None = None,
) -> float:
    """Aggregate parents' corrected qualities into a single input-quality factor.

    A source node (no parents) returns the neutral element 1.0, so its effective
    skill equals its own skill (no upstream degradation).

    For any fixed parent set the modes order as:  product <= min <= weighted_mean
    (product of values in [0,1] is <= each; a mean is >= the min). Product is
    therefore the most pessimistic / highest-masking merge.

    Args:
        parent_qualities: sigma_corr of each parent in [0, 1].
        mode: aggregation semantics.
        weights: required for WEIGHTED_MEAN, one per parent.

    Returns:
        float: aggregated input-quality factor in [0, 1].

    Raises:
        ValueError: if WEIGHTED_MEAN is used without matching weights.

    Complexity:
        Time  O(k).  Space O(1).   (k = number of parents)
    """
    if not parent_qualities:
        return 1.0
    if mode is Aggregation.PRODUCT:
        return math.prod(parent_qualities)
    if mode is Aggregation.MIN:
        return min(parent_qualities)
    # WEIGHTED_MEAN
    if weights is None or len(weights) != len(parent_qualities):
        raise ValueError("WEIGHTED_MEAN requires one weight per parent")
    total_weight: float = sum(weights)
    if total_weight <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return sum(w * q for w, q in zip(weights, parent_qualities, strict=True)) / total_weight


def effective_skill(
    skill: float,
    parent_qualities: Sequence[float],
    mode: Aggregation = Aggregation.PRODUCT,
    *,
    weights: Sequence[float] | None = None,
) -> float:
    """Effective skill at a node given its inputs (eq. 7).

        sigma_skill_eff(v) = sigma_skill(v) * AGG(parents)

    Complexity:
        Time  O(k).  Space O(1).
    """
    return skill * aggregate(parent_qualities, mode, weights=weights)


@dataclass(frozen=True)
class Node:
    """A delegation node: its intrinsic skill, corrector, and parents."""

    name: str
    skill: float
    catch_rate: float
    parents: tuple[str, ...] = ()
    aggregation: Aggregation = Aggregation.PRODUCT
    weights: tuple[float, ...] | None = None


@dataclass(frozen=True)
class NodeResult:
    """Propagated steady-state quantities at a node."""

    raw: float  # sigma*_raw  (competence signal -> authorization)
    corrected: float  # sigma*_corr (delivered quality)
    masking: float  # M* = corrected / raw


@dataclass
class DelegationGraph:
    """A delegation DAG. Nodes keyed by name; edges implied by `parents`."""

    nodes: dict[str, Node] = field(default_factory=dict)

    def add(self, node: Node) -> None:
        """Insert a node. Complexity: O(1) amortized."""
        self.nodes[node.name] = node

    def topological_order(self) -> list[str]:
        """Kahn's algorithm: parents before children.

        Returns node names in a valid processing order.

        Raises:
            ValueError: if the graph contains a cycle (not a DAG).

        Complexity:
            Time  O(V + E).  Space O(V).
        """
        indegree: dict[str, int] = {name: 0 for name in self.nodes}
        children: dict[str, list[str]] = {name: [] for name in self.nodes}
        for name, node in self.nodes.items():
            for parent in node.parents:
                if parent not in self.nodes:
                    raise ValueError(f"node '{name}' references unknown parent '{parent}'")
                indegree[name] += 1
                children[parent].append(name)

        queue: list[str] = [n for n, d in indegree.items() if d == 0]
        order: list[str] = []
        while queue:
            current: str = queue.pop()
            order.append(current)
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("graph has a cycle; delegation graphs must be DAGs")
        return order

    def propagate(
        self,
        *,
        observation_rate: float = 10.0,
        forget_rate: float = 2.0,
        prior_support: float = 0.0,
        coverage: float = 1.0,
    ) -> dict[str, NodeResult]:
        """Propagate competence/quality through the DAG (eq. 7 + Return Operator).

        For each node in topological order: aggregate parents' corrected quality
        into an effective skill (eq. 7), then apply the Return Operator to obtain
        the raw fixed point (eq. 5) and corrected quality (eq. 6), then M*.

        Returns:
            dict[str, NodeResult]: per-node raw, corrected, and masking index.

        Complexity:
            Time  O(V + E).  Space O(V).
        """
        results: dict[str, NodeResult] = {}
        for name in self.topological_order():
            node: Node = self.nodes[name]
            parent_quality: list[float] = [results[p].corrected for p in node.parents]
            skill_eff: float = effective_skill(
                node.skill, parent_quality, node.aggregation, weights=node.weights
            )
            raw: float = raw_fixed_point(
                skill_eff,
                observation_rate=observation_rate,
                forget_rate=forget_rate,
                prior_support=prior_support,
            )
            corrected: float = corrected_quality(raw, catch_rate=node.catch_rate, coverage=coverage)
            masking: float = corrected / raw if raw > 0.0 else math.inf
            results[name] = NodeResult(raw=raw, corrected=corrected, masking=masking)
        return results


def chain_ceiling(
    skill: float,
    depth: int,
    *,
    catch_rate: float,
    observation_rate: float = 10.0,
    forget_rate: float = 2.0,
    prior_support: float = 0.0,
    coverage: float = 1.0,
) -> list[float]:
    """Recursive operational ceiling of a linear chain, Cop(D) (eq. 11).

        sigma*_corr(i) = R( skill * sigma*_corr(i-1) ),   sigma*_corr(0) = 1
    where R applies the Return Operator (raw fixed point -> corrected). Cop(D)
    is the last element. Each layer feeds its corrected output to the next.

    Args:
        skill: intrinsic per-layer skill sigma_skill.
        depth: number of layers D (>= 0).
        catch_rate: per-layer corrector catch rate c.

    Returns:
        list[float]: [sigma*_corr(0)=1, sigma*_corr(1), ..., sigma*_corr(D)].

    Complexity:
        Time  O(D).  Space O(D).
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    qualities: list[float] = [1.0]  # sigma*_corr(0) = 1
    for _ in range(depth):
        skill_eff: float = skill * qualities[-1]
        raw: float = raw_fixed_point(
            skill_eff,
            observation_rate=observation_rate,
            forget_rate=forget_rate,
            prior_support=prior_support,
        )
        qualities.append(corrected_quality(raw, catch_rate=catch_rate, coverage=coverage))
    return qualities
