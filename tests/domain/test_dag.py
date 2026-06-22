"""Module 3 tests: DAG propagation, aggregation, recursive chain ceiling.

Reproduces structural facts from Azevedo (2026):
  - Aggregation ordering product <= min <= weighted_mean for any parent set.
  - Recursive chain ceiling (eq. 11) starts at 1.0 and decreases with depth.
  - Masking compounds: M* increases monotonically with depth in a linked chain.
  - Topological propagation processes parents before children; cycles are rejected.
"""

import math

import pytest

from delegation_lab.domain.dag import (
    Aggregation,
    DelegationGraph,
    Node,
    aggregate,
    chain_ceiling,
    effective_skill,
)

TOL = 1e-3


def test_aggregation_ordering():
    parents = [0.8, 0.6, 0.9]
    prod = aggregate(parents, Aggregation.PRODUCT)
    mn = aggregate(parents, Aggregation.MIN)
    wm = aggregate(parents, Aggregation.WEIGHTED_MEAN, weights=[1, 1, 1])
    assert prod <= mn <= wm
    assert math.isclose(prod, 0.8 * 0.6 * 0.9, abs_tol=1e-12)
    assert math.isclose(mn, 0.6, abs_tol=1e-12)


def test_source_node_has_neutral_aggregation():
    # no parents -> AGG returns 1.0 -> effective skill equals own skill
    assert aggregate([]) == 1.0
    assert math.isclose(effective_skill(0.7, []), 0.7, abs_tol=1e-12)


def test_weighted_mean_requires_weights():
    with pytest.raises(ValueError):
        aggregate([0.5, 0.6], Aggregation.WEIGHTED_MEAN)


def test_chain_ceiling_base_cases_and_monotonicity():
    q = chain_ceiling(0.55, 5, catch_rate=0.65, coverage=0.5)
    assert len(q) == 6  # sigma*_corr(0..5)
    assert q[0] == 1.0  # eq. 11 base case
    # ceiling decreases (each layer attenuates) and converges
    assert all(q[i + 1] <= q[i] for i in range(5))


def test_masking_increases_with_depth_in_linked_chain():
    g = DelegationGraph()
    prev: str | None = None
    for i in range(1, 6):
        parents = (prev,) if prev is not None else ()
        g.add(Node(name=f"L{i}", skill=0.55, catch_rate=0.65, parents=parents))
        prev = f"L{i}"
    res = g.propagate(coverage=0.5)
    masking = [res[f"L{i}"].masking for i in range(1, 6)]
    assert all(masking[i + 1] > masking[i] for i in range(4))  # strictly increasing
    assert masking[-1] > 2.1  # paper reports ~2.20 at layer 5


def test_diamond_propagation_and_topo_order():
    # A -> B, A -> C, (B,C) -> D  (the diamond motif)
    g = DelegationGraph()
    g.add(Node("A", skill=0.8, catch_rate=0.6))
    g.add(Node("B", skill=0.7, catch_rate=0.6, parents=("A",)))
    g.add(Node("C", skill=0.7, catch_rate=0.6, parents=("A",)))
    g.add(Node("D", skill=0.7, catch_rate=0.6, parents=("B", "C")))
    order = g.topological_order()
    assert order.index("A") < order.index("B")
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")
    res = g.propagate()
    assert set(res) == {"A", "B", "C", "D"}
    # merge node D sees product aggregation -> lower effective skill than a source
    assert res["D"].raw < res["A"].raw


def test_cycle_is_rejected():
    g = DelegationGraph()
    g.add(Node("X", skill=0.7, catch_rate=0.6, parents=("Y",)))
    g.add(Node("Y", skill=0.7, catch_rate=0.6, parents=("X",)))
    with pytest.raises(ValueError):
        g.topological_order()


def test_unknown_parent_is_rejected():
    g = DelegationGraph()
    g.add(Node("Z", skill=0.7, catch_rate=0.6, parents=("ghost",)))
    with pytest.raises(ValueError):
        g.topological_order()
