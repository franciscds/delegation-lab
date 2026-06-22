"""Generate result figures from the domain + simulation code (reproducible).

Run:  python scripts/generate_figures.py
Every figure is produced by calling the library itself, so the plots always
reflect the current implementation (no hard-coded numbers).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, no display
import matplotlib.pyplot as plt  # noqa: E402

from delegation_lab.domain.allocation import allocation_density  # noqa: E402
from delegation_lab.domain.capacity import delegation_capacity  # noqa: E402
from delegation_lab.domain.dag import DelegationGraph, Node, chain_ceiling  # noqa: E402
from delegation_lab.domain.return_operator import raw_fixed_point  # noqa: E402
from delegation_lab.simulation import (  # noqa: E402
    autonomy_time_scaling,
    simulate_return_operator,
)

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True})


def fig_convergence() -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for skill in (0.55, 0.80):
        traj = simulate_return_operator(skill, steps=400, seed=1)
        theory = raw_fixed_point(skill, observation_rate=10, forget_rate=2)
        t = [i * 0.01 for i in range(len(traj))]
        line = ax.plot(t, traj, lw=1, alpha=0.8, label=f"sim (skill={skill})")[0]
        ax.axhline(theory, ls="--", color=line.get_color(), label=f"theory sigma*={theory:.3f}")
    ax.set(
        xlabel="time",
        ylabel="sigma_raw",
        title="Return Operator: simulation converges to closed-form fixed point",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "01_return_operator_convergence.png")
    plt.close(fig)


def fig_autonomy_scaling() -> None:
    drifts = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    slope, times = autonomy_time_scaling(
        buffer=0.1, drifts=drifts, noise=0.005, n_seeds=20, dt=0.01, max_steps=200_000
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(drifts, times, "o", ms=6, label="simulated mean FPT")
    # fitted line through the cloud
    c = math.log(times[0]) - slope * math.log(drifts[0])
    fit = [math.exp(slope * math.log(mu) + c) for mu in drifts]
    ax.loglog(drifts, fit, "-", color="crimson", label=f"fit slope = {slope:.3f}  (paper: -0.99)")
    ax.set(
        xlabel="drift mu_eff",
        ylabel="autonomy time T_auto",
        title="Autonomy time scales as 1/mu (drift-dominated first passage)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "02_autonomy_time_scaling.png")
    plt.close(fig)


def fig_water_filling() -> None:
    xs = [i / 500 for i in range(1, 500)]
    ys = [allocation_density(s) for s in xs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, lw=2)
    ax.axvline(0.75, ls="--", color="crimson", label="peak at sigma = 0.75")
    ax.set(
        xlabel="raw competence sigma",
        ylabel="allocation density",
        title="Water-filling allocation peaks at intermediate competence (eq. 8)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "03_water_filling_peak.png")
    plt.close(fig)


def fig_chain_masking() -> None:
    g = DelegationGraph()
    prev: str | None = None
    for i in range(1, 6):
        parents = (prev,) if prev is not None else ()
        g.add(Node(name=f"L{i}", skill=0.55, catch_rate=0.65, parents=parents))
        prev = f"L{i}"
    res = g.propagate(coverage=0.5)
    layers = list(range(1, 6))
    masking = [res[f"L{i}"].masking for i in layers]
    ceiling = chain_ceiling(0.55, 5, catch_rate=0.65, coverage=0.5)[1:]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(layers, masking, "o-", color="darkorange")
    a1.set(
        xlabel="chain depth (layer)",
        ylabel="masking index M*",
        title="Masking compounds with depth",
    )
    a2.plot(layers, ceiling, "s-", color="teal")
    a2.set(
        xlabel="chain depth (layer)",
        ylabel="operational ceiling Cop",
        title="Recursive chain quality attenuates (eq. 11)",
    )
    fig.tight_layout()
    fig.savefig(OUT / "04_chain_masking_ceiling.png")
    plt.close(fig)


def fig_capacity() -> None:
    budgets = [i / 50 for i in range(51)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for skill, c in ((0.55, 0.65), (0.70, 0.70), (0.80, 0.70)):
        caps = [delegation_capacity(skill, c, b) for b in budgets]
        ax.plot(budgets, caps, lw=2, label=f"skill={skill}, c={c}")
    ax.set(
        xlabel="review budget B",
        ylabel="capacity (bits)",
        title="Delegation capacity grows with review budget (eq. 13)",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "05_delegation_capacity.png")
    plt.close(fig)


def main() -> None:
    fig_convergence()
    fig_autonomy_scaling()
    fig_water_filling()
    fig_chain_masking()
    fig_capacity()
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
