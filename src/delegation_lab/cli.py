"""Command-line adapter over the pure domain core (an inbound shell, like the API).

Each command is thin: parse args -> call a domain function -> print. No business
logic lives here. Built with Typer so the type annotations drive parsing,
validation and --help automatically (same ergonomics as the FastAPI shell).

Run after `pip install -e .`:
    delegation-lab masking --skill 0.8 --catch-rate 0.7
Or without installing:
    python -m delegation_lab.cli --help
"""

from __future__ import annotations

from typing import Annotated

import typer

from delegation_lab.domain.allocation import (
    corrector_threshold,
    governed_allocation,
    solve_water_level,
)
from delegation_lab.domain.capacity import (
    autonomy_buffer,
    autonomy_time,
    delegation_capacity,
    quality_target_feasible,
)
from delegation_lab.domain.return_operator import (
    corrected_quality,
    masking_index,
    raw_fixed_point,
)

app = typer.Typer(help="MSO delegation-lab calculator (Azevedo, 2026).", no_args_is_help=True)


@app.command()
def masking(
    skill: Annotated[float, typer.Option(help="agent skill sigma_skill in [0,1]")],
    catch_rate: Annotated[float, typer.Option(help="corrector catch rate c in [0,1]")],
    observation_rate: Annotated[float, typer.Option(help="eta > 0")] = 10.0,
    forget_rate: Annotated[float, typer.Option(help="delta >= 0")] = 2.0,
    prior_support: Annotated[float, typer.Option(help="sigma_0 in [0,1]")] = 0.0,
    coverage: Annotated[float, typer.Option(help="review coverage K/N in [0,1]")] = 1.0,
) -> None:
    """Raw competence (eq. 5), corrected quality (eq. 6) and masking index M*."""
    raw = raw_fixed_point(
        skill,
        observation_rate=observation_rate,
        forget_rate=forget_rate,
        prior_support=prior_support,
    )
    corr = corrected_quality(raw, catch_rate=catch_rate, coverage=coverage)
    typer.echo(f"raw={raw:.4f}  corrected={corr:.4f}  M*={masking_index(raw, corr):.4f}")


@app.command()
def allocate(
    sigma: Annotated[list[float], typer.Option(help="per-point sigma; repeat the flag")],
    p_min: Annotated[float, typer.Option(help="delivery target in [0,1]")],
    alpha_max: Annotated[float, typer.Option(help="trust ceiling in [0,1]")] = 1.0,
) -> None:
    """Water-filling allocation (eq. 8) over a scope of competence points."""
    water_level = solve_water_level(sigma, p_min, alpha_max=alpha_max)
    alphas = [governed_allocation(s, water_level, alpha_max=alpha_max) for s in sigma]
    pretty = ", ".join(f"{a:.4f}" for a in alphas)
    typer.echo(f"lambda={water_level:.4f}  alphas=[{pretty}]")


@app.command()
def threshold(
    p_min: Annotated[float, typer.Option(help="delivery target in [0,1]")],
    raw: Annotated[float, typer.Option(help="raw competence in [0,1)")],
    catch_rate: Annotated[float, typer.Option(help="corrector catch rate in (0,1]")],
) -> None:
    """Minimum corrector coverage K/N to sustain the target."""
    kn = corrector_threshold(p_min, raw, catch_rate)
    feasible = "yes" if kn <= 1.0 else "NO (infeasible at full review)"
    typer.echo(f"K/N_required={kn:.4f}  feasible={feasible}")


@app.command()
def capacity(
    skill: Annotated[float, typer.Option(help="agent skill in [0,1]")],
    catch_rate: Annotated[float, typer.Option(help="corrector catch rate in [0,1]")],
    budget: Annotated[float, typer.Option(help="review budget B in [0,1]")],
    p_min: Annotated[float | None, typer.Option(help="optional target to test feasibility")] = None,
) -> None:
    """Governed-delegation channel capacity in bits (eq. 13)."""
    cap = delegation_capacity(skill, catch_rate, budget)
    line = f"capacity={cap:.4f} bits"
    if p_min is not None:
        line += f"  feasible={quality_target_feasible(p_min, cap)}"
    typer.echo(line)


@app.command()
def autonomy(
    ceiling: Annotated[float, typer.Option(help="operational ceiling C_op in [0,1]")],
    p_min: Annotated[float, typer.Option(help="quality target in [0,1]")],
    lam: Annotated[float, typer.Option(help="governance gap lambda > 0")],
    entropy: Annotated[float, typer.Option(help="process entropy H(W) >= 0")] = 0.0,
    drift: Annotated[float, typer.Option(help="effective drift mu_eff > 0")] = 0.012,
) -> None:
    """Effective autonomy buffer (eq. 16) and autonomy time (eq. 17)."""
    beff = autonomy_buffer(ceiling, p_min, lam, entropy)
    t = autonomy_time(ceiling, p_min, lam, entropy, drift)
    typer.echo(f"B_eff={beff:.4f}  T_auto={t:.2f}  feasible={beff > 0.0}")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="bind host")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="bind port")] = 8000,
    reload: Annotated[bool, typer.Option(help="auto-reload on changes")] = False,
) -> None:
    """Launch the FastAPI server (requires the 'api' extra)."""
    import uvicorn  # local import: CLI calc commands don't need the api extra

    uvicorn.run("delegation_lab.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
