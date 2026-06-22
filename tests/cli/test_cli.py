"""CLI tests: invoke commands in-process (CliRunner) and assert paper numbers.

Same philosophy as the API tests: the adapter must reproduce the domain's
values exactly, proving it only translates.
"""

from typer.testing import CliRunner

from delegation_lab.cli import app

runner = CliRunner()


def test_masking_command_matches_paper():
    result = runner.invoke(app, ["masking", "--skill", "0.8", "--catch-rate", "0.7"])
    assert result.exit_code == 0
    assert "raw=0.6667" in result.stdout
    assert "M*=1.3500" in result.stdout


def test_autonomy_command_semi_real():
    result = runner.invoke(
        app,
        [
            "autonomy",
            "--ceiling",
            "0.86",
            "--p-min",
            "0.75",
            "--lam",
            "0.02",
            "--entropy",
            "2.3",
            "--drift",
            "0.012",
        ],
    )
    assert result.exit_code == 0
    assert "B_eff=0.0640" in result.stdout
    assert "T_auto=5.33" in result.stdout
    assert "feasible=True" in result.stdout


def test_threshold_command_worked_example():
    result = runner.invoke(
        app, ["threshold", "--p-min", "0.80", "--raw", "0.55", "--catch-rate", "0.65"]
    )
    assert result.exit_code == 0
    assert "K/N_required=0.8547" in result.stdout


def test_allocate_command_repeated_sigma():
    result = runner.invoke(
        app,
        ["allocate", "--sigma", "0.4", "--sigma", "0.6", "--sigma", "0.8", "--p-min", "0.3"],
    )
    assert result.exit_code == 0
    assert "lambda=" in result.stdout
    assert "alphas=[" in result.stdout


def test_capacity_command_with_feasibility():
    result = runner.invoke(
        app,
        ["capacity", "--skill", "0.9", "--catch-rate", "0.9", "--budget", "1.0", "--p-min", "0.6"],
    )
    assert result.exit_code == 0
    assert "capacity=" in result.stdout
    assert "feasible=True" in result.stdout


def test_infeasible_allocation_errors_out():
    # MSO infeasible -> domain raises ValueError -> Typer exits non-zero
    result = runner.invoke(app, ["allocate", "--sigma", "0.4", "--sigma", "0.5", "--p-min", "0.95"])
    assert result.exit_code != 0


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("masking", "allocate", "threshold", "capacity", "autonomy", "serve"):
        assert cmd in result.stdout
