# ADR 0001 — Architecture and quality gate

Status: accepted

## Context
delegation-lab implements the closed-form quantities of the MSO paper. The
domain is essentially pure mathematics (no external I/O), exposed to users
through more than one delivery mechanism.

## Decisions

1. **Functional core, imperative shell.** All math lives in `domain/` as pure,
   stdlib-only functions. Delivery mechanisms (FastAPI, CLI) are thin adapters.
   Rationale: a pure-math domain has nothing to isolate behind hexagonal ports;
   the functional-core pattern gives ~90% of the testability benefit at ~30% of
   the ceremony. Adapters are disposable; the core is the asset.

2. **Errors as domain ValueError, translated at the boundary.** The core raises
   plain `ValueError` for broken contracts (e.g. MSO infeasible). Each adapter
   maps it to its own vocabulary (HTTP 422; non-zero CLI exit). The core never
   imports a web/CLI framework.

3. **Quality gate = ruff (incl. ANN) + mypy strict + pytest.** ANN enforces the
   *presence* of type annotations (including return types); mypy enforces their
   *correctness*. Both run in CI. Tests are exempt from annotation-only rules.

4. **Target Python >= 3.10.** StrEnum (3.11+) is the only 3.11 feature we needed;
   `(str, Enum)` is equivalent and portable, so we widened support to 3.10 and
   test on 3.10/3.11/3.12. `UP042` is suppressed for this reason.

5. **Core stays dependency-free.** `domain/` and `simulation.py` use only the
   stdlib (`math`, `random`, `statistics`). FastAPI, Typer and matplotlib are
   optional extras (`api`, `cli`, `viz`), never imported by the core.
