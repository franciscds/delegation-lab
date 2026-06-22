"""Module 1 tests: reproduce the worked example from Azevedo (2026), Section 1.

The paper states explicitly (eq. 5-6 discussion):
  - eta=10, delta=2, skill=0.80, prior=0.0  -> sigma*_raw = 0.667
  - same params with prior=0.5              -> sigma*_raw = 0.750
  - sigma*_raw=0.667, c=0.70                -> sigma*_corr = 0.90, M* = 1.35
"""

import math

import pytest

from delegation_lab.domain.return_operator import (
    corrected_quality,
    masking_index,
    raw_fixed_point,
)

# Paper's exact numbers carry ~3 significant figures; assert to that tolerance.
TOL = 1e-3


def test_raw_fixed_point_conservative_prior():
    # eq. 5 with sigma_0 = 0  ->  eta*skill/(eta+delta) = 8/12
    sigma = raw_fixed_point(0.80, observation_rate=10, forget_rate=2, prior_support=0.0)
    assert math.isclose(sigma, 2 / 3, abs_tol=TOL)


def test_raw_fixed_point_uninformative_prior():
    # eq. 5 with sigma_0 = 0.5  ->  (10*0.8 + 2*0.5)/12 = 9/12 = 0.75
    sigma = raw_fixed_point(0.80, observation_rate=10, forget_rate=2, prior_support=0.5)
    assert math.isclose(sigma, 0.75, abs_tol=TOL)


def test_corrected_quality_and_masking_worked_example():
    raw = raw_fixed_point(0.80, observation_rate=10, forget_rate=2, prior_support=0.0)
    corr = corrected_quality(raw, catch_rate=0.70)  # full review (K/N = 1)
    assert math.isclose(corr, 0.90, abs_tol=TOL)  # paper: sigma*_corr = 0.90
    assert math.isclose(masking_index(raw, corr), 1.35, abs_tol=TOL)  # paper: M* = 1.35


def test_coverage_reduces_correction():
    # With only K/N = 0.5 reviewed, the corrector rescues half as much error mass.
    raw = 0.6
    full = corrected_quality(raw, catch_rate=0.7, coverage=1.0)
    half = corrected_quality(raw, catch_rate=0.7, coverage=0.5)
    assert half < full
    assert math.isclose(half, raw + (1 - raw) * 0.7 * 0.5, abs_tol=1e-12)


def test_masking_index_undefined_at_zero_raw():
    with pytest.raises(ValueError):
        masking_index(0.0, 0.5)


def test_degenerate_rates_rejected():
    with pytest.raises(ValueError):
        raw_fixed_point(0.8, observation_rate=0, forget_rate=0)
