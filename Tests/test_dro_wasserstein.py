import numpy as np
import pytest
from dro_wasserstein import (
    wasserstein_penalty, dro_wasserstein_update,
    estimate_means, stress_mean_by_std
)

def test_penalty_monotonicity():
    rng = np.random.default_rng(0)
    mu_o = np.array([0.01, 0.02, 0.015])
    mu_p_small = mu_o - 0.001
    mu_p_big   = mu_o - 0.01
    p_small = wasserstein_penalty(mu_p_small, mu_o, rho=0.1)
    p_big   = wasserstein_penalty(mu_p_big,   mu_o, rho=0.1)
    assert p_big > p_small

def test_update_invariants_sum1_nonneg():
    w0 = np.array([0.5, 0.3, 0.2])
    mu_o = np.array([0.01, 0.02, 0.015])
    mu_p = mu_o - 0.005
    w1 = dro_wasserstein_update(w0, mu_p, mu_o, rho=0.1)
    assert np.all(w1 >= 0)
    assert np.isclose(w1.sum(), 1.0)

def test_helpers_with_returns():
    rng = np.random.default_rng(1)
    T, N = 1000, 3
    R = rng.normal(0.001, [0.01, 0.012, 0.009], size=(T, N))
    mu_o = estimate_means(R)
    mu_p = stress_mean_by_std(R, k=1.0, direction=-1)
    # penalty should be positive
    p = wasserstein_penalty(mu_p, mu_o, rho=0.2)
    assert p > 0
