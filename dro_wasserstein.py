"""
Wasserstein DRO Module (reference implementation).

Penalty forms:
    penalty = rho * ||mu_p - mu_o||_2          (default)
    penalty = rho * ||mu_p - mu_o||_2^2        (if squared=True)

Update rule:
    w' = ΠΔ( max(clip_min, w * (1 - penalty)) )

Helpers:
- estimate_means: column means of returns (T x N)
- stress_mean_by_std: mu_p = mu_o - k * std(R)  (downward stress)
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-12

def projected_simplex(v: np.ndarray, s: float = 1.0) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if s <= 0:
        raise ValueError("s must be > 0")
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    sw = w.sum()
    if sw <= 0 or not np.isfinite(sw):
        raise ValueError("projection failed; input may be too pathological")
    return w / sw

def wasserstein_penalty(mu_p: np.ndarray, mu_o: np.ndarray, rho: float = 0.1, squared: bool = False) -> float:
    mu_p = np.asarray(mu_p, dtype=float).ravel()
    mu_o = np.asarray(mu_o, dtype=float).ravel()
    if mu_p.shape != mu_o.shape:
        raise ValueError("mu_p and mu_o must have the same shape")
    d = np.linalg.norm(mu_p - mu_o)
    return float(rho * (d * d if squared else d))

def dro_wasserstein_update(weights: np.ndarray, mu_p: np.ndarray, mu_o: np.ndarray, rho: float = 0.1, squared: bool = False, clip_min: float = 1e-12) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        raise ValueError("weights must have positive, finite sum")
    w = w / s

    pen = wasserstein_penalty(mu_p, mu_o, rho=rho, squared=squared)
    new_w = np.maximum(w * (1.0 - pen), clip_min)
    return projected_simplex(new_w, s=1.0)

def estimate_means(returns: np.ndarray) -> np.ndarray:
    R = np.asarray(returns, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns must be 2D (T, N)")
    return R.mean(axis=0)

def stress_mean_by_std(returns: np.ndarray, k: float = 1.0, direction: int = -1) -> np.ndarray:
    """Stress mean by +/- k * std per asset (direction=-1 => downward)."""
    R = np.asarray(returns, dtype=float)
    mu_o = estimate_means(R)
    std = R.std(axis=0, ddof=1)
    return mu_o + direction * k * std
