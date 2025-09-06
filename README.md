# Wasserstein Distributionally Robust Optimization (DRO) Module (Paper 1)

This repository contains a production‑ready implementation of a **Wasserstein DRO penalty** for portfolio updates. The idea is to hedge against distribution shift by penalizing divergence between the historical mean return vector and a stressed (perturbed) mean.

**Penalty (generic form)**
`penalty = rho * || mu_p - mu_o ||_2`
Use `squared=True` to apply `rho * ||·||_2^2` instead.

**Weight update**
`w' = Π_Δ( max(clip_min, w * (1 - penalty)) )`

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python examples.py
```

## Usage

```python
import numpy as np
from dro_wasserstein import (
    wasserstein_penalty, dro_wasserstein_update,
    estimate_means, stress_mean_by_std
)

# toy returns (T x N)
rng = np.random.default_rng(0)
T, N = 1500, 3
R = rng.normal(0.001, [0.01, 0.012, 0.009], size=(T, N))

mu_o = estimate_means(R)
mu_p = stress_mean_by_std(R, k=1.0)  # stress one std-dev downward
w0 = np.array([1/3, 1/3, 1/3])

pen = wasserstein_penalty(mu_p, mu_o, rho=0.1, squared=False)
w1  = dro_wasserstein_update(w0, mu_p, mu_o, rho=0.1)

print("penalty:", pen)
print("w1:", w1, "sum:", w1.sum())
```

## Files included

* `dro_wasserstein.py` — penalty, simplex projection, helpers, and update.
* `tests/test_dro_wasserstein.py` — invariants and monotonicity checks.
* `tests/test_artifacts_exist.py` — checks presence of the paper PDF and Test.zip (skips gracefully if missing).
* `.github/workflows/ci.yml` — run tests on push/PR.
* `.github/workflows/release.yml` — GitHub Release on tags (works with Zenodo integration).
* `CITATION.cff` — includes your ORCID.
* `.zenodo.json` — Zenodo deposition metadata.
* `requirements.txt`, `examples.py`, `CHANGELOG.md`, `LICENSE-CODE`, `LICENSE-DOCS`, `.gitignore`.

## ORCID & Zenodo

* Author ORCID: **[https://orcid.org/0009-0004-9601-5617](https://orcid.org/0009-0004-9601-5617)**.
* With GitHub↔Zenodo connected, pushing a tag (e.g., `v0.1.0`) will mint a DOI.

**Publish checklist**

1. Commit code, paper, and tests.
2. Update versions in `CHANGELOG.md` and `CITATION.cff`.
3. Tag a release: `git tag v0.1.0 && git push --tags`.
4. When DOI appears on Zenodo, add badge here and add to `CITATION.cff -> identifiers`.
5. Ensure the DOI shows in your ORCID Works (add manually if needed).

## Citing

See `CITATION.cff`. Replace the badge DOI after first release.
