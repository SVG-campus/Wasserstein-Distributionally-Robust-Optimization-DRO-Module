import numpy as np
from dro_wasserstein import wasserstein_penalty, dro_wasserstein_update, estimate_means, stress_mean_by_std

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, N = 1500, 3
    R = rng.normal(0.001, [0.01, 0.012, 0.009], size=(T, N))

    mu_o = estimate_means(R)
    mu_p = stress_mean_by_std(R, k=1.0, direction=-1)

    w0 = np.array([1/3, 1/3, 1/3])
    pen = wasserstein_penalty(mu_p, mu_o, rho=0.1)
    w1  = dro_wasserstein_update(w0, mu_p, mu_o, rho=0.1)

    print("penalty:", pen)
    print("w1:", w1, "sum:", w1.sum())
