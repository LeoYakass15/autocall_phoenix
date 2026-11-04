"""Correlated Monte Carlo of the basket under local volatility.

Each stock is simulated in log forward moneyness X = log(S_t / F_t):
    dX = -sigma^2/2 dt + sigma dW,   sigma = sigma_loc(t, X)
so the forwards are matched by construction and the local vol is read at k = X.
The Brownian motions are correlated with a Cholesky factor. With a flat surface this is
exactly Black-Scholes.
"""

import numpy as np

from .vol_surface import local_vol

K_GRID = np.linspace(-3.0, 2.0, 251)  # log-moneyness grid on which the local vol is tabulated


def simulate(market, obs_times, n_paths=100_000, steps_per_month=4, seed=42):
    """Performances S_t / S_0 on the observation dates, shape (n_paths, n_dates, n_assets).

    Antithetic variates: the second half of the paths uses the opposite random numbers.
    """
    obs_times = np.asarray(obs_times, float)
    starts = np.r_[0.0, obs_times[:-1]]
    grid = np.r_[0.0, np.concatenate([np.linspace(a, b, steps_per_month + 1)[1:] for a, b in zip(starts, obs_times)])]
    dt = np.diff(grid)
    mid = (grid[:-1] + grid[1:]) / 2

    # local vol tables: one row per time step, one column per point of K_GRID
    tables = [local_vol(s, K_GRID[None, :], mid[:, None]) for s in market.surfaces]
    chol = np.linalg.cholesky(market.corr)
    rng = np.random.default_rng(seed)

    n_assets = len(market.names)
    half = n_paths // 2
    x = np.zeros((2 * half, n_assets))
    out = np.empty((2 * half, len(obs_times), n_assets))
    forwards = market.forward(obs_times)
    obs = 0
    for step, h in enumerate(dt):
        z = rng.standard_normal((half, n_assets))
        dw = np.vstack([z, -z]) @ chol.T * np.sqrt(h)
        sigma = np.column_stack([np.interp(x[:, i], K_GRID, tables[i][step]) for i in range(n_assets)])
        x += -0.5 * sigma**2 * h + sigma * dw
        if (step + 1) % steps_per_month == 0:
            out[:, obs] = forwards[obs] * np.exp(x)
            obs += 1
    return out
