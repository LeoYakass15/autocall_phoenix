"""Implied vol surface (SSVI), its calibration to quotes, and Dupire local vol.

Notation: k = log(K / F_T) is the log forward moneyness and w(k, T) = sigma_implied^2 * T
is the total implied variance.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SSVI:
    """SSVI surface (Gatheral & Jacquier, 2014):

        w(k, T) = theta/2 * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2))
        phi     = eta / (theta^gamma * (1 + theta)^(1 - gamma))

    theta(T) is the at-the-money total variance, linear between the pillars.
    rho sets the skew, eta the curvature. With rho = eta = 0 the surface is flat (Black-Scholes).
    vol_shift moves every implied vol up or down (used for scenarios and vega).
    """

    times: np.ndarray  # pillar maturities
    theta: np.ndarray  # ATM total variance at each pillar
    rho: float = 0.0
    eta: float = 0.0
    gamma: float = 0.4
    vol_shift: float = 0.0

    def atm_variance(self, T):
        T = np.asarray(T, float)
        t_last, theta_last = self.times[-1], self.theta[-1]
        inside = np.interp(T, np.r_[0.0, self.times], np.r_[0.0, self.theta])
        return np.where(T > t_last, theta_last * T / t_last, inside)  # flat ATM vol after the last pillar

    def total_variance(self, k, T):
        theta = np.maximum(self.atm_variance(T), 1e-10)
        phi = self.eta / (theta**self.gamma * (1 + theta) ** (1 - self.gamma))
        w = theta / 2 * (1 + self.rho * phi * k + np.sqrt((phi * k + self.rho) ** 2 + 1 - self.rho**2))
        if self.vol_shift:
            w = np.maximum(np.sqrt(w / T) + self.vol_shift, 0.01) ** 2 * T
        return w

    def implied_vol(self, k, T):
        return np.sqrt(self.total_variance(k, T) / T)


def flat_surface(vol):
    """Constant implied vol (Black-Scholes)."""
    return SSVI(times=np.array([1.0]), theta=np.array([vol**2]))


def calibrate_ssvi(times, k, vols):
    """Fit an SSVI surface to implied vol quotes vols[i, j] (maturity times[i], moneyness k[j]).

    theta is read from the ATM quotes; rho, eta and gamma are fitted by least squares.
    eta is written as 2*s / (1 + |rho|) with s <= 1, which keeps the surface free of
    butterfly arbitrage.
    """
    atm = np.array([np.interp(0.0, k, row) for row in vols])
    theta = np.maximum.accumulate(atm**2 * times)  # ATM variance must increase with maturity

    def surface(params):
        rho, s, gamma = params
        return SSVI(times, theta, rho, 2 * s / (1 + abs(rho)), gamma)

    def errors(params):
        return (surface(params).implied_vol(k[None, :], times[:, None]) - vols).ravel()

    fit = least_squares(errors, x0=[-0.3, 0.5, 0.3], bounds=([-0.99, 0.001, 0.0], [0.99, 1.0, 0.5]))
    return surface(fit.x)


def local_vol(surface, k, T, dk=1e-3, dT=1e-4):
    """Dupire local vol from the implied surface, with Gatheral's formula in total variance:

        sigma_loc^2 = (dw/dT) / g
        g = (1 - k w'/(2w))^2 - w'^2/4 * (1/w + 1/4) + w''/2

    Derivatives by finite differences. g > 0 means no butterfly arbitrage, dw/dT > 0 no
    calendar arbitrage. The result is floored at 5% and capped at 200%.
    """
    k, T = np.broadcast_arrays(np.asarray(k, float), np.asarray(T, float))
    w = surface.total_variance(k, T)
    w_up, w_down = surface.total_variance(k + dk, T), surface.total_variance(k - dk, T)
    w_k = (w_up - w_down) / (2 * dk)
    w_kk = (w_up - 2 * w + w_down) / dk**2
    w_T = (surface.total_variance(k, T + dT) - surface.total_variance(k, np.maximum(T - dT, 1e-6))) / (2 * dT)
    g = (1 - k * w_k / (2 * w)) ** 2 - w_k**2 / 4 * (1 / w + 0.25) + w_kk / 2
    variance = np.nan_to_num(w_T / g, nan=0.05**2)
    return np.sqrt(np.clip(variance, 0.05**2, 2.0**2))
