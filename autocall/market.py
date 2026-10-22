"""Market data: spots, vol surfaces, correlation, rates, dividends and issuer funding."""

from dataclasses import dataclass, replace

import numpy as np

from .vol_surface import SSVI, calibrate_ssvi, flat_surface


@dataclass(frozen=True)
class Market:
    """Spots are relative to the initial fixing (1.0 at launch).

    Forwards grow at rate - dividend yield. The note's cash flows are discounted at
    rate + funding_spread: the issuer borrows at its own funding rate.
    """

    names: list
    surfaces: list
    corr: np.ndarray
    rate: float
    funding_spread: float
    divs: np.ndarray
    spots: np.ndarray = None
    quotes: list = None  # (times, k, vols) used for the calibration, kept for the charts

    def __post_init__(self):
        if self.spots is None:
            object.__setattr__(self, "spots", np.ones(len(self.names)))

    def discount(self, t):
        return np.exp(-(self.rate + self.funding_spread) * np.asarray(t))

    def forward(self, t):
        """Forwards (len(t), n_assets), relative to the initial fixing."""
        t = np.atleast_1d(t)[:, None]
        return self.spots * np.exp((self.rate - self.divs) * t)

    # Scenarios used by the Greeks and the coupon levers

    def bump_vol(self, shift, asset=None):
        surfaces = [replace(s, vol_shift=s.vol_shift + shift) if asset in (None, i) else s
                    for i, s in enumerate(self.surfaces)]
        return replace(self, surfaces=surfaces)

    def bump_spot(self, factor, asset):
        spots = self.spots.copy()
        spots[asset] *= factor
        return replace(self, spots=spots)

    def bump_corr(self, shift):
        corr = np.clip(self.corr + shift, -0.99, 0.99)
        np.fill_diagonal(corr, 1.0)
        return replace(self, corr=corr)

    def with_corr(self, rho):
        corr = np.full_like(self.corr, rho)
        np.fill_diagonal(corr, 1.0)
        return replace(self, corr=corr)

    def flat_vols(self, vols):
        """Black-Scholes version of the market: one constant vol per stock."""
        return replace(self, surfaces=[flat_surface(v) for v in vols])

    def without(self, asset):
        keep = [i for i in range(len(self.names)) if i != asset]
        return replace(self, names=[self.names[i] for i in keep], surfaces=[self.surfaces[i] for i in keep],
                       corr=self.corr[np.ix_(keep, keep)], divs=self.divs[keep], spots=self.spots[keep])


# Stylised market for UBER / DASH / COST (November 2025) -----------------------------------
# No option data feed here: quotes are generated from these reference surfaces plus noise,
# then calibrated exactly as real quotes would be.

QUOTE_TIMES = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
QUOTE_K = np.log([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5])

#                ATM vol on QUOTE_TIMES                    rho    eta
REFERENCE = {
    "UBER": ([0.40, 0.38, 0.36, 0.345, 0.335, 0.325], -0.45, 1.0),
    "DASH": ([0.46, 0.44, 0.42, 0.400, 0.390, 0.375], -0.40, 1.0),
    "COST": ([0.21, 0.21, 0.21, 0.212, 0.215, 0.220], -0.60, 1.0),
}
CORRELATION = np.array([[1.0, 0.55, 0.30], [0.55, 1.0, 0.25], [0.30, 0.25, 1.0]])


def reference_market(rate=0.037, funding_spread=0.010, divs=(0.0, 0.0, 0.0055), noise=0.003, seed=7):
    rng = np.random.default_rng(seed)
    surfaces, quotes = [], []
    for atm, rho, eta in REFERENCE.values():
        true = SSVI(QUOTE_TIMES, np.array(atm) ** 2 * QUOTE_TIMES, rho, eta)
        vols = true.implied_vol(QUOTE_K[None, :], QUOTE_TIMES[:, None])
        vols = vols + noise * rng.standard_normal(vols.shape)
        surfaces.append(calibrate_ssvi(QUOTE_TIMES, QUOTE_K, vols))
        quotes.append((QUOTE_TIMES, QUOTE_K, vols))
    return Market(list(REFERENCE), surfaces, CORRELATION, rate, funding_spread, np.array(divs), quotes=quotes)
