"""Market data: spots, vol surfaces, correlation, rates, dividends and issuer funding.

The market is built from real data:
  - Bloomberg OVDV screens: implied vols (80% to 120% of spot) and implied forwards,
  - Bloomberg CDS of Barclays: the issuer's funding spread,
  - daily closing prices (Yahoo): the correlation between the three stocks.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .vol_surface import calibrate_ssvi, flat_surface


@dataclass(frozen=True)
class Market:
    """Spots are relative to the note's initial fixing (1.0 at launch).

    Forwards grow at rate - divs, where divs is everything that lowers the forward
    (dividends and stock borrow). Cash flows are discounted at rate + funding_spread:
    the issuer borrows at its own funding rate.
    """

    names: list
    surfaces: list
    corr: np.ndarray
    rate: float
    funding_spread: float
    divs: np.ndarray
    spots: np.ndarray = None
    quotes: pd.DataFrame = None  # Bloomberg quotes used for the calibration, kept for the charts

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

    def flat_wings(self, moneyness=0.8):
        """Same surfaces, but the smile is kept flat below `moneyness` (no extrapolated skew)."""
        return replace(self, surfaces=[replace(s, k_floor=np.log(moneyness)) for s in self.surfaces])

    def without(self, asset):
        keep = [i for i in range(len(self.names)) if i != asset]
        return replace(self, names=[self.names[i] for i in keep], surfaces=[self.surfaces[i] for i in keep],
                       corr=self.corr[np.ix_(keep, keep)], divs=self.divs[keep], spots=self.spots[keep])


# Building the market from the data files ------------------------------------------------------


def load_quotes(path, min_years=1 / 12, max_years=4.0):
    """Bloomberg vol quotes with their maturity T (years) and log forward moneyness k = log(K / F).

    Maturities under one month are dropped (earnings make them noisy) and so are those after
    4 years (the note lasts 3 years).
    """
    q = pd.read_csv(path, parse_dates=["date", "expiry"])
    q["T"] = (q["expiry"] - q["date"]).dt.days / 365.25
    q["k"] = np.log(q["moneyness"] * q["spot"] / q["forward"])
    return q[(q["T"] >= min_years) & (q["T"] <= max_years)].reset_index(drop=True)


def carry(quotes, ticker, years=3.0):
    """Implied carry b = log(F / S) / T read from the Bloomberg forward closest to `years`."""
    q = quotes[(quotes["ticker"] == ticker) & (quotes["moneyness"] == 1.0)]
    row = q.iloc[(q["T"] - years).abs().argmin()]
    return np.log(row["forward"] / row["spot"]) / row["T"]


def historical_correlation(prices, date, days=252):
    """Correlation of daily log returns over the year before `date`."""
    returns = np.log(prices.loc[:date]).diff().dropna().iloc[-days:]
    return returns.corr().to_numpy()


def bloomberg_market(quote_file, prices, date, fixing, rate, cds_bp, horizon=3.0):
    """Market on `date`: SSVI surfaces calibrated to Bloomberg, spots relative to the fixing.

    rate:    USD risk-free rate (discounting)
    cds_bp:  Barclays CDS spread, used as the issuer's funding spread
    horizon: maturity (years) at which the forwards are matched to Bloomberg
    """
    quotes = load_quotes(quote_file)
    names = list(prices.columns)
    surfaces = []
    for name in names:
        q = quotes[quotes["ticker"] == name]
        surfaces.append(calibrate_ssvi(q["T"], q["k"], q["implied_vol"]))
    # dividends + borrow, so that the forwards match Bloomberg at the horizon (can be slightly < 0)
    divs = np.array([rate - carry(quotes, name, horizon) for name in names])
    spots = (prices.loc[date] / fixing).to_numpy()
    return Market(names, surfaces, historical_correlation(prices, date), rate, cds_bp / 1e4, divs, spots, quotes)
