"""Historical back-test: launch the note every month on real prices and see what happened.

At each launch the coupon is re-solved for the target price with a simple Black-Scholes
market built from the past three months of prices (realised vols and correlations), the way
a desk would have priced it that day. The real price path then decides the outcome.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .market import Market
from .pricer import solve_coupon
from .product import payoff
from .vol_surface import flat_surface


@dataclass(frozen=True)
class Settings:
    target_price: float = 0.98
    rate: float = 0.03
    funding_spread: float = 0.01
    window: int = 63  # trading days used for realised vol and correlation
    n_paths: int = 20_000


def load_prices(path, tickers):
    """Daily closes from a CSV with a 'Date' column and one column per ticker."""
    prices = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return prices[tickers].ffill().dropna()


def realised_market(history, settings):
    """Black-Scholes market from the last `window` days of returns."""
    returns = np.log(history).diff().dropna().iloc[-settings.window:]
    vols = returns.std().to_numpy() * np.sqrt(252)
    return Market(list(history.columns), [flat_surface(v) for v in vols], returns.corr().to_numpy(),
                  settings.rate, settings.funding_spread, np.zeros(len(vols)))


def observation_dates(index, launch, n_obs):
    """First trading day on or after each monthly anniversary (None if beyond the data)."""
    targets = [launch + pd.DateOffset(months=i + 1) for i in range(n_obs)]
    pos = index.searchsorted(targets)
    return None if pos[-1] >= len(index) else index[pos]


def launch_dates(prices, settings):
    """First trading day of each month, with enough history before and data after."""
    index = prices.index
    for month in prices.resample("MS").first().index:
        launch = index[index.searchsorted(month)]
        if index.get_loc(launch) >= settings.window:
            yield launch


def irr(times, flows):
    """Annual internal rate of return of an investment of 1 at time 0."""
    return brentq(lambda y: np.sum(flows / (1 + y) ** times) - 1, -0.99, 2.0)


def run_backtest(note, prices, settings=Settings()):
    """One row per launch: coupon offered and what the investor actually got."""
    rows = []
    for launch in launch_dates(prices, settings):
        dates = observation_dates(prices.index, launch, note.n_obs)
        if dates is None:
            break
        coupon = solve_coupon(note, realised_market(prices.loc[:launch], settings), settings.target_price,
                              n_paths=settings.n_paths, steps_per_month=1, seed=1)
        worst = (prices.loc[dates] / prices.loc[launch]).min(axis=1).to_numpy()[None, :]
        f = payoff(note, worst, np.ones(note.n_obs))
        end = f["exit_date"][0]
        times = (dates - launch).days.to_numpy()[: end + 1] / 365.25
        flows = coupon / 12 * f["coupon_paid"][0, : end + 1]
        flows[end] += f["principal"][0]
        outcome = "Autocalled" if f["called"][0] else ("Capital loss" if f["loss"][0] else "Redeemed at par")
        rows.append({"launch": launch, "coupon": coupon, "outcome": outcome, "life_years": times[end],
                     "coupons_paid": int(f["coupon_paid"][0].sum()), "redemption": f["principal"][0],
                     "irr": irr(times, flows)})
    return pd.DataFrame(rows)
