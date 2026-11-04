"""Daily P&L explain of the desk on the real Barclays note, from launch to today.

The desk sells the note at its model value and delta-hedges it every day with the three
stocks. Each day, the change in the note's value is split with its Greeks:

    dV = delta * dS + 1/2 gamma * dS^2 + cross-gamma * dS_i * dS_j + vega * dvol
         + correlation sensitivity * dcorr + theta + residual

The desk is short the note and long delta of each stock, so the delta part cancels: the
desk's P&L is minus all the other parts, plus carry.
Carry is the interest on the desk's cash (premium received minus stocks bought).

Vol input: we only have Bloomberg surfaces on two dates, so day to day the note is priced with
Black-Scholes and the trailing realised vol and correlation of each stock (window set in Settings).
"""

import numpy as np
import pandas as pd

from .backtest import Settings, realised_market
from .hedging import note_greeks
from .live_note import schedule


def pnl_explain(note, prices, launch, end, coupon, settings=Settings(), n_paths=20_000, seed=42):
    obs_dates = schedule(launch, note.n_obs)
    fixing = prices.loc[launch].to_numpy()
    days = prices.index[(prices.index >= launch) & (prices.index <= end)]
    funding = settings.rate + settings.funding_spread
    z = np.random.default_rng(seed).standard_normal((n_paths // 2, note.n_obs, len(fixing)))
    z = np.concatenate([z, -z])

    rows, k, prev = [], 0, None
    for j, date in enumerate(days):
        s = prices.loc[date].to_numpy() / fixing
        market = realised_market(prices.loc[:date], settings)
        corr = market.corr[np.triu_indices(len(s), 1)].mean()  # average pairwise correlation

        paid = 0.0  # cash flow of the note today (coupon on an observation date)
        if k < note.n_obs and date >= obs_dates[k]:
            paid = coupon / 12 if s.min() >= note.coupon_barrier else 0.0
            if s.min() >= note.call_levels[k]:
                raise NotImplementedError("the note was autocalled: stop the explain here")
            k += 1

        times = (obs_dates[k:] - date).days.to_numpy() / 365.25
        gap = ((days[j + 1] - date).days if j + 1 < len(days) else 1) / 365.25
        g = note_greeks(note, market, s, times, k, coupon, z, day=gap)

        if prev is not None:
            p = prev
            ds = s - p["s"]
            dt = (date - p["date"]).days / 365.25
            note_pnl = g["value"] + paid - p["value"]
            parts = {
                "delta": p["delta"] @ ds,
                "gamma": 0.5 * p["gamma"] @ ds**2,
                "cross_gamma": 0.5 * ds @ p["cross_gamma"] @ ds,
                "vega": p["vega"] @ (g["vols"] - p["vols"]),
                "correlation": p["corr"] * (corr - p["avg_corr"]),
                "theta": p["theta"],
            }
            residual = note_pnl - sum(parts.values())
            carry = (p["value"] - p["delta"] @ p["s"]) * funding * dt  # interest on the desk's cash
            row = {"date": date, "desk_pnl": -note_pnl + parts["delta"] + carry, "unhedged": -note_pnl}
            row.update({name: -x for name, x in parts.items() if name != "delta"})
            row.update({"carry": carry, "residual": -residual, "note_value": g["value"], "worst": s.min()})
            rows.append(row)
        prev = {**g, "date": date, "s": s, "avg_corr": corr}
    return pd.DataFrame(rows)
