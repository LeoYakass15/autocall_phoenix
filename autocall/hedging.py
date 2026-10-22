"""Back-test of the desk's hedge: sell the note on each launch date, hedge it until it ends.

Three strategies, run side by side on the real price paths:
  - unhedged: the desk just pays the coupons and the redemption;
  - delta:    hold dV/dS of each stock, rebalanced every week;
  - delta + vega: also sell 1-year at-the-money calls on each stock so that the book has
              no vega, then delta-hedge the note and the calls together.

At each rebalancing date the remaining note is re-priced with that day's spots and realised
vols and correlations (Black-Scholes, as at launch). Greeks are bump-and-revalue. Under
Black-Scholes a spot bump just scales the paths and a vol bump reuses the same random
numbers, so all Greeks come from one set of draws: fast, and without Monte Carlo noise
between bumps. Cash earns the issuer funding rate.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .backtest import Settings, launch_dates, observation_dates, realised_market
from .pricer import solve_coupon
from .product import payoff

STRATEGIES = ["Unhedged", "Delta", "Delta + vega"]


def bs_call(s, k, tau, vol, r):
    """Black-Scholes call: price, delta, vega (per 1.00 of vol)."""
    d1 = (np.log(s / k) + (r + vol**2 / 2) * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    price = s * norm.cdf(d1) - k * np.exp(-r * tau) * norm.cdf(d2)
    return price, norm.cdf(d1), s * norm.pdf(d1) * np.sqrt(tau)


def note_greeks(note, market, spots, times, first_obs, coupon, z, bump=0.02):
    """Value, deltas and vegas of the live note, with the observation `times` still to come.

    z: standard normals (n_paths, n_dates, n_assets), reused for every bump.
    """
    vols = np.array([np.sqrt(s.theta[0]) for s in market.surfaces])  # flat surfaces
    dt = np.diff(times, prepend=0.0)
    w = np.cumsum(z[:, : len(times)] @ np.linalg.cholesky(market.corr).T * np.sqrt(dt)[:, None], axis=1)
    discount = market.discount(times)

    def path(i, s, vol):  # (n_paths, n_dates) performance of stock i
        return s * np.exp((market.rate - vol**2 / 2) * times + vol * w[:, :, i])

    def value(stock_paths):
        f = payoff(note, np.minimum.reduce(stock_paths), discount, first_obs)
        return np.mean(f["principal"] + coupon * f["coupon_leg"])

    base = [path(i, spots[i], vols[i]) for i in range(len(spots))]
    pv = value(base)
    deltas, vegas = np.zeros(len(spots)), np.zeros(len(spots))
    for i in range(len(spots)):
        up, down, vol_up = list(base), list(base), list(base)
        up[i], down[i] = base[i] * (1 + bump), base[i] * (1 - bump)
        vol_up[i] = path(i, spots[i], vols[i] + 0.01)
        deltas[i] = (value(up) - value(down)) / (2 * bump * spots[i])
        vegas[i] = (value(vol_up) - pv) / 0.01
    return pv, deltas, vegas


def hedge_one_note(note, prices, launch, coupon, settings=Settings(), rebalance_every=5, n_paths=10_000, seed=42):
    """Desk P&L of each strategy at the end of the note, discounted to launch, in % of notional."""
    obs_dates = observation_dates(prices.index, launch, note.n_obs)
    fixing = prices.loc[launch].to_numpy()
    n = len(fixing)
    funding = settings.rate + settings.funding_spread
    z = np.random.default_rng(seed).standard_normal((n_paths // 2, note.n_obs, n))
    z = np.concatenate([z, -z])

    books = {name: {"cash": 0.0, "stocks": np.zeros(n), "calls": np.zeros(n)} for name in STRATEGIES}
    strike, expiry = np.ones(n), np.zeros(n)
    days = prices.index[(prices.index >= launch) & (prices.index <= obs_dates[-1])]
    rebalance_dates = set(days[::rebalance_every]) | set(obs_dates)
    k, previous, model_value = 0, launch, None  # k = next observation date

    for date in days:
        if date not in rebalance_dates:
            continue
        s = prices.loc[date].to_numpy() / fixing
        t = (date - launch).days / 365.25
        for b in books.values():  # interest on cash since the last rebalancing
            b["cash"] *= np.exp(funding * (date - previous).days / 365.25)
        previous = date
        market = realised_market(prices.loc[:date], settings)
        vols = np.array([np.sqrt(surface.theta[0]) for surface in market.surfaces])
        call_value = bs_call(s, strike, np.maximum(expiry - t, 1e-6), vols, settings.rate)[0]

        if date == obs_dates[k]:  # pay the note's cash flows
            worst = s.min()
            paid = coupon / 12 if worst >= note.coupon_barrier else 0.0
            called, final = worst >= note.call_levels[k], k == note.n_obs - 1
            if called or final:
                paid += worst if (final and not called and worst < note.capital_barrier) else 1.0
            for b in books.values():
                b["cash"] -= paid
            k += 1
            if called or final:  # the note is over: close the hedges
                for b in books.values():
                    b["cash"] += b["stocks"] @ s + b["calls"] @ call_value
                break

        # re-price the remaining note and rebalance
        times = (obs_dates[k:] - date).days.to_numpy() / 365.25
        pv, deltas, vegas = note_greeks(note, market, s, times, k, coupon, z)
        if model_value is None:  # launch: the desk sells the note at its model value
            model_value = pv
            for b in books.values():
                b["cash"] += pv

        # vega hedge: close the old calls, sell new 1-year ATM calls to cancel the note's vega
        vb = books["Delta + vega"]
        vb["cash"] += vb["calls"] @ call_value
        strike, expiry = s.copy(), np.full(n, t + 1.0)
        price, call_delta, call_vega = bs_call(s, strike, 1.0, vols, settings.rate)
        vb["calls"] = vegas / call_vega  # note vega < 0, so the desk sells calls
        vb["cash"] -= vb["calls"] @ price

        # delta hedge: the desk is short the note, so it buys delta of each stock
        for name, target in [("Delta", deltas), ("Delta + vega", deltas - vb["calls"] * call_delta)]:
            b = books[name]
            b["cash"] -= (target - b["stocks"]) @ s
            b["stocks"] = target

    discount = np.exp(-funding * (previous - launch).days / 365.25)
    return {"launch": launch, "end": previous, "model_value": model_value,
            **{name: b["cash"] * discount for name, b in books.items()}}


def run_hedge_backtest(note, prices, settings=Settings(), **kwargs):
    """Same launches as the investor back-test; one row per launch."""
    rows = []
    for launch in launch_dates(prices, settings):
        if observation_dates(prices.index, launch, note.n_obs) is None:
            break
        coupon = solve_coupon(note, realised_market(prices.loc[:launch], settings), settings.target_price,
                              n_paths=settings.n_paths, steps_per_month=1, seed=1)
        rows.append({**hedge_one_note(note, prices, launch, coupon, settings, **kwargs), "coupon": coupon})
    return pd.DataFrame(rows)
