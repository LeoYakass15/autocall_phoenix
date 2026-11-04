"""The real note after launch: what happened so far, what it is worth today, and why.

Launch: 7 November 2025. Valuation: 29 October 2026, with the Bloomberg screens of that day.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from .pricer import Pricing


def schedule(launch, n_obs):
    """Monthly observation dates: each monthly anniversary, moved to the next weekday."""
    dates = [pd.Timestamp(launch) + pd.DateOffset(months=i + 1) for i in range(n_obs)]
    return pd.DatetimeIndex([pd.offsets.BDay().rollforward(d) for d in dates])


def note_history(note, prices, launch, today):
    """One row per observation date already past: performances, worst, coupon paid, autocall."""
    dates = schedule(launch, note.n_obs)
    past = dates[dates <= pd.Timestamp(today)]
    fixing = prices.loc[launch]
    rows = []
    for i, date in enumerate(past):
        day = prices.index[prices.index.searchsorted(date)]  # first trading day on or after the date
        perf = prices.loc[day] / fixing
        worst = perf.min()
        rows.append({"date": day, **perf.to_dict(), "worst": worst,
                     "coupon_paid": note.coupon / 12 if worst >= note.coupon_barrier else 0.0,
                     "autocalled": worst >= note.call_levels[i]})
    return pd.DataFrame(rows)


def remaining(note, launch, today):
    """Remaining observation times (years from today) and number of observations already past."""
    dates = schedule(launch, note.n_obs)
    first_obs = int((dates <= pd.Timestamp(today)).sum())
    times = (dates[first_obs:] - pd.Timestamp(today)).days.to_numpy() / 365.25
    return times, first_obs


def value_bridge(note, launch_market, today_market, times, first_obs, **mc):
    """Move from the launch value to today's value one input at a time (same random numbers).

    1. launch value
    2. time:  today's schedule, launch market (spots at 100%, launch vols, rates, credit)
    3. spots: + today's stock prices
    4. vols:  + today's Bloomberg vol surfaces
    5. rates, credit and correlation of today  =  today's value
    """
    m = launch_market
    t = today_market
    steps = [
        ("Value at launch", m, note.obs_times, 0),
        ("Time passed", m, times, first_obs),
        ("Stock prices", replace(m, spots=t.spots), times, first_obs),
        ("Vol surfaces", replace(m, spots=t.spots, surfaces=t.surfaces), times, first_obs),
        ("Rates, credit, correlation", t, times, first_obs),
    ]
    values = [Pricing(note, mk, times=tt, first_obs=k, **mc).price() for _, mk, tt, k in steps]
    bridge = pd.Series(np.r_[values[0], np.diff(values)], index=[s[0] for s in steps], name="value")
    bridge["Value today"] = values[-1]
    return bridge


def value_vs_level(note, market, times, first_obs, asset, levels, **mc):
    """Today's value of the note and its delta when one stock moves (the others stay put)."""
    values = []
    for level in levels:
        spots = market.spots.copy()
        spots[asset] = level
        values.append(Pricing(note, replace(market, spots=spots), times=times, first_obs=first_obs, **mc).price())
    values = np.array(values)
    return pd.DataFrame({"level": levels, "value": values, "delta": np.gradient(values, levels)})
