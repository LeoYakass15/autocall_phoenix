import numpy as np
import pandas as pd

from autocall.backtest import Settings, irr, observation_dates
from autocall.hedging import bs_call, hedge_one_note
from autocall.product import Phoenix, payoff


def test_irr_of_a_bond():
    assert np.isclose(irr(np.array([1.0, 2.0]), np.array([0.05, 1.05])), 0.05)


def test_call_greeks_match_finite_differences():
    s, k, tau, vol, r, h = 1.05, 1.0, 0.8, 0.3, 0.03, 1e-4
    _, delta, vega = bs_call(s, k, tau, vol, r)
    assert np.isclose(delta, (bs_call(s + h, k, tau, vol, r)[0] - bs_call(s - h, k, tau, vol, r)[0]) / (2 * h))
    assert np.isclose(vega, (bs_call(s, k, tau, vol + h, r)[0] - bs_call(s, k, tau, vol - h, r)[0]) / (2 * h))


def test_unhedged_pnl_is_model_value_minus_what_the_note_paid():
    rng = np.random.default_rng(5)
    index = pd.bdate_range("2010-01-01", "2014-06-30")
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.015, (len(index), 2)), axis=0)),
                          index=index, columns=["A", "B"])
    note, settings, coupon = Phoenix(), Settings(), 0.15
    launch = index[100]
    result = hedge_one_note(note, prices, launch, coupon, settings, n_paths=2_000)

    dates = observation_dates(index, launch, note.n_obs)
    worst = (prices.loc[dates] / prices.loc[launch]).min(axis=1).to_numpy()[None, :]
    years = (dates - launch).days.to_numpy() / 365.25
    f = payoff(note, worst, np.exp(-(settings.rate + settings.funding_spread) * years))
    paid = f["principal"][0] + coupon * f["coupon_leg"][0]
    assert np.isclose(result["Unhedged"], result["model_value"] - paid)
