import numpy as np
from scipy.stats import norm

from autocall.market import Market, reference_market
from autocall.pricer import Pricing
from autocall.product import Phoenix, payoff
from autocall.vol_surface import flat_surface


def one_stock_market(vol=0.3, rate=0.03):
    return Market(["A"], [flat_surface(vol)], np.eye(1), rate, 0.0, np.zeros(1))


def flows(note, worst):
    f = payoff(note, np.array([worst]), np.ones(len(worst)))
    return f, f["principal"][0] + note.coupon * f["coupon_leg"][0]


def test_no_autocall_before_first_call_date():
    note = Phoenix(coupon=0.12, years=1, first_call=3)
    f, value = flows(note, [1.2, 1.2, 1.1] + [0.9] * 9)
    assert f["called"][0] and f["exit_date"][0] == 2
    assert np.isclose(value, 1 + 3 * 0.01)


def test_capital_loss_below_the_barrier():
    note = Phoenix(coupon=0.12, years=1)
    f, value = flows(note, [0.9] * 11 + [0.4])
    assert f["loss"][0]
    assert np.isclose(value, 0.4 + 11 * 0.01)


def test_digital_matches_black_scholes():
    # Barriers out of reach except the final capital barrier: the note is a bond minus
    # a put spread whose price is known in closed form.
    vol, r, T, B = 0.3, 0.03, 1.0, 0.7
    note = Phoenix(coupon=0.0, coupon_barrier=10, autocall_level=10, capital_barrier=B, years=1)
    pricing = Pricing(note, one_stock_market(vol, r), n_paths=200_000, steps_per_month=1)
    d2 = (np.log(1 / B) + (r - vol**2 / 2) * T) / (vol * np.sqrt(T))
    d1 = d2 + vol * np.sqrt(T)
    put = B * np.exp(-r * T) * norm.cdf(-d2) - norm.cdf(-d1)
    expected = np.exp(-r * T) - put - (1 - B) * np.exp(-r * T) * norm.cdf(-d2)
    assert abs(pricing.price() - expected) < 4 * pricing.std_error()


def test_coupon_solver_hits_the_target_and_blocks_add_up():
    pricing = Pricing(Phoenix(), reference_market(), n_paths=20_000, steps_per_month=2)
    coupon = pricing.solve_coupon(0.98)
    assert np.isclose(pricing.price(coupon), 0.98)
    blocks = pricing.building_blocks(coupon)
    assert np.isclose(blocks.iloc[:4].sum(), 0.98)
