"""Price the note, solve the coupon for a target price, and split it into building blocks."""

import numpy as np
import pandas as pd

from .monte_carlo import simulate
from .product import payoff


class Pricing:
    """Result of one Monte Carlo run: price for any coupon, probabilities, building blocks."""

    def __init__(self, note, market, n_paths=100_000, steps_per_month=4, seed=42):
        self.note = note
        paths = simulate(market, note.obs_times, n_paths, steps_per_month, seed)
        self.flows = payoff(note, paths.min(axis=2), market.discount(note.obs_times))

    def values(self, coupon=None):
        """Discounted payoff of each path."""
        c = self.note.coupon if coupon is None else coupon
        return self.flows["principal"] + c * self.flows["coupon_leg"]

    def price(self, coupon=None):
        return self.values(coupon).mean()

    def std_error(self, coupon=None):
        v = self.values(coupon)
        half = len(v) // 2
        pairs = (v[:half] + v[half:]) / 2  # antithetic pairs are independent
        return pairs.std() / np.sqrt(half)

    def solve_coupon(self, target_price):
        """Coupon p.a. that gives the target price.

        On fixed paths the price is linear in the coupon: price(c) = A + c * B,
        with A the value of the principal and B the value of a 100% coupon. So c = (target - A) / B,
        exact, from a single simulation.
        """
        return (target_price - self.flows["principal"].mean()) / self.flows["coupon_leg"].mean()

    def summary(self):
        f = self.flows
        return {
            "price": self.price(),
            "std_error": self.std_error(),
            "autocall_probability": f["called"].mean(),
            "capital_loss_probability": f["loss"].mean(),
            "expected_life_years": ((f["exit_date"] + 1) / 12).mean(),
        }

    def redemption_probabilities(self):
        """Probability that the note ends (autocall or maturity) on each date."""
        return np.bincount(self.flows["exit_date"], minlength=self.note.n_obs) / len(self.flows["exit_date"])

    def building_blocks(self, coupon=None, issue_price=1.0):
        """Note = zero-coupon bond (repaid at the call or maturity date) + digital coupons
        - down-and-in put on the worst-of. With a European barrier B, that put is exactly a put
        struck at B plus a digital put paying (1 - B) below B. The blocks add up to the price."""
        c = self.note.coupon if coupon is None else coupon
        f, b = self.flows, self.note.capital_barrier
        blocks = {
            "Zero-coupon bond (random maturity)": f["zero_coupon"].mean(),
            "Digital coupons": c * f["coupon_leg"].mean(),
            f"Short put struck at {b:.0%}": f["put"].mean(),
            f"Short digital put at {b:.0%}": f["digital_put"].mean(),
        }
        price = sum(blocks.values())
        blocks.update({"Note price": price, "Issuer margin": issue_price - price, "Issue price": issue_price})
        return pd.Series(blocks, name="value")


def solve_coupon(note, market, target_price=0.98, **mc):
    return Pricing(note, market, **mc).solve_coupon(target_price)
