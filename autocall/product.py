"""The Phoenix note: its terms and its payoff on simulated (or historical) paths."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Phoenix:
    """Worst-of Phoenix autocallable, per unit of notional, with monthly observations.

    Each month, with W the worst performance of the basket (S / S0):
      - coupon: paid if W >= coupon_barrier;
      - autocall: from month `first_call`, if W >= autocall_level the note repays 100% and stops;
      - at maturity, if never called: 100% if W >= capital_barrier, otherwise W (capital loss).

    Default terms: Barclays Phoenix AutoCallable Notes due 2028 on UBER / DASH / COST.
    """

    coupon: float = 0.1825  # per annum
    coupon_barrier: float = 0.60
    autocall_level: float = 1.00
    capital_barrier: float = 0.50
    years: int = 3
    first_call: int = 6  # first month with an autocall observation

    @property
    def n_obs(self):
        return 12 * self.years

    @property
    def obs_times(self):
        return np.arange(1, self.n_obs + 1) / 12

    @property
    def call_levels(self):
        """Autocall level on each date (infinite = no autocall that month)."""
        months = np.arange(1, self.n_obs + 1)
        return np.where(months >= self.first_call, self.autocall_level, np.inf)


def payoff(note: Phoenix, worst, discount, first_obs=0):
    """Discounted cash flows of the note on each path.

    worst:     (n_paths, n_dates) worst performance on the remaining observation dates
    discount:  (n_dates,) discount factor to each date
    first_obs: number of observation dates already past (used when hedging a live note)

    The coupon leg is returned per 1 unit of coupon, so that for any coupon c:
        value = principal + c * coupon_leg
    The principal is split into a zero-coupon bond and the short down-and-in put.
    """
    n_paths, n_dates = worst.shape
    call_levels = note.call_levels[first_obs:]
    dates = np.arange(n_dates)

    called = (worst >= call_levels).any(axis=1)
    exit_date = np.where(called, (worst >= call_levels).argmax(axis=1), n_dates - 1)
    alive = dates <= exit_date[:, None]

    coupon_paid = (worst >= note.coupon_barrier) & alive
    coupon_leg = (coupon_paid * discount).sum(axis=1) / 12

    final = worst[:, -1]
    loss = ~called & (final < note.capital_barrier)
    barrier = note.capital_barrier
    zero_coupon = discount[exit_date]
    put = -np.where(loss, barrier - final, 0.0) * discount[-1]  # put struck at the barrier
    digital_put = -np.where(loss, 1 - barrier, 0.0) * discount[-1]  # gap between barrier and 100%

    return {
        "principal": zero_coupon + put + digital_put,
        "coupon_leg": coupon_leg,
        "zero_coupon": zero_coupon,
        "put": put,
        "digital_put": digital_put,
        "coupon_paid": coupon_paid,
        "called": called,
        "loss": loss,
        "exit_date": exit_date,
    }
