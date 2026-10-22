"""What moves the coupon (structuring view) and what risks the desk keeps (trading view).

Every number is a bump-and-revalue with the same random numbers (same seed), so the
differences come from the bump, not from Monte Carlo noise.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from .pricer import Pricing, solve_coupon


def coupon_levers(note, market, target_price=0.98, **mc):
    """Fair coupon at a fixed price after one change of market or term sheet."""
    m, n = market, note
    scenarios = {
        "Volatility +5 pts": (n, m.bump_vol(0.05)),
        "Skew x1.5": (n, replace(m, surfaces=[replace(s, rho=1.5 * s.rho) for s in m.surfaces])),
        "Correlation +10 pts": (n, m.bump_corr(0.10)),
        "Dividends +1%": (n, replace(m, divs=m.divs + 0.01)),
        "Interest rates +50 bp": (n, replace(m, rate=m.rate + 0.005)),
        "Issuer funding spread +50 bp": (n, replace(m, funding_spread=m.funding_spread + 0.005)),
        "Capital barrier 50% -> 60%": (replace(n, capital_barrier=0.60), m),
        "Coupon barrier 60% -> 70%": (replace(n, coupon_barrier=0.70), m),
        "Autocall level 100% -> 90%": (replace(n, autocall_level=0.90), m),
        "Non-call period 6 -> 12 months": (replace(n, first_call=12), m),
        "Maturity 3 -> 4 years": (replace(n, years=4), m),
        "Two stocks instead of three (no COST)": (n, m.without(2)),
    }
    base = solve_coupon(note, market, target_price, **mc)
    rows = [{"lever": name, "coupon": solve_coupon(nn, mm, target_price, **mc)} for name, (nn, mm) in scenarios.items()]
    table = pd.DataFrame(rows)
    table["change"] = table["coupon"] - base
    return base, table


def coupon_grid(note, market, vol_shifts, correlations, target_price=0.98, **mc):
    """Fair coupon for a parallel vol shift (columns) and a uniform correlation (rows)."""
    return pd.DataFrame(
        [[solve_coupon(note, market.with_corr(rho).bump_vol(dv), target_price, **mc) for dv in vol_shifts]
         for rho in correlations],
        index=pd.Index(correlations, name="correlation"),
        columns=pd.Index(vol_shifts, name="vol shift"),
    )


def greeks(note, market, **mc):
    """Sensitivities of the note (central bumps), in % of notional, and the desk's side.

    Delta is dV / d(S/S0): the stock position, as a fraction of notional, that hedges the note.
    """
    def change(up, down, size):
        return (Pricing(note, up, **mc).price() - Pricing(note, down, **mc).price()) / (2 * size)

    m, rows = market, []
    for i, name in enumerate(m.names):
        rows.append(("Delta", name, change(m.bump_spot(1.01, i), m.bump_spot(0.99, i), 0.01)))
    for i, name in enumerate(m.names):
        rows.append(("Vega, per vol pt", name, change(m.bump_vol(0.01, i), m.bump_vol(-0.01, i), 1)))
    rows += [
        ("Correlation, per pt", "all pairs", change(m.bump_corr(0.01), m.bump_corr(-0.01), 1)),
        ("Dividends, per 10 bp", "all", change(replace(m, divs=m.divs + 0.001), replace(m, divs=m.divs - 0.001), 1)),
        ("Rates, per bp", "", change(replace(m, rate=m.rate + 1e-4), replace(m, rate=m.rate - 1e-4), 1)),
    ]
    table = pd.DataFrame(rows, columns=["greek", "stock", "note"])
    table["desk"] = -table["note"]  # the desk is short the note
    return table
