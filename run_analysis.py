"""Run every study on the Barclays Phoenix note and save the charts and tables in results/.

Part 1  Launch (7 Nov 2025): calibrate the vols, price the note, building blocks, coupon levers
Part 2  Today (29 Oct 2026): what happened so far, value today, Greeks
Part 3  History: back-test of the note and of the desk's hedge on UBER / DASH / COST
Part 4  Daily P&L explain of the desk on the real note since launch
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from autocall import plots
from autocall.backtest import Settings, run_backtest
from autocall.hedging import STRATEGIES, run_hedge_backtest
from autocall.live_note import note_history, remaining, value_bridge, value_vs_level
from autocall.market import bloomberg_market
from autocall.pnl_explain import pnl_explain
from autocall.pricer import Pricing
from autocall.product import Phoenix
from autocall.risks import coupon_grid, coupon_levers, greeks

# --- Inputs ---------------------------------------------------------------------------------
LAUNCH, TODAY = pd.Timestamp("2025-11-07"), pd.Timestamp("2026-10-29")
BARCLAYS_VALUE = 0.952  # Barclays' estimated value, from the pricing supplement

# Rates: at launch, read from COST's Bloomberg forward plus its 0.465% dividend yield;
# today, the 3-year USD SOFR swap (USOSFR3, 4.52%).
RATE_LAUNCH, RATE_TODAY = 0.0377, 0.0452
# Issuer funding: Barclays 5-year CDS (Bloomberg), in bp.
CDS_LAUNCH, CDS_TODAY = 71.5, 74.6

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--quick", action="store_true", help="fewer paths, for a fast check")
args = parser.parse_args()
n = 20_000 if args.quick else 100_000

out = Path("results")
out.mkdir(exist_ok=True)
start = time.time()


def step(message):
    print(f"[{time.time() - start:5.0f}s] {message}")


prices = pd.read_csv("data/prices.csv", parse_dates=["Date"]).set_index("Date")
fixing = prices.loc[LAUNCH]  # initial levels of the note = closing prices on the launch date
note = Phoenix()  # Barclays term sheet: 18.25% coupon, barriers 60% / 50%, autocall 100% from month 6

# --- Part 1: launch -------------------------------------------------------------------------
step("Calibrating the vol surfaces on Bloomberg (7 Nov 2025)")
market = bloomberg_market("data/bloomberg/vols_2025-11-07.csv", prices, LAUNCH, fixing, RATE_LAUNCH, CDS_LAUNCH)
fit = []
for name, surface in zip(market.names, market.surfaces):
    q = market.quotes[market.quotes["ticker"] == name]
    error = surface.implied_vol(q["k"].to_numpy(), q["T"].to_numpy()) - q["implied_vol"].to_numpy()
    fit.append({"stock": name, "rho": surface.rho, "eta": surface.eta, "gamma": surface.gamma,
                "rmse_vol_pts": 100 * np.sqrt(np.mean(error**2)), "quotes": len(q),
                "atm_vol_3y": surface.implied_vol(0.0, 3.0), "vol_at_50pct_3y": surface.implied_vol(np.log(0.5), 3.0)})
fit = pd.DataFrame(fit)
fit.to_csv(out / "calibration.csv", index=False)
print(fit.round(3).to_string(index=False))
plots.implied_surfaces(market, out / "01_vol_surfaces.png")
plots.smiles_with_barriers(market, out / "02_smiles_vs_bloomberg.png")
plots.local_vs_implied(market, out / "03_local_vs_implied_vol.png")

step("Pricing at launch")
pricing = Pricing(note, market, n_paths=n)
summary = pricing.summary()
print(f"  price at 18.25%: {summary['price']:.2%} (+/- {1.96 * summary['std_error']:.2%}), Barclays {BARCLAYS_VALUE:.1%}")
atm = [s.implied_vol(0.0, 3.0) for s in market.surfaces]
at_barrier = [s.implied_vol(np.log(note.capital_barrier), 3.0) for s in market.surfaces]
models = {"Black-Scholes\nATM vol": market.flat_vols(atm),
          "Black-Scholes\nvol at 50%": market.flat_vols(at_barrier),
          "Local vol": market,
          "Local vol, smile\nflat below 80%": market.flat_wings(0.8)}
runs = {name: Pricing(note, m, n_paths=n) for name, m in models.items()}
comparison = pd.DataFrame({"price": {k: r.price() for k, r in runs.items()},
                           f"fair coupon at {BARCLAYS_VALUE:.1%}": {k: r.solve_coupon(BARCLAYS_VALUE) for k, r in runs.items()}})
comparison.rename(index=lambda s: s.replace("\n", " ")).to_csv(out / "model_comparison.csv")
print(comparison.rename(index=lambda s: s.replace("\n", " ")).round(4).to_string())
plots.model_prices(comparison["price"], BARCLAYS_VALUE, out / "04_model_prices.png")

step("Building blocks and redemption profile")
blocks = pricing.building_blocks()
blocks.to_csv(out / "building_blocks.csv")
plots.waterfall(blocks, out / "05_building_blocks.png")
plots.redemption(pricing, out / "06_redemption_profile.png")
pd.Series(summary).to_csv(out / "pricing_summary.csv")

step("Coupon levers (coupon re-solved at Barclays' value)")
base, levers = coupon_levers(note, market, BARCLAYS_VALUE, n_paths=n // 2)
levers.to_csv(out / "coupon_levers.csv", index=False)
plots.tornado(base, levers, out / "07_coupon_levers.png")
grid = coupon_grid(note, market, [-0.10, -0.05, 0.0, 0.05, 0.10], [0.0, 0.2, 0.4, 0.6, 0.8], BARCLAYS_VALUE,
                   n_paths=n // 4, steps_per_month=2)
grid.to_csv(out / "coupon_vol_correlation.csv")
plots.coupon_heatmap(grid, out / "08_coupon_vol_correlation.png")

# --- Part 2: today --------------------------------------------------------------------------
step("The note today (29 Oct 2026)")
history = note_history(note, prices, LAUNCH, TODAY)
history.to_csv(out / "note_history.csv", index=False)
plots.note_since_launch(prices.loc[:TODAY], history, LAUNCH, out / "09_note_since_launch.png")
today = bloomberg_market("data/bloomberg/vols_2026-10-29.csv", prices, TODAY, fixing, RATE_TODAY, CDS_TODAY, horizon=2.0)
times, done = remaining(note, LAUNCH, TODAY)
live = Pricing(note, today, n_paths=n, times=times, first_obs=done)
coupons_received = history["coupon_paid"].sum()
print(f"  value today {live.price():.2%}, coupons received {coupons_received:.2%}")
pd.Series({**live.summary(), "coupons_received": coupons_received}).to_csv(out / "value_today.csv")
bridge = value_bridge(note, market, today, times, done, n_paths=n // 2)
bridge.to_csv(out / "value_bridge.csv")
plots.value_bridge(bridge, coupons_received, out / "10_value_bridge.png")
print(bridge.round(4).to_string())

step("Greeks of the desk today")
risk = greeks(note, today, n_paths=n // 2, steps_per_month=2, times=times, first_obs=done)
risk.to_csv(out / "greeks_today.csv", index=False)
print(risk.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
worst = int(np.argmin(today.spots))
profile = value_vs_level(note, today, times, done, worst, np.linspace(0.4, 1.0, 25), n_paths=n // 4, steps_per_month=2)
profile.to_csv(out / "value_vs_uber.csv", index=False)
plots.value_vs_level(profile, today.names[worst], today.spots[worst], out / "11_value_vs_uber.png")

# --- Part 3: history ------------------------------------------------------------------------
# DASH is listed since December 2020: launches from March 2021 (3 months of history needed),
# until the last note that has fully ended. Flat 3% rate and 0.75% funding spread.
step("Historical back-test of the note on UBER / DASH / COST")
settings = Settings(target_price=BARCLAYS_VALUE, rate=0.03, funding_spread=0.0075)
basket = prices.dropna()
backtest = run_backtest(note, basket, settings)
backtest.to_csv(out / "backtest.csv", index=False)
plots.backtest(backtest, out / "12_backtest.png")
print(backtest.groupby("outcome")["irr"].agg(["count", "mean"]).to_string())

step("Hedging back-test (unhedged, delta, delta + vega)")
hedged = run_hedge_backtest(note, basket, settings, n_paths=2_000 if args.quick else 10_000)
hedged.to_csv(out / "hedge_backtest.csv", index=False)
plots.hedge_backtest(hedged, out / "13_hedge_backtest.png")
print(hedged[STRATEGIES].agg(["mean", "std", "min"]).T.to_string(float_format=lambda v: f"{v:+.2%}"))

# --- Part 4: daily P&L explain --------------------------------------------------------------
step("Daily P&L explain of the desk on the real note")
# Vol input: 1-year realised vol, smoother day to day than a 3-month window (closer to how implied vol moves).
explain = pnl_explain(note, prices, LAUNCH, TODAY, note.coupon,
                      Settings(rate=RATE_LAUNCH, funding_spread=CDS_LAUNCH / 1e4, window=252),
                      n_paths=4_000 if args.quick else 20_000)
explain.to_csv(out / "pnl_explain.csv", index=False)
plots.pnl_explain(explain, out / "14_pnl_explain.png")
print(explain.drop(columns=["date", "note_value", "worst"]).sum().to_string(float_format=lambda v: f"{v:+.2%}"))

step("Done, results in results/")
