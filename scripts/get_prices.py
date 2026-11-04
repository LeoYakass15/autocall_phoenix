"""Download the daily closing prices of UBER, DASH and COST from Yahoo Finance.

Run it on your computer:
    pip install yfinance
    python3 scripts/get_prices.py
It creates data/prices.csv (run it from the project folder).
"""

from pathlib import Path

import yfinance as yf

TICKERS = ["UBER", "DASH", "COST"]

# Closing prices without dividend adjustment: the note pays on price performance only.
data = yf.download(TICKERS, start="2019-05-01", auto_adjust=False, progress=False)["Close"]
data = data[TICKERS].dropna(how="all")
data.index.name = "Date"

out = Path("data/prices.csv")
data.to_csv(out, float_format="%.4f")
print(f"{len(data)} days, from {data.index[0].date()} to {data.index[-1].date()}")
print(f"Saved in {out}")
