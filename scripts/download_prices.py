"""Download daily closes with yfinance into the CSV format used by the back-test.

    pip install yfinance
    python scripts/download_prices.py UBER DASH COST --start 2019-01-01 --out data/prices.csv
    python run_analysis.py --prices data/prices.csv --basket UBER,DASH,COST
"""

import argparse

import yfinance as yf


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--out", default="data/prices.csv")
    args = parser.parse_args()

    closes = yf.download(args.tickers, start=args.start, auto_adjust=False, progress=False)["Close"]
    closes.index.name = "Date"
    closes.dropna(how="all").to_csv(args.out)
    print(f"{len(closes)} rows written to {args.out}")


if __name__ == "__main__":
    main()
