# Bloomberg data (not included)

The Bloomberg screens used by the project are licensed, so they are not published here.
To run the analysis, put two files in this folder, copied from the OVDV screen (Mid, Listed, moneyness):

- `vols_2025-11-07.csv`: implied vols of UBER, DASH and COST on the launch date
- `vols_2026-10-29.csv`: the same on the valuation date

One row per quote:

```
date,ticker,expiry,spot,forward,moneyness,implied_vol
2025-11-07,ABC,2026-01-16,100.0,101.0,0.9,0.3500
```

`moneyness` is strike / spot (0.8 to 1.2), `implied_vol` is in decimals, `forward` is the implied forward of that expiry.
