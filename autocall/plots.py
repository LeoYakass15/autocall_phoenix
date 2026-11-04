"""Charts saved as PNG in results/."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from .vol_surface import local_vol  # noqa: E402

BLUE, RED, GREY, ORANGE, GREEN = "#2a78d6", "#e34948", "#9a9993", "#eb6834", "#1baf7a"
STOCK_COLORS = [BLUE, ORANGE, GREEN]
OUTCOME_COLORS = {"Autocalled": "#0ca30c", "Redeemed at par": "#fab219", "Capital loss": "#d03b3b"}
BARRIERS = {"capital barrier 50%": 0.5, "coupon barrier 60%": 0.6, "autocall 100%": 1.0}
PCT = PercentFormatter(1.0)

plt.rcParams.update({"axes.grid": True, "grid.color": "#e4e3df", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
                     "font.size": 9, "legend.frameon": False})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def implied_surfaces(market, path):
    """3D implied vol surface of each stock (SSVI fit) with the Bloomberg quotes as dots."""
    strikes, times = np.meshgrid(np.linspace(0.45, 1.3, 50), np.linspace(0.25, 3.5, 50))
    fig = plt.figure(figsize=(14, 4.6))
    for i, (name, surface) in enumerate(zip(market.names, market.surfaces)):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        k = np.log(strikes) - (market.rate - market.divs[i]) * times
        ax.plot_surface(strikes, times, surface.implied_vol(k, times), cmap="Blues", alpha=0.9)
        q = market.quotes[market.quotes["ticker"] == name]
        ax.scatter(q["moneyness"], q["T"], q["implied_vol"], color="black", s=4)
        ax.set(title=f"{name}: SSVI fit (dots: Bloomberg)", xlabel="strike / spot", ylabel="maturity (y)")
        ax.zaxis.set_major_formatter(PCT)
        ax.view_init(elev=22, azim=-125)
    save(fig, path)


def local_vs_implied(market, path, T=1.0):
    strikes = np.linspace(0.4, 1.6, 200)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for i, ax in enumerate(axes):
        k = np.log(strikes) - (market.rate - market.divs[i]) * T
        ax.plot(strikes, market.surfaces[i].implied_vol(k, T), color=STOCK_COLORS[i], label="implied vol")
        ax.plot(strikes, local_vol(market.surfaces[i], k, T), color=STOCK_COLORS[i], ls="--", label="local vol")
        ax.set(title=f"{market.names[i]}, 1-year smile", xlabel="strike / initial")
        ax.xaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_formatter(PCT)
        ax.legend()
    save(fig, path)


def smiles_with_barriers(market, path, maturities=(0.25, 1.0, 3.0)):
    """Calibrated smiles (lines) against the Bloomberg quotes (dots), with the note's barriers.

    Bloomberg quotes stop at 80% of spot: below that, the smile is extrapolated by SSVI.
    """
    strikes = np.linspace(0.35, 1.3, 200)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=False)
    for i, ax in enumerate(axes):
        q = market.quotes[market.quotes["ticker"] == market.names[i]]
        for T, color in zip(maturities, ["#86b6ef", "#3987e5", "#184f95"]):
            T_quote = q["T"].iloc[(q["T"] - T).abs().argmin()]  # closest Bloomberg maturity
            dots = q[q["T"] == T_quote]
            k = np.log(strikes) - (market.rate - market.divs[i]) * T_quote
            ax.plot(strikes, market.surfaces[i].implied_vol(k, T_quote), color=color, label=f"{T_quote:.2f}y")
            ax.scatter(dots["moneyness"], dots["implied_vol"], color=color, s=14, zorder=3)
        ax.axvspan(0.35, 0.8, color=GREY, alpha=0.08)
        for label, level in BARRIERS.items():
            ax.axvline(level, color=GREY, lw=1, ls=":")
            if i == 0:
                ax.annotate(label, (level, 0.02), xycoords=("data", "axes fraction"), rotation=90,
                            va="bottom", ha="right", fontsize=7)
        ax.set(title=f"{market.names[i]} (shaded: no quotes, extrapolated)", xlabel="strike / spot")
        ax.xaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_formatter(PCT)
    axes[-1].legend(title="maturity")
    save(fig, path)


def model_prices(prices, issuer_value, path):
    """Note price at the term sheet coupon under each model, against Barclays' estimate."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [GREY, GREY, BLUE, "#86b6ef"]
    bars = ax.bar(prices.index, prices.values, color=colors[: len(prices)], width=0.6)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in prices.values], fontsize=9)
    ax.axhline(issuer_value, color=RED, ls="--", lw=1.2, label=f"Barclays' estimated value {issuer_value:.1%}")
    ax.set(title="Price of the note at launch (coupon 18.25%) by model", ylabel="% of notional",
           ylim=(0.8, 1.0))
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="upper right")
    save(fig, path)


def waterfall(blocks, path):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    level = 0.0
    for i, (name, v) in enumerate(blocks.iloc[:4].items()):
        ax.bar(i, v, bottom=level, color=BLUE if v >= 0 else RED, width=0.6)
        ax.annotate(f"{v:+.1%}", (i, level + max(v, 0)), ha="center", va="bottom", fontsize=8)
        level += v
    price, margin = blocks["Note price"], blocks["Issuer margin"]
    ax.bar(4, price, color=GREY, width=0.6)
    ax.bar(5, margin, bottom=price, color="#86b6ef", width=0.6)
    ax.bar(6, price + margin, color=GREY, width=0.6)
    for i, v, top in [(4, price, price), (5, margin, price + margin), (6, price + margin, price + margin)]:
        ax.annotate(f"{v:.1%}", (i, top), ha="center", va="bottom", fontsize=8)
    labels = [name.replace(" (", "\n(").replace("Short ", "Short\n") for name in blocks.index]
    ax.set_xticks(range(7), labels, fontsize=8)
    ax.set_ylim(0, blocks.iloc[:4].clip(lower=0).sum() * 1.08)
    ax.set(title="What the investor buys: building blocks of the note", ylabel="% of notional")
    ax.yaxis.set_major_formatter(PCT)
    save(fig, path)


def redemption(pricing, path):
    s = pricing.summary()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(np.arange(1, pricing.note.n_obs + 1), pricing.redemption_probabilities(), color=BLUE)
    ax.annotate(f"P(autocall) {s['autocall_probability']:.0%}   P(capital loss) {s['capital_loss_probability']:.0%}"
                f"   expected life {s['expected_life_years']:.1f}y", (0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top")
    ax.set(title="When does the note end?", xlabel="month", ylabel="probability")
    ax.yaxis.set_major_formatter(PCT)
    save(fig, path)


def tornado(base, levers, path):
    data = levers.reindex(levers["change"].abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.barh(data["lever"], data["change"], color=[BLUE if v >= 0 else RED for v in data["change"]])
    ax.bar_label(bars, labels=[f"{v:+.1%}" for v in data["change"]], fontsize=8, padding=3)
    ax.axvline(0, color="black", lw=0.8)
    lim = data["change"].abs().max() * 1.3
    ax.set(title=f"What moves the coupon (base {base:.1%} p.a. at a fixed price)", xlim=(-lim, lim),
           xlabel="change in fair coupon p.a.")
    ax.xaxis.set_major_formatter(PCT)
    ax.grid(axis="y", visible=False)
    save(fig, path)


def coupon_heatmap(grid, path):
    values = grid.to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    image = ax.imshow(values, cmap="Blues", origin="lower", aspect="auto")
    for (r, c), v in np.ndenumerate(values):
        ax.text(c, r, f"{v:.1%}", ha="center", va="center", fontsize=8, color="white" if v > values.mean() else "black")
    ax.set_xticks(range(len(grid.columns)), [f"{v:+.0%}" for v in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f"{v:.0%}" for v in grid.index])
    ax.set(title="Fair coupon p.a.: volatility vs correlation", xlabel="parallel vol shift", ylabel="correlation")
    ax.grid(False)
    fig.colorbar(image, ax=ax, format=PCT)
    save(fig, path)


def backtest(results, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for outcome, color in OUTCOME_COLORS.items():
        sel = results[results["outcome"] == outcome]
        if len(sel):
            axes[0].scatter(sel["launch"], sel["irr"], color=color, s=22, label=f"{outcome} ({len(sel)})")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(title="Back-test: investor return (IRR) by launch date", ylabel="IRR p.a.")
    axes[0].legend(loc="upper right")
    axes[1].plot(results["launch"], results["coupon"], color=BLUE)
    axes[1].set(title="Coupon offered at launch", ylabel="coupon p.a.")
    for ax in axes:
        ax.yaxis.set_major_formatter(PCT)
    save(fig, path)


def hedge_backtest(results, path):
    colors = {"Unhedged": GREY, "Delta": ORANGE, "Delta + vega": BLUE}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.4, 1]})
    for name, color in colors.items():
        axes[0].plot(results["launch"], results[name], color=color, lw=1.4, marker="o", ms=3, label=name)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(title="Desk P&L at the end of each note (sold at model value)", ylabel="% of notional")
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].legend(loc="lower right", ncol=3)
    std = results[list(colors)].std()[::-1]
    bars = axes[1].barh(std.index, std, color=[colors[n] for n in std.index])
    axes[1].bar_label(bars, labels=[f"{v:.1%}" for v in std], padding=3)
    axes[1].set(title="P&L dispersion (std dev)", xlim=(0, std.max() * 1.3))
    axes[1].xaxis.set_major_formatter(PCT)
    axes[1].grid(axis="y", visible=False)
    save(fig, path)


def note_since_launch(prices, history, launch, path):
    """Performance of the three stocks since launch, the barriers and the monthly observations."""
    perf = prices.loc[launch:] / prices.loc[launch]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for name, color in zip(perf.columns, STOCK_COLORS):
        ax.plot(perf.index, perf[name], color=color, lw=1.5, label=name)
    for label, level, style in [("autocall 100%", 1.0, "--"), ("coupon barrier 60%", 0.6, ":"),
                                ("capital barrier 50%", 0.5, ":")]:
        ax.axhline(level, color=GREY, ls=style, lw=1)
        ax.annotate(label, (perf.index[-1], level), va="bottom", ha="right", fontsize=7)
    ax.scatter(history["date"], history["worst"], color="black", s=18, zorder=3, label="worst stock on each observation")
    ax.set(title=f"The note since launch: {int((history['coupon_paid'] > 0).sum())} coupons paid out of "
                 f"{len(history)}, no autocall", ylabel="price / initial price", ylim=(0.4, 1.3))
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="lower right", ncol=4, fontsize=8)
    save(fig, path)


def value_bridge(bridge, coupons_received, path):
    """Waterfall from the launch value to today's value, one input at a time."""
    fig, ax = plt.subplots(figsize=(10, 4.4))
    level = 0.0
    names = list(bridge.index)
    for i, name in enumerate(names):
        v = bridge[name]
        if name in ("Value at launch", "Value today"):
            ax.bar(i, v, color=GREY, width=0.6)
            ax.annotate(f"{v:.1%}", (i, v), ha="center", va="bottom", fontsize=8)
            level = v
        else:
            ax.bar(i, v, bottom=level, color=BLUE if v >= 0 else RED, width=0.6)
            ax.annotate(f"{v:+.1%}", (i, level + max(v, 0)), ha="center", va="bottom", fontsize=8)
            level += v
    i = len(names)
    ax.bar(i, coupons_received, bottom=bridge["Value today"], color=GREEN, width=0.6)
    ax.annotate(f"+{coupons_received:.1%}", (i, bridge["Value today"] + coupons_received), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(i + 1), [n.replace(", ", ",\n").replace(" ", "\n", 1) for n in names] + ["Coupons\nalready received"], fontsize=8)
    ax.set(title="What the note is worth today, and why it moved since launch", ylabel="% of notional",
           ylim=(0.6, 1.05))
    ax.yaxis.set_major_formatter(PCT)
    save(fig, path)


def value_vs_level(profile, name, current, path):
    """Today's value and delta of the note when one stock moves."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    axes[0].plot(profile["level"], profile["value"], color=BLUE)
    axes[1].plot(profile["level"], profile["delta"], color=ORANGE)
    for ax in axes:
        ax.axvline(current, color="black", lw=1, ls="--")
        ax.annotate(f"today {current:.0%}", (current, 0.02), xycoords=("data", "axes fraction"), fontsize=8)
        for level in (0.5, 0.6):
            ax.axvline(level, color=GREY, lw=1, ls=":")
        ax.set_xlabel(f"{name} price / initial price")
        ax.xaxis.set_major_formatter(PCT)
    axes[0].set(title=f"Note value today if {name} moves", ylabel="% of notional")
    axes[0].yaxis.set_major_formatter(PCT)
    axes[1].set(title=f"Delta of the note to {name}", ylabel="dV / d(S/S0)")
    save(fig, path)


def pnl_explain(explain, path):
    """Cumulative daily P&L of the delta-hedged desk, split by Greek."""
    parts = {"gamma": BLUE, "cross_gamma": "#86b6ef", "theta": RED, "vega": ORANGE,
             "correlation": "#a35fd1", "carry": GREEN, "residual": GREY}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [2, 1]})
    for name, color in parts.items():
        axes[0].plot(explain["date"], explain[name].cumsum(), color=color, lw=1.3, label=name.replace("_", "-"))
    axes[0].plot(explain["date"], explain["desk_pnl"].cumsum(), color="black", lw=2, label="desk P&L (delta-hedged)")
    axes[0].plot(explain["date"], explain["unhedged"].cumsum(), color="black", lw=1, ls="--", label="desk P&L (unhedged)")
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set(title="Desk P&L since launch, explained by the Greeks (cumulative)", ylabel="% of notional")
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].legend(fontsize=7, ncol=2, loc="lower left")
    totals = explain[list(parts)].sum()
    bars = axes[1].barh([n.replace("_", "-") for n in totals.index][::-1], totals.values[::-1],
                        color=[parts[n] for n in totals.index][::-1])
    axes[1].bar_label(bars, labels=[f"{v:+.1%}" for v in totals.values[::-1]], fontsize=8, padding=3)
    axes[1].axvline(0, color="black", lw=0.8)
    lim = totals.abs().max() * 1.4
    axes[1].set(title=f"Total: {explain['desk_pnl'].sum():+.1%}", xlim=(-lim, lim))
    axes[1].xaxis.set_major_formatter(PCT)
    axes[1].grid(axis="y", visible=False)
    save(fig, path)
