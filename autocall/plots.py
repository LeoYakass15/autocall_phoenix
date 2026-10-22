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
    strikes, times = np.meshgrid(np.linspace(0.45, 1.55, 50), np.linspace(0.25, 3.0, 50))
    fig = plt.figure(figsize=(14, 4.6))
    for i, (name, surface) in enumerate(zip(market.names, market.surfaces)):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        k = np.log(strikes) - (market.rate - market.divs[i]) * times
        ax.plot_surface(strikes, times, surface.implied_vol(k, times), cmap="Blues", alpha=0.95)
        q_times, q_k, q_vols = market.quotes[i]
        kq, tq = np.meshgrid(q_k, q_times)
        strike_q = np.exp(kq + (market.rate - market.divs[i]) * tq)
        show = (strike_q > 0.45) & (strike_q < 1.55)
        ax.scatter(strike_q[show], tq[show], q_vols[show], color="black", s=4)
        ax.set(title=f"{name} implied vol (dots: quotes)", xlabel="strike / initial", ylabel="maturity (y)")
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


def smiles_with_barriers(market, path, maturities=(0.5, 1.0, 3.0)):
    strikes = np.linspace(0.35, 1.6, 200)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for i, ax in enumerate(axes):
        for T, color in zip(maturities, ["#86b6ef", "#3987e5", "#184f95"]):
            k = np.log(strikes) - (market.rate - market.divs[i]) * T
            ax.plot(strikes, market.surfaces[i].implied_vol(k, T), color=color, label=f"{T:g}y")
        for label, level in BARRIERS.items():
            ax.axvline(level, color=GREY, lw=1, ls=":")
            if i == 0:
                ax.annotate(label, (level, 0.02), xycoords=("data", "axes fraction"), rotation=90,
                            va="bottom", ha="right", fontsize=7)
        ax.set(title=f"{market.names[i]} smiles and the note's barriers", xlabel="strike / initial")
        ax.xaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_formatter(PCT)
    axes[-1].legend(title="maturity")
    save(fig, path)


def model_comparison(table, termsheet_coupon, path):
    """table: one row per model, one column per target price, values = fair coupon."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    table.plot.bar(ax=ax, color=["#86b6ef", "#184f95"], rot=0, width=0.7)
    for bars in ax.containers:
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in bars.datavalues], fontsize=8)
    ax.axhline(termsheet_coupon, color=GREY, ls="--", lw=1, label=f"term sheet {termsheet_coupon:.2%}")
    ax.set(title="Fair coupon p.a. by volatility model", ylabel="coupon p.a.", xlabel="")
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="upper left")
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
