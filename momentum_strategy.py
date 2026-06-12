"""
Cross-Sectional Momentum Strategy — Rolling 5-Year Windows
============================================================
Runs the same momentum backtest across consecutive 5-year periods
to show how the strategy behaves across different market regimes:

  2000–2004: Dot-com bust and recovery
  2005–2009: Pre-crisis bull run and GFC
  2010–2014: Post-crisis recovery
  2015–2019: Low-volatility bull market
  2020–2024: COVID crash, recovery, rate hike cycle
  2021–2026: (partial overlap for recency)

Each period produces its own plot saved as e.g. "data/momentum_2000_2004.png"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from pandas.tseries.offsets import MonthEnd

# ── 0. Load data ──────────────────────────────────────────────────────────────
returns_monthly = pd.read_parquet("data/returns_monthly.parquet")
shares_monthly  = pd.read_parquet("data/shares_monthly.parquet")
ff3_monthly     = pd.read_parquet("data/ff3_monthly.parquet")

returns_monthly["date"] = pd.to_datetime(returns_monthly["date"])
shares_monthly["date"]  = pd.to_datetime(shares_monthly["date"])
ff3_monthly["date"]     = pd.to_datetime(ff3_monthly["date"])

returns_monthly = returns_monthly.sort_values(["symbol", "date"])

# ── 1. Compute momentum signal once on full sample ────────────────────────────
# 12-2 momentum: cumulative return t-12 to t-2 (skip last month)
def compute_momentum(df):
    r = df["returns_monthly"]
    mom = (
        (1 + r)
        .shift(1)
        .rolling(window=11, min_periods=9)
        .apply(np.prod, raw=True)
        - 1
    )
    df["momentum"] = mom
    return df

returns_monthly = (
    returns_monthly
    .groupby("symbol", group_keys=False)
    .apply(compute_momentum)
)

# ── 2. Merge market cap ───────────────────────────────────────────────────────
data_full = (
    returns_monthly
    .merge(shares_monthly, on=["symbol", "date"], how="left")
    .assign(
        mktcap=lambda x: x["adj_close"] * x["shares_outstanding"],
        mktcap_lag=lambda x: x.groupby("symbol")["mktcap"].shift(1),
    )
    .dropna(subset=["momentum", "returns_monthly", "mktcap_lag"])
)

# ── 3. Define 5-year windows ──────────────────────────────────────────────────
windows = [
    ("2000", "2004"),
    ("2005", "2009"),
    ("2010", "2014"),
    ("2015", "2019"),
    ("2020", "2024"),
    ("2022", "2026"),  # most recent partial window
]

# ── 4. Helper functions ───────────────────────────────────────────────────────
def assign_deciles(df):
    try:
        df["decile"] = pd.qcut(
            df["momentum"], q=10, labels=False, duplicates="drop"
        ) + 1
    except Exception:
        df["decile"] = np.nan
    return df

def vw_return(df):
    """Value-weighted return, falls back to equal-weight if mktcap unavailable."""
    if df["mktcap_lag"].isna().all():
        return df["returns_monthly"].mean()
    w = df["mktcap_lag"].fillna(0)
    if w.sum() == 0:
        return df["returns_monthly"].mean()
    w = w / w.sum()
    return (w * df["returns_monthly"]).sum()

def run_window(data, ff3, start_year, end_year):
    """Run full momentum analysis for a given date window."""

    mask = (
        (data["date"] >= f"{start_year}-01-01") &
        (data["date"] <= f"{end_year}-12-31")
    )
    d = data[mask].copy()

    if len(d) < 100:
        print(f"  Skipping {start_year}-{end_year}: insufficient data")
        return None

    # Assign deciles monthly
    d = (
        d.groupby("date", group_keys=False)
        .apply(assign_deciles)
        .dropna(subset=["decile"])
    )

    # Value-weighted returns per decile
    decile_returns = (
        d.groupby(["date", "decile"])
        .apply(vw_return)
        .reset_index()
        .rename(columns={0: "ret"})
    )

    # Pivot and construct WML
    port_wide = decile_returns.pivot(
        index="date", columns="decile", values="ret"
    )
    port_wide.columns = [f"D{int(c)}" for c in port_wide.columns]

    if "D1" not in port_wide.columns or "D10" not in port_wide.columns:
        print(f"  Skipping {start_year}-{end_year}: missing deciles")
        return None

    port_wide["WML"] = port_wide["D10"] - port_wide["D1"]

    # Merge FF3
    results = (
        port_wide.reset_index()
        .merge(ff3, on="date", how="inner")
        .dropna()
    )

    for col in [f"D{i}" for i in range(1, 11) if f"D{i}" in results.columns] + ["WML"]:
        results[f"{col}_excess"] = results[col] - results["RF"]

    return results

def compute_stats(results):
    """Compute summary stats for the window."""
    wml = results["WML_excess"]
    sp500 = results["Mkt-RF"]

    stats = {
        "Ann. WML Return (%)":   wml.mean() * 12 * 100,
        "Ann. WML Sharpe":       wml.mean() / wml.std() * np.sqrt(12),
        "Ann. SP500 Sharpe":     sp500.mean() / sp500.std() * np.sqrt(12),
        "Max Drawdown (%)":      (
            (1 + wml).cumprod() / (1 + wml).cumprod().cummax() - 1
        ).min() * 100,
        "N months":              len(wml),
    }

    # CAPM alpha
    X = sm.add_constant(results["Mkt-RF"])
    capm = sm.OLS(results["WML_excess"], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6}
    )
    stats["CAPM Alpha (%/mo)"] = capm.params["const"] * 100
    stats["CAPM Alpha t-stat"] = capm.tvalues["const"]

    # FF3 alpha
    X3 = sm.add_constant(results[["Mkt-RF", "SMB", "HML"]])
    ff3_reg = sm.OLS(results["WML_excess"], X3).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6}
    )
    stats["FF3 Alpha (%/mo)"]  = ff3_reg.params["const"] * 100
    stats["FF3 Alpha t-stat"]  = ff3_reg.tvalues["const"]
    stats["FF3 R²"]            = ff3_reg.rsquared

    return stats, capm, ff3_reg

def make_plot(results, stats, start_year, end_year):
    """Produce the 4-panel plot for a window."""

    label = f"{start_year}–{end_year}"
    wml   = results["WML_excess"]

    # Decile mean returns
    decile_means = {
        i: results[f"D{i}_excess"].mean() * 100
        for i in range(1, 11)
        if f"D{i}_excess" in results.columns
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Cross-Sectional Momentum — S&P 500  |  {label}\n"
        f"WML: {stats['Ann. WML Return (%)']:.1f}% p.a.  |  "
        f"Sharpe: {stats['Ann. WML Sharpe']:.2f}  |  "
        f"CAPM α: {stats['CAPM Alpha (%/mo)']:.2f}%/mo (t={stats['CAPM Alpha t-stat']:.1f})  |  "
        f"FF3 α: {stats['FF3 Alpha (%/mo)']:.2f}%/mo (t={stats['FF3 Alpha t-stat']:.1f})",
        fontsize=11
    )

    # Panel A: Mean excess return by decile
    ax = axes[0, 0]
    colors = [
        "#d73027" if i <= 2 else "#fee08b" if i <= 4
        else "#d9ef8b" if i <= 7 else "#1a9850"
        for i in decile_means.keys()
    ]
    ax.bar(list(decile_means.keys()), list(decile_means.values()), color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("A. Mean Monthly Excess Return by Decile")
    ax.set_xlabel("Momentum Decile (1=Losers, 10=Winners)")
    ax.set_ylabel("Mean Excess Return (%)")
    ax.set_xticks(list(decile_means.keys()))
    ax.set_xticklabels([f"D{i}" for i in decile_means.keys()])

    # Panel B: Cumulative returns
    ax = axes[0, 1]
    cum_wml   = (1 + wml).cumprod()
    cum_sp500 = (1 + results["Mkt-RF"]).cumprod()
    ax.plot(results["date"], cum_wml,   label="Momentum WML", color="#1a9850")
    ax.plot(results["date"], cum_sp500, label="S&P 500",      color="#2166ac")
    ax.axhline(1, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("B. Cumulative Returns: WML vs S&P 500")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel C: Drawdown
    ax = axes[1, 0]
    drawdown = (cum_wml / cum_wml.cummax() - 1) * 100
    ax.fill_between(results["date"], drawdown, 0, color="#d73027", alpha=0.6)
    ax.set_title(f"C. WML Drawdown (Max: {stats['Max Drawdown (%)']:.1f}%)")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Panel D: Rolling 12-month Sharpe
    ax = axes[1, 1]
    rolling_sharpe = (
        wml.rolling(12)
        .apply(lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan)
    )
    ax.plot(results["date"], rolling_sharpe, color="#762a83")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(
        stats["Ann. WML Sharpe"], color="black", linewidth=0.8,
        linestyle="--",
        label=f"Full-period Sharpe: {stats['Ann. WML Sharpe']:.2f}"
    )
    ax.set_title("D. Rolling 12-Month Sharpe Ratio (WML)")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fname = f"data/momentum_{start_year}_{end_year}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")

# ── 5. Run all windows ────────────────────────────────────────────────────────
summary_rows = []

for start_year, end_year in windows:
    print(f"\nProcessing {start_year}–{end_year}...")

    results = run_window(data_full, ff3_monthly, start_year, end_year)
    if results is None:
        continue

    stats, capm_reg, ff3_reg = compute_stats(results)
    make_plot(results, stats, start_year, end_year)

    summary_rows.append({"Period": f"{start_year}–{end_year}", **stats})

# ── 6. Cross-period summary table ─────────────────────────────────────────────
summary = pd.DataFrame(summary_rows).set_index("Period")
cols = [
    "Ann. WML Return (%)", "Ann. WML Sharpe", "Ann. SP500 Sharpe",
    "Max Drawdown (%)", "CAPM Alpha (%/mo)", "CAPM Alpha t-stat",
    "FF3 Alpha (%/mo)", "FF3 Alpha t-stat", "FF3 R²"
]
print("\n\n── Cross-Period Summary ──────────────────────────────────────────────")
print(summary[cols].round(3).to_string())

summary[cols].round(3).to_csv("data/momentum_summary.csv")
print("\nSummary saved to data/momentum_summary.csv")

# Summary