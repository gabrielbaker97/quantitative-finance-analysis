import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from pandas.tseries.offsets import MonthEnd

ff3_monthly  = pd.read_parquet("data/ff3_monthly.parquet")
rf = ff3_monthly[["date", "RF"]]
rf['RF'] = 0

# ── 0. Configuration ──────────────────────────────────────────────────────────
# Faber (2007) uses 5 asset classes; ETFs are standard proxies
assets = {
    "US_EQ":  "SPY",   # US equities
    "INTL_EQ": "EFA",  # International equities
    "BONDS":  "IEF",   # Intermediate US bonds
    "COMM":   "GSG",   # Commodities
    "REITS":  "VNQ",   # Real estate
}
cash_proxy = "SHY"     # Short-term Treasuries — "risk-off" holding
start_date = "2002-01-01"   # GSG/VNQ limit history; adjust if needed
end_date   = "2026-05-31"
sma_window = 10        # Faber's 10-month moving average signal

# ── 1. Download monthly price data ───────────────────────────────────────────
tickers = list(assets.values()) + [cash_proxy]

raw = yf.download(tickers, start=start_date, end=end_date, interval="1mo", auto_adjust=True)
prices = raw["Close"].copy()
prices.index = pd.to_datetime(prices.index) + MonthEnd(0)  # normalize to month-end

# ── 2. Compute signals ───────────────────────────────────────────────────────
# Faber rule: hold asset if price > 10-month SMA, else hold cash proxy
sma = prices.rolling(window=sma_window).mean()
in_mkt = prices > sma                                     
in_mkt_lagged = in_mkt.shift(1)

rets = (prices
    .pct_change()
    .reset_index()
    .rename(columns={"Date": "date"})
    .merge(rf, on = 'date', how = 'left')
    .set_index("date")
)

asset_cols = asset_tickers + [cash_proxy]
rets_excess = (rets
    .copy()
    .assign(**{col: rets[col] - rets["RF"] for col in asset_cols})
)


# ── 3. Build portfolios ──────────────────────────────────────────────────────
asset_tickers = list(assets.values())
n = len(asset_tickers)

# --- 3a. Faber TAA ---
def faber_portfolio(rets, in_mkt, asset_tickers, cash_proxy, weight=1/5):
    port_rets = []
    dates     = rets.index

    for date in dates:
        if date not in in_mkt.index:
            port_rets.append(np.nan)
            continue
        r = 0.0
        for tkr in asset_tickers:
            signal  = in_mkt.loc[date, tkr] if date in in_mkt.index else False
            holding = tkr if signal else cash_proxy
            r      += weight * rets.loc[date, holding]
        port_rets.append(r)

    return pd.Series(port_rets, index=dates, name="TAA")

taa      = faber_portfolio(rets_excess, in_mkt_lagged, asset_tickers, cash_proxy)
ew_ret   = rets_excess[asset_tickers].mean(axis=1).rename("Equal-Weighted")
ret_6040 = (0.60 * rets_excess["SPY"] + 0.40 * rets_excess["IEF"]).rename("60/40")
portfolios = pd.concat([taa, ew_ret, ret_6040], axis=1).dropna()


# ── 5. Performance metrics ────────────────────────────────────────────────────
def performance_table(ret_df):
    ann_ret = (1 + ret_df).prod() ** (12 / len(ret_df)) - 1
    ann_vol = ret_df.std() * np.sqrt(12)
    sharpe  = ann_ret / ann_vol          # already excess returns
    max_dd  = ((1 + ret_df).cumprod()
               .div((1 + ret_df).cumprod().cummax())
               .sub(1).min())

    return pd.DataFrame({
        "Ann. Excess Return (%)": (ann_ret * 100).round(2),
        "Ann. Volatility (%)":    (ann_vol * 100).round(2),
        "Sharpe Ratio":           sharpe.round(3),
        "Max Drawdown (%)":       (max_dd * 100).round(2),
    })

def window_performance(ret_df, window_years=5):
    results      = []
    window_start = ret_df.index.min()
    end          = ret_df.index.max()

    while window_start < end:
        window_end = min(
            window_start + pd.DateOffset(years=window_years) - MonthEnd(1),
            end
        )
        window = ret_df.loc[window_start:window_end]
        if len(window) < 12:
            break

        stats       = performance_table(window)
        label       = f"{window_start.strftime('%Y-%m')} to {window_end.strftime('%Y-%m')}"
        stats.index = pd.MultiIndex.from_product([[label], stats.index])
        results.append(stats)

        window_start = window_start + pd.DateOffset(years=window_years)

    return pd.concat(results)

print(performance_table(portfolios))
wp = window_performance(portfolios)
print(wp)

# ── 6. Plot cumulative returns ────────────────────────────────────────────────
os.makedirs("saa_taa/results", exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 5))
cumrets = (1 + portfolios).cumprod()

for col in cumrets.columns:
    ax.plot(cumrets.index, cumrets[col], label=col, linewidth=1.5)

ax.set_title("Tactical vs. Strategic Allocation (Faber 2007 replication)")
ax.set_ylabel("Growth of $1")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("saa_taa/results/taa_vs_strategic.png", dpi=150)
plt.show()


# Extract window labels and strategy names from the MultiIndex
windows   = wp.index.get_level_values(0).unique().tolist()
strats    = (wp
    .index.get_level_values(1)
    .unique()
    .tolist()
)
# Build the sharpe dict dynamically
sharpe = {
    strat: [wp.loc[(window, strat), "Sharpe Ratio"] for window in windows]
    for strat in strats
}

def plot_metric_by_window(wp, metric="Sharpe Ratio", title=None, savepath=None):
    windows = wp.index.get_level_values(0).unique().tolist()
    strats  = wp.index.get_level_values(1).unique().tolist()

    data = {
        strat: [wp.loc[(window, strat), metric] for window in windows]
        for strat in strats
    }

    colors = {"TAA": "#1D9E75", "Equal-Weighted": "#D85A30", "60/40": "#378ADD"}
    y, width = np.arange(len(windows)), 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, values) in enumerate(data.items()):
        offset = (i - 1) * width
        bars = ax.barh(y + offset, values, height=width,
                       label=label, color=colors.get(label, "gray"), alpha=0.9)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9,
                     color=colors.get(label, "gray"))

    ax.set_yticks(y)
    ax.set_yticklabels([f"{w[:4]}–{w[11:15]}" for w in windows])
    ax.set_xlabel(metric)
    ax.set_title(title or f"{metric} by window")
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
    ax.invert_yaxis()   # most recent window at the bottom
    ax.legend(loc = 'lower left')
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()
    plt.close()

def slugify(s):
    return s.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('.', '')

for metric in wp.columns.tolist():
    plot_metric_by_window(wp,
        metric=metric,
        savepath=f"saa_taa/results/{slugify(metric)}.png")