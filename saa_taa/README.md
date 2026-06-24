# Strategic vs. Tactical Asset Allocation
Inspired by Faber (2007), a tactical asset allocation (TAA) portfolio is constructed and benchmarked against a strategic 60/40 portfolio (US equities / US bonds) and an equal-weight buy-and-hold portfolio. The TAA signal rotates each asset to cash (SHY) when its price falls below its 10-month simple moving average. Analysis covers 2002–2026 using ETF proxies (SPY, EFA, IEF, GSG, VNQ). Performance is evaluated on annualised excess returns, volatility, Sharpe ratio, and maximum drawdown, reported both over the full period and across five-year subperiods.

## Covered periods:
    2002–2007: Dot-com recovery
    2007–2012: GFC and aftermath
    2012–2017: US equity bull market
    2017–2022: Late cycle, COVID crash, reopening
    2022–2026: Rate shock and AI rally


## Key Results
| Strategy       | Ann. Excess Return | Volatility | Sharpe | Max Drawdown |
|----------------|--------------------|------------|--------|--------------|
| TAA            | 6.52%              | 6.33%      | 1.03   | −10.2%       |
| Equal-weighted | 6.82%              | 12.16%     | 0.56   | −46.6%       |
| 60/40          | 8.40%              | 9.13%      | 0.92   | −29.4%       |

> TAA delivers competitive risk-adjusted returns (Sharpe 1.03 vs. 0.92 for 60/40) at roughly half the volatility and a fraction of the drawdown. The cost is modest underperformance in raw returns during US-equity-dominated regimes.

## Methodology 
Each asset is held at equal weight (20%) when its price exceeds its 10-month simple moving average, and rotated to cash (SHY) otherwise. The signal is lagged one period to avoid look-ahead bias. Returns are computed in excess of the Ken French risk-free rate. Performance is evaluated on annualised excess return, volatility, Sharpe ratio, and maximum drawdown both over the full period and across five-year subperiods.

## Data
 - Risk-free rate via the French data library
 - ETF for 5 asset classes from Yahoo Finance.
