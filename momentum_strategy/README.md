# Cross-Sectional Momentum Strategy — Rolling 5-Year Windows
Backtests a standard 12-2 cross-sectional momentum strategy on US equities across 5 year windows, using CRSP monthly returns and Ken French FF3 factors. The rolling structure is designed to show how momentum alpha, Sharpe ratio, and drawdown profile vary across distinct market regimes rather the averaging across them. 

## Covered periods:
    1990-1994: Early 90s recession and recovery
    1995-1999: Dot-com bull market
    2000–2004: Dot-com bust and recovery
    2005–2009: Pre-crisis bull run and 2008 financial crisis 
    2010–2014: Post-crisis recovery
    2015–2019: Low-volatility bull market
    2020–2024: COVID crash, recovery, rate hike cycle
    2021–2026: AI/momentum cycle

Each period produces its own plot saved as e.g. "results/momentum_2000_2004.png"

## Key Results

| Period    | WML Return (%) | WML Sharpe | S&P 500 Sharpe | Max Drawdown (%) | CAPM α (%/mo) | CAPM t | FF3 α (%/mo) | FF3 t | FF3 R² |
|-----------|---------------|------------|----------------|------------------|---------------|--------|--------------|-------|--------|
| 1990–1994 | 16.1          | 0.66       | 0.38           | -34.2            | 1.42          | 1.55   | 1.65         | 2.01  | 0.18   |
| 1995–1999 | 19.6          | 0.67       | 1.43           | -34.1            | 1.80          | 1.67   | 1.88         | 1.82  | 0.20   |
| 2000–2004 | 7.4           | 0.13       | -0.18          | -73.1            | 0.18          | 0.13   | -2.49        | -1.88 | 0.43   |
| 2005–2009 | 7.9           | 0.16       | -0.02          | -86.4            | 0.61          | 0.35   | 0.66         | 0.37  | 0.37   |
| 2010–2014 | 6.4           | 0.31       | 1.15           | -26.5            | 1.08          | 1.79   | 0.96         | 1.63  | 0.10   |
| 2015–2019 | 10.0          | 0.34       | 0.87           | -41.4            | 1.49          | 1.40   | 0.81         | 0.76  | 0.32   |
| 2020–2024 | 24.5          | 0.55       | 0.68           | -65.0            | 3.19          | 3.16   | 2.53         | 3.23  | 0.47   |
| 2022–2026 | 45.4          | 1.09       | 0.32           | -44.3            | 4.12          | 3.78   | 2.60         | 2.27  | 0.55   |

FF3 alpha is positive in all periods but statistically insignificant during the dot-com reversal (2000–2004) and pre-crisis bull run and 2008 financial crisis  (2005–2009). Drawdowns of 73% and 86% make the gross returns uninvestable in practice. The 2020–2024 and 2022–2026 windows show the strongest alpha in the sample (FF3 t > 3), likely driven by concentration in large-cap technology, with rising FF3 $R^2$ suggesting increasing factor exposure. The 2022–2026 window overlaps substantially with 2020–2024 and should not be treated as an independent observation. All returns are gross of transaction costs. Alphas estimated via OLS with HAC standard errors (6 lags).
