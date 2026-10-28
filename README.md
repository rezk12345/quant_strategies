# Algorithmic Trading: ML & Quant Strategies in Python

Three independent research strategies, each with its own backtest against a
benchmark:

| # | Strategy | Universe | Rebalance | Benchmark |
|---|---|---|---|---|
| 1 | Unsupervised learning (K-Means clustering) + max-Sharpe portfolio | S&P 500 | monthly | SPY |
| 2 | Twitter sentiment (engagement ratio) | NASDAQ stocks | monthly | QQQ |
| 3 | GARCH volatility forecast + intraday RSI/Bollinger | one asset, 5-min bars | daily / intraday | buy & hold |

> Research and education only. Not financial advice. Backtests are not live results.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py all --source synthetic   # everything offline, ~1 minute
python main.py unsupervised             # real S&P 500 data (yfinance + Wikipedia + Ken French)
python main.py sentiment                # needs data/sentiment_data.csv
python main.py intraday                 # needs data/simulated_daily_data.csv + data/simulated_5min_data.csv
pytest -q
```

Charts and return CSVs go to `reports/`. All parameters live in `config.yaml`.

---

## 1 · Unsupervised learning strategy (`quantlab/strategies/unsupervised.py`)

1. **Daily features** per stock: Garman-Klass volatility, RSI(20), Bollinger
   bands on log price, normalised ATR(14), normalised MACD, dollar volume.
2. **Monthly panel**: month-end values; keep the 150 most liquid stocks by
   rolling average dollar volume.
3. **Return features**: 1, 2, 3, 6, 9 and 12-month returns (winsorised,
   expressed as monthly rates).
4. **Fama-French betas**: rolling 24-month OLS of each stock's return on the
   five factors, lagged one month to avoid look-ahead.
5. **K-Means (4 clusters)**. Initial centroids are seeded on RSI at
   30 / 45 / 55 / 70. That keeps cluster 3 meaning "high-RSI momentum names"
   every month, instead of the labels reshuffling between months.
6. **Portfolio**: the stocks in cluster 3 are weighted for maximum Sharpe ratio
   (PyPortfolioOpt), using the past year of prices. Each weight is between half
   of equal weight and max(10%, 2× equal weight). The portfolio is held for the
   next month. If optimisation is infeasible, it falls back to equal weight.

**Note:** the stock list comes from *today's* S&P 500 membership, so real-data
backtests include survivorship bias and will look better than reality.

## 2 · Twitter sentiment strategy (`quantlab/strategies/sentiment.py`)

The engagement ratio is comments ÷ likes. Raw likes mostly track company size;
the ratio tracks how much people are actually *debating* a stock. Each month:
filter thin stock-days (likes < 20, comments < 10), average the ratio, and buy
the top 5 equally weighted for the next month.

**Data**: the Twitter API no longer offers this for free, so the strategy reads
a CSV with these columns:

```
date, symbol, twitterPosts, twitterComments, twitterLikes, twitterImpressions
```

`python main.py make-sample-data` writes a synthetic example in that format.

## 3 · Intraday GARCH strategy (`quantlab/strategies/intraday_garch.py`)

- **Daily layer**: each day, fit GARCH(1,3) on the prior 180 days of returns
  and forecast tomorrow's variance. The forecast minus recent realised variance
  is z-scored. z > 1.5 means a volatility shock is expected (bias short);
  z < −1.5 means calm is expected (bias long).
- **Intraday layer**: on signal days, use RSI(20) and Bollinger(20, 2) on
  5-minute closes. Go long on an oversold dip when the daily bias is long;
  go short on an overbought pop when the bias is short.
- **Execution**: enter at the next bar's open, exit at the session close,
  one trade per day.

**Data**: two CSVs, the same shape as the synthetic ones from `make-sample-data`.

```
daily:    Date, Open, High, Low, Close, Adj Close, Volume
intraday: datetime, open, high, low, close, volume     (5-minute bars)
```

## Synthetic data

`--source synthetic` generates realistic stand-ins: a factor-model stock panel
with slow-moving momentum, engagement data where higher ratios carry a small
real drift, and GARCH-clustered daily and intraday prices. This proves the code
works end to end. It does **not** show the strategies make money: the edges
were planted by construction (strategies 1–2), or are absent by construction
(strategy 3).

## Limitations

- No transaction costs in strategies 1–2. Monthly turnover can be high; add
  ~5–10 bps per trade before trusting results.
- Strategy 3 uses daily-close exits only, with no stop-loss and no slippage.
- yfinance and Wikipedia are unofficial sources and occasionally change format.
