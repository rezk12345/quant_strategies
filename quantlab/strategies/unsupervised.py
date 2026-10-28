"""Strategy 1 — Unsupervised learning (K-Means) on S&P 500 stocks.

Pipeline, re-run at the end of every month:
  1. Daily features per stock: Garman-Klass vol, RSI, Bollinger bands, ATR,
     MACD, dollar volume.
  2. Aggregate to month-end; keep the `universe_size` most liquid stocks
     (by rolling average dollar volume).
  3. Add monthly return features at several lags and rolling Fama-French
     factor betas.
  4. Cluster that month's stocks with K-Means. Centroids are seeded on RSI so
     cluster k always means "the RSI≈centroid[k] group" — without this, cluster
     labels would shuffle every month.
  5. Take the momentum cluster (seeded at RSI 70), find max-Sharpe weights
     from the past year of returns, and hold them for the next month.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..config import Config
from ..indicators import atr, bollinger, garman_klass_vol, macd, rsi

log = logging.getLogger(__name__)


# ------------------------------------------------------------- features
def daily_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Indicators computed on wide date x ticker tables (vectorised across stocks)."""
    w = {c: panel[c].unstack("ticker") for c in ["open", "high", "low", "close", "adj_close", "volume"]}
    ac = w["adj_close"]
    lp = np.log1p(ac)
    bb_low, bb_mid, bb_high = bollinger(lp, 20, 2)
    a = atr(w["high"], w["low"], w["close"], 14)
    m = macd(ac)
    # ATR and MACD are scaled per stock so expensive and cheap stocks are comparable
    feats = {
        "adj_close": ac,
        "gk_vol": garman_klass_vol(w["open"], w["high"], w["low"], w["close"]),
        "rsi": rsi(ac, 20),
        "bb_low": bb_low, "bb_mid": bb_mid, "bb_high": bb_high,
        "atr": (a - a.mean()) / a.std(),
        "macd": (m - m.mean()) / m.std(),
        "dollar_volume": ac * w["volume"] / 1e6,
    }
    return pd.concat({k: v.stack(future_stack=True) for k, v in feats.items()}, axis=1)


def monthly_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    u = cfg.unsupervised
    feat_cols = ["adj_close", "gk_vol", "rsi", "bb_low", "bb_mid", "bb_high", "atr", "macd"]
    wide_last = (df[feat_cols].unstack("ticker").resample("ME").last()
                 .stack("ticker", future_stack=True))
    dv = df["dollar_volume"].unstack("ticker").resample("ME").mean()
    dv_roll = dv.rolling(u.liquidity_window_months, min_periods=12).mean()
    m = wide_last.join(dv_roll.stack().rename("dollar_volume")).dropna()

    rank = m.groupby(level="date")["dollar_volume"].rank(ascending=False)
    m = m[rank <= u.universe_size].drop(columns="dollar_volume")

    # monthly return features, winsorised
    def _rets(x: pd.DataFrame) -> pd.DataFrame:
        x = x.droplevel("ticker")
        for lag in u.return_lags:
            r = x["adj_close"].pct_change(lag)
            lo, hi = r.quantile(u.outlier_clip), r.quantile(1 - u.outlier_clip)
            x[f"ret_{lag}m"] = (r.clip(lo, hi) + 1) ** (1 / lag) - 1
        return x

    m = (m.groupby(level="ticker", group_keys=True).apply(_rets)
         .swaplevel().sort_index().dropna())
    return m


def factor_betas(monthly: pd.DataFrame, factors: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Rolling OLS betas of each stock's 1-month return on the five factors.

    Betas are lagged one month (we only know the beta estimated through last
    month when we form this month's clusters).
    """
    from statsmodels.regression.rolling import RollingOLS
    import statsmodels.api as sm

    w = cfg.unsupervised.factor_window
    data = monthly[["ret_1m"]].join(factors, on="date").dropna()
    counts = data.groupby(level="ticker").size()
    data = data[data.index.get_level_values("ticker").isin(counts[counts >= 10].index)]
    fcols = list(factors.columns)

    def _roll(x: pd.DataFrame) -> pd.DataFrame:
        x = x.droplevel("ticker")
        win = min(w, len(x))
        res = RollingOLS(x["ret_1m"], sm.add_constant(x[fcols]), window=win,
                         min_nobs=len(fcols) + 1).fit(params_only=True)
        return res.params.drop(columns="const")

    betas = data.groupby(level="ticker", group_keys=True).apply(_roll).swaplevel().sort_index()
    return betas.groupby(level="ticker").shift()


def cluster_month(x: pd.DataFrame, cfg: Config) -> pd.Series:
    u = cfg.unsupervised
    cols = [c for c in x.columns if c != "adj_close"]
    X = x[cols].copy()
    # standardise everything except RSI, which keeps its natural 0-100 scale
    # so the RSI-seeded centroids dominate the cluster meaning
    other = [c for c in cols if c != "rsi"]
    X[other] = (X[other] - X[other].mean()) / X[other].std().replace(0, 1)
    X[other] = X[other].fillna(0)
    init = np.zeros((u.n_clusters, len(cols)))
    init[:, cols.index("rsi")] = u.rsi_centroids
    km = KMeans(n_clusters=u.n_clusters, init=init, n_init=1, random_state=0).fit(X.to_numpy())
    return pd.Series(km.labels_, index=x.index, name="cluster")


# ------------------------------------------------------------- optimiser
def max_sharpe_weights(prices: pd.DataFrame, cfg: Config) -> dict[str, float]:
    from pypfopt import EfficientFrontier, expected_returns, risk_models

    o = cfg.unsupervised.optimisation
    n = prices.shape[1]
    lo = 0.5 / n if o.min_weight_floor == "half_equal" else 0.0
    # with few names a fixed cap can force equal weights (e.g. 10% cap, 7 stocks),
    # so always leave the optimiser room up to 2x equal weight
    hi = min(1.0, max(o.max_weight, 2.0 / n))
    try:
        mu = expected_returns.mean_historical_return(prices, frequency=252)
        S = risk_models.sample_cov(prices, frequency=252)
        ef = EfficientFrontier(mu, S, weight_bounds=(lo, hi), solver="SCS")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ef.max_sharpe()
        w = ef.clean_weights()
        if abs(sum(w.values()) - 1) > 0.02:
            raise ValueError("weights don't sum to 1")
        return dict(w)
    except Exception as e:  # infeasible / all expected returns below rf, etc.
        log.debug("max-Sharpe failed (%s) -> equal weight", e)
        return {c: 1.0 / n for c in prices.columns}


# ------------------------------------------------------------- backtest
def run(panel: pd.DataFrame, factors: pd.DataFrame, benchmark: pd.Series, cfg: Config) -> dict:
    u = cfg.unsupervised
    log.info("computing daily features for %d tickers...", panel.index.get_level_values(1).nunique())
    daily = daily_features(panel)
    monthly = monthly_features(daily, cfg)
    betas = factor_betas(monthly, factors, cfg)
    data = monthly.join(betas, how="left")
    data[betas.columns] = data.groupby(level="ticker")[list(betas.columns)].transform(lambda s: s.fillna(s.mean()))
    data = data.dropna()

    clusters = data.groupby(level="date", group_keys=False).apply(lambda x: cluster_month(x, cfg))
    data = data.join(clusters)

    prices = panel["adj_close"].unstack("ticker")
    daily_ret = np.log(prices).diff()
    port = []
    picks_log = {}
    months = sorted(data.index.get_level_values("date").unique())
    for m_end in months:
        picks = data.xs(m_end, level="date")
        picks = picks[picks.cluster == u.target_cluster].index.tolist()
        if len(picks) < 2:
            continue
        hold_start = m_end + pd.offsets.Day(1)
        hold_end = m_end + pd.offsets.MonthEnd(1)
        hist = prices.loc[m_end - pd.Timedelta(days=u.optimisation.lookback_days):m_end, picks].dropna(axis=1)
        if hist.shape[1] < 2:
            continue
        w = max_sharpe_weights(hist, cfg)
        picks_log[m_end.date()] = w
        r = daily_ret.loc[hold_start:hold_end, list(w)]
        if r.empty:
            continue
        port.append((r * pd.Series(w)).sum(axis=1))

    strat = np.expm1(pd.concat(port).sort_index())   # log -> simple returns
    strat = strat[~strat.index.duplicated()]
    bench = benchmark.pct_change().reindex(strat.index).fillna(0)
    return {"returns": {"K-Means momentum cluster": strat, f"{u.benchmark} buy & hold": bench},
            "clusters": data[["rsi", "cluster"]], "weights": picks_log}
