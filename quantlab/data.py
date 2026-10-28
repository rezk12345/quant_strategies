"""Data loading for all three strategies.

live      : S&P 500 list from Wikipedia, prices from yfinance, factors from the
            Ken French data library, sentiment/intraday from CSV files you supply.
synthetic : generated equivalents with the same shape, so every strategy runs
            offline and the tests don't need network access.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, Config

log = logging.getLogger(__name__)


# =========================================================== S&P 500 panel
def sp500_symbols() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tbl = pd.read_html(url)[0]
    return sorted(tbl["Symbol"].str.replace(".", "-", regex=False).tolist())


def download_panel(symbols: list[str], start: str, end: str, cfg: Config) -> pd.DataFrame:
    """Long-format OHLCV panel indexed by (date, ticker)."""
    cache = ROOT / cfg.data.cache_dir / f"panel_{start}_{end}_{len(symbols)}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    import yfinance as yf

    raw = yf.download(symbols, start=start, end=end, auto_adjust=False, progress=False, group_by="column")
    if raw.empty:
        raise RuntimeError("yfinance returned nothing (offline?) — try --source synthetic")
    df = raw.stack(level=1, future_stack=True)
    df.index.names = ["date", "ticker"]
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(subset=["adj_close"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df


def download_close(symbol: str, start: str, end: str) -> pd.Series:
    import yfinance as yf

    s = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return s.squeeze().rename(symbol)


def fama_french_monthly(start: str) -> pd.DataFrame:
    """Fama-French 5 factors (monthly, decimal) from Ken French's site."""
    import requests

    url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_CSV.zip")
    z = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=30).content))
    txt = z.read(z.namelist()[0]).decode("latin1").splitlines()
    head = next(i for i, l in enumerate(txt) if l.startswith(",Mkt-RF"))
    rows = []
    for l in txt[head + 1:]:
        parts = l.split(",")
        if len(parts[0].strip()) != 6:   # annual block starts -> stop
            break
        rows.append(parts)
    ff = pd.DataFrame(rows, columns=["date"] + txt[head].split(",")[1:])
    ff["date"] = pd.to_datetime(ff["date"].str.strip(), format="%Y%m") + pd.offsets.MonthEnd(0)
    ff = ff.set_index("date").astype(float) / 100
    return ff.drop(columns=["RF"]).loc[start:]


# =========================================================== synthetic data
def synthetic_panel(start: str, end: str, n: int = 200, seed: int = 3):
    """OHLCV panel with a market factor, sector factors and persistent momentum.

    Each stock has a slowly drifting 'quality' alpha, so stocks that have done
    well recently tend to keep doing slightly better — the kind of structure a
    momentum-cluster strategy is meant to exploit. Returns (panel, factors, spy).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    mkt = 0.0004 + 0.011 * rng.standard_normal(T)
    sectors = 0.006 * rng.standard_normal((T, 8))
    smb = 0.004 * rng.standard_normal(T)
    hml = 0.004 * rng.standard_normal(T)
    sec_of = rng.integers(0, 8, n)
    beta = rng.uniform(0.6, 1.5, n)
    b_smb = rng.normal(0, 0.6, n)
    b_hml = rng.normal(0, 0.6, n)
    idio_vol = rng.uniform(0.010, 0.025, n)
    alpha = np.zeros((T, n))
    a = rng.normal(0, 0.0004, n)
    for t in range(T):
        a = 0.997 * a + 0.00004 * rng.standard_normal(n)
        alpha[t] = a
    rets = (alpha + mkt[:, None] * beta + sectors[:, sec_of] + smb[:, None] * b_smb
            + hml[:, None] * b_hml + idio_vol * rng.standard_normal((T, n)))
    close = 50 * np.exp(np.cumsum(rets, axis=0))
    tickers = [f"S{i:03d}" for i in range(n)]
    rng_hl = np.abs(rng.normal(0, 0.008, (T, n)))
    open_ = close * np.exp(rng.normal(0, 0.004, (T, n)))
    high = np.maximum(open_, close) * (1 + rng_hl)
    low = np.minimum(open_, close) * (1 - rng_hl)
    volume = rng.lognormal(13, 1.0, n) * np.exp(rng.normal(0, 0.3, (T, n)))
    frames = {"open": open_, "high": high, "low": low, "close": close,
              "adj_close": close, "volume": volume}
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = pd.DataFrame({k: v.reshape(-1) for k, v in frames.items()}, index=idx)

    daily_f = pd.DataFrame({"Mkt-RF": mkt, "SMB": smb, "HML": hml,
                            "RMW": 0.003 * rng.standard_normal(T),
                            "CMA": 0.003 * rng.standard_normal(T)}, index=dates)
    factors = (1 + daily_f).resample("ME").prod() - 1
    spy = pd.Series(100 * np.exp(np.cumsum(mkt)), index=dates, name="SPY")
    return panel, factors, spy


def synthetic_sentiment(start: str, end: str, n: int = 60, seed: int = 5):
    """Daily Twitter-engagement CSV + matching prices + a QQQ-like benchmark.

    Stocks with genuinely higher comment-to-like ratios get a small positive
    drift, so the sentiment strategy has a real (but noisy) signal to find.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    syms = [f"N{i:03d}" for i in range(n)]
    base_ratio = rng.uniform(0.05, 0.6, n)
    rows = []
    for t, d in enumerate(dates):
        likes = rng.lognormal(6, 1.2, n)
        ratio = base_ratio * np.exp(rng.normal(0, 0.35, n))
        comments = likes * ratio
        rows.append(pd.DataFrame({"date": d, "symbol": syms,
                                  "twitterPosts": rng.poisson(30, n),
                                  "twitterComments": comments.round(),
                                  "twitterLikes": likes.round(),
                                  "twitterImpressions": (likes * rng.uniform(20, 60, n)).round()}))
    sent = pd.concat(rows, ignore_index=True)
    mkt = 0.0005 + 0.014 * rng.standard_normal(T)
    drift = 0.0012 * (base_ratio - base_ratio.mean()) / base_ratio.std()
    rets = drift + mkt[:, None] * rng.uniform(0.8, 1.4, n) + 0.02 * rng.standard_normal((T, n))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=syms)
    qqq = pd.Series(100 * np.exp(np.cumsum(mkt + 0.0001)), index=dates, name="QQQ")
    return sent, prices, qqq


def synthetic_intraday(start: str = "2018-01-01", end: str = "2023-09-29", seed: int = 9):
    """Daily OHLCV with GARCH-style clustered volatility + 5-minute bars for the last ~2 years."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    T = len(days)
    var = np.empty(T)
    r = np.empty(T)
    var[0] = 0.012 ** 2
    for t in range(T):   # GARCH(1,1) data-generating process
        if t:
            var[t] = 0.000004 + 0.09 * r[t - 1] ** 2 + 0.88 * var[t - 1]
        # vol spikes tend to be followed by short-term reversals
        r[t] = 0.0003 + np.sqrt(var[t]) * rng.standard_normal()
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[100, close[:-1]] * np.exp(rng.normal(0, 0.002, T))
    rngp = np.abs(rng.normal(0, 0.5, T)) * np.sqrt(var)
    daily = pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * (1 + rngp),
                          "Low": np.minimum(open_, close) * (1 - rngp), "Close": close,
                          "Adj Close": close, "Volume": rng.lognormal(16, 0.3, T).round()},
                         index=pd.Index(days, name="Date"))

    bars = []
    for d, row in daily.loc[days[-500]:].iterrows():
        n = 78                                            # 09:30-16:00 in 5-min bars
        sig = np.sqrt(var[days.get_loc(d)] / n)
        steps = rng.normal(0, sig, n)
        steps -= steps.mean() - np.log(row.Close / row.Open) / n   # land on the daily close
        path = row.Open * np.exp(np.cumsum(steps))
        o = np.r_[row.Open, path[:-1]]
        wig = np.abs(rng.normal(0, sig * 0.5, n))
        ts = pd.date_range(d + pd.Timedelta(hours=9, minutes=30), periods=n, freq="5min")
        bars.append(pd.DataFrame({"open": o, "high": np.maximum(o, path) * (1 + wig),
                                  "low": np.minimum(o, path) * (1 - wig), "close": path,
                                  "volume": rng.lognormal(10, 0.5, n).round()},
                                 index=pd.Index(ts, name="datetime")))
    return daily, pd.concat(bars)


def write_synthetic_csvs(cfg: Config) -> None:
    """Create the CSVs strategies 2 and 3 read, so the live-mode code path is exercised too."""
    s = cfg.sentiment
    sent, _, _ = synthetic_sentiment(s.start, s.end)
    p = ROOT / s.csv
    p.parent.mkdir(parents=True, exist_ok=True)
    sent.to_csv(p, index=False)
    daily, intra = synthetic_intraday()
    daily.to_csv(ROOT / cfg.intraday.daily_csv)
    intra.to_csv(ROOT / cfg.intraday.intraday_csv)
