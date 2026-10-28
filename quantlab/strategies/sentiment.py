"""Strategy 2 — Twitter sentiment investing on NASDAQ stocks.

Idea: raw like counts mostly measure how big a company is. The *engagement
ratio* (comments / likes) measures how much people actually argue about a
stock, which is a better proxy for attention that hasn't been priced in yet.

Each month:
  1. drop stock-days with too few likes or comments (bot-like noise)
  2. engagement ratio = comments / likes, averaged over the month
  3. rank stocks; hold the top N equally weighted for the next month
Compared against buying and holding QQQ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def monthly_rankings(sent: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    s = cfg.sentiment
    df = sent.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df.twitterLikes >= s.min_likes) & (df.twitterComments >= s.min_comments)]
    df["engagement_ratio"] = df.twitterComments / df.twitterLikes
    monthly = (df.groupby([pd.Grouper(key="date", freq="ME"), "symbol"])["engagement_ratio"]
               .mean().rename("engagement_ratio").reset_index())
    monthly["rank"] = monthly.groupby("date")["engagement_ratio"].rank(ascending=False)
    return monthly


def portfolio_schedule(rankings: pd.DataFrame, cfg: Config) -> dict[pd.Timestamp, list[str]]:
    """Picks formed at month-end M are held from the first trading day of M+1."""
    top = rankings[rankings["rank"] <= cfg.sentiment.top_n]
    return {d + pd.offsets.Day(1): g.symbol.tolist() for d, g in top.groupby("date")}


def run(sent: pd.DataFrame, prices: pd.DataFrame, benchmark: pd.Series, cfg: Config) -> dict:
    rankings = monthly_rankings(sent, cfg)
    schedule = portfolio_schedule(rankings, cfg)
    rets = np.log(prices).diff()
    pieces = []
    for start, syms in schedule.items():
        end = start + pd.offsets.MonthEnd(0)
        syms = [x for x in syms if x in rets.columns]
        if not syms:
            continue
        pieces.append(rets.loc[start:end, syms].mean(axis=1))   # equal weight
    strat = np.expm1(pd.concat(pieces).sort_index())
    bench = benchmark.pct_change().reindex(strat.index).fillna(0)
    return {"returns": {f"Top-{cfg.sentiment.top_n} engagement ratio": strat,
                        f"{cfg.sentiment.benchmark} buy & hold": bench},
            "rankings": rankings, "schedule": schedule}


def load_prices_for(sent: pd.DataFrame, cfg: Config):
    """Live mode: download the sentiment universe + benchmark from yfinance."""
    import yfinance as yf

    s = cfg.sentiment
    syms = sorted(sent.symbol.unique())
    px = yf.download(syms + [s.benchmark], start=s.start, end=s.end, auto_adjust=True, progress=False)["Close"]
    return px.drop(columns=s.benchmark), px[s.benchmark]
