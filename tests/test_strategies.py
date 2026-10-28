"""Offline tests — run with:  pytest -q"""
import logging

import numpy as np
import pandas as pd
import pytest

from quantlab.config import load_config
from quantlab.data import synthetic_intraday, synthetic_panel, synthetic_sentiment
from quantlab.indicators import garman_klass_vol, rsi
from quantlab.metrics import stats
from quantlab.strategies import intraday_garch, sentiment, unsupervised

logging.disable(logging.INFO)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_garman_klass_nonnegative_on_flat_bar():
    s = pd.Series([100.0])
    assert garman_klass_vol(s, s * 1.01, s * 0.99, s).iloc[0] > 0


def test_rsi_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi(up, 14).iloc[-1] > 99


def test_stats_on_known_series():
    r = pd.Series([0.01] * 252)
    st = stats(r)
    assert st["total_return"] == pytest.approx(1.01 ** 252 - 1)
    assert st["max_drawdown"] == 0


def test_kmeans_clusters_keep_rsi_order(cfg):
    u = cfg.unsupervised
    panel, ff, spy = synthetic_panel("2017-01-01", "2021-12-31", n=120)
    cfg2 = load_config()
    cfg2["unsupervised"]["universe_size"] = 80
    res = unsupervised.run(panel, ff, spy, cfg2)
    means = res["clusters"].groupby("cluster").rsi.mean()
    assert list(means.index) == list(range(u.n_clusters))
    assert means.is_monotonic_increasing          # cluster 3 is always the high-RSI group
    for w in res["weights"].values():
        assert sum(w.values()) == pytest.approx(1, abs=0.02)


def test_sentiment_picks_top_ratio(cfg):
    d = pd.Timestamp("2022-01-03")
    sent = pd.DataFrame({"date": [d] * 7, "symbol": list("ABCDEFG"),
                         "twitterPosts": 1, "twitterLikes": 100,
                         "twitterComments": [10, 70, 20, 60, 30, 50, 5]})
    sched = sentiment.portfolio_schedule(sentiment.monthly_rankings(sent, cfg), cfg)
    (start, picks), = sched.items()
    assert start == pd.Timestamp("2022-02-01")
    assert set(picks) == {"B", "D", "F", "E", "C"}       # top 5 ratios; G filtered (comments < 10)


def test_sentiment_runs(cfg):
    sent, px, qqq = synthetic_sentiment("2021-01-01", "2021-12-31", n=20)
    res = sentiment.run(sent, px, qqq, cfg)
    assert len(res["returns"]["QQQ buy & hold"]) > 150


def test_intraday_no_lookahead_and_one_trade_per_day(cfg):
    daily, intra = synthetic_intraday("2019-01-01", "2023-09-29")
    fake_var = (np.log(daily["Adj Close"]).diff() ** 2).rolling(5).mean().shift()   # cheap stand-in for GARCH
    res = intraday_garch.run(daily, intra, cfg, garch_var=fake_var)
    t = res["trades"]
    assert t.day.is_unique
    if len(t):
        assert (t.entry_time.dt.normalize() == t.day).all()
    # signals on day d must not change if we drop data after d
    cut = daily.index[-200]
    full = intraday_garch.daily_signals(daily, cfg, fake_var)["signal_daily"]
    part = intraday_garch.daily_signals(daily.loc[:cut], cfg, fake_var.loc[:cut])["signal_daily"]
    assert (full.loc[:cut] == part).all()
