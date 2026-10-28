"""Strategy 3 — Intraday strategy using a GARCH volatility model.

Two layers:

Daily layer (decided before the session):
  * Each day, fit GARCH(p, q) on the previous `window_days` of daily log
    returns and forecast tomorrow's variance.
  * Compare that forecast with a simple realised-variance estimate. A forecast
    far ABOVE realised vol means the model expects a volatility shock that
    the market hasn't shown yet; far BELOW means calm is expected.
  * z-score that gap over a rolling window: z > +threshold -> daily signal -1
    (expect turbulence, lean short / fade rallies); z < -threshold -> +1
    (expect calm, lean long / buy dips).

Intraday layer (5-minute bars on signal days):
  * RSI and Bollinger bands on the 5-min closes.
  * Daily +1 and an oversold dip (RSI < lower, close < lower band) -> go long.
  * Daily -1 and an overbought pop (RSI > upper, close > upper band) -> go short.
  * Enter on the bar after the trigger, hold until the session close.
    At most one trade per day.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from ..config import Config
from ..indicators import bollinger, rsi

log = logging.getLogger(__name__)


def rolling_garch_forecast(log_ret: pd.Series, cfg: Config) -> pd.Series:
    """One-step-ahead variance forecast for each day, using only prior data."""
    from arch import arch_model

    g = cfg.intraday.garch
    w = g.window_days
    r = log_ret.dropna() * 100       # arch works best with percent returns
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(w, len(r)):
            train = r.iloc[i - w:i]
            try:
                fit = arch_model(train, mean="Constant", vol="GARCH", p=g.p, q=g.q,
                                 dist="normal", rescale=False).fit(disp="off", show_warning=False)
                out[r.index[i]] = fit.forecast(horizon=1, reindex=False).variance.iloc[-1, 0] / 1e4
            except Exception as e:
                log.debug("GARCH fit failed on %s: %s", r.index[i], e)
    return pd.Series(out, name="garch_var")


def daily_signals(daily: pd.DataFrame, cfg: Config, garch_var: pd.Series | None = None) -> pd.DataFrame:
    c = cfg.intraday
    df = daily.copy()
    df["log_ret"] = np.log(df["Adj Close"]).diff()
    # realised variance: rolling mean of squared returns over the past `bb_period` days
    df["real_var"] = (df["log_ret"] ** 2).rolling(c.bb_period).mean().shift()
    if garch_var is None:
        garch_var = rolling_garch_forecast(df["log_ret"], cfg)
    df["garch_var"] = garch_var
    df["premium"] = df["garch_var"] - df["real_var"]
    roll = df["premium"].rolling(c.signal_z_window, min_periods=30)
    df["premium_z"] = (df["premium"] - roll.mean()) / roll.std()
    df["signal_daily"] = 0
    df.loc[df["premium_z"] > c.signal_z_threshold, "signal_daily"] = -1
    df.loc[df["premium_z"] < -c.signal_z_threshold, "signal_daily"] = 1
    return df


def intraday_trades(intra: pd.DataFrame, daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    c = cfg.intraday
    bars = intra.copy()
    bars["day"] = bars.index.normalize()
    bars = bars.join(daily["signal_daily"], on="day")
    bars["rsi"] = rsi(bars["close"], c.rsi_period)
    lo, _, hi = bollinger(bars["close"], c.bb_period, c.bb_std)
    bars["bb_low"], bars["bb_high"] = lo, hi

    long_trig = (bars.signal_daily == 1) & (bars.rsi < c.rsi_lower) & (bars.close < bars.bb_low)
    short_trig = (bars.signal_daily == -1) & (bars.rsi > c.rsi_upper) & (bars.close > bars.bb_high)
    bars["trigger"] = np.where(long_trig, 1, np.where(short_trig, -1, 0))

    trades = []
    for day, g in bars.groupby("day"):
        hits = np.flatnonzero(g["trigger"].to_numpy() != 0)
        if not len(hits) or hits[0] + 1 >= len(g):
            continue
        i = hits[0]
        side = int(g["trigger"].iloc[i])
        entry = g["open"].iloc[i + 1]          # next bar's open: no same-bar look-ahead
        exit_ = g["close"].iloc[-1]
        trades.append({"day": day, "side": side, "entry_time": g.index[i + 1],
                       "entry": entry, "exit": exit_,
                       "return": side * np.log(exit_ / entry)})
    return pd.DataFrame(trades)


def run(daily: pd.DataFrame, intra: pd.DataFrame, cfg: Config, garch_var: pd.Series | None = None) -> dict:
    sig = daily_signals(daily, cfg, garch_var)
    trades = intraday_trades(intra, sig, cfg)
    days = pd.Index(sorted(intra.index.normalize().unique()))
    strat = pd.Series(0.0, index=days)
    if len(trades):
        strat.loc[trades.day] = np.expm1(trades["return"].to_numpy())
    bh = np.log(daily["Adj Close"]).diff().reindex(days).fillna(0)
    return {"returns": {"GARCH + intraday signals": strat, "buy & hold": np.expm1(bh)},
            "daily": sig, "trades": trades}
