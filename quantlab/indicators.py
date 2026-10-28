"""Technical indicators, implemented with pandas only (no TA-Lib dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def garman_klass_vol(o, h, l, c) -> pd.Series:
    """Single-bar Garman-Klass variance estimate (uses the full OHLC range)."""
    return 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return out.where(~((dn == 0) & up.notna()), 100.0)    # no losses at all -> RSI 100


def bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return mid - n_std * sd, mid, mid + n_std * sd


def atr(h, l, c, period: int = 14) -> pd.Series:
    """Works on a Series or on a wide DataFrame (one column per ticker)."""
    prev = c.shift()
    tr = np.maximum(np.maximum(h - l, (h - prev).abs()), (l - prev).abs())
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / (sd if sd else 1.0)
