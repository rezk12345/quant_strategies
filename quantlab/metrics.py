"""Performance statistics and the comparison chart shared by all three strategies."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def stats(daily_ret: pd.Series, periods: int = 252) -> dict:
    r = daily_ret.dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / periods
    dd = (eq / eq.cummax() - 1).min()
    vol = r.std() * np.sqrt(periods)
    return {"total_return": eq.iloc[-1] - 1,
            "cagr": eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan,
            "volatility": vol,
            "sharpe": r.mean() * periods / vol if vol else np.nan,
            "max_drawdown": dd,
            "days": len(r)}


def compare(series: dict[str, pd.Series], periods: int = 252) -> pd.DataFrame:
    return pd.DataFrame({k: stats(v, periods) for k, v in series.items()}).T


def plot_cumulative(series: dict[str, pd.Series], title: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, r in series.items():
        ax.plot((1 + r.fillna(0)).cumprod() - 1, label=name, lw=1.4)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title)
    ax.set_ylabel("cumulative return")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
