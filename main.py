"""Run any of the three strategies.

    python main.py unsupervised            # K-Means momentum cluster on S&P 500
    python main.py sentiment               # Twitter engagement-ratio portfolio vs QQQ
    python main.py intraday                # GARCH + 5-minute indicators
    python main.py all --source synthetic  # everything, offline
    python main.py make-sample-data        # write synthetic CSVs for strategies 2 and 3
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings

import pandas as pd

from quantlab.config import ROOT, load_config
from quantlab.metrics import compare, plot_cumulative

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger("main")
OUT = ROOT / "reports"


def show(name: str, res: dict) -> None:
    print(f"\n=== {name} ===")
    print(compare(res["returns"]).to_string(float_format=lambda v: f"{v:,.3f}"))
    png = plot_cumulative(res["returns"], name, OUT / f"{name.split()[0].lower()}.png")
    pd.DataFrame(res["returns"]).to_csv(OUT / f"{name.split()[0].lower()}_returns.csv")
    print(f"chart -> {png}")


def run_unsupervised(cfg, synthetic: bool) -> None:
    from quantlab import data
    from quantlab.strategies import unsupervised

    u = cfg.unsupervised
    if synthetic:
        panel, factors, bench = data.synthetic_panel(u.start, u.end)
    else:
        syms = data.sp500_symbols()
        log.info("downloading %d S&P 500 stocks (first run takes a few minutes)...", len(syms))
        panel = data.download_panel(syms, u.start, u.end, cfg)
        factors = data.fama_french_monthly(u.start)
        bench = data.download_close(u.benchmark, u.start, u.end)
    res = unsupervised.run(panel, factors, bench, cfg)
    show("Unsupervised learning strategy", res)
    last = max(res["weights"])
    print(f"latest portfolio ({last}):",
          json.dumps({k: round(v, 3) for k, v in sorted(res["weights"][last].items(), key=lambda x: -x[1])}))


def run_sentiment(cfg, synthetic: bool) -> None:
    from quantlab import data
    from quantlab.strategies import sentiment

    s = cfg.sentiment
    if synthetic:
        sent, prices, bench = data.synthetic_sentiment(s.start, s.end)
    else:
        path = ROOT / s.csv
        if not path.exists():
            raise SystemExit(f"{path} not found — put your Twitter sentiment CSV there "
                             "(see README) or run with --source synthetic")
        sent = pd.read_csv(path)
        prices, bench = sentiment.load_prices_for(sent, cfg)
    res = sentiment.run(sent, prices, bench, cfg)
    show("Sentiment strategy", res)
    last = max(res["schedule"])
    print(f"holding from {last.date()}: {res['schedule'][last]}")


def run_intraday(cfg, synthetic: bool) -> None:
    from quantlab import data
    from quantlab.strategies import intraday_garch

    c = cfg.intraday
    if synthetic:
        daily, intra = data.synthetic_intraday()
    else:
        for p in (c.daily_csv, c.intraday_csv):
            if not (ROOT / p).exists():
                raise SystemExit(f"{ROOT / p} not found — supply it (see README) or use --source synthetic")
        daily = pd.read_csv(ROOT / c.daily_csv, index_col=0, parse_dates=True)
        intra = pd.read_csv(ROOT / c.intraday_csv, index_col=0, parse_dates=True)
    log.info("fitting rolling GARCH(%d,%d) — about a minute...", c.garch.p, c.garch.q)
    res = intraday_garch.run(daily, intra, cfg)
    show("Intraday GARCH strategy", res)
    t = res["trades"]
    if len(t):
        print(f"trades: {len(t)}  long {int((t.side == 1).sum())} / short {int((t.side == -1).sum())}"
              f"  hit rate {(t['return'] > 0).mean():.0%}")
        t.to_csv(OUT / "intraday_trades.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="ML & quant trading strategies")
    p.add_argument("command", choices=["unsupervised", "sentiment", "intraday", "all", "make-sample-data"])
    p.add_argument("--source", choices=["live", "synthetic"])
    p.add_argument("--config")
    a = p.parse_args()
    cfg = load_config(a.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("yfinance", "peewee", "cvxpy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    OUT.mkdir(exist_ok=True)
    synthetic = (a.source or cfg.data.source) == "synthetic"

    if a.command == "make-sample-data":
        from quantlab.data import write_synthetic_csvs
        write_synthetic_csvs(cfg)
        print("wrote", cfg.sentiment.csv, cfg.intraday.daily_csv, cfg.intraday.intraday_csv)
        return
    jobs = {"unsupervised": run_unsupervised, "sentiment": run_sentiment, "intraday": run_intraday}
    for name in (jobs if a.command == "all" else [a.command]):
        jobs[name](cfg, synthetic)


if __name__ == "__main__":
    main()
