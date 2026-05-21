"""Data loaders for the AAPL/SPY pipeline + EWM daily volatility (AFML Snippet 3.1).

The CSVs under ``data/raw/`` have a column-header bug: the header reads
``Open,High,Low,Close,Adj Close,Volume`` but the underlying yfinance frame was
saved after a ``sort_index(axis=1)`` so the actual column order is alphabetical:
``Adj Close, Close, High, Low, Open, Volume``. We override the headers on load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

ACTUAL_COLUMN_ORDER = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume", "company_name"]


def load_ohlcv(ticker: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load a single-ticker OHLCV CSV from ``data/raw/``, fixing the column order."""
    data_dir = data_dir or DATA_DIR
    path = data_dir / f"{ticker}_stock_data_2010_2024.csv"
    df = pd.read_csv(path, header=0, names=ACTUAL_COLUMN_ORDER, skiprows=1)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]


def load_aapl_with_spy() -> pd.DataFrame:
    """Merged AAPL + SPY frame for market-relative features. Index = trading dates."""
    aapl = load_ohlcv("AAPL")
    spy = load_ohlcv("SPY")[["Adj Close", "Volume"]].rename(
        columns={"Adj Close": "SPY_Close", "Volume": "SPY_Volume"}
    )
    return aapl.join(spy, how="inner")


def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """EWM daily-return volatility — AFML Snippet 3.1 (BonusPDF p.26).

    Used to set the horizontal barrier widths in triple-barrier labeling. Output
    is forward-fill safe: NaNs only at the leading edge before EWM warmup.
    """
    returns = close.pct_change()
    return returns.ewm(span=span).std()


def cumulative_returns_path(close: pd.Series, t0, t1) -> pd.Series:
    """Return path from t0 to t1 expressed as ``close/close[t0] - 1``."""
    return close.loc[t0:t1] / close.loc[t0] - 1
