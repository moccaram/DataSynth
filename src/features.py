"""Fractional differentiation — AFML Ch.5 §5.4 (BonusPDF p.46).

Why this module exists
----------------------
Log-returns achieve stationarity but destroy memory: the binomial weights
``(1-B)^d`` collapse to ``[1, -1, 0, 0, ...]`` at ``d=1``. For ``d ∈ (0, 1)``
the weights decay as a long power-law tail, so the series stays stationary
while retaining a long memory of past prices (Table 5.1 in AFML shows most
liquid futures reach ADF stationarity at ``d < 0.6``, and the majority at
``d < 0.3``).

This is a port of AFML Snippets 5.1, 5.3, 5.4 (BonusPDF pp.48, 51, 53).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gamma


def get_ffd_weights(d: float, thres: float = 1e-5, max_size: int = 1024) -> np.ndarray:
    """Binomial-series weights for the fractional-differencing operator ``(1-B)^d``.

    Cuts the series off once ``|w_k| < thres``. Uses ``scipy.special.gamma`` for
    a vectorized closed form rather than the recursive loop in AFML Snippet 5.1
    — same values, faster and avoids accumulated float error in long series.

    Returns
    -------
    np.ndarray of shape ``(n,)`` ordered from oldest to newest:
        ``[w_{n-1}, w_{n-2}, ..., w_1, w_0]`` so the dot product with
        ``series[t-n+1 : t+1]`` is the differenced value at ``t``.
    """
    k = np.arange(max_size)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (-1) ** k * gamma(d + 1) / (gamma(k + 1) * gamma(d - k + 1))
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    cutoff = np.argmax(np.abs(w) < thres) if np.any(np.abs(w) < thres) else max_size
    if cutoff == 0:
        cutoff = max_size
    return w[:cutoff][::-1]


def frac_diff_ffd(series: pd.Series | pd.DataFrame, d: float, thres: float = 1e-5) -> pd.DataFrame:
    """Fixed-width fractional differencing — AFML Snippet 5.3 (BonusPDF p.51).

    The fixed-width window keeps weights stable through time (unlike the
    expanding-window variant in Snippet 5.2 which downweights early observations).
    """
    if isinstance(series, pd.Series):
        series = series.to_frame()
    w = get_ffd_weights(d, thres=thres)  # shape (width+1,)
    width = len(w) - 1
    out = {}
    for col in series.columns:
        s = series[[col]].ffill().dropna()
        if len(s) <= width:
            out[col] = pd.Series(index=s.index[width:], dtype=float)
            continue
        values = s[col].to_numpy()
        # Vectorized: build a (n_out, width+1) sliding-window matrix and dot with w
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(values, width + 1)
        diffed = windows @ w
        out[col] = pd.Series(diffed, index=s.index[width:])
    return pd.concat(out, axis=1)


def find_min_d(series: pd.Series, d_range=(0.0, 1.0), n_steps: int = 11, thres: float = 1e-5) -> pd.DataFrame:
    """Sweep ``d`` and return ADF stat + correlation — AFML Snippet 5.4 (BonusPDF p.53).

    Use to pick the smallest ``d`` for which the FFD-differenced log-price passes
    the ADF stationarity test at 95% (statistic < critical value ≈ -2.86).
    Returns a frame indexed by ``d`` with columns: ``adf_stat, p_value, n_obs,
    crit_95, corr_with_original``.
    """
    from statsmodels.tsa.stattools import adfuller

    log_series = np.log(series.dropna()).to_frame(name=series.name or "value")
    results = {}
    for d in np.linspace(d_range[0], d_range[1], n_steps):
        diffed = frac_diff_ffd(log_series, d, thres=thres).dropna()
        if len(diffed) < 50:
            continue
        col = diffed.columns[0]
        adf = adfuller(diffed[col], maxlag=1, regression="c", autolag=None)
        aligned = log_series.loc[diffed.index, col]
        corr = float(aligned.corr(diffed[col]))
        results[round(d, 3)] = {
            "adf_stat": adf[0],
            "p_value": adf[1],
            "n_obs": adf[3],
            "crit_95": adf[4]["5%"],
            "corr_with_original": corr,
        }
    return pd.DataFrame(results).T.rename_axis("d")


def rolling_zscore(series: pd.Series, window: int = 252, min_periods: int | None = None) -> pd.Series:
    """Rolling z-score with leak-free statistics (uses only the trailing window).

    Stronger than a single fit-on-train ``StandardScaler`` because regime shifts
    don't carry stale means forward into the test set.
    """
    min_periods = min_periods or max(window // 4, 20)
    mu = series.rolling(window=window, min_periods=min_periods).mean()
    sd = series.rolling(window=window, min_periods=min_periods).std()
    return (series - mu) / sd.replace(0, np.nan)
