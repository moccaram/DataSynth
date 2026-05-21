"""Triple-barrier labeling — AFML Ch.3 (BonusPDF pp.26-34).

The triple-barrier method assigns each event one of three labels based on which
of three barriers is hit first:

- ``+1`` — upper (profit-taking) horizontal barrier hit first
- ``-1`` — lower (stop-loss) horizontal barrier hit first
- ``0``  — vertical (max holding period) barrier hit first

The horizontal barriers are scaled by a per-event volatility estimate (typically
EWM daily vol, ``get_daily_vol`` in ``src/data.py``). This is a port of AFML
Snippets 3.2-3.5 and Rambo's cleaner ``get_triple_barrier_label`` (his repo,
``Chapter_3.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_pt_sl_on_t1(
    close: pd.Series, events: pd.DataFrame, pt_sl: tuple[float, float]
) -> pd.DataFrame:
    """AFML Snippet 3.2 (BonusPDF p.27). Find time of first barrier touch.

    Parameters
    ----------
    close : pd.Series
        Closing-price series, indexed by date.
    events : pd.DataFrame
        Required columns: ``t1`` (vertical-barrier date or NaT), ``target``
        (vol estimate at the event), ``side`` (+1 for long, -1 for short; if
        we don't know side, pass +1 for all).
    pt_sl : (float, float)
        Profit-taking and stop-loss multipliers of ``target``. Pass 0 to disable
        a barrier.

    Returns
    -------
    pd.DataFrame indexed like ``events`` with columns ``t1, pt, sl`` containing
    the first-touch timestamps (NaT if never touched).
    """
    out = events[["t1"]].copy()
    pt = pt_sl[0] * events["target"] if pt_sl[0] > 0 else pd.Series(np.nan, index=events.index)
    sl = -pt_sl[1] * events["target"] if pt_sl[1] > 0 else pd.Series(np.nan, index=events.index)

    for t0, t1 in events["t1"].fillna(close.index[-1]).items():
        path_prices = close.loc[t0:t1]
        path_returns = (path_prices / close.loc[t0] - 1) * events.at[t0, "side"]
        sl_hits = path_returns[path_returns < sl[t0]]
        pt_hits = path_returns[path_returns > pt[t0]]
        out.at[t0, "sl"] = sl_hits.index.min() if len(sl_hits) else pd.NaT
        out.at[t0, "pt"] = pt_hits.index.min() if len(pt_hits) else pd.NaT
    return out


def add_vertical_barrier(
    close: pd.Series, t_events: pd.DatetimeIndex, num_days: int
) -> pd.Series:
    """AFML Snippet 3.4 (BonusPDF p.30). Vertical (time-limit) barriers.

    Returns a Series indexed by ``t_events`` whose values are ``num_days`` later,
    snapped to the next available trading day; events too close to the end of
    the series are dropped.
    """
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    return pd.Series(close.index[t1], index=t_events[: len(t1)])


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: tuple[float, float],
    target: pd.Series,
    min_ret: float,
    num_days: int | None = None,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """AFML Snippet 3.3 (BonusPDF p.29). Run triple-barrier for a batch of events.

    Returns a DataFrame indexed by event start time with columns:

    - ``t1`` (timestamp of the *first* barrier hit — earliest of vertical/pt/sl)
    - ``vertical_t1`` (the original vertical-barrier date)
    - ``barrier_hit`` (one of ``"vertical"`` / ``"pt"`` / ``"sl"`` — what was hit
      first; used by ``get_bins`` to produce the {-1, 0, +1} label)
    - ``target`` (vol estimate at the event)

    If ``side`` is provided, it is propagated for downstream meta-labeling.
    """
    target = target.reindex(t_events).dropna()
    target = target[target > min_ret]

    if num_days is not None:
        vertical_t1 = add_vertical_barrier(close, target.index, num_days)
    else:
        vertical_t1 = pd.Series(pd.NaT, index=target.index)

    if side is None:
        side_ = pd.Series(1.0, index=target.index)
    else:
        side_ = side.reindex(target.index).fillna(1.0)

    events = pd.concat(
        {"t1": vertical_t1, "target": target, "side": side_}, axis=1
    ).dropna(subset=["target"])
    touches = apply_pt_sl_on_t1(close, events, pt_sl)

    # Drop events where no barrier ever fires (can't happen with a vertical
    # barrier present, but defensive against future config changes).
    touches = touches.dropna(subset=["t1", "pt", "sl"], how="all")
    events = events.loc[touches.index]

    # Earliest touch among (vertical, pt, sl); record which barrier won.
    all_touches = touches[["t1", "pt", "sl"]]
    earliest = all_touches.min(axis=1)
    # Manual row-wise argmin: pandas' idxmin chokes on all-NaT slices.
    barrier_hit = pd.Series("vertical", index=all_touches.index)
    pt_arr = all_touches["pt"]
    sl_arr = all_touches["sl"]
    vert_arr = all_touches["t1"]
    # Replace NaT with a very large date for comparison purposes
    far = pd.Timestamp.max
    cmp = pd.DataFrame(
        {
            "pt": pt_arr.fillna(far),
            "sl": sl_arr.fillna(far),
            "vertical": vert_arr.fillna(far),
        }
    )
    barrier_hit = cmp.idxmin(axis=1)

    events["vertical_t1"] = events["t1"]
    events["t1"] = earliest
    events["barrier_hit"] = barrier_hit.astype(str)
    if side is None:
        events = events.drop("side", axis=1)
    return events.dropna(subset=["t1"])


def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """AFML Snippet 3.5 (BonusPDF p.30). Convert event outcomes to {-1, 0, +1}.

    Full triple-barrier semantics: the label depends on which barrier was hit
    *first*:

    - ``barrier_hit == "pt"``  → ``+1`` (profit-taking, scaled by ``side``)
    - ``barrier_hit == "sl"``  → ``-1`` (stop-loss, scaled by ``side``)
    - ``barrier_hit == "vertical"`` → ``0`` (no signal; the time limit ran out
      before either horizontal barrier was hit)

    If meta-labeling (``side`` column present), maps to ``{0, 1}`` for
    "don't act" vs "act in this side".
    """
    events_ = events.dropna(subset=["t1"]).copy()
    px_idx = events_.index.union(events_["t1"].values).unique()
    px = close.reindex(px_idx, method="bfill")

    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index].values - 1
    if "side" in events_.columns:
        out["ret"] *= events_["side"].values

    if "barrier_hit" in events_.columns:
        # Full triple-barrier: 0 when the vertical barrier (time limit) wins.
        out["bin"] = 0
        out.loc[events_["barrier_hit"] == "pt", "bin"] = 1
        out.loc[events_["barrier_hit"] == "sl", "bin"] = -1
        if "side" in events_.columns:
            # meta-labeling: collapse to {0, 1} = "don't act / act"
            out.loc[out["ret"] <= 0, "bin"] = 0
            out.loc[out["bin"] != 0, "bin"] = 1
    else:
        # Fallback to AFML Snippet 3.5 default (sign of return)
        out["bin"] = np.sign(out["ret"]).astype(int)
    out["bin"] = out["bin"].astype(int)
    return out


def drop_labels(events: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
    """AFML Snippet 3.8 (BonusPDF p.34). Drop labels with < ``min_pct`` support.

    Repeats until every remaining label has at least ``min_pct`` of observations
    or fewer than 3 classes remain.
    """
    while True:
        counts = events["bin"].value_counts(normalize=True)
        if counts.min() > min_pct or len(counts) < 3:
            break
        smallest = counts.idxmin()
        events = events[events["bin"] != smallest]
        print(f"Dropped label {smallest}: {100 * counts.min():.2f}% of observations")
    return events


def cusum_filter(series: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter — AFML §2.5.2 (general technique).

    Generates event start times where the cumulative sum of returns (in either
    direction) exceeds ``threshold``. Resets after each event. Returns a
    DatetimeIndex of event-trigger timestamps.

    Avoids the "predict on every bar" inefficiency by only labeling at
    statistically interesting moments.
    """
    t_events, s_pos, s_neg = [], 0.0, 0.0
    diff = series.diff().fillna(0)
    for t, d in diff.items():
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)
        if s_neg < -threshold:
            s_neg = 0.0
            t_events.append(t)
        elif s_pos > threshold:
            s_pos = 0.0
            t_events.append(t)
    return pd.DatetimeIndex(t_events)
