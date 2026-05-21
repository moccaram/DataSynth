"""Purged k-fold cross-validation — AFML Ch.7 (BonusPDF pp.62-67).

Standard k-fold leaks information in finance because labels span time intervals.
If a training label's interval ``[t_i, t1_i]`` overlaps a test label's interval
``[t_j, t1_j]``, the two share underlying price information and the train/test
boundary is fictitious. ``PurgedKFold`` drops the offending training samples;
an additional ``pctEmbargo`` buffer drops samples immediately *after* each test
fold to prevent reverse leakage from the test set into a later train fold.

This is a port of AFML Snippets 7.2-7.3 (BonusPDF pp.65-66). The canonical class
inherits from sklearn's ``_BaseKFold`` so it works as a drop-in replacement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection._split import _BaseKFold


class PurgedKFold(_BaseKFold):
    """K-fold CV with purging + optional embargo. AFML Snippet 7.3 (BonusPDF p.66)."""

    def __init__(self, n_splits: int = 5, t1: pd.Series | None = None, pct_embargo: float = 0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("`t1` must be a pd.Series of label-end timestamps")
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        if not X.index.equals(self.t1.index):
            raise ValueError("X.index must equal t1.index")
        indices = np.arange(X.shape[0])
        embargo_size = int(X.shape[0] * self.pct_embargo)
        test_ranges = [(arr[0], arr[-1] + 1) for arr in np.array_split(indices, self.n_splits)]

        for i, j in test_ranges:
            t0 = self.t1.index[i]
            test_indices = indices[i:j]
            max_t1_in_test = self.t1.iloc[test_indices].max()
            max_t1_pos = self.t1.index.searchsorted(max_t1_in_test)
            # left train: rows whose label ended before test starts
            left_train = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            # right train: rows starting after max-t1 + embargo
            if max_t1_pos < X.shape[0]:
                right_train = indices[max_t1_pos + embargo_size :]
            else:
                right_train = np.array([], dtype=int)
            train_indices = np.concatenate([left_train, right_train])
            yield train_indices, test_indices


def get_embargo_times(times: pd.DatetimeIndex, pct_embargo: float) -> pd.Series:
    """AFML Snippet 7.2 (BonusPDF p.65). Map each timestamp to its embargo end."""
    step = int(times.shape[0] * pct_embargo)
    if step == 0:
        return pd.Series(times, index=times)
    embargo = pd.Series(times[step:], index=times[:-step])
    return pd.concat([embargo, pd.Series(times[-1], index=times[-step:])])


def binomial_pvalue(n_correct: int, n_total: int, p_null: float = 0.5) -> float:
    """One-sided binomial p-value: ``P(X >= n_correct | n=n_total, p=p_null)``.

    Used to test whether observed accuracy or directional accuracy exceeds the
    null. For three-class targets, pass ``p_null=1/3``; for binary direction
    after dropping 0-labels, pass ``p_null=0.5``.
    """
    return float(stats.binomtest(n_correct, n_total, p=p_null, alternative="greater").pvalue)


def proportion_ci(n_correct: int, n_total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson 95% CI for an accuracy proportion. More accurate than normal-approx for small n."""
    if n_total == 0:
        return (np.nan, np.nan)
    ci = stats.binomtest(n_correct, n_total).proportion_ci(
        confidence_level=1 - alpha, method="wilson"
    )
    return float(ci.low), float(ci.high)
