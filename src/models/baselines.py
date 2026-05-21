"""Naïve and SES baselines for triple-barrier classification.

The original notebook found SES beat the LSTM under rolling evaluation, which
was the most interesting result. We keep both baselines under the new label
scheme to see whether that finding survives a fair (purged-CV) comparison.

Both classes follow a uniform fit/predict_proba/predict interface so the
training driver can iterate over models polymorphically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


class MajorityClassClassifier:
    """Predicts the most common class from the training set every time.

    The honest "do nothing" baseline. A useful sanity check: any model that
    fails to beat this on accuracy isn't doing anything.
    """

    def __init__(self):
        self.majority_class_: int | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        counts = np.bincount((y - self.classes_.min()).astype(int))
        self.majority_class_ = int(self.classes_[np.argmax(counts)])
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_class_, dtype=int)

    def predict_proba(self, X):
        n = len(X)
        proba = np.zeros((n, len(self.classes_)))
        idx = int(np.where(self.classes_ == self.majority_class_)[0][0])
        proba[:, idx] = 1.0
        return proba


class SESClassifier:
    """Simple exponential smoothing applied to the *label series*, then sign-mapped.

    Approach: fit ``SimpleExpSmoothing`` on the train labels (treated as a
    continuous signal in ``{-1, 0, +1}``), forecast next-step level, and round
    back to the nearest class. Not a real classifier — a sanity check that the
    label sequence has any short-horizon autocorrelation at all.
    """

    def __init__(self, smoothing_level: float | None = None):
        self.smoothing_level = smoothing_level
        self.model_ = None
        self.last_forecast_: float = 0.0
        self.classes_: np.ndarray | None = None

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, dtype=float)
        self.classes_ = np.array(sorted(np.unique(y.astype(int))))
        self.model_ = SimpleExpSmoothing(y, initialization_method="estimated").fit(
            smoothing_level=self.smoothing_level, optimized=self.smoothing_level is None
        )
        fc = self.model_.forecast(1)
        self.last_forecast_ = float(fc[0] if hasattr(fc, "__getitem__") else fc)
        return self

    def predict(self, X):
        # SES gives a single forecast; broadcast it across the test window.
        # The "label series has very weak structure" finding is intentional —
        # this is meant to be a sanity baseline.
        n = len(X)
        forecast = self.last_forecast_
        return np.full(n, self._nearest_class(forecast), dtype=int)

    def predict_proba(self, X):
        n = len(X)
        pred_class = self._nearest_class(self.last_forecast_)
        proba = np.zeros((n, len(self.classes_)))
        idx = int(np.where(self.classes_ == pred_class)[0][0])
        proba[:, idx] = 1.0
        return proba

    def _nearest_class(self, value: float) -> int:
        return int(self.classes_[np.argmin(np.abs(self.classes_ - value))])
