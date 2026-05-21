"""ARIMA wrapper for triple-barrier classification.

ARIMA forecasts a continuous next-step return; we threshold it into ``{-1, 0, +1}``
using ``±k·σ`` where ``σ`` is the daily-vol estimate at the event time. The
``k`` factor matches the profit-taking / stop-loss multiplier used for labeling
so that the discretization is consistent with the label scheme.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ARIMAClassifier:
    """Wraps statsmodels ARIMA so it can sit in the same fit/predict loop as XGB/LSTM.

    The model is fit on the log-price series implied by the training rows (the
    feature matrix carries the volatility estimate per row, used to threshold).

    Required X columns: ``frac_diff_close`` (used as a proxy for the underlying
    log-price level we want to forecast) and ``target_vol`` (per-event vol used
    to set the ±k·σ threshold).
    """

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1), threshold_k: float = 0.5):
        self.order = order
        self.threshold_k = threshold_k
        self.fitted_ = None
        self.train_tail_value_: float = 0.0
        self.classes_: np.ndarray = np.array([-1, 0, 1])

    def fit(self, X, y, sample_weight=None):
        series = X["frac_diff_close"].astype(float).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fitted_ = ARIMA(series, order=self.order).fit()
        self.train_tail_value_ = float(series[-1])
        return self

    def predict(self, X):
        n = len(X)
        forecast = self.fitted_.forecast(steps=n)
        # convert forecast deltas back to per-step returns vs the tail of training
        last = self.train_tail_value_
        per_step_return = np.diff(np.concatenate([[last], np.asarray(forecast)]))

        thresholds = self.threshold_k * X["target_vol"].astype(float).to_numpy()
        preds = np.zeros(n, dtype=int)
        preds[per_step_return > thresholds] = 1
        preds[per_step_return < -thresholds] = -1
        return preds

    def predict_proba(self, X):
        # ARIMA isn't probabilistic in the triple-barrier sense; collapse hard
        # predictions into a one-hot for log-loss calculation.
        preds = self.predict(X)
        proba = np.zeros((len(preds), 3))
        for i, c in enumerate(self.classes_):
            proba[preds == c, i] = 1.0
        return proba
