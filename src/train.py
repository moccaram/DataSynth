"""CV-aware training driver — one harness for all five models.

The driver expects each model to expose ``fit(X, y, sample_weight=None)``,
``predict(X)``, and (optionally) ``predict_proba(X)``. The triple-barrier label
``{-1, 0, +1}`` is shared across all of them.

Sample weights come from AFML Ch.4 — observations whose label intervals overlap
contribute less unique information, so they should count less in the loss. The
simplest implementation is to weight inversely by the number of overlapping
labels (Snippet 4.1); for now the driver supports passing pre-computed weights
or falling back to uniform.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .cv import PurgedKFold
from .eval import fold_metrics


def fit_predict_one_fold(
    model_builder: Callable[[], Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    sample_weight_train: np.ndarray | None = None,
    standardize: bool = True,
) -> tuple[np.ndarray, Any]:
    """Fit on the train fold, predict on the test fold. Returns (y_pred, fitted_model)."""
    if standardize:
        scaler = StandardScaler().fit(X_train.values)
        X_train_s = pd.DataFrame(
            scaler.transform(X_train.values), index=X_train.index, columns=X_train.columns
        )
        X_test_s = pd.DataFrame(
            scaler.transform(X_test.values), index=X_test.index, columns=X_test.columns
        )
    else:
        X_train_s, X_test_s = X_train, X_test
    model = model_builder()
    model.fit(X_train_s, y_train.values, sample_weight=sample_weight_train)
    return model.predict(X_test_s), model


def run_cv(
    model_name: str,
    model_builder: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: PurgedKFold,
    sample_weight: pd.Series | None = None,
    standardize: bool = True,
    extra_columns: dict | None = None,
) -> pd.DataFrame:
    """Run a model across all CV folds. Returns one row per fold."""
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        sw_train = sample_weight.iloc[train_idx].values if sample_weight is not None else None

        y_pred, _ = fit_predict_one_fold(
            model_builder=model_builder,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            sample_weight_train=sw_train,
            standardize=standardize,
        )

        metrics = fold_metrics(y_test.values, y_pred)
        row = {"model": model_name, "fold": fold_idx, **metrics}
        if extra_columns:
            row.update(extra_columns)
        rows.append(row)
    return pd.DataFrame(rows)


def uniqueness_weights(t1: pd.Series) -> pd.Series:
    """Approximate AFML Ch.4 sample-uniqueness weights.

    For each event, count how many other events have overlapping
    ``[start, t1]`` intervals, and weight inversely. Not the rigorous Snippet
    4.1 (which counts overlap proportionally), but the right order of magnitude
    and much faster.
    """
    weights = pd.Series(1.0, index=t1.index)
    t1_arr = t1.values
    start_arr = t1.index.values
    n = len(t1)
    for i in range(n):
        overlap = np.sum((start_arr <= t1_arr[i]) & (t1_arr >= start_arr[i]))
        weights.iloc[i] = 1.0 / max(overlap, 1)
    # normalize so the weights sum to n (mean weight = 1)
    weights *= n / weights.sum()
    return weights
