"""Evaluation metrics with statistical significance — triple-barrier era.

The original notebook reported directional accuracy without binomial p-values;
49.9% over 499 days is statistically indistinguishable from 50%. This module
makes that explicit by attaching a p-value to every accuracy figure.

Metric conventions
------------------
- For 3-class labels ``{-1, 0, +1}``, the null is uniform random: ``p_null=1/3``.
- For *directional accuracy when acting*, restrict to predictions ``in {-1, +1}``
  (i.e. ignore "no-action" 0 predictions), compare to ``p_null=1/2``.
- Both metrics use a one-sided binomial test (we only care if it beats chance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from .cv import binomial_pvalue


def directional_accuracy_when_acting(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, int, int]:
    """Accuracy conditioned on the model predicting a non-zero direction.

    Returns ``(accuracy, n_correct, n_acting)``. If ``n_acting`` is 0, returns
    ``(nan, 0, 0)``.
    """
    acting_mask = y_pred != 0
    n_acting = int(acting_mask.sum())
    if n_acting == 0:
        return float("nan"), 0, 0
    correct = int(((y_pred == y_true) & acting_mask).sum())
    return correct / n_acting, correct, n_acting


def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-fold metric bundle. Designed to be one row in the comparison CSV."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    n_acc_correct = int((y_true == y_pred).sum())
    dir_acc, n_dir_correct, n_acting = directional_accuracy_when_acting(y_true, y_pred)

    return {
        "n_test": n,
        "accuracy": acc,
        "binom_p_acc": binomial_pvalue(n_acc_correct, n, p_null=1 / 3),
        "n_acting": n_acting,
        "dir_acc_when_acting": dir_acc,
        "binom_p_dir": (
            binomial_pvalue(n_dir_correct, n_acting, p_null=0.5) if n_acting > 0 else float("nan")
        ),
    }


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold rows to per-model summary with mean ± std."""
    keep = ["accuracy", "binom_p_acc", "dir_acc_when_acting", "binom_p_dir"]
    grouped = results.groupby("model")[keep]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{c}_{stat}" for c, stat in summary.columns]
    summary["n_folds"] = results.groupby("model").size()
    return summary.reset_index()


def confusion_table(y_true: np.ndarray, y_pred: np.ndarray, labels=(-1, 0, 1)) -> pd.DataFrame:
    """Confusion matrix as a labeled DataFrame (rows=true, cols=pred)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(
        cm, index=[f"true_{c}" for c in labels], columns=[f"pred_{c}" for c in labels]
    )
