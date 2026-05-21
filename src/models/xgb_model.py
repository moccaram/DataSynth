"""XGBoost classifier for triple-barrier labels.

Per Jansen Ch.12, gradient-boosted trees are the natural baseline for tabular
financial features and routinely beat LSTMs on these problems. The hyper-
parameters here are conservative (shallow trees, moderate n_estimators) to
avoid overfitting on small per-fold training sets in the purged CV scheme.
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier


def build_xgb_classifier(random_state: int = 42) -> XGBClassifier:
    """Returns a fresh XGBClassifier for one CV fold.

    Output classes use the XGBoost-internal indexing ``{0, 1, 2}`` for
    ``{-1, 0, +1}`` since XGBoost requires non-negative integer labels. The
    training driver wraps this with an encoder.
    """
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=4,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )


class XGBTripleBarrier:
    """Thin wrapper that owns the label encoding from ``{-1, 0, 1}`` ↔ ``{0, 1, 2}``."""

    def __init__(self, random_state: int = 42):
        self.model = build_xgb_classifier(random_state=random_state)
        self.classes_ = np.array([-1, 0, 1])

    def fit(self, X, y, sample_weight=None):
        y_enc = np.asarray(y).astype(int) + 1  # {-1, 0, 1} -> {0, 1, 2}
        self.model.fit(X, y_enc, sample_weight=sample_weight)
        return self

    def predict(self, X):
        y_pred_enc = self.model.predict(X)
        return y_pred_enc - 1

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_
