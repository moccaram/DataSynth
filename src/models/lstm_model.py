"""Refined LSTM for triple-barrier classification.

Architectural choices vs the original notebook
-----------------------------------------------
The original model (128→64 units, MSE, no clipnorm) collapsed to predicting the
mean. The refinements here are each anchored to a specific reference:

- **Smaller**: 32→16 units, ~10× fewer params. Jansen's univariate-LSTM notebook
  uses 10 units on S&P daily. Karpathy warns that over-parameterized RNNs
  *"do not always show convincing signs of generalizing in the correct way."*
- **Gradient clipping** (``clipnorm=1.0``): Goodfellow §10.11.1, eq 10.48-49
  (PDF p.414). Without it, the 60-step BPTT chain has the "cliff" landscape
  shown in figure 10.17 and SGD updates can be catastrophically large.
- **Recurrent dropout** (``recurrent_dropout=0.1``): Goodfellow §10.11.2
  (PDF p.415). Drops the time-axis connections, which is where the
  generalization problem lives — sequence-level Dropout drops feature
  dimensions and misses this.
- **Softmax over 3 classes** with ``categorical_crossentropy``: aligns the
  loss with the directional-accuracy metric, fixing the original's MSE-vs-
  direction mismatch.
- **Forget-gate bias = 1**: Keras default (``unit_forget_bias=True``), kept
  explicit so a reader sees Goodfellow §10.10.2 (PDF p.412) is honored.

Sample weighting
----------------
AFML Pitfall #7 (Table 1.2) — non-IID samples need uniqueness weighting. The
training driver passes a ``sample_weight`` array if available; ``categorical_
crossentropy`` honors it natively via Keras.
"""

from __future__ import annotations

import numpy as np


def build_lstm(
    sequence_length: int,
    n_features: int,
    n_classes: int = 3,
    lstm_units: tuple[int, int] = (32, 16),
    dropout: float = 0.2,
    recurrent_dropout: float = 0.1,
    learning_rate: float = 1e-3,
    clipnorm: float = 1.0,
):
    """Build the refined LSTM. Import inside the function so tensorflow doesn't load at module import."""
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            Input(shape=(sequence_length, n_features)),
            LSTM(
                lstm_units[0],
                return_sequences=True,
                recurrent_dropout=recurrent_dropout,
                unit_forget_bias=True,  # Goodfellow §10.10.2 (PDF p.412)
            ),
            Dropout(dropout),
            LSTM(
                lstm_units[1],
                return_sequences=False,
                recurrent_dropout=recurrent_dropout,
                unit_forget_bias=True,
            ),
            Dropout(dropout),
            Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_sequences(
    X: np.ndarray, y: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``(n_obs, n_features)`` into ``(n_seq, sequence_length, n_features)``.

    The target at sequence index ``i`` is ``y[i + sequence_length - 1]`` — the
    model predicts the label at the END of each window, not the next step
    beyond it (the next-step view is handled at the event level by the
    triple-barrier ``t1``).
    """
    n = len(X) - sequence_length + 1
    if n <= 0:
        return np.empty((0, sequence_length, X.shape[1])), np.empty((0,))
    X_seq = np.stack([X[i : i + sequence_length] for i in range(n)])
    y_seq = y[sequence_length - 1 :]
    return X_seq, y_seq


class LSTMTripleBarrier:
    """Wraps the refined LSTM with the same fit/predict interface as other models.

    Owns label encoding ``{-1, 0, +1} -> {0, 1, 2}`` and sequence construction
    so the training driver doesn't have to special-case it.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 8,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 15,
        verbose: int = 0,
        random_state: int = 42,
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state
        self.model = None
        self.classes_ = np.array([-1, 0, 1])
        self.history_ = None

    def fit(self, X, y, sample_weight=None):
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.utils import to_categorical

        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = np.asarray(X)
        y_enc = np.asarray(y).astype(int) + 1
        X_seq, y_seq = build_sequences(X_arr, y_enc, self.sequence_length)
        if len(X_seq) == 0:
            raise ValueError(f"Not enough rows ({len(X_arr)}) for sequence_length={self.sequence_length}")
        y_onehot = to_categorical(y_seq, num_classes=3)

        sw_seq = None
        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight)
            sw_seq = sw_arr[self.sequence_length - 1 :]

        self.model = build_lstm(
            sequence_length=self.sequence_length,
            n_features=X_arr.shape[1],
        )
        callbacks = [
            EarlyStopping(monitor="loss", patience=self.patience, restore_best_weights=True)
        ]
        self.history_ = self.model.fit(
            X_seq,
            y_onehot,
            sample_weight=sw_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=callbacks,
            shuffle=False,
        )
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X)
        # Always pad the start with `sequence_length - 1` copies of the first row
        # so the output has exactly one prediction per input row. (Without this
        # pad we'd lose the first 59 rows of every test fold.)
        n_pad = self.sequence_length - 1
        pad = np.tile(X_arr[:1], (n_pad, 1))
        X_padded = np.vstack([pad, X_arr])
        X_seq, _ = build_sequences(
            X_padded, np.zeros(len(X_padded)), self.sequence_length
        )
        return self.model.predict(X_seq, verbose=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
