# References

The four sources that anchored this work. Each entry lists *what I actually
used from it* — not just a citation but the specific chapters / pages / code
that shaped the implementation.

---

## 1. López de Prado, *Advances in Financial Machine Learning*

**Wiley, 2018.** The single most influential reference for this project.
Chapters 3, 5, and 7 each diagnose a pitfall that the v1 notebook had
committed and prescribe a concrete fix with runnable code.

| Chapter | Topic | Used for |
|---|---|---|
| **Ch.3** | Triple-barrier labeling, meta-labeling | Replaced fixed-time-horizon `log_return.shift(-1)` target with the full barrier scheme. Snippets 3.1–3.5, 3.8 ported into [src/labeling.py](../src/labeling.py). |
| **Ch.5** | Fractional differentiation | Replaced `log_return` features with fractionally-differenced log-price (`d ≈ 0.4`). Snippets 5.1, 5.3, 5.4 ported into [src/features.py](../src/features.py). The big idea: integer differentiation destroys memory; fractional preserves it while still achieving stationarity. |
| **Ch.7** | Cross-validation in finance | Replaced single chronological train/val/test split with PurgedKFold + embargo. Snippet 7.3 ported into [src/cv.py](../src/cv.py). |
| Ch.1 (Table 1.2) | Common Pitfalls in Financial ML | The canonical 10-row checklist of what goes wrong in financial ML and which chapter fixes it. Print this and put it above your desk. |
| Ch.4 (Snippet 4.1) | Sample uniqueness weighting | Used an *approximate* version of this in [src/train.py](../src/train.py) — exact version flagged as future work. |
| Ch.12 (future work) | Combinatorial purged CV | Flagged in REPORT §7 as the next step beyond single-path purged k-fold. |

**Companion code (and the actual port I used):** Charles Rambo's Python
reimplementation at [github.com/charlesrambo/advances_in_financial_ML](https://github.com/charlesrambo/advances_in_financial_ML)
(`Chapter_3.py`, `Chapter_5.py`, `Chapter_7.py`). My ports add type hints,
defensive handling for all-NaT slices, and vectorized FFD via
`numpy.lib.stride_tricks.sliding_window_view`.

---

## 2. Goodfellow, Bengio, Courville, *Deep Learning*

**MIT Press, 2016. Free online at [deeplearningbook.org](https://www.deeplearningbook.org/).**
The textbook anchor for the LSTM rework.

| Section | Topic | Used for |
|---|---|---|
| **§10.7** | The challenge of long-term dependencies | The math reason vanilla RNNs fail past ~10-20 steps — Jacobian eigenvalues `λ^t` either vanish or explode. Provides the *necessity* argument for the LSTM cell state. |
| **§10.10.1** | LSTM forward equations | Eq 10.40-10.44 are the forget/input/output gates and cell state. Provides the architectural anchor for [src/models/lstm_model.py](../src/models/lstm_model.py). |
| **§10.10.2** | LSTM variants | Jozefowicz et al. (2015) finding that *"adding a bias of 1 to the LSTM forget gate ... makes the LSTM as strong as the best of the explored architectural variants."* Keras default; kept explicit in the model. |
| **§10.11.1** | Gradient clipping | Eq 10.48-10.49 and figure 10.17 (the "cliff" landscape). **The single biggest fix to the v1 model.** Without `clipnorm` the 60-step BPTT chain catastrophically diverges — the loss flatlining at epoch 2 was exactly this failure. |
| **§10.11.2** | Regularization for info flow | Pascanu et al. (2013) regularizer. Keras doesn't expose this; `recurrent_dropout` is the operational proxy. |
| Ch.11 §11.1 | Performance metrics | *"Your error metric will guide all of your future actions."* The reason MSE-on-returns + report-directional-accuracy is a contradiction. |

---

## 3. Jansen, *Machine Learning for Algorithmic Trading*, 2nd ed.

**Packt, 2020.** The most practical of the four — every chapter has runnable
notebooks at [github.com/stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading).

| Chapter / Notebook | Topic | Used for |
|---|---|---|
| **Ch.12** | Gradient-boosted machines | The argument that GBMs are the canonical strong baseline on small tabular financial datasets. *The reason XGBoost was the headline winner here.* |
| **Ch.19 NB 01** (`01_univariate_time_series_regression.ipynb`) | Univariate LSTM on S&P daily | Showed me a credible LSTM baseline used **10 units**, not 128/64. Anchors the downsizing in [src/models/lstm_model.py](../src/models/lstm_model.py). |
| Ch.19 NB 02 (`02_stacked_lstm_with_feature_embeddings.ipynb`) | Calendar features as embeddings | Flagged in REPORT §7 — replace the integer `day_of_week` with an `Embedding(7, 2)` layer. Future work for the LSTM. |
| Ch.6 | Sample weighting, cross-validation in time-series | Reinforces the AFML Ch.7 case for purged CV from a slightly different angle. |

---

## 4. Karpathy (2015), *The Unreasonable Effectiveness of Recurrent Neural Networks*

**Free at [karpathy.github.io/2015/05/21/rnn-effectiveness/](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).**

Read this first if you've never trained an RNN. The mental model lasts:

- *Input `x_t`, hidden state `h_t` as a running summary, output `y_t` mediated through `h_t`.*
- *"RNNs do not always show convincing signs of generalizing in the correct way."* — the generalization caveat that motivates downsizing.

Doesn't appear directly in the implementation but informed the v2 decisions
to (a) downsize the LSTM, (b) treat the result with skepticism, (c) lean
toward simpler baselines.

---

## How these four sources fit together

```
                Karpathy blog         ← read in 30 min, mental model
                       │
                       ▼
                Goodfellow Ch.10      ← mathematical foundation, the "why" of LSTM
                       │
                       ▼
              AFML Ch.3, 5, 7         ← why standard ML fails in finance
                       │
                       ▼
              Jansen Ch.12, 19        ← practical runnable code, the "how"
                       │
                       ▼
                  THIS PROJECT        ← apply all of the above to AAPL
```

The order matters. Reading Jansen first (the most practical) without
Goodfellow (the most foundational) produced the v1 notebook that *looked*
correct but had a misaligned loss/metric and no gradient clipping. Reading
AFML last would have been fine but reading it third was right — without
the LSTM-mechanics grounding, the AFML "labels are the problem" message
would have felt like a sidetrack.

---

## Other sources consulted (not used directly)

- The AFML "BonusPDF" supplement (`GIL2476_AdvancesFinancial_BonusPDF.pdf`) —
  the canonical equations and code snippets. Useful as a reference; not a
  substitute for the full chapter prose.
- scikit-learn `_BaseKFold` source code — for the inheritance pattern in
  `PurgedKFold`.
- statsmodels `adfuller` documentation — for understanding the ADF stat /
  critical value interpretation.
