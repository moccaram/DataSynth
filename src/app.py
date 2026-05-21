"""Gradio demo — AAPL triple-barrier direction classifier (educational).

Loads the XGBoost model (the headline winner in this study, mean test accuracy
~38% vs 33% random) and lets the user pick any date in the available range to
inspect the next-10-day direction prediction with class probabilities.

This is a *portfolio artifact*. The directional accuracy when the model
actually picks a side is ~36% — worse than random. Do not trade on this.
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_aapl_with_spy, get_daily_vol
from src.features import frac_diff_ffd
from src.labeling import cusum_filter, get_events, get_bins, drop_labels
from src.models.xgb_model import XGBTripleBarrier


CLASS_LABELS = {-1: "DOWN (stop-loss first)", 0: "FLAT (time-out, no signal)", 1: "UP (profit-taking first)"}


def build_features_and_labels():
    """Rebuild the full feature matrix + triple-barrier labels at startup."""
    df = load_aapl_with_spy()
    close = df["Adj Close"]
    log_returns = np.log(close).diff().dropna()
    daily_vol = get_daily_vol(close, span=100)

    features = pd.DataFrame(index=df.index)
    features["frac_diff_close"] = frac_diff_ffd(np.log(close).to_frame("c"), 0.4, thres=1e-5)["c"]
    features["frac_diff_volume"] = frac_diff_ffd(
        np.log(df["Volume"].replace(0, np.nan)).to_frame("v"), 0.4, thres=1e-5
    )["v"]
    features["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    features["spy_return"] = np.log(df["SPY_Close"]).diff()
    features["volatility_20d"] = log_returns.rolling(20).std()
    features["rolling_beta"] = (
        log_returns.rolling(30).cov(features["spy_return"])
        / features["spy_return"].rolling(30).var()
    )
    features["day_of_week"] = df.index.dayofweek
    features["vol_regime"] = daily_vol / daily_vol.rolling(252, min_periods=60).median()
    features = features.dropna()

    t_events = cusum_filter(np.log(close), threshold=float(daily_vol.median()))
    events = get_events(
        close=close, t_events=t_events, pt_sl=(2.0, 2.0),
        target=daily_vol, min_ret=0.005, num_days=10,
    )
    labels = get_bins(events, close)
    events_with_labels = events.join(labels[["bin"]])
    events_with_labels = drop_labels(events_with_labels, min_pct=0.05)
    labels = labels.loc[events_with_labels.index]

    aligned = features.index.intersection(labels.index)
    return df, close, features, labels.loc[aligned, "bin"].astype(int), features.loc[aligned]


print("Loading data and training XGBoost (one-time, ~10 sec)...")
DF, CLOSE, FEATURES_FULL, Y_TRAIN, X_TRAIN_ALIGNED = build_features_and_labels()

from sklearn.preprocessing import StandardScaler
SCALER = StandardScaler().fit(X_TRAIN_ALIGNED.values)
MODEL = XGBTripleBarrier(random_state=42)
MODEL.fit(
    pd.DataFrame(SCALER.transform(X_TRAIN_ALIGNED.values), index=X_TRAIN_ALIGNED.index, columns=X_TRAIN_ALIGNED.columns),
    Y_TRAIN.values,
)
print(f"Model trained on {len(X_TRAIN_ALIGNED)} labeled events. Ready.")

VALID_DATES = FEATURES_FULL.index
DEFAULT_DATE = VALID_DATES[-1]


def predict(date_str: str):
    try:
        date = pd.Timestamp(date_str)
    except Exception:
        return "Invalid date format. Use YYYY-MM-DD.", None, None

    available = FEATURES_FULL.index[FEATURES_FULL.index <= date]
    if len(available) == 0:
        return f"No features available on or before {date.date()}. Try a later date.", None, None
    use_date = available[-1]

    x_row = FEATURES_FULL.loc[[use_date]]
    x_scaled = pd.DataFrame(SCALER.transform(x_row.values), index=x_row.index, columns=x_row.columns)
    proba = MODEL.predict_proba(x_scaled)[0]
    pred_class = int(MODEL.classes_[np.argmax(proba)])

    proba_df = pd.DataFrame(
        {"class": [CLASS_LABELS[c] for c in MODEL.classes_], "probability": [f"{p:.1%}" for p in proba]}
    )

    end_idx = DF.index.get_loc(use_date)
    start_idx = max(0, end_idx - 59)
    chart_data = DF["Adj Close"].iloc[start_idx : end_idx + 1]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(chart_data.index, chart_data.values, color="black", lw=1.0)
    ax.scatter([chart_data.index[-1]], [chart_data.iloc[-1]], color="red", s=40, zorder=3, label=f"As-of: {use_date.date()}")
    ax.set_title(f"AAPL adjusted close — 60 days ending {use_date.date()}")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    summary = (
        f"**As-of date:** {use_date.date()}  \n"
        f"**Last close:** ${chart_data.iloc[-1]:.2f}  \n"
        f"**Prediction (next 10 trading days):** {CLASS_LABELS[pred_class]}  \n"
        f"**Confidence (max class probability):** {proba.max():.1%}"
    )
    return summary, proba_df, fig


def build_interface():
    import gradio as gr

    caveat = """
> ⚠️ **This is an educational portfolio artifact, NOT a trading signal.**
>
> Under 5-fold purged k-fold cross-validation (López de Prado, *AFML*, Ch.7), this XGBoost
> classifier reaches mean accuracy ~38% on a 3-class triple-barrier label set (random baseline
> = 33%, p<0.05 in 3 of 5 folds). However, **directional accuracy *when the model picks a side*
> is ~36% — worse than coin flip**. The model is mildly informative about "will something
> happen vs nothing" but uninformative about "up vs down." Do not trade real money on this.
"""

    with gr.Blocks(title="AAPL Triple-Barrier Direction Classifier") as demo:
        gr.Markdown("# AAPL Triple-Barrier Direction Classifier (educational)")
        gr.Markdown(caveat)
        gr.Markdown(
            "Reference-backed financial-ML pipeline: triple-barrier labeling "
            "(AFML Ch.3), fractional differentiation (Ch.5), purged k-fold CV (Ch.7), "
            "XGBoost classifier. Repo: this folder."
        )

        with gr.Row():
            with gr.Column(scale=1):
                date_input = gr.Textbox(
                    label="As-of date (YYYY-MM-DD)",
                    value=str(DEFAULT_DATE.date()),
                    info=f"Valid range: {VALID_DATES[0].date()} → {VALID_DATES[-1].date()}",
                )
                predict_btn = gr.Button("Predict next 10-day direction", variant="primary")
                summary_md = gr.Markdown()
                proba_table = gr.Dataframe(headers=["class", "probability"], label="Class probabilities")

            with gr.Column(scale=2):
                chart = gr.Plot(label="60-day price context")

        predict_btn.click(
            fn=predict, inputs=[date_input], outputs=[summary_md, proba_table, chart]
        )

        gr.Markdown(
            "---\n"
            "Headline result table (mean over 5 purged folds):\n\n"
            "| Model     | Accuracy | Beat random (p<0.05) | Dir.acc when acting |\n"
            "|-----------|----------|----------------------|---------------------|\n"
            "| Majority  | 35.0%    | 0/5 folds            | N/A                 |\n"
            "| SES       | 36.8%    | 2/5 folds            | always abstains     |\n"
            "| ARIMA     | 36.8%    | 2/5 folds            | always abstains     |\n"
            "| LSTM      | 35.8%    | 2/5 folds            | 33% (worse than 50%) |\n"
            "| **XGBoost** | **37.8%** | **3/5 folds**     | 36% (worse than 50%) |\n"
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, share=False)
