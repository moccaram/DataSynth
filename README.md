# DataSynth — Stock Market Time-Series Forecasting

End-to-end stock market analysis and forecasting project using 15 years of daily market data (2010–2024) for SPY, AAPL, AMZN, GOOGL, and MSFT.

The project covers:
- Data collection and preparation
- Feature engineering for AAPL return forecasting
- Classical + deep learning forecasting models (ARIMA, Prophet, LSTM)
- Static split and rolling-window evaluation
- Visual and tabular model comparison

## Project Goals
- Build a reproducible forecasting pipeline for stock returns/prices
- Compare model behavior under realistic temporal evaluation
- Produce analysis artifacts suitable for reporting and portfolio presentation

## Repository Structure

```text
DataSynth/
├── notebooks/
│   └── DataSynthis_ML_JobTask.ipynb
├── src/
│   └── app.py
├── data/
│   ├── raw/            # downloaded and combined stock datasets
│   ├── processed/      # metadata and prepared intermediate files
│   ├── features/       # engineered/selected feature datasets
│   └── splits/         # train/val/test files and LSTM tensors
├── models/             # trained artifacts (ARIMA/LSTM/scaler)
├── reports/
│   ├── figures/        # charts for EDA and model evaluation
│   ├── tables/         # CSV comparison/metric outputs
│   └── docs/           # project reports and assignment PDFs
├── archive/
│   └── Second round/   # unrelated task files retained for reference
├── configs/
│   └── project_config.yaml
├── .gitignore
├── .gitattributes
├── requirements.txt
└── README.md
```

## Data & Workflow
1. Download OHLCV data with `yfinance` for SPY + major tech tickers
2. Build raw datasets in wide and long formats
3. Engineer features on AAPL (lags, volatility, momentum, market-relative features)
4. Prepare target variable (`target_return`) and chronological train/val/test splits
5. Train and evaluate:
   - ARIMA
   - Prophet (with regressors)
   - LSTM
6. Compare static vs rolling-window performance
7. Export figures/tables for reporting

## Main Outputs
- `data/raw/*`: source and merged stock price datasets
- `data/features/*`: engineered feature sets and selected feature lists
- `models/*`: saved model artifacts (`.pkl`, `.h5`)
- `reports/figures/*`: EDA/modeling visualizations
- `reports/tables/*`: numerical model comparison files

## Setup
### 1) Create environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Launch notebook workflow
```bash
jupyter notebook notebooks/DataSynthis_ML_JobTask.ipynb
```

### 4) Optional: run demo app
```bash
python src/app.py
```

## Reproducibility Notes
- Preserve chronological order in splits (no random shuffling for time series).
- Keep model and scaler artifacts versioned together.
- If artifact sizes exceed GitHub limits, store them using Git LFS.

## Suggested Next Improvements
- Add `src/` pipeline scripts (`prepare_data.py`, `train_models.py`, `evaluate.py`) extracted from notebook cells
- Add unit tests for feature engineering and metric functions
- Add a small sample dataset for lightweight quickstart
- Add CI checks (lint + smoke test)

## License
Add a license before publishing (recommended: MIT for portfolio use).
