# Bank Stock Price & Volatility Forecasting

Vietnamese bank stock prediction using machine learning and deep learning models.

## Project Structure

```
.
├── langkinh1_xgboost_shap.py              # Phase 1: XGBoost + SHAP analysis
├── langkinh2_neuralprophet_seasonality.py # Phase 1: NeuralProphet seasonality
├── langkinh3_tft_memory.py                # Phase 1: TFT memory analysis
├── run_4fold_all_models_both_targets.py   # Phase 2: 4-fold CV evaluation
├── run_perday_5models.py                  # Phase 2: Per-day trading simulation
├── run_market_event_validation.py          # Phase 2: High-volatility days analysis
├── run_sensitivity_analysis.py             # Phase 2: Hyperparameter sensitivity
├── split_by_bank.py                       # Data preprocessing
│
├── thesis_documentation/                   # Thesis documentation (Markdown)
├── TienXuLy/                             # Data processing scripts
│
├── banks_BID_dataset.csv                 # BIDV stock data
├── banks_CTG_dataset.csv                 # VietinBank stock data
├── banks_VCB_dataset.csv                 # Vietcombank stock data
└── requirements.txt                      # Python dependencies
```

## Data

- **3 banks**: BID (BIDV), CTG (VietinBank), VCB (Vietcombank)
- **Period**: 2016-2026 (~2,500 trading days)
- **Features**: OHLCV + Technical indicators + Macro indicators (VNIndex, USD/VND, Interest Rate)

## Models

| Model | Type | Purpose |
|-------|------|---------|
| Naive | Baseline | Predict last value |
| XGBoost | Gradient Boosting | Volatility prediction |
| NeuralProphet | Facebook Time Series | Seasonality capture |
| TFT | Temporal Fusion Transformer | Volatility with attention |
| Hybrid | GARCH + Ridge | Price prediction |

## Key Results

| Target | Winner | Improvement vs Naive |
|--------|--------|---------------------|
| Volatility | TFT | +30-34% |
| Price | Hybrid (GARCH + Ridge) | +35-41% |

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
# Phase 1 - Exploratory Analysis
python langkinh1_xgboost_shap.py
python langkinh2_neuralprophet_seasonality.py
python langkinh3_tft_memory.py

# Phase 2 - Model Evaluation
python run_4fold_all_models_both_targets.py
python run_perday_5models.py
python run_market_event_validation.py
python run_sensitivity_analysis.py
```

## Dependencies

- Python 3.8+
- pandas, numpy
- scikit-learn
- xgboost
- neuralprophet
- pytorch (for TFT)
- arch (for GARCH)
- shap