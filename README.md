# Bank Stock Prediction - Vietnam

## Clone & Run

```bash
# Clone
git clone https://github.com/luuquanghuy-ui/bank-forecast-models.git
cd bank-forecast-models

# Setup
python -m venv thesis_env
thesis_env\Scripts\activate
pip install -r requirements.txt
```

## Run Models

```bash
# 4-fold Cross-Validation (chính)
python run_4fold_all_models_both_targets.py

# Sensitivity Analysis
python run_sensitivity_analysis.py

# Market Event Validation
python run_market_event_validation.py
```

## Results

| Target | Best Model | vs Baseline |
|--------|-----------|-------------|
| Volatility | XGBoost | +9-29% |
| Price | Hybrid | +35-41% |

## Documentation

Đọc `thesis_documentation/README.md` để biết thứ tự đọc và chi tiết.
