# GARCH-XGBoost Main Model Results

## Setup

- Main model: `GARCH-XGBoost main`
- Benchmarks: `Naive`, `GARCH-only`, `XGBoost`
- Target: `|log_return_{t+1}|`
- Evaluation: `walk-forward` with 4 folds
- Model-specific outputs: GARCH params, sigma plot, feature importance, latest-fold forecast

## Mean Results By Bank

### BID

| Model | Mean MAE | Mean RMSE |
|---|---:|---:|
| GARCH-only | 0.011417 | 0.015048 |
| Naive | 0.013681 | 0.019328 |
| GARCH-XGBoost main | 0.019577 | 0.022086 |
| XGBoost | 0.019654 | 0.022166 |

### CTG

| Model | Mean MAE | Mean RMSE |
|---|---:|---:|
| GARCH-only | 0.011099 | 0.014661 |
| GARCH-XGBoost main | 0.014427 | 0.017389 |
| XGBoost | 0.014823 | 0.017659 |
| Naive | 0.013881 | 0.019454 |

### VCB

| Model | Mean MAE | Mean RMSE |
|---|---:|---:|
| GARCH-only | 0.009030 | 0.011984 |
| GARCH-XGBoost main | 0.011731 | 0.014224 |
| XGBoost | 0.012423 | 0.014777 |
| Naive | 0.010351 | 0.015013 |

## GARCH Parameters

| Bank | Fold | Alpha | Beta | Alpha+Beta |
|---|---:|---:|---:|---:|
| BID | 1 | 0.0663 | 0.8542 | 0.9205 |
| BID | 2 | 0.0558 | 0.9256 | 0.9814 |
| BID | 3 | 0.0630 | 0.9111 | 0.9740 |
| BID | 4 | 0.0600 | 0.9191 | 0.9791 |
| CTG | 1 | 0.0865 | 0.8866 | 0.9731 |
| CTG | 2 | 0.0881 | 0.8917 | 0.9797 |
| CTG | 3 | 0.0825 | 0.8958 | 0.9783 |
| CTG | 4 | 0.0869 | 0.8921 | 0.9790 |
| VCB | 1 | 0.0671 | 0.9180 | 0.9852 |
| VCB | 2 | 0.0517 | 0.9334 | 0.9851 |
| VCB | 3 | 0.0691 | 0.9013 | 0.9704 |
| VCB | 4 | 0.0715 | 0.8907 | 0.9622 |

## Overall Mean

| Model | Mean MAE | Mean RMSE |
|---|---:|---:|
| GARCH-only | 0.010515 | 0.013897 |
| GARCH-XGBoost main | 0.015245 | 0.017900 |
| Naive | 0.012638 | 0.017931 |
| XGBoost | 0.015634 | 0.018201 |

Best overall RMSE: **GARCH-only**

## Output Files

- Summary CSV: `garch_xgboost_main_results.csv`
- Main report: `garch_xgboost_main_report.md`
- GARCH params CSV: `garch_params_all_folds.csv`
- Output folder: `garch_xgboost_main_outputs`
