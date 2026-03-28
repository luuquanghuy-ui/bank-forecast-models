# NeuralProphet Main Model Results

## Setup

- Main model: `NeuralProphet` with lagged regressors
- Benchmark: `Naive`
- Target: `close`
- Split: `70% train`, `15% validation`, `15% test`
- Model-specific outputs: forecast, components, parameters

## BID

- Rows: `2510` | Train: `1757` | Val: `376` | Test: `377`

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| Naive | 0.4180 | 0.5770 | 0.4610 | 0.7367 |
| NeuralProphet main | 0.8751 | 1.0774 | 1.2455 | 2.4746 |

## CTG

- Rows: `2495` | Train: `1746` | Val: `374` | Test: `375`

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| Naive | 0.2334 | 0.3372 | 0.3446 | 0.5227 |
| NeuralProphet main | 0.4878 | 0.6326 | 0.6551 | 0.9842 |

## VCB

- Rows: `2510` | Train: `1757` | Val: `376` | Test: `377`

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| Naive | 0.5069 | 0.6958 | 0.5802 | 0.9682 |
| NeuralProphet main | 1.1897 | 1.4762 | 1.6428 | 3.0071 |

## Output Files

- Summary CSV: `neuralprophet_main_results.csv`
- Output folder: `neuralprophet_main_outputs`
