# Temporal Fusion Transformer Main Model Results

## Setup

- Main model: `TFT + covariates`
- Benchmark: `Naive`
- Target: `close`
- Split: `70% train`, `15% validation`, `15% test`
- Model-specific outputs: forecast, attention, encoder/decoder/static variable importance

## BID

- Rows: `2510` | Train: `1757` | Val: `376` | Test: `377`

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| Naive | 0.4180 | 0.5770 | 0.4610 | 0.7367 |
| TFT main | 3.4433 | 4.7124 | 1.2659 | 2.5825 |

## CTG

- Rows: `2495` | Train: `1746` | Val: `374` | Test: `375`

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| Naive | 0.2334 | 0.3372 | 0.3446 | 0.5227 |
| TFT main | 0.8067 | 1.0318 | 5.2738 | 7.1174 |

## VCB

- Rows: `2510` | Train: `1757` | Val: `376` | Test: `377`

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| Naive | 0.5069 | 0.6958 | 0.5802 | 0.9682 |
| TFT main | 6.1379 | 6.8794 | 2.3158 | 3.2478 |

## Output Files

- Summary CSV: `tft_main_results.csv`
- Output folder: `tft_main_outputs`
