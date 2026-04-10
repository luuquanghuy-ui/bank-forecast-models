# THESIS DOCUMENTATION: Stock Volatility Prediction for Vietnam Commercial Banks

**Author:** Luu Quang Huy
**Date:** 2026-04-10
**Models:** GARCH, NeuralProphet, TFT, Hybrid GARCH+Ridge
**Target:** Volatility prediction (|log_return|) and Price prediction

---

## TABLE OF CONTENTS

### PART 0: FOUNDATION
- [0.1 Data Description](PART_0_FOUNDATION/0.1_data_description.md) - Data sources, structure, preprocessing
- [0.2 Problem Formulation](PART_0_FOUNDATION/0.2_problem_formulation.md) - Martingale property, volatility clustering
- [0.3 Fair Comparison Framework](PART_0_FOUNDATION/0.3_fair_comparison_framework.md) - Why we compare models differently

### PART 1: MODELS
- [1.1 GARCH Model](PART_1_MODELS/1.1_garch_model.md) - Methodology, parameters, sensitivity
- [1.2 NeuralProphet Model](PART_1_MODELS/1.2_neuralprophet_model.md) - Methodology, results, failure analysis
- [1.3 TFT Model](PART_1_MODELS/1.3_tft_model.md) - Methodology, results, failure analysis
- [1.4 Hybrid GARCH+Ridge Model](PART_1_MODELS/1.4_hybrid_model.md) - Methodology, sensitivity, ensemble

### PART 2: EVALUATION
- [2.1 Four-Fold Walk-Forward](PART_2_EVALUATION/2.1_four_fold_cv.md) - Robustness validation
- [2.2 Per-Day All Models Comparison](PART_2_EVALUATION/2.2_perday_comparison.md) - Daily predictions
- [2.3 Statistical Tests](PART_2_EVALUATION/2.3_statistical_tests.md) - DM test, p-values, ACF/PACF
- [2.4 Sensitivity Analysis](PART_2_EVALUATION/2.4_sensitivity_analysis.md) - All sensitivity results
- [2.5 Market Event Validation](PART_2_EVALUATION/2.5_market_events.md) - Worst days analysis

### PART 3: SYNTHESIS
- [3.1 Root Cause Analysis](PART_3_SYNTHESIS/3.1_root_cause_analysis.md) - Why models succeed or fail
- [3.2 Final Results Summary](PART_3_SYNTHESIS/3.2_final_results.md) - Complete results table
- [3.3 Conclusions](PART_3_SYNTHESIS/3.3_conclusions.md) - Key findings and thesis contribution

---

## QUICK SUMMARY

### Best Model for Volatility: Hybrid GARCH+Ridge
| Bank | Naive | GARCH | Hybrid | Improvement |
|------|-------|-------|--------|-------------|
| BID | 0.0114 | 0.0095 | **0.0080** | +30% |
| CTG | 0.0112 | 0.0097 | **0.0093** | +17% |
| VCB | 0.0093 | 0.0083 | **0.0066** | +29% |

### Best Model for Price: Naive (Martingale)
| Bank | Naive | NeuralProphet | TFT |
|------|-------|---------------|-----|
| BID | **0.46** | 1.25 | 1.27 |
| CTG | **0.34** | 0.66 | 5.27 |
| VCB | **0.58** | 1.64 | 2.32 |

### Key Insight
> **Price is unpredictable (martingale). Volatility is predictable (clustering). Complex deep learning models (NP, TFT) fail on price prediction but can be converted to volatility for fair comparison.**

---

## READING ORDER

For complete understanding, read in this order:

1. **START HERE** → `README.md` (this file)
2. `PART_0_FOUNDATION/0.1_data_description.md` → Understand the data
3. `PART_0_FOUNDATION/0.2_problem_formulation.md` → Understand WHY some models fail
4. `PART_0_FOUNDATION/0.3_fair_comparison_framework.md` → Understand HOW to compare
5. `PART_1_MODELS/1.1_garch_model.md` → GARCH methodology
6. `PART_1_MODELS/1.4_hybrid_model.md` → Hybrid ensemble
7. `PART_2_EVALUATION/2.1_four_fold_cv.md` → Validation approach
8. `PART_2_EVALUATION/2.2_perday_comparison.md` → Daily results
9. `PART_3_SYNTHESIS/3.1_root_cause_analysis.md` → Complete analysis
10. `PART_3_SYNTHESIS/3.2_final_results.md` → Summary table
11. `PART_3_SYNTHESIS/3.3_conclusions.md` → Final conclusions

---

## FILE STRUCTURE

```
thesis_documentation/
├── README.md                          ← Entry point (this file)
├── PART_0_FOUNDATION/
│   ├── 0.1_data_description.md
│   ├── 0.2_problem_formulation.md
│   └── 0.3_fair_comparison_framework.md
├── PART_1_MODELS/
│   ├── 1.1_garch_model.md
│   ├── 1.2_neuralprophet_model.md
│   ├── 1.3_tft_model.md
│   └── 1.4_hybrid_model.md
├── PART_2_EVALUATION/
│   ├── 2.1_four_fold_cv.md
│   ├── 2.2_perday_comparison.md
│   ├── 2.3_statistical_tests.md
│   ├── 2.4_sensitivity_analysis.md
│   └── 2.5_market_events.md
└── PART_3_SYNTHESIS/
    ├── 3.1_root_cause_analysis.md
    ├── 3.2_final_results.md
    └── 3.3_conclusions.md
```

---

## DATA FILES

Original datasets: `banks_BID_dataset.csv`, `banks_CTG_dataset.csv`, `banks_VCB_dataset.csv`

Model outputs: See respective model directories:
- `GARCH_XGBoost/garch_xgboost_main_results.csv`
- `Hybrid_GARCH_DL/detailed_analysis/`
- `NeuralProphet/neuralprophet_main_outputs/`
- `TemporalFusionTransformer/tft_main_outputs/`

Analysis outputs:
- `four_fold_analysis/` - 4-fold walk-forward results
- `perday_all_models/` - Per-day comparison results
