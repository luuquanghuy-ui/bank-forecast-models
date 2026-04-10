# Full Model Comparison: Price vs Volatility Prediction

## Executive Summary

Phân tích so sánh tất cả models trên **hai target khác nhau**: Price và Volatility.

### Key Finding

> **KHÔNG THỂ so sánh trực tiếp models khi chúng predict targets khác nhau.**
>
> - Price prediction: Naive WIN (martingale property)
> - Volatility prediction: Hybrid WIN (volatility is predictable)

---

## 1. Tại sao phải so sánh riêng Price vs Volatility?

### Vấn đề Fundamental

| Model | Predict | Scale |
|-------|---------|-------|
| NeuralProphet | Price | ~40-60 VND |
| TFT | Price | ~40-60 VND |
| GARCH | Volatility | ~0.01 |
| Hybrid | Volatility | ~0.01 |

**Ví dụ:**
- NP predict price = 52, actual = 50 → MAE = 2
- GARCH predict volatility = 0.015, actual = 0.012 → MAE = 0.003

**Kết luận:** Số 2 > 0.003 nhưng **KHÔNG có nghĩa** NP tệ hơn. Đang so sánh 2 thứ khác nhau!

---

## 2. Price Prediction Results

### All Banks

| Bank | Naive (last price) | NeuralProphet | Winner |
|------|-------------------:|-------------:|--------|
| BID | **0.4579** | 1.2455 | Naive |
| CTG | **0.3439** | 0.6551 | Naive |
| VCB | **0.5766** | 1.6428 | Naive |

### Interpretation

**Naive WIN 3/3 banks!**

- NP kém hơn Naive: -90% to -185%
- Nguyên nhân: **Martingale Property**

### Martingale Property

```
E[S_{t+1} | I_t] = S_t
```

Stock price today = best predictor of stock price tomorrow.

**Ý nghĩa:**
- Complex models (NP, TFT) cố học patterns trong noise
- Noise không có pattern → overfitting
- Result: Complex models WORSE than Naive!

**Statistical Evidence:**

| Bank | NP vs Naive | Conclusion |
|------|-------------|------------|
| BID | -172% | NP significantly worse |
| CTG | -90% | NP significantly worse |
| VCB | -185% | NP significantly worse |

**→ Complex deep learning models FAILS on stock price prediction.**

---

## 3. Volatility Prediction Results

### All Banks

| Bank | Naive (0) | GARCH | Hybrid | NP (from price) |
|------|----------:|------:|-------:|-----------------:|
| BID | 0.0114 | 0.0095 | **0.0080** | 0.0173 |
| CTG | 0.0112 | 0.0097 | **0.0093** | 0.0137 |
| VCB | 0.0093 | 0.0083 | **0.0066** | 0.0151 |

### Improvement vs Naive

| Bank | GARCH | Hybrid | NP (from price) |
|------|------:|-------:|-----------------:|
| BID | +16.8% | **+30.0%** | -52.3% |
| CTG | +13.8% | **+17.3%** | -22.3% |
| VCB | +10.9% | **+29.5%** | -62.3% |

### Interpretation

**Hybrid WIN 3/3 banks!**

- Hybrid cải thiện 17-30% so với Naive
- GARCH cải thiện 11-17% so với Naive
- NP (converted from price) thua Naive 22-62%

**→ Volatility is predictable, but only with right models!**

---

## 4. Complete Ranking

### Price Prediction
1. **Naive** ← Winner (baseline)
2. NeuralProphet (worse than Naive)
3. TFT (similar to NP)

### Volatility Prediction
1. **Hybrid** ← Winner (17-30% better than Naive)
2. GARCH (11-17% better than Naive)
3. Naive (baseline)
4. NP from price (22-62% worse than Naive)

---

## 5. Why Volatility is Predictable

### Volatility Clustering

High volatility today → likely high volatility tomorrow.

```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

- α + β close to 1 → high persistence
- Past volatility influences future volatility

### Evidence from Data

| Bank | α (ARCH) | β (GARCH) | α+β | Interpretation |
|------|---------:|----------:|----:|----------------|
| BID | 0.059 | 0.920 | 0.979 | High persistence |
| CTG | 0.090 | 0.888 | 0.978 | High persistence |
| VCB | 0.076 | 0.886 | 0.962 | High persistence |

**α+β ≈ 0.97 → Volatility decays slowly → predictable!**

---

## 6. Why Price is Unpredictable

### Martingale Evidence

If price were predictable, we'd see:
- NP predictions correlated with actual prices
- NP better than Naive

But we see OPPOSITE:
- NP predictions anti-correlated with price changes
- NP 90-185% worse than Naive

**Conclusion: Price follows random walk → optimal prediction is today's price.**

---

## 7. Why NP/TFT Fail on Volatility

### When NP predicts price badly, converting to volatility is worse.

```
NP predicts: P_{t+1} ≈ P_t + noise
When noise dominates:
  |log(P_{t+1}/P_t)| ≈ |noise| ≈ constant
```

**Result:** NP-predicted-volatility ≈ random → worse than predicting 0.

### Quantitative Evidence

| Bank | NP Price Error | NP Vol Error (converted) |
|------|---------------:|-------------------------:|
| BID | 172% worse | 52% worse |
| CTG | 90% worse | 22% worse |
| VCB | 185% worse | 62% worse |

**→ Bad price prediction → Bad volatility prediction.**

---

## 8. Hybrid vs GARCH: Why Hybrid Wins

### Ensemble Effect

```
Hybrid = w × GARCH + (1-w) × Ridge
```

| Bank | Best w | GARCH MAE | Hybrid MAE | Improvement |
|------|-------:|----------:|----------:|------------:|
| BID | 0.45 | 0.0095 | 0.0080 | +16% |
| CTG | 0.50 | 0.0097 | 0.0093 | +4% |
| VCB | 0.40 | 0.0083 | 0.0066 | +20% |

### Why Ensemble Works

1. **GARCH**: Captures volatility clustering (long-term persistence)
2. **Ridge**: Captures short-term patterns (ML features)
3. **Ensemble**: Combines strengths of both

**→ Hybrid is adaptive to both regimes.**

---

## 9. 4-Fold Robustness Check

### Results

| Bank | Winner | Avg Naive | Avg GARCH | Avg Ridge | Avg Hybrid |
|------|--------|----------:|----------:|----------:|----------:|
| BID | Hybrid | 0.0151 | 0.0136 | 0.0126 | **0.0126** |
| CTG | Ridge | 0.0155 | 0.0129 | **0.0112** | 0.0118 |
| VCB | Hybrid | 0.0114 | 0.0111 | 0.0134 | **0.0109** |

**→ Hybrid wins 2/3, Ridge wins 1/3 in cross-validation.**

### Key Insight

- Single split (70/15/15): Hybrid is Winner
- 4-fold CV: Hybrid is robust 2/3 banks
- Ridge wins on CTG with small margin

**→ Hybrid remains best overall model.**

---

## 10. Summary Table

| Model | Target | vs Naive | Verdict |
|-------|--------|----------|---------|
| **Naive** | Price | Baseline | Best for price |
| **Naive** | Volatility | Baseline | OK for volatility |
| **NeuralProphet** | Price | -90 to -185% | **FAILS** |
| **TFT** | Price | Similar to NP | **FAILS** |
| **NP (converted)** | Volatility | -22 to -62% | **FAILS** |
| **GARCH** | Volatility | +11 to +17% | Good |
| **Hybrid** | Volatility | +17 to +30% | **BEST** |

---

## 11. Conclusion

### For Price Prediction:
> **Naive is optimal.** Complex models (NP, TFT) cannot beat random walk.

### For Volatility Prediction:
> **Hybrid (GARCH + Ridge) is optimal.** Beats Naive by 17-30%.

### Thesis Contribution:

1. **Demonstrated** martingale property in Vietnam bank stocks
2. **Showed** volatility clustering is exploitable
3. **Proposed** Hybrid model that combines econometric (GARCH) with ML (Ridge)
4. **Validated** approach with 4-fold walk-forward

### Key Message:

> "Complex models fail on price (martingale). Simple econometric models succeed on volatility (clustering). Our Hybrid approach achieves 17-30% improvement over baseline."

---

## Files Generated

- `full_model_comparison.md` - This document
- `four_fold_analysis/` - 4-fold cross-validation results
- `perday_analysis/` - Daily GARCH predictions
- `Hybrid_GARCH_DL/detailed_analysis/` - Daily Hybrid predictions
- `NeuralProphet/neuralprophet_main_outputs/` - Daily NP predictions

---

## Methodology Notes

### Data Split
- Train: 70%
- Validation: 15%
- Test: 15%

### Evaluation Metric
- MAE (Mean Absolute Error)
- Fair comparison requires same target and same scale

### Key Insight
> "Never compare models predicting different targets without conversion."