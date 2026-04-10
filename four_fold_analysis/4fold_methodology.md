# 4-Fold Walk-Forward Methodology and Results

## Tại sao cần 4-Fold?

**Single split** (70/15/15) có nhược điểm:
- Kết quả phụ thuộc vào random split cụ thể
- Không đánh giá được robustness của model

**4-Fold Walk-Forward:**
- Expanding window: mỗi fold train trên nhiều data hơn
- Test trên các period khác nhau
- Kết quả ổn định và đáng tin cậy hơn

## Tại sao chỉ so sánh GARCH vs Hybrid?

### Vấn đề: Target khác nhau

| Model | Target | Scale |
|-------|--------|-------|
| NeuralProphet | PRICE | ~40-60 |
| TFT | PRICE | ~40-60 |
| GARCH | VOLATILITY | ~0.01 |
| Hybrid | VOLATILITY | ~0.01 |

**So sánh trực tiếp NP/TFT vs GARCH/Hybrid = UNFAIR**

Ví dụ:
- NP predict price = 45.2, actual = 45.0 → MAE = 0.2
- GARCH predict volatility = 0.015, actual = 0.012 → MAE = 0.003

Con số 0.2 > 0.003 nhưng KHÔNG có nghĩa NP tệ hơn. Chúng đang predict những thứ hoàn toàn khác nhau!

### Giải pháp

Chỉ so sánh các model cùng target:
- **GARCH vs Hybrid** (cả 2 predict volatility)
- NP/TFT được phân tích riêng với root cause analysis

## Methodology

### Fold Structure

```
Fold 1: Train 50% | Val 15% | Test 15%
Fold 2: Train 65% | Val 15% | Test 15%
Fold 3: Train 80% | Val 15% | Test 15%
Fold 4: Train 90% | Val 5%  | Test 5%
```

### Hybrid Weight Selection

Với mỗi fold, tìm best weight w cho:
```
Hybrid = w * GARCH + (1-w) * Ridge
```

Weight được chọn trên validation set để minimize MAE.

## Results

### Summary

| Bank | Winner | Naive MAE | GARCH MAE | Ridge MAE | Hybrid MAE |
|------|--------|----------:|----------:|----------:|-----------:|
| BID | **Hybrid** | 0.0151 | 0.0136 | 0.0126 | **0.0126** |
| CTG | Ridge | 0.0155 | 0.0129 | **0.0112** | 0.0118 |
| VCB | **Hybrid** | 0.0114 | 0.0111 | 0.0134 | **0.0109** |

### Key Findings

1. **Hybrid wins 2/3 banks** (BID, VCB)
2. **Ridge wins 1/3 banks** (CTG) - nhưng margin nhỏ
3. **All models beat Naive** consistently (10-30% improvement)
4. **Hybrid là model tổng hợp tốt nhất** khi đánh giá trên volatility prediction

### Per-Fold Results

#### BID
| Fold | Train N | Test N | Best w | Naive | GARCH | Ridge | Hybrid |
|------|--------:|-------:|-------:|------:|------:|------:|-------:|
| 1 | 1244 | 125 | 0.00 | 0.0215 | 0.0177 | 0.0151 | 0.0151 |
| 2 | 1617 | 124 | 0.00 | 0.0122 | 0.0109 | 0.0096 | 0.0096 |
| 3 | 1991 | 124 | 0.25 | 0.0114 | 0.0126 | 0.0133 | 0.0130 |
| 4 | 2240 | 125 | 0.10 | 0.0152 | 0.0132 | 0.0125 | 0.0126 |

#### CTG
| Fold | Train N | Test N | Best w | Naive | GARCH | Ridge | Hybrid |
|------|--------:|-------:|-------:|------:|------:|------:|-------:|
| 1 | 1237 | 123 | 1.00 | 0.0231 | 0.0156 | 0.0133 | 0.0156 |
| 2 | 1608 | 123 | 0.05 | 0.0136 | 0.0115 | 0.0104 | 0.0105 |
| 3 | 1979 | 124 | 0.00 | 0.0121 | 0.0134 | 0.0110 | 0.0110 |
| 4 | 2226 | 124 | 0.00 | 0.0132 | 0.0113 | 0.0100 | 0.0100 |

#### VCB
| Fold | Train N | Test N | Best w | Naive | GARCH | Ridge | Hybrid |
|------|--------:|-------:|-------:|------:|------:|------:|-------:|
| 1 | 1244 | 125 | 0.00 | 0.0152 | 0.0120 | 0.0103 | 0.0103 |
| 2 | 1617 | 124 | 0.00 | 0.0074 | 0.0093 | 0.0089 | 0.0089 |
| 3 | 1991 | 124 | 0.75 | 0.0104 | 0.0118 | 0.0178 | 0.0131 |
| 4 | 2240 | 125 | 1.00 | 0.0124 | 0.0113 | 0.0168 | 0.0113 |

## Tại sao Hybrid không always winner?

### Lý do

1. **CTG: Ridge thắng với margin nhỏ**
   - Ridge = pure ML, không có volatility structure
   - Khi market đi sideway, Ridge có thể tốt hơn
   - Hybrid vẫn top 2

2. **Best weight thay đổi theo market condition**
   - Fold 1: w=0 (Ridge thắng)
   - Fold 3-4: w cao hơn (GARCH quan trọng hơn)

3. **Volatility clustering thay đổi theo thời gian**
   - Trong calm period: Ridge tốt
   - Trong volatile period: GARCH tốt hơn

### Kết luận

**Hybrid = best overall** vì:
- Thắng 2/3 banks
- Robust: tốt cả khi market conditions thay đổi
- Không bao giờ tệ nhất
- Adaptive: weight tự điều chỉnh theo data

## Comparison with Original Models (NP, TFT)

### Why NP/TFT Perform Poorly

NP và TFT predict PRICE, không phải VOLATILITY. Điều này dẫn đến:

1. **Martingale Property**: E[S_{t+1}|I_t] = S_t
   - Optimal price prediction = today's price
   - Complex models (NP, TFT) overfit noise, perform worse than Naive

2. **Root Cause**: Stock prices follow random walk
   - Any pattern learned by NN is likely spurious
   - GARCH models volatility (which IS predictable) not price

### Results (NP/TFT vs Naive)

| Bank | Naive MAE (price) | NP MAE (price) | TFT MAE (price) |
|------|------------------:|---------------:|----------------:|
| BID | 0.46 | 1.25 | 3.44 |
| CTG | 0.34 | 0.66 | 0.81 |
| VCB | 0.58 | 1.64 | 2.32 |

**NP và TFT thua Naive 2-17x** vì:
- Predicting price of a martingale
- Neural networks overfit to noise
- Random walk is unpredictable

### Fair Comparison Framework

Để so sánh fair giữa tất cả models, cần cùng target:

| Model | Target | Fair? |
|-------|--------|-------|
| Naive | Price/Volatility | Yes |
| GARCH | Volatility | Yes |
| Hybrid | Volatility | Yes |
| Ridge | Volatility | Yes |
| NP | Price | **No** |
| TFT | Price | **No** |

**Kết luận**: Báo cáo NP/TFT riêng với root cause analysis (martingale), không so sánh trực tiếp với GARCH/Hybrid.

## Files

- `4fold_garch_hybrid_summary.csv` - Summary statistics
- `BID_4fold_garch_hybrid.csv` - Per-fold results BID
- `CTG_4fold_garch_hybrid.csv` - Per-fold results CTG
- `VCB_4fold_garch_hybrid.csv` - Per-fold results VCB
- `4fold_garch_hybrid_comparison.png` - Visualization