# THESIS CẦN CẬP NHẬT SAU KHI SỬA LEAKAGE

**Ngày:** 2026-04-20
**Lý do:** Sửa 2 lỗi leakage làm số liệu không công bằng

---

## LÝ DO THAY ĐỔI

### 1. NeuralProphet `validation_df` Leakage
- **Vấn đề:** NeuralProphet được train với `validation_df=val_np`, khiến model "thấy" validation data trong quá trình huấn luyện
- **Hậu quả:** Số liệu NP tốt hơn thực tế (leakage benefit)
- **Sửa:** Bỏ `validation_df` parameter, train trực tiếp trên training set

### 2. Hybrid Volatility Target Leakage
- **Vấn đề:** Volatility target dùng contemporaneous return (cùng ngày) thay vì lagged volatility
- **Hậu quả:** Hybrid volatility prediction có lợi thế không công bằng
- **Sửa:** Đổi sang dùng lagged volatility làm target

---

## MODELS BỊ ẢNH HƯỞNG

| Model | Volatility | Price |
|-------|-----------|-------|
| Naive | Không đổi | Không đổi |
| XGBoost | Không đổi | Không đổi |
| **NeuralProphet** | Thay đổi NHIỀU (↓ 18-26%) | Thay đổi NHIỀU (↑ 69-252%) |
| **TFT** | Thay đổi ÍT (↓ 0.1% đến ↑ 0.5%) | Thay đổi ÍT (↓ 1% đến ↑ 4%) |
| **Hybrid** | Thay đổi ÍT (↓ 1-2% đến ↑ 1.4%) | Không đổi |

**Winner KHÔNG ĐỔI:** TFT thắng volatility, Hybrid thắng price.

---

## FILES BỊ ẢNH HƯỞNG

### 1. FOUR_FOLD_ALL_TARGETS

| File | Thay đổi |
|------|-----------|
| `four_fold_all_targets/4fold_5models_summary.csv` | Có |
| `four_fold_all_targets/BID_4fold_5models.csv` | Có |
| `four_fold_all_targets/CTG_4fold_5models.csv` | Có |
| `four_fold_all_targets/VCB_4fold_5models.csv` | Có |
| `four_fold_all_targets/4fold_5models_comparison.png` | Có (regenerate rồi) |

### 2. PERDAY_ALL_MODELS

| File | Thay đổi |
|------|-----------|
| `perday_all_models/perday_summary.csv` | Có |
| `perday_all_models/BID_perday_all.csv` | Có |
| `perday_all_models/CTG_perday_all.csv` | Có |
| `perday_all_models/VCB_perday_all.csv` | Có |
| `perday_all_models/BID_all_models_comparison.png` | Có (regenerate rồi) |
| `perday_all_models/CTG_all_models_comparison.png` | Có (regenerate rồi) |
| `perday_all_models/VCB_all_models_comparison.png` | Có (regenerate rồi) |

### 3. MARKET_EVENT_OUTPUTS

| File | Thay đổi |
|------|-----------|
| `market_event_outputs/market_event_summary.csv` | Có |
| `market_event_outputs/high_vol_days_BID.csv` | Có |
| `market_event_outputs/high_vol_days_CTG.csv` | Có |
| `market_event_outputs/high_vol_days_VCB.csv` | Có |
| `market_event_outputs/market_event_validation.png` | Có (regenerate rồi) |

### 4. SENSITIVITY_OUTPUTS

| File | Thay đổi |
|------|-----------|
| `sensitivity_outputs/sensitivity_analysis_charts.png` | Có (regenerate rồi) |
| `sensitivity_outputs/hybrid_ridge_alpha_sensitivity.csv` | Có |

---

## CHI TIẾT THAY ĐỔI SỐ LIỆU

### 4-FOLD WALK-FORWARD RESULTS

#### BID

| Cột | Git Cũ | Git Mới | Thay đổi |
|-----|--------|---------|-----------|
| `avg_np_vol` | 0.014408 | 0.010735 | ↓ 25.5% |
| `avg_tft_vol` | 0.010437 | 0.010437 | ↓ 0.0% (gần như =) |
| `avg_hybrid_vol` | 0.013073 | 0.012897 | ↓ 1.3% |
| `avg_np_price` | 2.604462 | 4.399113 | ↑ 68.9% |
| `avg_tft_price` | 0.660559 | 0.671474 | ↑ 1.7% |
| `avg_hybrid_price` | 0.543461 | 0.543461 | = (không đổi) |

#### CTG

| Cột | Git Cũ | Git Mới | Thay đổi |
|-----|--------|---------|-----------|
| `avg_np_vol` | 0.014686 | 0.011930 | ↓ 18.8% |
| `avg_tft_vol` | 0.010226 | 0.010217 | ↓ 0.1% |
| `avg_hybrid_vol` | 0.012127 | 0.011877 | ↓ 2.1% |
| `avg_np_price` | 1.648328 | 5.807224 | ↑ 252.3% |
| `avg_tft_price` | 0.475977 | 0.471380 | ↓ 1.0% |
| `avg_hybrid_price` | 0.382268 | 0.382268 | = (không đổi) |

#### VCB

| Cột | Git Cũ | Git Mới | Thay đổi |
|-----|--------|---------|-----------|
| `avg_np_vol` | 0.011175 | 0.009127 | ↓ 18.3% |
| `avg_tft_vol` | 0.007834 | 0.007876 | ↑ 0.5% |
| `avg_hybrid_vol` | 0.011699 | 0.011861 | ↑ 1.4% |
| `avg_np_price` | 3.323490 | 5.932745 | ↑ 78.5% |
| `avg_tft_price` | 0.796875 | 0.828252 | ↑ 3.9% |
| `avg_hybrid_price` | 0.646608 | 0.646608 | = (không đổi) |

---

### PER-DAY RESULTS

#### BID

| Cột | Git Cũ | Git Mới | Thay đổi |
|-----|--------|---------|-----------|
| `np_vol_mae` | 0.009450 | 0.008593 | ↓ 9.1% |
| `tft_vol_mae` | 0.007646 | 0.007613 | ↓ 0.4% |
| `hybrid_vol_mae` | 0.010823 | 0.010727 | ↓ 0.9% |
| `np_price_mae` | 3.405093 | 5.131523 | ↑ 50.7% |
| `tft_price_mae` | 0.460987 | 0.466950 | ↑ 1.3% |
| `hybrid_price_mae` | 0.437617 | 0.437617 | = (không đổi) |

#### CTG

| Cột | Git Cũ | Git Mới | Thay đổi |
|-----|--------|---------|-----------|
| `np_vol_mae` | 0.010140 | 0.009415 | ↓ 7.2% |
| `tft_vol_mae` | 0.007800 | 0.007762 | ↓ 0.5% |
| `hybrid_vol_mae` | 0.010677 | 0.010526 | ↓ 1.4% |
| `np_price_mae` | 3.967030 | 5.736363 | ↑ 44.6% |
| `tft_price_mae` | 0.319290 | 0.322787 | ↑ 1.1% |
| `hybrid_price_mae` | 0.310427 | 0.310427 | = (không đổi) |

#### VCB

| Cột | Git Cũ | Git Mới | Thay đổi |
|-----|--------|---------|-----------|
| `np_vol_mae` | 0.008607 | 0.008012 | ↓ 6.9% |
| `tft_vol_mae` | 0.006244 | 0.006233 | ↓ 0.2% |
| `hybrid_vol_mae` | 0.010170 | 0.010337 | ↑ 1.6% |
| `np_price_mae` | 3.046907 | 10.106359 | ↑ 231.7% |
| `tft_price_mae` | 0.577199 | 0.592252 | ↑ 2.6% |
| `hybrid_price_mae` | 0.550089 | 0.550089 | = (không đổi) |

---

## THESIS DOCUMENTATION CẦN UPDATE

1. `thesis_documentation/PART_2_EVALUATION/2.1_four_fold_cv.md` - Cập nhật số mới
2. `thesis_documentation/PART_2_EVALUATION/2.2_perday_comparison.md` - Cập nhật số mới
3. `thesis_documentation/PART_2_EVALUATION/2.5_market_events.md` - Cập nhật số mới
4. `thesis_documentation/PART_3_SYNTHESIS/3.2_final_results.md` - Cập nhật số mới

---

## KẾT LUẬN

1. **Kết luận thesis KHÔNG ĐỔI** - TFT thắng volatility, Hybrid thắng price
2. **Naive, XGBoost: Không thay đổi** - không bị leakage
3. **NP: Vol cải thiện nhưng Price tệ hơn rất nhiều** - do mất leakage từ validation_df
4. **TFT, Hybrid: Thay đổi rất ít** - chỉ vài %
5. **Số liệu bây giờ CÔNG BẰNG** - thesis nên dùng số mới
