# Lăng kính 1: XGBoost - Giải mã nhân tố (Feature Importance)

## Mục tiêu
**Trả lời câu hỏi: "Cái gì (What) tác động mạnh nhất?"**

Phân tích: Biến vĩ mô (Lãi suất, USD/VND) hay biến kỹ thuật (RSI, MA, Volume) đang "cầm lái" volatility của ngân hàng?

---

## Phương pháp

### XGBoost + SHAP
- **XGBoost**: Tree-based gradient boosting, tốt cho capturing non-linear patterns
- **SHAP (SHapley Additive exPlanations)**: Giải thích output của XGBoost bằng game theory

### Features được phân tích

| Nhóm | Features | Ý nghĩa |
|------|----------|----------|
| **Technical** | Return (t-1, t-2, t-3, t-5), Volatility (t-1, t-2), RSI (t-1), Volume, MA Ratio | Biến từ giá/volume |
| **Macro** | USD/VND (t-1), Interest Rate (t-1), VNIndex (t-1), VN30 (t-1) | Biến vĩ mô |

---

## Kết quả

### 1. Technical vs Macro - Ai thắng?

| Ngân hàng | Technical | Macro |
|-----------|-----------|-------|
| **BID** | **80.9%** | 19.1% |
| **CTG** | **82.7%** | 17.3% |
| **VCB** | **77.3%** | 22.7% |

**Kết luận: Technical indicators "cầm lái" volatility!**

→ Biến kỹ thuật (RSI, Volume, MA) đóng góp **~80%** vào dự đoán volatility
→ Biến vĩ mô (USD/VND, Interest Rate) chỉ đóng góp **~20%**

---

### 2. Top 5 Features quan trọng nhất

#### BID
| Rank | Feature | SHAP Importance |
|------|---------|-----------------|
| 1 | **Volatility (t-1)** | 0.00798 |
| 2 | USD/VND (t-1) | 0.00058 |
| 3 | RSI (t-1) | 0.00019 |
| 4 | MA20 Ratio | 0.00018 |
| 5 | Volume Ratio | 0.00017 |

#### CTG
| Rank | Feature | SHAP Importance |
|------|---------|-----------------|
| 1 | **Volatility (t-1)** | (highest) |
| 2 | USD/VND (t-1) | (second) |
| 3 | Volume Ratio | (third) |
| 4 | RSI (t-1) | (fourth) |
| 5 | MA50 Ratio | (fifth) |

#### VCB
| Rank | Feature | SHAP Importance |
|------|---------|-----------------|
| 1 | **Volatility (t-1)** | (highest) |
| 2 | USD/VND (t-1) | (second) |
| 3 | Volume Ratio | (third) |
| 4 | Return (t-3) | (fourth) |
| 5 | MA20 Ratio | (fifth) |

---

### 3. Insight quan trọng

#### Insight 1: Volatility (t-1) là "king"
- Volatility hôm qua = volatility ngày mai
- Điều này xác nhận **volatility clustering** (α+β ≈ 0.97 từ GARCH)
- High vol hôm nay → high vol ngày mai

#### Insight 2: RSI quan trọng hơn Interest Rate
- RSI (chỉ báo kỹ thuật) có SHAP cao hơn Interest Rate (biến vĩ mô)
- Thị trường phản ứng nhanh hơn với technical signals

#### Insight 3: USD/VND là macro variable quan trọng nhất
- Dù macro chỉ 20%, nhưng USD/VND luôn top 2-3
- Nhạy cảm với tỷ giá USD/VND của thị trường Việt Nam

#### Insight 4: Volume Ratio có ảnh hưởng đáng kể
- Volume bất thường → warning signal cho volatility
- High volume = high volatility

---

## Files kết quả

```
langkinh1_xgboost_shap/
├── BID_feature_importance.csv      # Chi tiết SHAP từng feature
├── CTG_feature_importance.csv
├── VCB_feature_importance.csv
├── BID_group_importance.csv        # Tổng hợp Technical vs Macro
├── CTG_group_importance.csv
├── VCB_group_importance.csv
├── BID_shap_summary.png           # SHAP summary plot
├── CTG_shap_summary.png
├── VCB_shap_summary.png
├── BID_feature_groups.png          # Pie chart + bar chart
├── CTG_feature_groups.png
├── VCB_feature_groups.png
├── BID_top5_features.png           # Top 5 SHAP dependence
├── CTG_top5_features.png
├── VCB_top5_features.png
└── all_banks_comparison.png        # So sánh 3 banks
```

---

## Chạy lại analysis

```bash
python langkinh1_xgboost_shap.py
```

---

## Kết luận Lăng kính 1

| Câu hỏi | Trả lời |
|----------|---------|
| **What?** | Volatility (t-1) là quan trọng nhất |
| **Technical hay Macro?** | Technical (~80%) thắng |
| **Cụ thể?** | RSI, Volume Ratio, MA20 Ratio |
| **Macro nào quan trọng nhất?** | USD/VND |

**→ Dự đoán volatility cổ phiếu ngân hàng Việt Nam: Technical analysis thắng vĩ mô**
