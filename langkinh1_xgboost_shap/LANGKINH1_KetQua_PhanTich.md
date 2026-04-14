# Lăng kính 1: Kết quả & Phân tích

## Target: |log_return| — Absolute daily return (proxy cho daily volatility)
## Method: XGBoost + SHAP → Feature importance + Cross-bank DNA

---

## 1. Model Performance

| Bank | Test MAE | R² | Đánh giá |
|------|----------|-----|----------|
| BID | 0.008349 | 0.2461 | XGBoost giải thích ~25% variance — fit tốt nhất |
| CTG | 0.010565 | -0.1240 | R² âm — model tệ hơn predict trung bình |
| VCB | 0.009722 | 0.0641 | XGBoost giải thích ~6% variance — yếu |

### Nhận xét

- BID dễ predict nhất: R²=0.25 nghĩa là 25% biến động daily của BID có thể giải thích bằng 15 features này
- CTG R² âm: model tệ hơn cả việc predict giá trị trung bình. CTG biến động "bất thường" hơn, khó dự đoán hơn
- VCB R²=0.06: rất yếu, gần random

Sự khác biệt R² giữa 3 banks cho thấy: dù cùng ngành, mức độ "predictable" rất khác nhau. Không thể dùng 1 model cho tất cả.

---

## 2. Part A — Technical vs Macro

| Bank | Technical | Macro |
|------|-----------|-------|
| BID | **86.7%** | 13.3% |
| CTG | **76.7%** | 23.3% |
| VCB | **78.3%** | 21.7% |

### Nhận xét

- Technical indicators chiếm ưu thế ở cả 3 banks (77-87%)
- BID thiên technical nhất (86.7%) — biến động BID chủ yếu driven bởi momentum, volume, RSI
- CTG có tỷ lệ Macro cao nhất (23.3%) — CTG nhạy hơn với tỷ giá và thị trường chung
- VCB tương tự CTG (21.7% Macro)

Ý nghĩa: Trader dùng technical analysis sẽ hiệu quả hơn ở BID. Với CTG, cần kết hợp thêm thông tin vĩ mô.

---

## 3. Part B — Cross-bank DNA: Top Features từng bank

### Ranking đầy đủ

| Feature | BID rank | CTG rank | VCB rank | Khác biệt? |
|---------|----------|----------|----------|------------|
| Volume Ratio | **#1** | **#1** | #3 | Nhất quán top 3 |
| Price / MA20 | **#2** | **#3** | **#1** | Nhất quán top 3 |
| RSI (t-1) | **#3** | #6 | **#2** | BID/VCB cao, CTG thấp |
| Volatility 20d (t-1) | #4 | #4 | #5 | Tương đối nhất quán |
| USD/VND (t-1) | #5 | **#2** | #6 | CTG nhạy USD/VND nhất |
| Return (t-1) | #6 | #7 | #8 | Nhất quán |
| Volatility 20d (t-2) | #7 | #5 | #9 | |
| Price / MA50 | #8 | #10 | #10 | |
| Volume (t-1) | #9 | #13 | #7 | Khác biệt vừa |
| VNIndex (t-1) | #10 | #11 | #11 | |
| Return (t-2) | #11 | #12 | #12 | |
| Return (t-3) | #12 | #8 | #14 | CTG khác biệt |
| Return (t-5) | #13 | #15 | #13 | |
| VN30 (t-1) | #14 | #9 | **#4** | VCB rất khác! |
| Interest Rate (t-1) | #15 | #14 | #15 | Gần 0 — đúng kỳ vọng |

### Phân tích theo từng bank

**BID — "Retail momentum bank"**
- Top 3: Volume Ratio, Price/MA20, RSI → toàn technical momentum indicators
- Macro yếu nhất (13.3%)
- VN30 rank #14 → BID ít đi theo thị trường chung
- Giải thích: BID có nhiều giao dịch từ nhà đầu tư cá nhân → technical signals mạnh

**CTG — "Macro-sensitive bank"**
- USD/VND rank **#2** → nhạy tỷ giá nhất trong 3 banks
- RSI chỉ rank #6 → ít bị chi phối bởi momentum đơn giản
- Return (t-3) rank #8 → CTG có "memory" xa hơn BID/VCB
- R² âm → biến động CTG phức tạp, khó giải thích bằng features này

**VCB — "Market-following bank"**
- VN30 rank **#4** → VCB đi theo thị trường chung (VN30) mạnh nhất
- RSI rank #2 → vẫn nhạy momentum
- Volume Ratio chỉ #3 (thay vì #1 như BID/CTG) → volume ít ảnh hưởng hơn
- Giải thích: VCB là blue-chip lớn nhất, giá VCB phần nào phản ánh thị trường chung

### Features khác biệt lớn nhất giữa 3 banks

| Feature | Variance | Chi tiết | Giải thích |
|---------|----------|----------|-----------|
| **VN30 (t-1)** | 16.7 | BID=#14, CTG=#9, VCB=#4 | VCB là blue-chip, đi theo VN30 |
| **Volume (t-1)** | 6.2 | BID=#9, CTG=#13, VCB=#7 | CTG thanh khoản cao, volume ít ảnh hưởng |
| **Return (t-3)** | 6.2 | BID=#12, CTG=#8, VCB=#14 | CTG có memory xa hơn |
| **USD/VND (t-1)** | 2.9 | BID=#5, CTG=#2, VCB=#6 | CTG nhạy tỷ giá |
| **RSI (t-1)** | 2.9 | BID=#3, CTG=#6, VCB=#2 | BID/VCB nhạy momentum hơn CTG |

---

## 4. Part C — Dependence Patterns

(Xem các file `*_partC_top5_dependence.png` trong thư mục output)

Các pattern cần chú ý từ scatter plots:
- Volume Ratio > 2 (volume bất thường gấp đôi trung bình) → SHAP tăng → biến động tăng
- Price/MA20 xa 1.0 (giá xa đường MA) → SHAP tăng → mean-reversion signal
- RSI extreme (< 30 hoặc > 70) → SHAP có thể thay đổi hướng

---

## 5. Interest Rate — Gần bằng 0

Interest Rate rank #14-15 ở cả 3 banks. Lý do: lãi suất VN chỉ có 7 giá trị unique và thay đổi 10 lần trong ~2500 ngày giao dịch. Gần như constant → XGBoost không split được → SHAP ≈ 0.

Đây KHÔNG có nghĩa lãi suất không quan trọng — mà là trong giai đoạn này (2016-2026), lãi suất ít biến động nên không drive daily volatility. Nếu có sự kiện siết/nới lãi suất mạnh, kết quả có thể khác.

---

## 6. Kết luận Lăng kính 1

### Kết luận chính

1. **Technical indicators chiếm 77-87% importance** — mức độ biến động daily bị chi phối bởi momentum, volume, xu hướng giá ngắn hạn
2. **Mỗi bank có DNA riêng**: BID = retail momentum, CTG = macro-sensitive, VCB = market-following
3. **USD/VND là macro variable quan trọng nhất** — đặc biệt ở CTG (rank #2)
4. **Interest Rate gần bằng 0** — lãi suất VN quá ít biến động để drive daily volatility
5. **R² rất khác nhau** giữa 3 banks (0.25 vs -0.12 vs 0.06) — không thể dùng 1 model cho tất cả

### Kết nối Phase 2

| Insight | Ảnh hưởng |
|---------|----------|
| Technical dominates | → Features cho Ridge/XGBoost nên ưu tiên technical |
| Volume Ratio + Price/MA20 luôn top 3 | → Đây là features cốt lõi cần giữ |
| Mỗi bank DNA riêng | → Train model riêng cho mỗi bank |
| R²(BID)=0.25 nhưng R²(CTG)=-0.12 | → Expect model performance khác nhau giữa banks |
| USD/VND quan trọng ở CTG | → Macro features vẫn nên giữ |

---

## Files output

```
langkinh1_xgboost_shap/
├── partA_group_comparison.png        # Pie chart Technical vs Macro, 3 banks
├── partA_feature_bars.png            # Bar chart 15 features, 3 banks
├── partB_crossbank_heatmap.png       # Heatmap ranking, 3 banks
├── partB_crossbank_comparison.csv    # Bảng so sánh ranking + variance
├── partB_radar_dna.png               # Radar chart DNA, 3 banks
├── BID_shap_summary.png             # SHAP beeswarm BID
├── CTG_shap_summary.png             # SHAP beeswarm CTG
├── VCB_shap_summary.png             # SHAP beeswarm VCB
├── BID_partC_top5_dependence.png    # Top 5 dependence BID
├── CTG_partC_top5_dependence.png    # Top 5 dependence CTG
├── VCB_partC_top5_dependence.png    # Top 5 dependence VCB
├── BID_feature_importance.csv       # Bảng SHAP chi tiết BID
├── CTG_feature_importance.csv       # Bảng SHAP chi tiết CTG
├── VCB_feature_importance.csv       # Bảng SHAP chi tiết VCB
├── BID_group_importance.csv         # Technical vs Macro BID
├── CTG_group_importance.csv         # Technical vs Macro CTG
└── VCB_group_importance.csv         # Technical vs Macro VCB
```
