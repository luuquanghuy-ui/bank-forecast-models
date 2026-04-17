# Đồ án: 3 Lăng kính + 5 Models Prediction

---

## Vấn đề gặp phải

Nhóm đã làm **Phase 2 trước Phase 1**:

- Phase 2 (đã xong): Train 5 models (Naive, XGBoost, NP, TFT, Hybrid) dự đoán volatility + price. Kết quả: **TFT thắng volatility** (MAE thấp nhất, 30-34% tốt hơn Naive), price thì các model đều chưa tốt → nhóm phát triển Hybrid (GARCH + Ridge) → **Hybrid thắng price** (35-41% tốt hơn Naive).
- Phase 1 (chưa làm): 3 lăng kính — dùng từng model để khám phá, hiểu thị trường.

Lẽ ra Phase 1 phải đi trước vì nó là bước hiểu thị trường, từ đó mới thiết kế prediction ở Phase 2. Giờ cần bổ sung Phase 1, viết lại narrative cho đúng thứ tự logic.

**Flow đúng:**

```
Phase 1: 3 Lăng kính
   Lăng kính 1 (XGBoost + SHAP) → "Cái gì chi phối?"
   Lăng kính 2 (NeuralProphet)  → "Khi nào có pattern?"
   Lăng kính 3 (TFT + ACF)      → "Bao xa trí nhớ?"

Phase 2: 5 Models Prediction
   5 models x 2 targets (volatility + price)
   TFT thắng volatility (MAE thấp nhất, 30-34% tốt hơn Naive)
   Price chưa tốt → phát triển Hybrid → Hybrid thắng price (35-41% tốt hơn Naive)
```

---

## Lăng kính 1: XGBoost + SHAP — "Cái gì chi phối mức độ biến động ngân hàng VN?"

### Câu hỏi

Yếu tố nào tác động mạnh nhất đến mức độ biến động cổ phiếu ngân hàng? Technical indicators (RSI, Volume, MA) hay Macro variables (lãi suất, tỷ giá, VNIndex)? Và 3 ngân hàng BID, CTG, VCB — cùng ngành nhưng có bị chi phối giống nhau không?

### Tại sao XGBoost + SHAP?

XGBoost học được mối quan hệ phi tuyến mà không cần giả định. SHAP (dựa trên Shapley value từ game theory) đo chính xác mỗi feature đóng góp bao nhiêu vào prediction. Khác với correlation (chỉ bắt linear) hay feature importance mặc định của tree (bị bias theo cardinality).

### Thiết kế

**Target**: `|log_return|` — absolute daily return, proxy cho daily volatility.

Tại sao không dùng `log_return` (return thô)? Vì return gần random walk (mean ≈ 0, autocorrelation ≈ 0) → XGBoost gần như không predict được → SHAP values từ model yếu không meaningful. Absolute return có structure (volatility clustering) → XGBoost fit tốt hơn → SHAP meaningful.

Tại sao không dùng `volatility_20d`? Vì `volatility_20d` là rolling window 20 ngày → derived feature, phức tạp hơn. `|log_return|` đơn giản, trực tiếp, không lag.

Ghi chú từ data: `interest_rate` chỉ có 7 giá trị unique, thay đổi 10 lần trong ~2500 ngày → gần như constant → SHAP sẽ rất thấp. Giữ lại vì SHAP thấp cũng là insight (lãi suất VN ít biến động trong giai đoạn này).

**Features** (15 biến, 2 nhóm):

Technical (11 biến):
- Return lags: `return_lag1`, `return_lag2`, `return_lag3`, `return_lag5`
- Volatility lags: `volatility_lag1` (volatility_20d shifted), `volatility_lag2`
- `rsi_lag1` — momentum indicator
- `volume_lag1`, `volume_ratio` (volume / MA20 volume) — thanh khoản
- `ma_ratio` (close / MA20), `ma50_ratio` (close / MA50) — xu hướng

Macro (4 biến):
- `vnindex_lag1`, `vn30_lag1` — sentiment thị trường chung
- `usd_vnd_lag1` — rủi ro tỷ giá
- `interest_rate_lag1` — chi phí vốn

**Split**: 70% train / 15% validation / 15% test, walk-forward.

### Phân tích gồm 3 phần

**Phần A — Technical vs Macro**: Tổng SHAP importance của nhóm Technical vs nhóm Macro. Pie chart + bar chart cho mỗi bank. Xem nhóm nào chiếm ưu thế.

**Phần B — Cross-bank DNA** (phần chính): Bảng ranking SHAP cả 3 banks cạnh nhau. Tìm sự khác biệt: feature nào quan trọng ở bank này nhưng không ở bank kia? Ví dụ RSI quan trọng ở bank nào, USD/VND quan trọng ở bank nào. Giải thích sự khác biệt dựa trên đặc thù từng bank (VCB thiên quốc tế, BID thiên retail...).

**Phần C — SHAP Dependence**: Scatter plot SHAP value vs feature value cho top 5 features. Xem mối quan hệ linear hay non-linear. Có breaking point nào không (ví dụ RSI > 70 thì SHAP đổi hướng?).

### Kết nối Phase 2

- Biết features nào quan trọng → chọn feature set cho models ở Phase 2
- Nếu volatility lag dominates → justify thêm GARCH component vào Hybrid
- Nếu mỗi bank DNA riêng → justify việc train model riêng cho mỗi bank

### Code cần sửa

File `langkinh1_xgboost_shap.py`: đổi target từ `volatility_20d.shift(-1)` sang `|log_return|` (np.abs(log_return)). Thêm cross-bank heatmap/radar chart so sánh ranking.

---

## Lăng kính 2: NeuralProphet — "Khi nào cổ phiếu ngân hàng có pattern?"

### Câu hỏi

Có pattern theo lịch trong cổ phiếu ngân hàng VN không? Thứ 2 có khác thứ 6? Tháng 1 có khác tháng 7? Cuối quý có khác giữa quý? Nếu có, 3 bank có cùng pattern không?

### Tại sao NeuralProphet?

NP được thiết kế để decompose time series thành Trend + Seasonality + AR + Residual. Đây là thế mạnh riêng mà XGBoost hay TFT không có — NP tự động tách các thành phần và cho phép nhìn từng thành phần riêng lẻ.

### Thiết kế — 3 phần

**Phần A — NP Decomposition (visual exploration)**

Dùng NP decompose **daily close price** cho mỗi bank:
- Trend component → giá tăng/giảm dài hạn
- Weekly seasonality → ngày nào trong tuần giá có xu hướng?
- Yearly seasonality → tháng nào trong năm?
- AR component → phần autoregressive (phụ thuộc quá khứ gần)

Plot từng component riêng. So sánh 3 banks: trend VCB vs BID có giống nhau không? Weekly pattern có giống nhau không?

Cũng chạy thêm NP decomposition trên **daily log return** để so sánh. Return gần random walk nên seasonal component sẽ yếu hơn — bản thân sự khác biệt giữa price decomposition và return decomposition là insight.

**Phần B — Statistical calendar tests (phần chính — không thể bắt bẻ)**

Test trực tiếp trên daily return, không cần NP:

| Test | Cách làm | Method |
|------|---------|--------|
| Day-of-week effect | Mean return mỗi ngày (Mon→Fri), boxplot | Kruskal-Wallis (non-parametric ANOVA) |
| Monday effect | Monday return vs tất cả ngày khác | Mann-Whitney U test |
| Month-of-year effect | Mean return mỗi tháng (Jan→Dec), boxplot | Kruskal-Wallis |
| January effect | January return vs tất cả tháng khác | Mann-Whitney U test |
| Quarter-end effect | Return 5 ngày cuối quý vs bình thường | Mann-Whitney U test |

Tại sao dùng Kruskal-Wallis và Mann-Whitney thay vì t-test/ANOVA?
→ Return distribution có fat tails, không normal → non-parametric test chính xác hơn.

Tại sao test trên return chứ không trên price?
→ Price non-stationary → statistical test vô nghĩa. Return stationary → test hợp lệ.

Mỗi test chạy riêng cho BID, CTG, VCB. Report p-value, có significant hay không (α = 0.05).

**Phần C — So sánh NP vs Statistics**

- NP weekly seasonality nói ngày nào mạnh/yếu?
- Statistical test (Kruskal-Wallis) nói có significant day-of-week effect không?
- Nếu khớp → NP captures real pattern
- Nếu không khớp → NP có thể overfit noise

### Kết quả: report trung thực

Kết quả thế nào report thế đó:
- Nếu có calendar effect → tìm thấy pattern thú vị, report
- Nếu không có → cũng là insight: "thị trường VN không có calendar effect rõ ràng ở tần suất daily"
- Nếu có ở bank này mà không ở bank kia → so sánh cross-bank

### Kết nối Phase 2

- Nếu seasonal patterns yếu → hiểu vì sao NP prediction performance không tốt ở Phase 2 (NP được thiết kế cho time series có seasonality mạnh)
- Nếu seasonal patterns mạnh → có thể cải thiện NP configuration ở Phase 2
- Calendar effect khác nhau giữa 3 banks → confirm mỗi bank cần tiếp cận riêng

### Code

File mới: `langkinh2_neuralprophet_seasonality.py`

---

## Lăng kính 3: TFT + ACF — "Bao xa trí nhớ thị trường?"

### Câu hỏi

Khi dự đoán giá ngân hàng, nên nhìn về bao xa? 5 ngày? 20 ngày? 60 ngày? Return ngày hôm nay có liên quan đến bao nhiêu ngày trước? Và 3 bank có "trí nhớ" giống nhau không?

### Tại sao TFT + ACF?

Dùng 2 công cụ khác nhau để cùng trả lời 1 câu hỏi:

- **ACF/PACF** (Autocorrelation Function): Công cụ thống kê kinh điển, đo mức tương quan giữa return ngày t và return ngày t-k. Đơn giản, robust, ai cũng chấp nhận.
- **TFT Attention**: TFT có multi-head attention mechanism, cho biết model focus vào time step nào trong quá khứ. Approach deep learning, experimental hơn.

Dùng cả 2 rồi so sánh → cross-validate kết quả.

### Thiết kế — 3 phần

**Phần A — ACF/PACF (nền tảng thống kê)**

Tính và plot cho mỗi bank (BID, CTG, VCB):

1. ACF của `log_return` (return thô) — 60 lags
   - Đo: Return hôm nay có tương quan với return bao nhiêu ngày trước?
   - Kỳ vọng: ACF nhanh chóng về 0 (return gần random walk)

2. ACF của `|log_return|` (absolute return = proxy volatility) — 60 lags
   - Đo: Volatility hôm nay có tương quan với volatility bao nhiêu ngày trước?
   - Kỳ vọng: ACF decay chậm hơn (volatility clustering)

3. PACF của cả 2 — xem partial correlation
   - PACF cutoff cho biết order tối ưu cho AR model

So sánh cross-bank:
- Bank nào ACF decay nhanh hơn? (→ "random" hơn, khó predict hơn)
- Bank nào volatility memory dài hơn?

**Phần B — TFT Attention (deep learning approach)**

- Train TFT trên mỗi bank (code đã có trong Phase 2)
- Dùng `tft.plot_interpretation()` hoặc extract attention weights thủ công
- Attention heatmap: model focus vào lag nào nhiều nhất?
- So sánh attention pattern 3 banks

Lưu ý khi trình bày:
- TFT là thí nghiệm — report kết quả thế nào ra thế đó
- Nếu attention noisy → report là noisy
- Nếu attention clear → report insight
- Không cần TFT "tốt" thì mới report — attention pattern là thông tin riêng, không phụ thuộc prediction accuracy

**Phần C — So sánh ACF vs TFT Attention**

Đặt cạnh nhau: ACF nói gì, TFT attention nói gì?

| Scenario | Ý nghĩa |
|----------|---------|
| ACF và TFT cùng chỉ ra memory ~X ngày | 2 approach khác nhau cho cùng kết luận → kết luận robust |
| ACF clear nhưng TFT noisy | Thống kê truyền thống cho insight rõ hơn DL trên dataset này |
| ACF khác nhau giữa 3 banks | Mỗi bank cần lookback window riêng |
| ACF(return) ≈ 0 nhưng ACF(\|return\|) significant | Return random nhưng volatility có structure → GARCH phù hợp |

### Kết quả: report trung thực

Report kết quả thế nào ra thế đó. TFT attention đẹp thì report đẹp, xấu thì report xấu. ACF cho thấy gì thì nói đó.

### Kết nối Phase 2

- ACF cho biết memory length → justify lookback window trong models (VD: TFT dùng lookback=24 có phù hợp không?)
- ACF(return) vs ACF(|return|) khác nhau → hiểu tại sao volatility prediction dễ hơn price prediction
- So sánh memory 3 banks → confirm hoặc bác bỏ giả thuyết "mỗi bank cần approach riêng"

### Code

File mới: `langkinh3_tft_memory.py`

---

## Tổng hợp 3 Lăng kính

| Lăng kính | Model | Câu hỏi | Method chính |
|-----------|-------|---------|-------------|
| 1 | XGBoost + SHAP | Cái gì chi phối? | SHAP feature importance + cross-bank comparison |
| 2 | NeuralProphet | Khi nào có pattern? | NP decomposition + statistical calendar tests |
| 3 | TFT + ACF | Bao xa trí nhớ? | ACF/PACF + TFT attention extraction |

Mỗi lăng kính dùng model như công cụ khám phá, không phải để so sánh accuracy. Kết quả tốt hay xấu đều report. Insights từ 3 lăng kính tạo foundation cho Phase 2 (5 models prediction).

---

## Files cần làm

| File | Status | Nội dung |
|------|--------|---------|
| `langkinh1_xgboost_shap.py` | ✅ Xong | Target → |log_return|, SHAP, cross-bank DNA |
| `langkinh2_neuralprophet_seasonality.py` | ✅ Xong | NP decomposition + statistical calendar tests + Bonferroni |
| `langkinh3_tft_memory.py` | ✅ Xong | ACF/PACF + TFT train + attention extraction |

