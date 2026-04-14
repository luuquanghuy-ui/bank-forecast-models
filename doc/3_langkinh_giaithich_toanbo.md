# 3 Lăng kính: Giải thích toàn bộ Code, Kết quả & Ý nghĩa

---

## Tổng quan

Trước khi dự đoán (Phase 2), nhóm dùng 3 model như 3 cái "kính lúp" để hiểu thị trường ngân hàng VN từ 3 góc khác nhau. Mỗi model có thế mạnh riêng, trả lời 1 câu hỏi riêng.

| Lăng kính | Model | Câu hỏi | Ví dụ dễ hiểu |
|-----------|-------|---------|---------------|
| 1 | XGBoost + SHAP | Cái gì chi phối? | "Giá ngân hàng bị kéo bởi cái gì? RSI? Tỷ giá? Volume?" |
| 2 | NeuralProphet | Khi nào có quy luật? | "Thứ 2 có khác thứ 4 không? Tháng 1 có khác tháng 7?" |
| 3 | TFT + ACF/PACF | Bao xa trí nhớ? | "Hôm nay biến động mạnh, bao lâu sau trở lại bình thường?" |

---

# LĂNG KÍNH 1: XGBoost + SHAP

## Model XGBoost là gì?

XGBoost là thuật toán machine learning dạng "cây quyết định". Nó hoạt động như sau:

Tưởng tượng mày là bác sĩ khám bệnh:
- Bệnh nhân đến, mày hỏi: "Sốt hơn 38 độ không?" → Có/Không
- Nếu Có: "Ho nhiều không?" → Có/Không
- Nếu Không: "Đau bụng không?" → ...
- Sau vài câu hỏi → chẩn đoán bệnh

XGBoost cũng vậy: nó hỏi hàng trăm câu hỏi kiểu "RSI > 70 không?", "Volume hôm qua > trung bình không?" và từ đó dự đoán kết quả. Mỗi "câu hỏi" gọi là 1 split trong cây.

XGBoost mạnh vì:
- Không cần giả định dữ liệu phải linear (thẳng hàng)
- Bắt được quan hệ phức tạp (ví dụ: "RSI > 70 VÀ volume tăng → giá sẽ giảm")
- Nhanh, ổn định

## SHAP là gì?

Sau khi XGBoost dự đoán, mày muốn biết: "Tại sao model cho đáp án này?" SHAP trả lời câu đó.

Ví dụ: XGBoost dự đoán ngày mai BID biến động 2%. SHAP nói:
- "Volume cao hôm nay đóng góp +0.8%"
- "RSI thấp đóng góp +0.5%"
- "USD/VND ổn định đóng góp -0.3%"
- Tổng = 2%

SHAP dựa trên **Shapley value** — một khái niệm từ lý thuyết trò chơi (game theory). Cách tính: thử bỏ từng yếu tố ra, xem kết quả thay đổi bao nhiêu → biết yếu tố đó đóng góp bao nhiêu.

## Code giải thích

### Target = |log_return|

```python
df['target'] = np.abs(df['log_return'])
```

`log_return` = log(giá hôm nay / giá hôm qua). Nếu giá tăng 1%, log_return ≈ +0.01. Nếu giảm 1%, log_return ≈ -0.01.

`|log_return|` = lấy giá trị tuyệt đối, bỏ dấu. Tăng 1% và giảm 1% đều = 0.01. Đây là **mức độ biến động** — không quan tâm hướng tăng/giảm, chỉ quan tâm biến động MẠNH hay YẾU.

Tại sao không dùng `log_return` thô? Vì log_return gần như random (hôm nay tăng, mai giảm, random). XGBoost dự đoán nó sẽ rất tệ → SHAP values từ model tệ thì vô nghĩa. |log_return| thì có quy luật (volatility clustering) → XGBoost fit được → SHAP meaningful.

### Features (15 biến đầu vào)

```python
# Lagged returns: return 1,2,3,5 ngày trước
df['return_lag1'] = df['log_return'].shift(1)

# Volatility lag: volatility_20d (trung bình biến động 20 ngày) hôm qua
df['volatility_lag1'] = df['volatility_20d'].shift(1)

# RSI lag: chỉ báo RSI hôm qua (RSI > 70 = overbought, < 30 = oversold)
df['rsi_lag1'] = df['rsi'].shift(1)

# Volume ratio: volume hôm nay / trung bình 20 ngày → volume bất thường?
df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

# MA ratio: giá / đường trung bình 20 ngày → giá đang trên/dưới xu hướng?
df['ma_ratio'] = df['close'] / df['ma20']

# Macro: tỷ giá USD/VND, lãi suất, VNIndex, VN30 hôm qua
df['usd_vnd_lag1'] = df['usd_vnd'].shift(1)
```

`shift(1)` = lấy giá trị của NGÀY HÔM QUA. Quan trọng vì: ta dự đoán hôm nay dựa trên thông tin hôm qua, KHÔNG ĐƯỢC nhìn vào hôm nay (đó là "data leakage" = gian lận).

### Train/Test Split

```python
train_end = int(n * 0.70)  # 70% đầu tiên = train
val_end = int(n * 0.85)    # 15% tiếp = validation
# 15% cuối = test
```

Chia THEO THỜI GIAN, không random. Vì time series: dùng quá khứ dự đoán tương lai, không thể trộn lẫn.

### SHAP computation

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)  # Tính SHAP cho mỗi row trong test set
```

Output: 1 bảng số, mỗi row = 1 ngày, mỗi cột = 1 feature, giá trị = đóng góp của feature đó cho dự đoán ngày đó.

## Kết quả & Ý nghĩa

### Model fit

| Bank | R² | Ý nghĩa |
|------|----|---------|
| BID | 0.246 | XGBoost giải thích 25% biến động BID — fit tốt nhất |
| VCB | 0.064 | Giải thích 6% — yếu |
| CTG | -0.124 | Tệ hơn cả predict trung bình — CTG khó hiểu nhất |

R² = "model giải thích được bao nhiêu %?" R² = 1 là hoàn hảo, R² = 0 là không giải thích gì, R² < 0 là tệ hơn cả predict giá trị trung bình.

BID dễ predict nhất, CTG khó nhất. **Cùng ngành nhưng khác nhau.**

### Technical vs Macro

| Bank | Technical | Macro | Nghĩa là |
|------|-----------|-------|----------|
| BID | **86.7%** | 13.3% | Biến động BID chủ yếu do RSI, volume, xu hướng giá |
| CTG | **76.7%** | 23.3% | CTG bị macro (tỷ giá, VNIndex) ảnh hưởng nhiều hơn |
| VCB | **78.3%** | 21.7% | VCB cũng nhạy macro hơn BID |

Ý nghĩa: Technical indicators (thông tin từ chính cổ phiếu) quan trọng hơn Macro indicators (thông tin kinh tế vĩ mô) ở TẤT CẢ banks. Nhưng mức độ khác nhau: BID gần như chỉ cần technical, CTG cần kết hợp macro.

### DNA mỗi bank — Khác biệt lớn nhất

| Feature | BID rank | CTG rank | VCB rank | Điều này cho thấy |
|---------|----------|----------|----------|-------------------|
| VN30 | #14 | #9 | **#4** | VCB đi theo thị trường chung nhất (blue-chip lớn nhất) |
| USD/VND | #5 | **#2** | #6 | CTG nhạy tỷ giá nhất (hướng quốc tế) |
| RSI | **#3** | #6 | **#2** | BID/VCB bị retail investor chi phối (dùng RSI giao dịch) |

### Model XGBoost + SHAP mang lại gì cho Lăng kính 1?

XGBoost + SHAP cho phép **bóc tách** đóng góp của từng yếu tố — điều mà correlation đơn giản hay nhìn mắt thường KHÔNG làm được. Nó trả lời chính xác: "1% thay đổi của RSI ảnh hưởng bao nhiêu đến biến động?" cho từng bank.

Không model nào khác (NP, TFT, GARCH) có khả năng giải thích features tốt như XGBoost + SHAP. Đây là lý do XGBoost được chọn cho Lăng kính 1.

---

# LĂNG KÍNH 2: NeuralProphet + Statistical Tests

## NeuralProphet là gì?

NeuralProphet là model dự báo time series dựa trên neural network, phát triển từ Facebook Prophet. Thế mạnh riêng:

Nó **tách** chuỗi thời gian thành các thành phần:
- **Trend** = xu hướng dài hạn (giá tăng hay giảm qua nhiều năm?)
- **Weekly seasonality** = quy luật theo ngày trong tuần (thứ 2 khác thứ 5?)
- **Yearly seasonality** = quy luật theo tháng trong năm (tháng 1 khác tháng 7?)
- **Residual** = phần còn lại không giải thích được

Ví dụ: NP nói "Giá BID hôm nay = trend tăng +0.5% + weekly seasonality -0.1% (thứ 2 thường xấu) + yearly seasonality +0.2% (tháng 1 tốt) + noise"

## Statistical Tests là gì? Tại sao thêm vào?

NP decompose cho đẹp, nhưng NP có thể "tìm" pattern ở chỗ không có (overfit noise). Để kiểm tra pattern CÓ THẬT không, dùng statistical test.

**Kruskal-Wallis test**: So sánh nhiều nhóm cùng lúc.
- Ví dụ: Chia return theo 5 ngày (Mon-Fri) → 5 nhóm
- Test hỏi: "5 nhóm này có giống nhau không?"
- Nếu p < 0.05 → "Không, ít nhất 1 ngày khác biệt" → Calendar effect CÓ THẬT

**Mann-Whitney U test**: So sánh 2 nhóm.
- Ví dụ: Monday return vs tất cả ngày khác
- Test hỏi: "2 nhóm này giống nhau không?"
- Nếu p < 0.05 → "Không, Monday khác thật" → Monday Effect CÓ

Tại sao dùng Kruskal-Wallis/Mann-Whitney chứ không phải t-test thông thường?
- t-test giả định dữ liệu phân phối chuẩn (đường cong hình chuông)
- Return cổ phiếu **KHÔNG** phân phối chuẩn — nó có "đuôi mập" (fat tails): ngày tăng/giảm cực mạnh xảy ra nhiều hơn lý thuyết
- Kruskal-Wallis / Mann-Whitney **KHÔNG cần** giả định phân phối chuẩn → chính xác hơn cho financial data

## Code giải thích

### NP Decomposition

```python
model = NeuralProphet(
    growth='linear',              # Trend dạng đường thẳng
    yearly_seasonality=True,      # Bật phân tích mùa vụ theo năm
    weekly_seasonality=True,      # Bật phân tích mùa vụ theo tuần
    daily_seasonality=False,      # Tắt (dữ liệu daily, không cần intra-day)
    n_lags=0,                     # Không dùng AR
    epochs=30,                    # Train 30 vòng
)
```

NP cần 2 cột: `ds` (ngày) và `y` (giá trị). Ở đây `y = close price`.

### Statistical Tests

```python
# Chia return theo ngày trong tuần
day_groups = [df[df['day_of_week'] == d]['log_return'].values for d in range(5)]

# Kruskal-Wallis: "5 ngày có giống nhau không?"
kw_stat, kw_p = stats.kruskal(*day_groups)
```

`*day_groups` = truyền 5 nhóm dữ liệu vào function. Scipy tự tính p-value.

```python
# Mann-Whitney: "Monday có khác tất cả ngày khác không?"
monday = df[df['day_of_week'] == 0]['log_return'].values
other_days = df[df['day_of_week'] != 0]['log_return'].values
mw_stat, mw_p = stats.mannwhitneyu(monday, other_days, alternative='two-sided')
```

`two-sided` = test cả 2 hướng (Monday có thể tốt HƠN hoặc tệ HƠN, không biết trước).

## Kết quả & Ý nghĩa

### January Effect — Phát hiện quan trọng nhất

**Cả 3 ngân hàng, tháng 1 return cao gấp 10-20 lần trung bình.**

| Bank | Return trung bình tháng 1 | Return trung bình các tháng khác | Gấp mấy lần |
|------|--------------------------|----------------------------------|------------|
| BID | +0.63% mỗi ngày | +0.03% | ~21 lần |
| CTG | +0.45% | +0.03% | ~15 lần |
| VCB | +0.46% | +0.03% | ~15 lần |

Tại sao tháng 1 tốt?
- **Window dressing**: Quỹ đầu tư bán cổ phiếu "xấu" cuối tháng 12 để báo cáo đẹp → giá giảm → tháng 1 mua lại → giá tăng
- **Dòng tiền đầu năm**: Nhà đầu tư phân bổ vốn mới đầu năm
- **Tâm lý lạc quan đầu năm**

Statistical test (Mann-Whitney): p = 0.003 - 0.023 → **97-99.7% đây là pattern thật**.

Lưu ý trung thực: Sau Bonferroni correction (vì test 18 lần), chỉ CTG và VCB January Effect còn significant chắc chắn (p ≈ 0.003). BID borderline (p = 0.023).

### Monday Effect — BID và VCB có, CTG không

| Ngày | BID | CTG | VCB |
|------|-----|-----|-----|
| **Mon** | **-0.20%** | -0.12% | **-0.11%** |
| Tue | +0.21% | +0.14% | +0.20% |
| **Wed** | **+0.33%** | **+0.26%** | **+0.21%** |
| Thu | -0.09% | -0.02% | +0.04% |
| Fri | +0.10% | +0.07% | -0.003% |

Pattern chung: **Thứ 2 xấu nhất, thứ 4 tốt nhất.** Nhưng chỉ BID/VCB significant, CTG thì không.

CTG không có Monday Effect → CTG "khác biệt" — confirm Lăng kính 1 (CTG có DNA riêng, bị chi phối bởi macro hơn retail behavior).

### Kết quả không có

- **Month-of-Year Effect**: Không significant → ngoài tháng 1, không tháng nào đặc biệt
- **Quarter-End Effect**: Không significant → cuối quý không khác gì
- **Friday Effect**: Chỉ VCB có nhẹ

### Model NeuralProphet mang lại gì cho Lăng kính 2?

NP có khả năng **tự động tách trend + seasonality** — nhìn thấy trực quan từng thành phần. KHÔNG model nào khác (XGBoost, TFT, GARCH) có khả năng decompose như NP.

Nhưng NP chỉ cho kết quả visual. Để CHỨNG MINH pattern có thật, cần statistical test (Kruskal-Wallis, Mann-Whitney). Hai công cụ bổ sung cho nhau:
- NP: "tháng 1 trông có vẻ tốt hơn" (visual)
- Statistical test: "tháng 1 tốt hơn thật, p = 0.003" (chứng minh)

---

# LĂNG KÍNH 3: TFT Attention + ACF/PACF

## ACF là gì?

ACF = Autocorrelation Function = **hàm tự tương quan**.

Nói đơn giản: "Giá trị hôm nay có liên quan đến giá trị X ngày trước không?"

- ACF(1) = mối liên quan giữa hôm nay và hôm qua
- ACF(5) = mối liên quan giữa hôm nay và 5 ngày trước
- ACF(20) = mối liên quan giữa hôm nay và 20 ngày trước

Giá trị ACF:
- ACF = 1: hoàn toàn giống nhau (hôm nay = hôm qua)
- ACF = 0: không liên quan (random)
- ACF = -1: ngược hoàn toàn (hôm nay tăng thì hôm qua giảm)

## PACF là gì?

PACF = Partial ACF = ACF nhưng **loại bỏ ảnh hưởng trung gian**.

Ví dụ: Hôm nay liên quan đến 3 ngày trước. Nhưng có thể vì "hôm nay → hôm qua → 2 ngày trước → 3 ngày trước" (chain effect). PACF loại bỏ chain, chỉ đo **trực tiếp**.

PACF cho biết: để predict, cần nhìn lùi **TỐI ĐA** bao nhiêu ngày.

## Ljung-Box test là gì?

ACF đo từng lag riêng lẻ. Ljung-Box đo **TỔNG THỂ**: "Nhìn chung, có autocorrelation đáng kể nào trong chuỗi không?"

- p < 0.05 → CÓ autocorrelation → chuỗi có quy luật
- p > 0.05 → KHÔNG autocorrelation → chuỗi random

## Tại sao tính ACF cho cả return VÀ |return|?

Đây là thí nghiệm quan trọng nhất:

- **ACF(return)** = "Hướng giá (tăng/giảm) hôm nay liên quan đến hôm qua không?"
- **ACF(|return|)** = "Mức độ biến động hôm nay liên quan đến hôm qua không?"

Nếu 2 cái này cho kết quả KHÁC NHAU → insight cực giá trị.

## Code giải thích

```python
# Tính ACF 60 lags, với confidence interval 95%
acf_vals, acf_ci = acf(series, nlags=60, alpha=0.05)
```

60 lags = 60 ngày giao dịch ≈ 3 tháng. Đủ xa để thấy "trí nhớ" dài nhất.

```python
# Đường confidence interval: nếu ACF nằm trong đường này = random
ci_bound = 1.96 / np.sqrt(n)
```

1.96 là giá trị z cho 95% confidence. `n` = số data points. Nếu ACF nằm trong ±ci_bound → **không significant** → random.

```python
# Ljung-Box test nhiều lags
lb_results = acorr_ljungbox(series, lags=[5, 10, 20, 40])
```

Test ở nhiều lags để xem: autocorrelation mạnh ở ngắn hạn hay dài hạn?

```python
# Half-life: ACF giảm xuống còn 1/2 ở lag nào?
def compute_halflife(acf_vals):
    target = acf_vals[1] / 2
    for lag in range(2, len(acf_vals)):
        if acf_vals[lag] <= target:
            return lag
```

Half-life = "thời gian quán tính giảm nửa". Biến động mạnh hôm nay, bao lâu sau ảnh hưởng giảm còn phân nửa?

## Kết quả & Ý nghĩa

### Return: GẦN NHƯ RANDOM WALK

| Bank | ACF(1) | Significant lags / 60 | Ljung-Box (lag 10) |
|------|--------|----------------------|-------------------|
| BID | +0.001 | 7 (rải rác) | p = 0.506 (random) |
| CTG | -0.035 | 7 (rải rác) | p = 0.011 |
| VCB | +0.029 | 6 (rải rác) | p = 0.028 |

ACF(1) gần 0 ở cả 3 banks → **giá tăng/giảm hôm qua KHÔNG giúp dự đoán hôm nay**. Thị trường ngân hàng VN hoạt động gần giống random walk — đúng với lý thuyết thị trường hiệu quả (Efficient Market Hypothesis).

Lưu ý: CTG có ACF(1) = -0.035 (âm nhẹ) → dấu hiệu "mean-reversion" rất yếu: nếu hôm nay tăng, hôm sau HƠI XU HƯỚNG giảm. Nhưng quá yếu để exploit.

### |Return|: VOLATILITY CLUSTERING CỰC MẠNH

| Bank | ACF(1) | Significant lags / 60 | Half-life | Ljung-Box (lag 10) |
|------|--------|----------------------|-----------|-------------------|
| BID | **+0.236** | **60/60** | **6 ngày** | p = 0.000000 |
| CTG | **+0.209** | **60/60** | **16 ngày** | p = 0.000000 |
| VCB | **+0.242** | **51/60** | **8 ngày** | p = 0.000000 |

Đây là kết quả QUAN TRỌNG NHẤT toàn bộ 3 lăng kính:

**ACF(1) = 0.21-0.24**: Biến động hôm qua giải thích ~21-24% biến động hôm nay. Rất cao!

**Significant lags = 60/60**: TẤT CẢ 60 lags đều significant. Biến động có "trí nhớ" kéo dài ít nhất 3 tháng!

**Ljung-Box p = 0.000000**: Không còn nghi ngờ gì — volatility clustering **CHẮC CHẮN 100%** tồn tại.

### Half-life — Phát hiện Cross-bank

| Bank | Half-life | Giải thích |
|------|-----------|-----------|
| BID | 6 ngày | Tin xấu → biến động 1 tuần → bình thường |
| VCB | 8 ngày | Trung bình |
| CTG | **16 ngày** | Tin xấu → biến động kéo dài **2-3 tuần**! |

CTG half-life GẤP ĐÔI BID. CTG biến động "dai" hơn, khó predict hơn. Điều này giải thích:
- Lăng kính 1: CTG R² = -0.12 (tệ nhất)
- Lăng kính 2: CTG không có Day-of-Week Effect (random hơn)

### Model ACF/PACF + TFT Attention mang lại gì cho Lăng kính 3?

ACF/PACF là công cụ thống kê **cổ điển, đơn giản, không thể bắt bẻ**. Nó đo trực tiếp quan hệ giữa hiện tại và quá khứ mà không cần train model phức tạp.

Nó phát hiện sự đối lập Return vs |Return| — insight mà không model machine learning nào cho thấy rõ bằng.

TFT Attention bổ sung: train TFT 15 epochs trên close price, rồi extract encoder_attention weights. Attention cho biết model tập trung vào ngày nào trong 24 ngày trước → so sánh với ACF. Nếu 2 approach cùng chỉ ra lags gần quan trọng → kết luận robust.

---

# TỔNG HỢP: 3 Lăng kính nối với nhau

## Mỗi lăng kính phát hiện gì?

```
Lăng kính 1 (XGBoost+SHAP): "CÁI GÌ?"
  → Volume Ratio, Price/MA20, RSI chi phối nhất
  → Technical 77-87%, Macro 13-23%
  → Mỗi bank DNA riêng: BID=retail, CTG=macro, VCB=market-following

Lăng kính 2 (NP + Statistics): "KHI NÀO?"
  → January Effect CÓ THẬT (p < 0.003-0.023)
  → Monday Effect ở BID/VCB, KHÔNG ở CTG
  → Calendar effects yếu nhìn chung → return gần random

Lăng kính 3 (TFT + ACF/PACF): "BAO XA?"
  → Return: random walk, không có trí nhớ
  → |Return|: volatility clustering CỰC MẠNH, memory 51-60 lags
  → CTG biến động dai nhất (half-life 16 ngày)
  → TFT Attention confirm: model focus vào lags gần nhất
```

## Insights chéo giữa 3 lăng kính

| Insight | Lăng kính 1 | Lăng kính 2 | Lăng kính 3 |
|---------|-------------|-------------|-------------|
| CTG khác biệt | R² = -0.12, Macro cao nhất | Không có Monday Effect | Half-life dài nhất (16 ngày) |
| Return random | - | Calendar effects yếu | ACF(return) ≈ 0 |
| Volatility có quy luật | Volatility lag là feature quan trọng | - | ACF(|return|) = 0.21-0.24, 60/60 lags |
| Mỗi bank khác nhau | DNA ranking khác nhau | Calendar effects khác nhau | Half-life khác nhau |

## Kết nối sang Phase 2 — Tại sao Hybrid?

3 lăng kính cho thấy:

1. **Return gần random walk** (LK2 + LK3) → Dự đoán HƯỚNG giá rất khó → Naive baseline sẽ mạnh cho price
2. **Volatility clustering CỰC MẠNH** (LK1 + LK3) → Dự đoán MỨC BIẾN ĐỘNG khả thi → GARCH + TFT đều phù hợp
3. **Technical features quan trọng** (LK1) → Ridge Regression với technical features sẽ giúp price prediction
4. **Mỗi model có chỗ đứng riêng**: TFT attention bắt được volatility clustering → TFT thắng volatility. NP seasonal yếu → NP thua. Hybrid kết hợp GARCH + Ridge → thắng price.

→ **Kết quả Phase 2: TFT thắng volatility (nhờ attention bắt vol clustering), Hybrid thắng price (nhờ GARCH + Ridge). Không model nào thắng tất cả — mỗi task cần approach phù hợp.**

Đây chính là câu chuyện: **Hiểu trước, thiết kế sau.**
