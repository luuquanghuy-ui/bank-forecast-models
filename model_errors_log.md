# LOG LỖI CÁC MÔ HÌNH

---

## 1. NEURALPROPHET V2 - LỖI CONCAT MISSING VALUES

### Lỗi 1: Column 'y' missing
```
ValueError: Column 'y' missing from dataframe
```
**Nguyên nhân:** NeuralProphet yêu cầu column tên "y" là target, nhưng code dùng "y_log_return" cho target.
**Cách fix:** Đổi tên column thành "y".

### Lỗi 2: NeuralProphet không chấp nhận extra column 'y_raw'
```
ValueError: Unexpected column y_raw in data
```
**Nguyên nhân:** NeuralProphet không chấp nhận column không phải "y" hoặc lagged regressors.
**Cách fix:** Giữ y_raw riêng, không truyền vào NeuralProphet dataframe.

### Lỗi 3: Missing values khi concat train + test
```
ValueError: Inputs/targets with missing values detected. Please either adjust imputation parameters, or set 'drop_missing' to True to drop those samples.
```
**Nguyên nhân:** NeuralProphet yêu cầu dữ liệu liên tục không gap. Khi concat train (2021-2023) + test (2024), các missing dates (weekends, holidays) không match → lỗi.
**Cách fix (thất bại):**
- Thử `drop_missing=True` → không hỗ trợ parameter này trong `predict()`
- Thử concat riêng rồi filter → vẫn lỗi
- Thử dùng `predict()` trực tiếp trên test_df → NeuralProphet cần context từ training data

**Root cause:** NeuralProphet không designed cho financial data với nhiều missing dates không liên tục.

---

## 2. TFT V2 - LỖI HIDDEN SIZE VÀ TRAINING

### Lỗi 1: Hidden_size=8 quá nhỏ
```
GPU available: False, used: False
`Trainer.fit` stopped: `max_epochs=15` reached.
```
**Vấn đề:** V1 hidden_size=8, epochs=15 → kết quả rất kém (thua Naive 3-13 lần).

**Cách fix V2:**
- hidden_size: 8 → 32
- attention_head: 1 → 4
- epochs: 15 → 80
- lr: 0.01 → 0.001

### Lỗi 2: V2 kết quả tệ hơn V1
```
BID TFT v2 test_rmse: 3.6952 (vs v1: 2.5825)
CTG TFT v2 test_rmse: 7.4914 (vs v1: 7.1174)
VCB TFT v2 test_rmse: 3.5037 (vs v1: 3.2478)
```
**Nguyên nhân:**
- Log-transform gây convexity bias (Jensen's inequality)
- Model lớn hơn không cải thiện vì fundamental problem là data không có predictable patterns
- Sample size quá nhỏ (1750 rows) cho deep learning

**Kết luận:** V2 không cải thiện. Vấn đề không phải hyperparameters mà là bản chất dữ liệu financial.

---

## 3. HYBRID MODELS - LỖI CONCAT VÀ MISSING VALUES

### Lỗi 1: NeuralProphet không chấp nhận drop_missing parameter
```
TypeError: NeuralProphet.predict() got an unexpected keyword argument 'drop_missing'
```
**Nguyên nhân:** `drop_missing` không phải parameter của `predict()`, chỉ có trong dataset creation.
**Cách fix:** Không dùng cách này.

### Lỗi 2: Concatenation tạo ra NaN values
```
ValueError: Inputs/targets with missing values detected.
```
**Nguyên nhân chung:** Khi concat train.tail(N_LAGS) + test, các columns khác (regressors) có NaN vì missing dates không align.
**Các cách thử (đều thất bại):**
- Concat rồi filter theo ds
- Thêm drop_missing=True vào predict
- Đổi cách xây dựng dataframe
- Dùng NeuralProphet với predict_mode

### Lỗi 3: Model training nhưng predict fail
```
ValueError: Inputs/targets with missing values detected. Please either adjust imputation parameters, or set 'drop_missing' to True to drop those samples.
```
**Nguyên nhân:** Dù model train thành công, nhưng khi concat để predict thì vẫn lỗi.

---

## 4. GARCH - KHÔNG CÓ LỖI

GARCH chạy thành công từ đầu đến cuối. Không có vấn đề với:
- Walk-forward prediction
- Missing values
- Concatenation

**Lý do:** GARCH chỉ cần returns array, không cần dataframe với dates/index.

---

## 5. ROOT CAUSE TỔNG HỢP

### Tại sao NeuralProphet/TFT lỗi concat:

1. **Financial data có nhiều missing dates** (weekends, holidays Việt Nam)
2. **NeuralProphet/TFT framework-based** → không linh hoạt với cách xử lý data đặc thù
3. **Không có cách đơn giản để concat** train context + test prediction mà không tạo NaN

### Tại sao GARCH không lỗi:

1. **GARCH chỉ cần numpy array** (returns)
2. **Không phụ thuộc vào dates/index**
3. **Walk-forward prediction đơn giản:** fit trên train → predict từng điểm test

### Giải pháp thay thế (thành công):

**GARCH + Ridge Ensemble:**
- GARCH: predict volatility (walk-forward)
- Ridge: simple linear regression với features
- Ensemble: weighted average của 2 predictions
- Không cần concat, không bị missing values

---

## 6. CÁC LỖI CỤ THỂ THEO THỨ TỰ

1. `ModuleNotFoundError: No module named 'torch'` - Chạy sai Python environment
2. `ValueError: Column 'y' missing` - NeuralProphet yêu cầu "y"
3. `ValueError: Unexpected column y_raw` - NeuralProphet không chấp nhận extra columns
4. `ValueError: Inputs/targets with missing values detected` - Concat train+test gây NaN
5. `TypeError: NeuralProphet.predict() got an unexpected keyword argument 'drop_missing'` - Sai parameter name
6. GARCH-XGBoost chạy OK từ đầu
7. NeuralProphet V1 chạy OK (nhưng thua Naive)
8. TFT V1 chạy OK (nhưng thua Naive rất nặng)
9. NeuralProphet V2 thua V1 (log-transform không cải thiện)
10. TFT V2 thua V1 (lớn hơn không cải thiện)
11. Hybrid NP approaches thất bại (concat issues)
12. GARCH + Ridge Ensemble THÀNH CÔNG
