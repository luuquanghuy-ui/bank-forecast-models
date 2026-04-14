# Bank Stock Volatility & Price Prediction

## Mô hình dự đoán biến động và giá cổ phiếu ngân hàng Việt Nam

### Các mô hình
1. **Naive** - Baseline (dự đoán giá trị cuối)
2. **XGBoost** - Gradient Boosting
3. **NeuralProphet** - Deep Learning time series
4. **TFT** - Temporal Fusion Transformer
5. **Hybrid** - GARCH + Ridge (mô hình phát triển)

### Kết quả chính
| Mục tiêu | Mô hình tốt nhất |
|----------|-----------------|
| **Volatility** | TFT (Temporal Fusion Transformer) |
| **Price** | Hybrid GARCH+Ridge (35-41% tốt hơn Naive) |

### Chạy mô hình

```bash
# 4-fold Walk-Forward Cross-Validation
python run_4fold_all_models_both_targets.py

# Sensitivity Analysis
python run_sensitivity_analysis.py

# Market Event Validation
python run_market_event_validation.py
```

### Cấu trúc thư mục
```
├── thesis_documentation/    # Tài liệu chi tiết
├── four_fold_all_targets/   # Kết quả 4-fold CV
├── sensitivity_outputs/     # Kết quả sensitivity analysis
├── market_event_outputs/    # Kết quả market event validation
```

### Đọc tài liệu
Xem `thesis_documentation/README.md` để biết thứ tự đọc và nội dung chi tiết.
