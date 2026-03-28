# Hướng Dẫn GARCH-XGBoost Main Model

## 1. File chính của nhánh này

Nếu bạn muốn chạy đúng nhánh `GARCH-XGBoost` như mô hình trung tâm của nhánh đầu tiên, dùng file:

- `GARCH_XGBoost\run_garch_xgboost_main.py`

## 2. Logic của nhánh main

Khác với `NeuralProphet` và `TFT`, nhánh này là mô hình lai nên benchmark bắt buộc vẫn phải giữ:

- `Naive`
- `GARCH-only`
- `XGBoost`
- `GARCH-XGBoost main`

Điểm chính là:

- `GARCH-XGBoost main` mới là model trung tâm;
- các model còn lại là benchmark logic bắt buộc để chứng minh phần hybrid có tạo giá trị hay không.

## 3. Output đặc trưng của nhánh này

Ngoài bảng `MAE`, `RMSE`, nhánh này còn xuất riêng:

- tham số `GARCH` theo từng fold;
- `garch sigma` plot của latest fold;
- `GARCH-XGBoost` feature importance;
- forecast CSV của latest fold;
- biểu đồ forecast so sánh giữa target, `Naive`, `GARCH-only`, `GARCH-XGBoost`.

## 4. Lệnh chạy

```powershell
python .\GARCH_XGBoost\run_garch_xgboost_main.py
```

## 5. File kết quả cần mở

- `GARCH_XGBoost\garch_xgboost_main_results.csv`
- `GARCH_XGBoost\garch_xgboost_main_report.md`
- `GARCH_XGBoost\garch_xgboost_main_outputs\`

## 6. File cần đọc tiếp sau khi có kết quả

- `GARCH_XGBoost\GARCH_XGBoost_can_lam_tiep_truoc_do_an.md`

## 7. Ý nghĩa

Đây là nhánh phù hợp nếu bạn muốn giữ đúng bản chất của `GARCH-XGBoost`: mô hình lai cần benchmark để chứng minh phần kết hợp có giá trị hay không, nhưng vẫn phải có output riêng của hybrid chứ không chỉ là một bảng benchmark khô.
