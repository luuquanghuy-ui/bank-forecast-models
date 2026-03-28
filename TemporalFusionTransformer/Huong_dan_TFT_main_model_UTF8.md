# Hướng Dẫn Temporal Fusion Transformer Main Model

## 1. File chính của nhánh này

Nếu bạn muốn chạy đúng nhánh `Temporal Fusion Transformer` như một mô hình chính có bản sắc riêng, dùng file:

- `TemporalFusionTransformer\run_tft_main.py`

Không dùng file `run_tft_3_banks.py` cho phần kết quả chính nếu bạn không muốn trình bày theo kiểu ablation.

## 2. Logic của nhánh main

Nhánh này chỉ giữ:

- `TFT main`
- `Naive` làm benchmark đối chiếu

Điểm khác biệt của nhánh main là ngoài forecast và chỉ số, nó còn xuất đúng artefact đặc trưng của TFT:

- attention weights;
- encoder variable importance;
- decoder variable importance;
- static variable importance;
- forecast CSV và plot test.

## 3. Lệnh chạy

```powershell
. .\.venv-neural\Scripts\Activate.ps1
python .\TemporalFusionTransformer\run_tft_main.py
```

## 4. File kết quả cần mở

- `TemporalFusionTransformer\tft_main_results.csv`
- `TemporalFusionTransformer\tft_main_report.md`
- `TemporalFusionTransformer\tft_main_outputs\`

## 5. File cần đọc tiếp sau khi có kết quả

- `TemporalFusionTransformer\TFT_can_lam_tiep_truoc_do_an.md`

## 6. Ý nghĩa

Đây mới là nhánh phù hợp nếu bạn muốn giữ đúng bản sắc của `TFT` trong báo cáo, thay vì biến nó thành một cụm benchmark giống hệt các nhánh khác.
