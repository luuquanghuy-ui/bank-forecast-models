# Hướng Dẫn NeuralProphet Main Model

## 1. File chính của nhánh này

Nếu bạn muốn chạy đúng nhánh `NeuralProphet` như một mô hình chính có bản sắc riêng, dùng file:

- `NeuralProphet\run_neuralprophet_main.py`

Không dùng file `run_neuralprophet_3_banks.py` cho phần kết quả chính nếu bạn không muốn trình bày theo kiểu ablation.

## 2. Logic của nhánh main

Nhánh này chỉ giữ:

- `NeuralProphet main`
- `Naive` làm benchmark đối chiếu

Điểm khác biệt của nhánh main là output không dừng ở RMSE/MAE, mà còn xuất riêng:

- forecast plot;
- components plot;
- parameters plot;
- forecast CSV cho test;
- full forecast CSV.

## 3. Lệnh chạy

```powershell
. .\.venv-neural\Scripts\Activate.ps1
python .\NeuralProphet\run_neuralprophet_main.py
```

## 4. File kết quả cần mở

- `NeuralProphet\neuralprophet_main_results.csv`
- `NeuralProphet\neuralprophet_main_report.md`
- `NeuralProphet\neuralprophet_main_outputs\`

## 5. File cần đọc tiếp sau khi có kết quả

- `NeuralProphet\NeuralProphet_can_lam_tiep_truoc_do_an.md`

## 6. Ý nghĩa

Đây mới là nhánh phù hợp nếu bạn muốn giữ đúng bản sắc của `NeuralProphet` trong báo cáo, thay vì biến nó thành một cụm benchmark giống hệt các nhánh khác.
