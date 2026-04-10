# README - Workspace Mô Hình Dự Báo Cho 3 Ngân Hàng

## QUICK START - Setup trong 5 phút

### 1. Clone về
```bash
git clone https://github.com/luuquanghuy-ui/bank-forecast-models.git
cd bank-forecast-models
```

### 2. Tạo môi trường Python
```bash
# Tạo virtual environment
python -m venv thesis_env

# Activate (Windows)
thesis_env\Scripts\activate

# Activate (Mac/Linux)
source thesis_env/bin/activate
```

### 3. Cài đặt packages
```bash
pip install -r requirements.txt
```

### 4. Chạy thử
```bash
# GARCH model (volatility)
python GARCH_XGBoost/run_garch_xgboost_main.py

# Hybrid ensemble
python Hybrid_GARCH_DL/run_ensemble.py
```

---

## 1. Mục đích

Workspace này được sắp xếp để làm việc với 3 ngân hàng tách riêng:

- `BID`
- `CTG`
- `VCB`

Và 3 nhánh mô hình chính:

- `GARCH_XGBoost`
- `NeuralProphet`
- `TemporalFusionTransformer`

Mỗi nhánh đều có:

- script chạy model chính;
- file kết quả chính;
- thư mục output riêng;
- file checklist các việc còn phải làm trước khi đưa vào đồ án.

## 2. Cấu trúc thư mục gốc

Trong `D:\labs 2\DOANPTDLKD`:

- `banks_master_dataset_final - banks_master_dataset_final.csv`: dữ liệu gốc
- `banks_BID_dataset.csv`: dữ liệu đã tách cho BID
- `banks_CTG_dataset.csv`: dữ liệu đã tách cho CTG
- `banks_VCB_dataset.csv`: dữ liệu đã tách cho VCB
- `split_by_bank.py`: script tách file gốc thành 3 CSV
- `GARCH_XGBoost\`: nhánh mô hình GARCH-XGBoost
- `NeuralProphet\`: nhánh mô hình NeuralProphet
- `TemporalFusionTransformer\`: nhánh mô hình TFT

## 3. Bước đầu tiên - Tách dữ liệu

Nếu muốn tạo lại 3 file CSV từ file gốc, chạy:

```powershell
python .\split_by_bank.py
```

Sau khi chạy xong sẽ có:

- `banks_BID_dataset.csv`
- `banks_CTG_dataset.csv`
- `banks_VCB_dataset.csv`

## 4. Mở terminal trong VS Code

- bấm `Ctrl + \``
- hoặc vào `Terminal -> New Terminal`

Khi đúng thư mục gốc, terminal sẽ giống như:

```powershell
PS D:\labs 2\DOANPTDLKD>
```

## 5. Cài đặt cần thiết

### 5.1. Cho GARCH_XGBoost

Nếu máy chưa có `xgboost` và `arch`, chạy:

```powershell
python -m pip install xgboost arch --target .vendor_real --no-cache-dir
```

### 5.2. Cho NeuralProphet và TFT

Hai nhánh này dùng chung môi trường Python `3.12` trong `.venv-neural`.

Nếu chưa có, chạy:

```powershell
uv venv .venv-neural -p 3.12 --seed --cache-dir .uvcache
. .\.venv-neural\Scripts\Activate.ps1
python -m pip install neuralprophet --no-cache-dir
python -m pip install pytorch-forecasting lightning --no-cache-dir
```

## 6. Nhánh chính 1 - GARCH_XGBoost

### 6.1. Script chính

- `GARCH_XGBoost\run_garch_xgboost_main.py`

### 6.2. Cách chạy

```powershell
python .\GARCH_XGBoost\run_garch_xgboost_main.py
```

### 6.3. Model trong nhánh này

- `Naive`
- `GARCH-only`
- `XGBoost`
- `GARCH-XGBoost main`

Lưu ý:

- ở nhánh này, benchmark là bắt buộc vì đây là mô hình lai;
- `GARCH-XGBoost main` là mô hình trung tâm, còn các model khác là đối chiếu logic.

### 6.4. Chỉ số và output

Chỉ số chính:

- `MAE`
- `RMSE`

Output riêng của nhánh này:

- tham số `GARCH` theo từng fold
- `garch sigma` plot
- `feature importance` của hybrid
- forecast latest fold
- plot forecast latest fold

### 6.5. Mở file nào để xem

- `GARCH_XGBoost\garch_xgboost_main_results.csv`
- `GARCH_XGBoost\garch_xgboost_main_report.md`
- `GARCH_XGBoost\garch_xgboost_main_outputs\`
- `GARCH_XGBoost\Huong_dan_GARCH_XGBoost_main_model_UTF8.md`
- `GARCH_XGBoost\GARCH_XGBoost_can_lam_tiep_truoc_do_an.md`

## 7. Nhánh chính 2 - NeuralProphet

### 7.1. Script chính

- `NeuralProphet\run_neuralprophet_main.py`

### 7.2. Cách chạy

```powershell
. .\.venv-neural\Scripts\Activate.ps1
python .\NeuralProphet\run_neuralprophet_main.py
```

### 7.3. Model trong nhánh này

- `Naive`
- `NeuralProphet main`

### 7.4. Chỉ số và output

Chỉ số chính:

- `MAE`
- `RMSE`

Output riêng của nhánh này:

- forecast plot
- components plot
- parameters plot
- full forecast CSV
- test forecast CSV

### 7.5. Mở file nào để xem

- `NeuralProphet\neuralprophet_main_results.csv`
- `NeuralProphet\neuralprophet_main_report.md`
- `NeuralProphet\neuralprophet_main_outputs\`
- `NeuralProphet\Huong_dan_NeuralProphet_main_model_UTF8.md`
- `NeuralProphet\NeuralProphet_can_lam_tiep_truoc_do_an.md`

## 8. Nhánh chính 3 - Temporal Fusion Transformer

### 8.1. Script chính

- `TemporalFusionTransformer\run_tft_main.py`

### 8.2. Cách chạy

```powershell
. .\.venv-neural\Scripts\Activate.ps1
python .\TemporalFusionTransformer\run_tft_main.py
```

### 8.3. Model trong nhánh này

- `Naive`
- `TFT main`

### 8.4. Chỉ số và output

Chỉ số chính:

- `MAE`
- `RMSE`

Output riêng của nhánh này:

- attention weights
- encoder variable importance
- decoder variable importance
- static variable importance
- test forecast CSV
- test forecast plot

### 8.5. Mở file nào để xem

- `TemporalFusionTransformer\tft_main_results.csv`
- `TemporalFusionTransformer\tft_main_report.md`
- `TemporalFusionTransformer\tft_main_outputs\`
- `TemporalFusionTransformer\Huong_dan_TFT_main_model_UTF8.md`
- `TemporalFusionTransformer\TFT_can_lam_tiep_truoc_do_an.md`

## 9. Nếu muốn chạy các bản phụ / ablation

Các file này vẫn được giữ lại để làm thử nghiệm mở rộng, không phải kết quả chính nên không cần ưu tiên khi gửi folder:

- `GARCH_XGBoost\fair_garch_xgboost_walkforward.py`
- `GARCH_XGBoost\pilot_garch_xgboost_3_banks.py`
- `NeuralProphet\run_neuralprophet_3_banks.py`
- `TemporalFusionTransformer\run_tft_3_banks.py`

## 10. Cách đọc chỉ số

- `MAE`: càng nhỏ càng tốt
- `RMSE`: càng nhỏ càng tốt

Nếu cần đọc nhanh, ưu tiên nhìn `RMSE` trước.

## 11. Thứ tự chạy khuyến nghị

1. Tách dữ liệu:

```powershell
python .\split_by_bank.py
```

2. Chạy GARCH_XGBoost main:

```powershell
python .\GARCH_XGBoost\run_garch_xgboost_main.py
```

3. Chạy NeuralProphet main:

```powershell
. .\.venv-neural\Scripts\Activate.ps1
python .\NeuralProphet\run_neuralprophet_main.py
```

4. Chạy TFT main:

```powershell
. .\.venv-neural\Scripts\Activate.ps1
python .\TemporalFusionTransformer\run_tft_main.py
```

## 12. File nên mở đầu tiên

Nếu chỉ cần xem nhanh để hiểu toàn bộ workspace, mở theo thứ tự này:

- `README.md`
- `GARCH_XGBoost\garch_xgboost_main_results.csv`
- `NeuralProphet\neuralprophet_main_results.csv`
- `TemporalFusionTransformer\tft_main_results.csv`

## 13. Ghi chú quan trọng

- Kết quả hiện tại là kết quả đã chạy thật trên dữ liệu này.
- Không phải model nào cũng thắng benchmark.
- Một model thua benchmark vẫn có thể đưa vào báo cáo nếu quy trình chạy đúng, có test set rõ ràng và phần thảo luận trung thực.
- Trước khi bỏ vào đồ án, nên đọc thêm file checklist của từng nhánh để biết còn phải bổ sung gì.
