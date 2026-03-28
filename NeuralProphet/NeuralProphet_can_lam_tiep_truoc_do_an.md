# Những Việc Cần Làm Tiếp Trước Khi Đưa NeuralProphet Vào Đồ Án

## 1. Kết quả hiện tại có ý nghĩa gì

Nhánh `NeuralProphet` trong workspace này đã chạy xong end-to-end, nghĩa là:

- đã train/validation/test theo thời gian;
- đã có benchmark `Naive`;
- đã có hai cấu hình `NeuralProphet baseline` và `NeuralProphet + lagged regressors`;
- đã xuất file kết quả, forecast CSV và hình.

Tuy nhiên, kết quả hiện tại chưa đủ để kết luận NeuralProphet là mô hình tốt nhất.

## 2. Việc cần làm tiếp trước khi đưa vào đồ án

### 2.1. Chốt vai trò của NeuralProphet trong bài

Với kết quả hiện tại, cách đặt vai trò an toàn nhất là:

- benchmark hiện đại cho bài toán dự báo chuỗi thời gian;
- hoặc nhánh thực nghiệm mở rộng.

Không nên mặc định xem NeuralProphet là mô hình chính nếu chưa có thêm tuning hoặc thêm bằng chứng thực nghiệm tốt hơn.

### 2.2. Thống nhất target với các nhánh mô hình khác

Hiện tại NeuralProphet đang dự báo:

- `close`

Nếu phần còn lại của đồ án đi theo hướng dự báo volatility/risk, bạn cần:

- hoặc chạy lại NeuralProphet với target thống nhất;
- hoặc tách riêng nhánh này thành bài toán dự báo giá, không trộn bảng kết quả với GARCH.

### 2.3. Tuning lại các tham số quan trọng

Cấu hình hiện tại mới là cấu hình chạy được. Nên thêm một vòng tuning có kiểm soát cho:

- `n_lags`
- `epochs`
- `batch_size`
- `learning_rate`
- tập `lagged regressors`

Việc tuning phải dựa trên validation chứ không nhìn test để chỉnh.

### 2.4. Làm rõ logic chọn regressor

Trong báo cáo phải nói rõ:

- biến nào là `lagged regressor`;
- vì sao dùng lagged regressor thay vì future regressor;
- vì sao các biến như `volume`, `rsi`, `volatility_20d`, `vnindex_close`, `vn30_close`, `usd_vnd`, `interest_rate` được đưa vào mô hình.

### 2.5. Kiểm tra trực quan forecast

Mở thư mục:

- `NeuralProphet\neuralprophet_outputs\`

để kiểm tra:

- forecast có lệch level mạnh không;
- có bị drift không;
- mô hình có bám được turning points không.

### 2.6. Viết phần thảo luận trung thực

Cách viết phù hợp với kết quả hiện tại là:

- NeuralProphet đã được triển khai đầy đủ trên ba ngân hàng;
- tuy nhiên benchmark `Naive` vẫn cho sai số test thấp hơn;
- điều này cho thấy trên bộ dữ liệu hiện tại, NeuralProphet chưa phát huy được lợi thế rõ rệt.

## 3. Những gì nên bổ sung vào báo cáo

### 3.1. Bảng cấu hình mô hình

Nên có bảng riêng ghi:

- target;
- `n_lags`;
- `epochs`;
- `batch_size`;
- `learning_rate`;
- danh sách regressors.

### 3.2. Bảng kết quả ngắn gọn

Nên tổng hợp:

- `val_mae`
- `val_rmse`
- `test_mae`
- `test_rmse`

cho từng model và từng ngân hàng.

### 3.3. Hình forecast minh họa

Ít nhất mỗi ngân hàng nên có:

- 1 hình `Actual vs Predicted`;
- 1 bảng forecast test mẫu nếu cần phụ lục.

## 4. Cách dùng kết quả hiện tại cho đúng

Bạn có thể dùng kết quả hiện tại để chứng minh rằng:

- đã chạy thật;
- có benchmark;
- có tập test;
- có file đầu ra để kiểm chứng.

Bạn không nên dùng kết quả hiện tại để khẳng định rằng:

- NeuralProphet là mô hình tối ưu nhất;
- NeuralProphet chắc chắn phù hợp nhất với dữ liệu này.

## 5. File nên mở cùng checklist này

- `NeuralProphet\neuralprophet_results.csv`
- `NeuralProphet\neuralprophet_results.md`
- `NeuralProphet\neuralprophet_outputs\`

## 6. Kết luận ngắn gọn

Nhánh NeuralProphet hiện tại đã đủ để:

- đưa vào báo cáo như benchmark/thực nghiệm mở rộng;
- viết phần mô tả pipeline và kết quả thực nghiệm trung thực.

Nhánh này chưa đủ để:

- tự động trở thành mô hình chính;
- bỏ vào đồ án mà không tuning, không thống nhất target và không giải thích vai trò của nó.
