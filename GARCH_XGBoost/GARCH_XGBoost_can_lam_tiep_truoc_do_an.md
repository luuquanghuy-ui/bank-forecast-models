# Những Việc Cần Làm Tiếp Trước Khi Đưa GARCH-XGBoost Vào Đồ Án

## 1. Kết quả hiện tại đã đủ tới đâu

Nhánh `GARCH-XGBoost` hiện tại đã có:

- walk-forward 4 folds;
- benchmark `Naive`, `GARCH-only`, `XGBoost`;
- mô hình trung tâm `GARCH-XGBoost main`;
- file forecast latest fold;
- tham số `GARCH`;
- feature importance của hybrid.

Tức là nhánh này đã vượt mức chạy thử đơn thuần và đã có đủ vật liệu để viết phần thực nghiệm.

## 2. Điều chưa được phép kết luận

Kết quả hiện tại chưa cho phép nói rằng:

- `GARCH-XGBoost` là mô hình tốt nhất trên dataset này.

Vì ở kết quả đang có:

- `GARCH-only` vẫn tốt hơn `GARCH-XGBoost main` trên cả 3 ngân hàng theo RMSE trung bình.

## 3. Việc cần làm tiếp về mặt học thuật

### 3.1. Chốt vai trò của hybrid trong bài

Bạn phải chọn một trong hai cách viết sạch nhất:

- `GARCH-XGBoost` là mô hình nghiên cứu trung tâm trong một bài comparative study;
- hoặc `GARCH-only` là mô hình mạnh nhất, còn `GARCH-XGBoost` là mô hình lai được kiểm định thực nghiệm.

### 3.2. Viết rõ lý do vẫn giữ `XGBoost` và `GARCH-only`

Ở nhánh này, benchmark không phải phần thừa. Nó là phần logic bắt buộc để trả lời hai câu hỏi:

- thêm ML vào `GARCH` có giúp gì không;
- thêm `GARCH` vào `XGBoost` có giúp gì không.

### 3.3. Giải thích đúng kết quả hiện tại

Cách viết phù hợp là:

- hybrid có cải thiện so với `XGBoost` thuần ở một số trường hợp;
- nhưng mức cải thiện đó chưa đủ để vượt `GARCH-only` trên bộ dữ liệu hiện tại.

## 4. Việc cần làm tiếp về mặt thực nghiệm

### 4.1. Bổ sung kiểm định so sánh dự báo

Nếu muốn phần này chặt hơn trước hội đồng, nên thêm:

- `Diebold-Mariano test`

để kiểm tra xem chênh lệch sai số giữa `GARCH-only` và `GARCH-XGBoost` có ý nghĩa thống kê hay không.

### 4.2. Kiểm tra độ ổn định của feature importance

Hiện tại đã có `feature importance`, nhưng nên đọc thêm:

- feature nào consistently quan trọng ở 3 ngân hàng;
- feature GARCH có thật sự xuất hiện trong top features không.

### 4.3. Xem latest-fold forecast bằng mắt

Mở các file trong:

- `GARCH_XGBoost\garch_xgboost_main_outputs\`

để xem:

- hybrid có bám được cụm biến động không;
- sigma của `GARCH` có đi cùng target không;
- hybrid thua `GARCH-only` vì lệch level hay vì không bắt được spike.

### 4.4. Nếu muốn cho hybrid thêm một cơ hội công bằng

Bạn có thể chạy thêm một vòng tuning có nguyên tắc cho:

- `max_depth`
- `learning_rate`
- `n_estimators`
- `subsample`
- `colsample_bytree`

Nhưng phải dùng validation/walk-forward, không nhìn test rồi chỉnh ngược lại.

## 5. Việc cần làm tiếp về mặt báo cáo

### 5.1. Bảng cấu hình mô hình

Nên có bảng riêng ghi:

- target;
- số folds;
- tham số `GARCH(1,1)`;
- các hyperparameters chính của `XGBoost`.

### 5.2. Bảng kết quả ngắn gọn

Nên tổng hợp riêng cho từng ngân hàng:

- `Naive`
- `GARCH-only`
- `XGBoost`
- `GARCH-XGBoost main`

với `MAE`, `RMSE`.

### 5.3. Hình minh họa bắt buộc nên đưa

Ít nhất nên có:

- 1 forecast plot latest fold cho mỗi ngân hàng;
- 1 sigma plot cho mỗi ngân hàng;
- 1 feature importance plot cho mỗi ngân hàng.

## 6. Cách dùng nhánh này cho đúng

Bạn có thể dùng ngay nhánh này để nói rằng:

- đã triển khai hybrid thật;
- đã so với benchmark logic bắt buộc;
- đã có output riêng của hybrid;
- kết quả cho thấy hybrid chưa vượt `GARCH-only` trên dataset hiện tại.

Bạn không nên dùng nhánh này để nói rằng:

- hybrid chắc chắn là mô hình tốt nhất;
- mọi mô hình ML đều vô dụng;
- chỉ cần có benchmark table là đủ cho đồ án.

## 7. File nên mở cùng checklist này

- `GARCH_XGBoost\garch_xgboost_main_results.csv`
- `GARCH_XGBoost\garch_xgboost_main_report.md`
- `GARCH_XGBoost\garch_xgboost_main_outputs\`

## 8. Kết luận ngắn gọn

Nhánh `GARCH-XGBoost` hiện tại đã đủ để:

- vào phần thực nghiệm chính của đồ án;
- làm mô hình nghiên cứu trung tâm theo hướng comparative study;
- trình bày rõ vai trò của hybrid và benchmark.

Nhánh này chưa đủ để:

- tự động được gọi là mô hình tốt nhất;
- bỏ thẳng vào đồ án mà không có phần thảo luận, giải thích benchmark và kiểm định bổ sung.
