# Những Việc Cần Làm Tiếp Trước Khi Đưa Temporal Fusion Transformer Vào Đồ Án

## 1. Không được coi kết quả hiện tại là bản đồ án cuối

Script đã chạy xong end-to-end, nhưng điều đó chỉ có nghĩa là:

- pipeline đã hoạt động;
- dữ liệu đã đi qua train, validation, test;
- bạn đã có benchmark, chỉ số và forecast.

Nó chưa có nghĩa là nhánh TFT đã đủ chuẩn để chèn thẳng vào đồ án mà không bổ sung gì thêm.

## 2. Việc phải làm tiếp về mặt học thuật

### 2.1. Chốt lại câu hỏi nghiên cứu

Bạn phải quyết định rõ TFT xuất hiện trong bài với vai trò nào:

- mô hình chính;
- benchmark deep-learning;
- hay mô hình mở rộng/phụ lục.

Với kết quả hiện tại, hướng an toàn nhất là:

- dùng TFT như benchmark deep-learning hoặc mô hình mở rộng,
- không nên gọi đây là mô hình tốt nhất.

### 2.2. Thống nhất target với toàn bộ đồ án

Hiện tại:

- nhánh GARCH/GARCH-XGBoost đang đi theo hướng dự báo biến động;
- nhánh NeuralProphet và TFT đang chạy trên `close`.

Nếu muốn so sánh tất cả mô hình trong cùng một bảng chính của đồ án, bạn phải:

- hoặc thống nhất toàn bộ về cùng một target;
- hoặc tách rõ hai nhóm bài toán khác nhau trong báo cáo.

### 2.3. Viết rõ lý do chọn TFT

Cần có một đoạn ngắn giải thích:

- vì sao TFT phù hợp với dữ liệu chuỗi thời gian có nhiều covariates;
- vì sao vẫn thử TFT dù benchmark đơn giản có thể mạnh hơn;
- vì sao việc kiểm định mô hình sâu vẫn có ý nghĩa trong bối cảnh dữ liệu tài chính.

## 3. Việc phải làm tiếp về mặt thực nghiệm

### 3.1. Tuning có nguyên tắc

Cấu hình hiện tại mới là cấu hình chạy được. Trước khi đưa vào đồ án, nên chạy thêm ít nhất một vòng tuning có kiểm soát cho các tham số:

- `max_encoder_length`
- `hidden_size`
- `attention_head_size`
- `dropout`
- `learning_rate`
- `max_epochs`

Lưu ý:

- tuning phải dựa trên validation;
- không được nhìn test rồi chỉnh ngược lại.

### 3.2. Kiểm tra độ ổn định qua nhiều seed hoặc nhiều split

Hiện tại kết quả là một lần chạy với seed cố định. Nếu muốn báo cáo chắc hơn, nên thêm:

- 2 đến 3 seed khác nhau;
- hoặc rolling / walk-forward evaluation.

### 3.3. Đọc forecast test bằng mắt

Không chỉ nhìn mỗi RMSE. Cần mở các file PNG và CSV trong:

- `TemporalFusionTransformer\tft_outputs\`

để xem:

- mô hình có bị lệch level không;
- có bám được turning points không;
- có bị trôi dự báo theo thời gian không.

### 3.4. So sánh với benchmark mạnh hơn nếu cần

Nếu thầy muốn nhánh deep-learning chặt hơn, có thể bổ sung một benchmark phù hợp hơn `Naive`, ví dụ:

- LSTM/GRU cơ bản;
- hoặc một mô hình forecast sâu khác đã chạy sạch sẽ.

Không bắt buộc, nhưng đây là hướng mở rộng hợp lý.

## 4. Việc phải làm tiếp về mặt báo cáo

### 4.1. Bổ sung bảng mô tả dữ liệu

Cần có:

- số quan sát từng ngân hàng;
- giai đoạn dữ liệu;
- thống kê mô tả của `close` và các covariates chính.

### 4.2. Bổ sung bảng cấu hình mô hình

Nên có một bảng riêng nêu rõ:

- encoder length;
- prediction length;
- learning rate;
- số epoch;
- batch size;
- hidden size;
- dropout;
- covariates sử dụng.

### 4.3. Viết phần diễn giải kết quả trung thực

Với kết quả hiện tại, cách viết phù hợp là:

- TFT đã được triển khai đầy đủ trên ba ngân hàng;
- tuy nhiên sai số test vẫn cao hơn benchmark `Naive`;
- do đó, trên bộ dữ liệu hiện tại, chưa có bằng chứng để kết luận TFT vượt trội.

### 4.4. Đính kèm hình và forecast mẫu

Trong phần phụ lục hoặc kết quả thực nghiệm, nên đưa:

- ít nhất 1 hình forecast cho mỗi ngân hàng;
- 1 bảng tóm tắt test RMSE;
- 1 đoạn mô tả model nào tốt nhất theo từng mã.

## 5. Cách dùng kết quả hiện tại cho đúng

Bạn có thể dùng ngay kết quả hiện tại để nói rằng:

- TFT đã được chạy end-to-end;
- pipeline đúng về mặt kỹ thuật;
- nhưng TFT chưa thắng benchmark đơn giản trên bộ dữ liệu này.

Điều không nên làm là:

- gọi kết quả hiện tại là bằng chứng TFT tốt nhất;
- dùng nhánh này làm mô hình chính mà không tuning hoặc không giải thích vai trò của nó;
- ghép thẳng bảng này với GARCH nếu target giữa hai nhánh chưa thống nhất.

## 6. File cần mở cùng với checklist này

- `TemporalFusionTransformer\tft_results.csv`
- `TemporalFusionTransformer\tft_results.md`
- `TemporalFusionTransformer\tft_outputs\`

## 7. Kết luận ngắn gọn

Nhánh TFT hiện tại đã đủ để:

- chứng minh bạn có triển khai mô hình thật;
- có benchmark;
- có kết quả thực nghiệm;
- có thể đưa vào báo cáo như benchmark deep-learning.

Nhánh này chưa đủ để:

- tự động trở thành mô hình chính;
- bỏ thẳng vào đồ án mà không bổ sung phần tuning, framing và diễn giải.
