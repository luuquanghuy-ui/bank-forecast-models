# Phần mới - Lăng kính giải mã mô hình

## Lăng kính 1: XGBoost (Giải mã nhân tố - Feature Importance)

**Ban đầu chọn đề:** Trả lời câu hỏi **Cái gì (What)** tác động mạnh nhất?

**Giải thích:** Nhóm dùng XGBoost kết hợp SHAP value để bóc tách: Biến vĩ mô (Lãi suất) hay biến kỹ thuật (RSI, MA) đang "cầm lái".

---

## Lăng kính 2: NeuralProphet (Giải mã chu kỳ - Seasonality)

**Ban đầu chọn đề:** Trả lời câu hỏi **Khi nào (When)** là thời điểm nhạy cảm?

**Giải thích:** Nhóm dùng nó để tách biệt Xu hướng dài hạn và Tính mùa vụ. Ví dụ: Tại sao cứ tháng 1 giá Bank lại tăng mạnh?

---

## Lăng kính 3: TFT (Giải mã sự chú ý - Attention Mechanism)

**Ban đầu chọn đề:** Trả lời câu hỏi **AI đang nhìn vào đâu (Attention)?**

**Giải thích:** Đây là mô hình hiện đại nhất, nhóm kỳ vọng nó sẽ chỉ ra "trí nhớ" của thị trường (ví dụ: thị trường nhìn về 60 ngày trước để phản ứng cho hôm nay).

---

## Tổng hợp 3 Lăng kính

| Lăng kính | Model | Câu hỏi | Ý nghĩa |
|------------|-------|---------|---------|
| 1 | XGBoost | What? | Biến nào tác động mạnh nhất |
| 2 | NeuralProphet | When? | Thời điểm nhạy cảm nào |
| 3 | TFT | Where? | AI nhìn vào đâu trong quá khứ |
