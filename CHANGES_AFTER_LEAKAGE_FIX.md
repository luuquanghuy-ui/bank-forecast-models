# Thay Đổi Sau Khi Sửa Leakage

**Ngày:** 2026-04-20
**Mục đích:** Tài liệu chi tiết cho bạn bè tham khảo các thay đổi sau khi fix leakage

---

## 2 Lỗi Leakage Đã Sửa

### 1. NeuralProphet `validation_df` Leakage
- **Vấn đề:** NeuralProphet được train với `validation_df=val_np`, khiến model "thấy" validation data trong quá trình huấn luyện
- **Hậu quả:** Số liệu NP tốt hơn thực tế (leakage benefit)
- **Sửa:** Bỏ `validation_df` parameter, train trực tiếp trên training set

### 2. Hybrid Volatility Target Leakage
- **Vấn đề:** Volatility target dùng contemporaneous return (cùng ngày) thay vì lagged volatility
- **Hậu quả:** Hybrid volatility prediction có lợi thế không công bằng
- **Sửa:** Đổi sang dùng lagged volatility làm target

---

## KẾT QUẢ CHÍNH XÁC - 4-FOLD WALK-FORWARD

### BID

| Model | Vol (Cũ) | Vol (Mới) | Chênh | Price (Cũ) | Price (Mới) | Chênh |
|-------|----------|-----------|--------|------------|-------------|--------|
| Naive | 0.01508 | 0.01508 | = | 0.8482 | 0.8482 | = |
| XGBoost | 0.01261 | 0.01261 | = | 2.9808 | 2.9808 | = |
| NP | 0.01441 | 0.01068 | ↓ 25.9% | 2.6045 | 5.3061 | ↑ 103.7% |
| TFT | 0.01044 | 0.01048 | ↑ 0.4% | 0.6606 | 0.6586 | ↓ 0.3% |
| Hybrid | 0.01307 | 0.01290 | ↓ 1.3% | 0.5435 | 0.5435 | = |

**Best:** TFT vol, Hybrid price ✓

### CTG

| Model | Vol (Cũ) | Vol (Mới) | Chênh | Price (Cũ) | Price (Mới) | Chênh |
|-------|----------|-----------|--------|------------|-------------|--------|
| Naive | 0.01550 | 0.01550 | = | 0.5846 | 0.5846 | = |
| XGBoost | 0.01106 | 0.01106 | = | 3.5682 | 3.5682 | = |
| NP | 0.01469 | 0.01181 | ↓ 19.6% | 1.6483 | 6.1653 | ↑ 274% |
| TFT | 0.01023 | 0.01032 | ↑ 0.9% | 0.4760 | 0.4685 | ↓ 1.6% |
| Hybrid | 0.01213 | 0.01188 | ↓ 2.1% | 0.3823 | 0.3823 | = |

**Best:** TFT vol, Hybrid price ✓

### VCB

| Model | Vol (Cũ) | Vol (Mới) | Chênh | Price (Cũ) | Price (Mới) | Chênh |
|-------|----------|-----------|--------|------------|-------------|--------|
| Naive | 0.01137 | 0.01137 | = | 1.1039 | 1.1039 | = |
| XGBoost | 0.01038 | 0.01038 | = | 4.3386 | 4.3386 | = |
| NP | 0.01117 | 0.00891 | ↓ 20.2% | 3.3235 | 7.1814 | ↑ 116% |
| TFT | 0.00783 | 0.00785 | ↑ 0.3% | 0.7969 | 0.7784 | ↓ 2.3% |
| Hybrid | 0.01170 | 0.01186 | ↑ 1.4% | 0.6466 | 0.6466 | = |

**Best:** TFT vol, Hybrid price ✓

---

## PER-DAY RESULTS

### BID

| Model | Vol (Cũ) | Vol (Mới) | Chênh | Price (Cũ) | Price (Mới) | Chênh |
|-------|----------|-----------|--------|------------|-------------|--------|
| Naive | 0.01149 | 0.01149 | = | 0.4818 | 0.4818 | = |
| XGBoost | 0.01013 | 0.01013 | = | 4.7867 | 4.7867 | = |
| NP | 0.00945 | 0.01032 | ↑ 9.2% | 3.4051 | 3.8676 | ↑ 13.6% |
| TFT | 0.00765 | 0.00759 | ↓ 0.8% | 0.4610 | 0.4633 | ↑ 0.5% |
| Hybrid | 0.01082 | 0.01073 | ↓ 0.8% | 0.4376 | 0.4376 | = |

### CTG

| Model | Vol (Cũ) | Vol (Mới) | Chênh | Price (Cũ) | Price (Mới) | Chênh |
|-------|----------|-----------|--------|------------|-------------|--------|
| Naive | 0.01128 | 0.01128 | = | 0.3126 | 0.3126 | = |
| XGBoost | 0.00971 | 0.00971 | = | 3.2926 | 3.2926 | = |
| NP | 0.01014 | 0.00934 | ↓ 7.9% | 3.9670 | 6.8082 | ↑ 71.6% |
| TFT | 0.00780 | 0.00780 | = | 0.3193 | 0.3189 | ↓ 0.1% |
| Hybrid | 0.01068 | 0.01053 | ↓ 1.4% | 0.3104 | 0.3104 | = |

### VCB

| Model | Vol (Cũ) | Vol (Mới) | Chênh | Price (Cũ) | Price (Mới) | Chênh |
|-------|----------|-----------|--------|------------|-------------|--------|
| Naive | 0.00911 | 0.00911 | = | 0.6150 | 0.6150 | = |
| XGBoost | 0.00891 | 0.00891 | = | 7.8088 | 7.8088 | = |
| NP | 0.00861 | 0.01094 | ↑ 27.1% | 3.0469 | 6.8503 | ↑ 124.8% |
| TFT | 0.00624 | 0.00621 | ↓ 0.5% | 0.5772 | 0.5738 | ↓ 0.6% |
| Hybrid | 0.01017 | 0.01034 | ↑ 1.7% | 0.5501 | 0.5501 | = |

---

## TÓM TẮT THAY ĐỔI THEO MODEL

### Naive - KHÔNG ĐỔI
| Bank | Vol | Price |
|------|-----|-------|
| BID | = | = |
| CTG | = | = |
| VCB | = | = |

### XGBoost - KHÔNG ĐỔI
| Bank | Vol | Price |
|------|-----|-------|
| BID | = | = |
| CTG | = | = |
| VCB | = | = |

### NeuralProphet - THAY ĐỔI NHIỀU NHẤT
| Bank | Vol (cũ→mới) | Price (cũ→mới) |
|------|---------------|-----------------|
| BID | 0.01441→0.01068 (↓26%) | 2.60→5.31 (↑104%) |
| CTG | 0.01469→0.01181 (↓20%) | 1.65→6.17 (↑274%) |
| VCB | 0.01117→0.00891 (↓20%) | 3.32→7.18 (↑116%) |

### TFT - Thay đổi ÍT
| Bank | Vol (cũ→mới) | Price (cũ→mới) |
|------|---------------|-----------------|
| BID | 0.01044→0.01048 (+0.4%) | 0.6606→0.6586 (-0.3%) |
| CTG | 0.01023→0.01032 (+0.9%) | 0.4760→0.4685 (-1.6%) |
| VCB | 0.00783→0.00785 (+0.3%) | 0.7969→0.7784 (-2.3%) |

### Hybrid - Thay đổi ÍT
| Bank | Vol (cũ→mới) | Price |
|------|---------------|-------|
| BID | 0.01307→0.01290 (-1.3%) | = |
| CTG | 0.01213→0.01188 (-2.1%) | = |
| VCB | 0.01170→0.01186 (+1.4%) | = |

---

## KẾT LUẬN

1. **Kết luận thesis KHÔNG ĐỔI** - TFT thắng volatility, Hybrid thắng price
2. **Naive, XGBoost: Không thay đổi** - không bị leakage
3. **NP: Vol cải thiện nhưng Price tệ hơn rất nhiều** - do mất leakage từ validation_df
4. **TFT, Hybrid: Thay đổi rất ít** - chỉ vài %
5. **Số liệu bây giờ CÔNG BẰNG** - thesis nên dùng số mới
