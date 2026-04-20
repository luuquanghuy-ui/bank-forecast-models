# CHƯƠNG 3: BỘ DỮ LIỆU, BIẾN SỐ VÀ QUY TRÌNH TIỀN XỬ LÝ

Chương này trình bày chi tiết toàn bộ quy trình từ khâu thu thập nguồn dữ liệu thô đến khi xây dựng hoàn chỉnh bộ dữ liệu phục vụ cho các mô hình dự báo. Do đặc thù của bài toán dự báo tài chính, dữ liệu không chỉ bao gồm thông tin nội tại của cổ phiếu mà còn chứa các chỉ báo kinh tế vĩ mô. Nội dung chương này tương đương với quy trình thiết lập, tiền xử lý và phân tích khám phá dữ liệu (EDA) nhằm đảm bảo tính hợp lệ, tính dừng và độ tin cậy của đầu vào trước khi tiến hành mô hình hóa.

---

## 3.1. Nguồn Dữ Liệu Và Cấu Trúc Bộ Dữ Liệu

### 3.1.1. Tổng Quan Nguồn Dữ Liệu

Để đảm bảo phản ánh đầy đủ bối cảnh kinh tế vĩ mô Việt Nam cũng như hành vi riêng biệt của từng cổ phiếu ngân hàng, nghiên cứu đã tiến hành thu thập dữ liệu từ ba nguồn độc lập uy tín. Khoảng thời gian khảo sát kéo dài 10 năm, từ **04/02/2016 đến 27/02/2026**, tương đương khoảng **2,495 - 2,510 ngày giao dịch** cho mỗi mã cổ phiếu.

*   **Yahoo Finance:** Cung cấp thông tin giao dịch vi mô hàng ngày của 3 mã ngân hàng (Giá OHLCV). Dữ liệu tại đây đã được điều chỉnh (adjusted) cho các sự kiện chia tách cổ phiếu hoặc trả cổ tức, đảm bảo chuỗi giá phản ánh chính xác giá trị thực tế của doanh nghiệp theo thời gian.
*   **Investing.com:** Cung cấp dữ liệu vĩ mô về chỉ số thị trường chứng khoán (VNIndex, VN30), đại diện cho xu hướng chung của toàn thị trường chứng khoán Việt Nam.
*   **Ngân Hàng Nhà Nước Việt Nam (NHNN):** Cung cấp dữ liệu vĩ mô về chi phí vốn và tỷ giá, bao gồm Tỷ giá trung tâm USD/VND và Lãi suất liên ngân hàng qua đêm.

Giai đoạn 10 năm này bao trùm nhiều chu kỳ kinh tế và các sự kiện lớn như: đại dịch COVID-19, chu kỳ thắt chặt/nới lỏng tiền tệ của FED và những biến động mạnh của tỷ giá USD/VND. Sự đa dạng về điều kiện thị trường giúp mô hình dự báo học được các mẫu hình (patterns) thay đổi linh hoạt từ trạng thái bình ổn đến khủng hoảng.

### 3.1.2. Ba Mã Cổ Phiếu Ngân Hàng Đại Diện

Nghiên cứu tập trung vào ba ngân hàng thương mại niêm yết lớn nhất Việt Nam. Việc lựa chọn không mang tính ngẫu nhiên mà đại diện cho ba phân khúc khác biệt trên thị trường, giúp kiểm chứng năng lực tổng quát hóa của các mô hình dự báo:
*   **BID (Ngân hàng TMCP Đầu tư và Phát triển Việt Nam - BIDV):** Ngân hàng quốc doanh có quy mô tài sản lớn nhất. Cổ phiếu có mức giá tương đối thấp so với cùng kỳ, thanh khoản trung bình và chịu ảnh hưởng đáng kể từ dòng tiền của nhà đầu tư cá nhân.
*   **CTG (Ngân hàng TMCP Công thương Việt Nam - VietinBank):** Ngân hàng quốc doanh có mức thanh khoản cực kỳ cao. Cổ phiếu CTG rất nhạy cảm với các yếu tố vĩ mô và sự thay đổi chính sách tiền tệ.
*   **VCB (Ngân hàng TMCP Ngoại thương Việt Nam - Vietcombank):** Ngân hàng thương mại cổ phần tư nhân lớn nhất (tính theo vốn hóa). Đây là cổ phiếu "blue-chip" mang tính dẫn dắt thị trường, thường có xu hướng biến động sát với chỉ số VNIndex.

### 3.1.3. Cấu Trúc 3 Bộ Dữ Liệu Hiện Tại (Dataset Structure)

Thay vì gộp chung vào một file master khổng lồ, dữ liệu được tiền xử lý và tách lập thành ba bộ dữ liệu độc lập tương ứng với ba mã cổ phiếu: `banks_BID_dataset.csv`, `banks_CTG_dataset.csv`, và `banks_VCB_dataset.csv`. Việc tổ chức như vậy đáp ứng trực tiếp yêu cầu huấn luyện mô hình dự báo chuỗi thời gian riêng biệt cho từng mã, cung cấp cho các thuật toán một luồng thông tin biệt lập để tránh hiện tượng rò rỉ dữ liệu (data leakage).

Sau khi hoàn tất Feature Engineering (trích xuất biến nội tại và tạo các biến độ trễ), mỗi bộ dữ liệu có kích thước dao động khoảng 2,450 dòng (sau khi drop các dòng khuyết thiếu do trượt thời gian). Từ 17 cột gốc, quá trình Feature Engineering tạo ra tổng cộng 25 features có thể sử dụng (bao gồm các biến lag, ratio, và derived indicators). Tuy nhiên, **Lăng Kính 1 (XGBoost) chỉ sử dụng 15 features** (11 Technical và 4 Macro) — các features được chọn dựa trên domain knowledge và availability tại thời điểm dự báo. Bảng dưới thể hiện toàn bộ 25 features có thể tạo ra:

| STT | Tên Biến | Nhóm | Loại | Mô Tả |
|:---|:---|:---|:---|:---|
| 1 | `open` | Technical | Raw | Giá mở cửa ngày giao dịch |
| 2 | `high` | Technical | Raw | Giá cao nhất ngày |
| 3 | `low` | Technical | Raw | Giá thấp nhất ngày |
| 4 | `close` | Technical | Raw | Giá đóng cửa ngày |
| 5 | `volume` | Technical | Raw | Khối lượng giao dịch |
| 6 | `log_return` | Technical | Derived | Lợi suất log: ln(close_t / close_{t-1}) |
| 7 | `volatility_20d`| Technical | Derived | Độ biến động trung bình 20 ngày |
| 8 | `ma20` | Technical | Derived | Đường trung bình động 20 ngày |
| 9 | `ma50` | Technical | Derived | Đường trung bình động 50 ngày |
| 10 | `rsi` | Technical | Derived | Relative Strength Index (chỉ báo RSI 14 ngày) |
| 11 | `volume_ratio` | Technical | Derived | Tỷ số (volume hôm nay) / (trung bình volume 20 ngày) |
| 12 | `ma_ratio` | Technical | Derived | Tỷ số close / ma20 |
| 13 | `return_lag1` | Technical | Lag | log_return của ngày t-1 |
| 14 | `return_lag2` | Technical | Lag | log_return của ngày t-2 |
| 15 | `return_lag3` | Technical | Lag | log_return của ngày t-3 |
| 16 | `return_lag5` | Technical | Lag | log_return của ngày t-5 |
| 17 | `volatility_lag1`| Technical | Lag | volatility_20d của ngày t-1 |
| 18 | `volatility_lag2`| Technical | Lag | volatility_20d của ngày t-2 |
| 19 | `rsi_lag1` | Technical | Lag | RSI của ngày t-1 |
| 20 | `vnindex_close` | Macro | Raw | VNIndex đóng cửa ngày |
| 21 | `vn30_close` | Macro | Raw | VN30 đóng cửa ngày |
| 22 | `usd_vnd` | Macro | Raw | Tỷ giá trung tâm USD/VND |
| 23 | `interest_rate` | Macro | Raw | Lãi suất liên ngân hàng qua đêm (%) |
| 24 | `vnindex_lag1` | Macro | Lag | VNIndex của ngày t-1 |
| 25 | `vn30_lag1` | Macro | Lag | VN30 của ngày t-1 |

---

## 3.2. Tiền Xử Lý Dữ Liệu Vàng (Data Preprocessing)

Chất lượng của dữ liệu quyết định đến 80% độ chính xác của các mô hình Machine Learning. Bộ dữ liệu sau khi thu thập thô đã được đưa qua quy trình làm sạch và feature engineering chặt chẽ.

### 3.2.1. Xử Lý Giá Trị Thiếu (Missing Values) Từ Quá Trình Lag

Bản chất dữ liệu tài chính (cột 1 đến 5 và 20 đến 23) có độ toàn vẹn 100%. Tuy nhiên, khi hệ thống trích xuất **10 biến Feature Engineering liên quan đến chu kỳ (Lagged Features & MA)**, hàng loạt ô dữ liệu (NaN) xuất hiện tự nhiên tại các dòng đầu tiên. (Ví dụ: `ma50` cần 50 ngày đầu để lấy trung bình, do đó 49 ngày đầu sẽ bị NaN).

**Quyết định xử lý:** Thay vì nội suy (Interpolation) hay điền giá trị trước đó (Forward Fill) vốn tạo ra các khu vực "bước nhảy phẳng" làm sai lệch tín hiệu của thuật toán, nghiên cứu quyết định **xóa bỏ (Drop Rows)**. Khoảng 50 dòng đầu tiên bị loại bỏ chỉ chiếm ~2% tổng số dòng, hoàn toàn không gây ảnh hưởng đến cấu trúc phân phối của tập dữ liệu.

### 3.2.2. Xử Lý Ngày Nghỉ Lễ (Holiday Handling)

Đặc thù của thị trường chứng khoán Việt Nam là kỳ nghỉ lễ Âm lịch kéo dài (Nghỉ Tết từ 7 đến 10 ngày).
*   **Không sử dụng Forward-Fill:** Nếu điền giá ngày cận Tết cho toàn bộ những ngày nghỉ, tỷ suất sinh lời của những ngày đó sẽ bị ép bằng 0 (ln(1) = 0). Điều này sai lệch về mặt kinh tế vì giá trị doanh nghiệp tiếp tục thay đổi trong dịp lễ. Không có dữ liệu không có nghĩa là giá không đổi.
*   **Giải pháp thực hiện:** Giữ nguyên các phần khuyết (Gap) thời gian. Log_return được tính toán dựa trên mức giá của "ngày giao dịch thực tế hiện hành" so với "ngày giao dịch thực tế gần nhất trước đó" (Gap-aware calculation).

### 3.2.3. Xử Lý Giá Trị Ngoại Lai (Outliers)

Theo phân phối chuẩn, tỷ lệ Outlier lớn hơn 3 độ lệch chuẩn chỉ khoảng 0.27% (1 ngày trong 1 năm rưỡi). Tuy nhiên khi quét z-score, `log_return` có tới **3% - 5%** giá trị Outliers. Sự chênh lệch kỳ dị này minh chứng cho hiện tượng **đuôi mập (fat tails)** đặc thù của chứng khoán. Nghiên cứu **quyết định giữ nguyên vẹn 100% Outliers** vì đây không phải sai số kỹ thuật, mà là "tín hiệu vàng" sinh ra từ biến cố COVID-19, giúp các thuật toán (XGBoost, GARCH) học được cách phản ứng với những cú sốc đứt gãy thị trường.

---

## 3.3. Xây Dựng Và Giải Thích Chi Tiết Các Biến Số Đầu Vào

### 3.3.1. Nhóm Biến Phân Tích Kỹ Thuật (Technical Indicators - 19 biến)

Các biến Technical phản ánh "hành vi nội tại của giá" – nơi phản chiếu sự kỳ vọng, lòng tham và nỗi sợ hãi của nhà đầu tư.

**1. Nhóm Giá Thô (Raw Data):** Gồm bộ ngũ `open`, `high`, `low`, `close` và `volume`. Khối lượng (Volume) làm nhiệm vụ xác nhận xu hướng: một sự phá vỡ giá (breakout) kèm theo Volume tăng vọt có độ tin cậy vượt trội so với Volume nhỏ rò rỉ.

**2. Lợi suất sinh lời Logarit (`log_return`):**
Thay vì đo lường bằng lợi suất đơn giản P(t)/P(t-1) - 1, nghiên cứu sử dụng logarit tự nhiên:
$$log\_return_t = \ln\left(\frac{close_t}{close_{t-1}}\right)$$
*Ý nghĩa:* Log-return có tính chất cộng dồn liên tục, loại trừ lỗi tính ngược lợi suất lũy kế. Đồng thời xử lý được tính xu hướng phình to của giá, giúp biến số đạt chuẩn "tính dừng".

**3. Khẩu độ Biến động 20 ngày (`volatility_20d`):**
Tính bằng độ lệch chuẩn của chuỗi `log_return` trong cửa sổ 20 ngày giao dịch:
$$volatility\_20d_t = std(log\_return_{t-19:t})$$
*Ý nghĩa:* Đây là thước đo chuẩn mực nhất để đại diện cho Rủi ro (Risk). Biến số này được sử dụng làm biến mục tiêu (Target) trung tâm cho bài toán dự báo bằng GARCH và học máy.

**4. Đường Trung bình Động (`ma20` và `ma50`) & Tỷ lệ MA (`ma_ratio`):**
Tính bằng trung bình cộng giá bóng cửa. Biến `ma_ratio` = close / ma20.
*Ý nghĩa:* Ứng dụng quy luật mean-reversion (hồi quy về mức trung bình). Khi `ma_ratio` vọt lên quá cao so với 1, chứng tỏ cổ phiếu đang tăng quá dốc rời xa MA20 và có áp lực tự nhiên hút ngược nó trở về đường trung tâm.

**5. Chỉ số RSI và Tỷ lệ Volume (`rsi`, `volume_ratio`):**
*   **RSI:** So sánh tốc độ tăng/giảm giá trong 14 ngày. RSI > 70 cảnh báo "Quá mua" (căng cứng, dễ sập); RSI < 30 cảnh báo "Quá bán".
*   **Volume Ratio (= volume/chuẩn 20 ngày):** Định lượng trực tiếp những pha "Nổ Volume" > 1.5 lần mức bình quân bình thường, rà quét dòng tiền của đại gia xâm nhập.

**6. Nhóm Biến Trễ (Lagged Variables):**
Bao gồm `return_lag1`, `volatility_lag1`,... Việc tạo ra biến trễ ép mô hình trí tuệ nhân tạo phải nhìn vào thông tin quá khứ của "ngày hôm qua" (t-1) để tiên lượng "ngày hôm nay" (t), xóa bỏ vĩnh viễn hành vi look-ahead (học trộm tương lai). Việc lấy lag1 đến lag5 (1 tuần) giúp mô hình đo được quán tính rơi rớt giá trị.

### 3.3.2. Nhóm Biến Kinh Tế Vĩ Mô (Macro Indicators - 6 biến)

Cổ phiếu ngân hàng "nhạy cảm Vĩ mô" cực độ. Đưa nhóm này vào đã giúp mô hình học máy tăng phần lớn năng lực giải thích R-squared.

**1. Chỉ số thị trường (`vnindex_close`, `vn30_close` và các biến trễ):**
*Ý nghĩa:* Chuyển động của VNIndex chính là "Trọng lực" ảnh hưởng lên ngân hàng. Đặc biệt VN30 chứa quỹ ETF khối ngoại, sự tháo chạy hoặc mua gom của VN30 sẽ đánh trực tiếp vào hệ trục giá trị của BID, CTG, VCB.

**2. Tỷ giá `usd_vnd` (USD/VND Exchange Rate):**
*Ý nghĩa:* Đối với các đại ngân hàng nắm giữ tài sản ngoại hối khổng lồ và tham gia tín dụng xuất nhập khẩu, USD vọt tăng sẽ đúc kết lợi nhuận kinh doanh ngoại hối nhưng đè bẹp khả năng phòng thủ lạm phát dự trữ của dòng tiền, tác động thẳng vào bảng cân đối kế toán.

**3. Lãi suất liên ngân hàng (`interest_rate`):**
*Ý nghĩa:* Là định lượng sinh tử cho "Thanh khoản liên thông". Khi NHNN hút tiền ròng đưa Interest rate lên cao, lập tức chi phí vốn (Cost of Funds) nhảy vọt. Nợ xấu (NPL) phình to, kỳ vọng NIM sụt gãy, đánh gục giá niêm yết cổ phiếu trên sàn trong vòng vài phiên sau đó.

---

## 3.4. Kiểm Định Tính Dừng (Stationarity Test)

Trong phân tích dự báo Time-series, thuật toán không thể phân tích một chuỗi số liệu "bay bổng" trôi dạt vô định. Dữ liệu phải có tính "Hồi quy dừng" (Stationary) - tức là trung bình (mean) và phương sai không tự do phình to mãi mãi. Nghiên cứu sử dụng **Augmented Dickey-Fuller (ADF) Test**.
*   **$H_0$:** Chuỗi tồn tại nghiệm đơn vị (Unit root) $\implies$ Không dừng.
*   **$H_1$:** Chuỗi không có nghiệm đơn vị $\implies$ Dừng (Với mức ý nghĩa p-value < 0.05).

| Biến Kiểm Định | BID (p-value) | CTG (p-value) | VCB (p-value) | Kết Luận | Ý Nghĩa Chuyên Sâu |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Giá đóng cửa (`close`)** | 0.7978 | 0.9781 | 0.6949 | **Không Dừng** | Mức giá thô mang xu hướng tích luỹ hướng lên (Martingale process). Việc trực tiếp đưa `close` vào mô hình sẽ gây ra lỗi hồi quy giả mạo (Spurious correlation). |
| **Lợi suất Log (`log_return`)** | **0.0000** | **0.0000** | **0.0000** | **Rất Dừng (Stationary)** | Sự thay đổi gia tốc giá bằng logarit đã loại trừ hoàn toàn lạm phát giá. Điều kiện vàng để nhận dạng. |
| **Biến động (`volatility_20d`)** | **0.0005** | **0.0001** | **0.0001** | **Dừng (Stationary)** | Rủi ro có tính chất quay về mức trung bình chứ không vỡ vụn vô hạn (mean-reversion), phù hợp tuyệt đối cho GARCH. |

*Hệ quả triển khai mô hình:* Thay vì dự đoán vào mức giá tuyệt đối `close` tĩnh, thuật toán sẽ đi dự án hàm phái sinh `log_return` và `volatility_20d`. Sau khi chốt kết quả sinh lời ở tương lai, hệ thống sẽ thực hiện hàm nghịch đảo khôi phục lại mức giá cổ phiếu.

---

## 3.5. Phân Tích Thống Kê Mô Tả Nâng Cao (Advanced EDA)

Việc hiểu sâu hình dạng phân phối (Distribution shape) của bộ dữ liệu đập tan các lý thuyết sách vở truyền thống trong việc đánh giá rủi ro ngân hàng. Phân phối lợi suất đạt phân phối chuẩn khi Độ lệch (Skewness) = 0 và Độ nhọn dư (Excess Kurtosis) = 0. Song thực tế lại hoàn toàn khác.

| Đặc trưng rủi ro | BID | CTG | VCB | Giải thích ý nghĩa |
| :--- | :--- | :--- | :--- | :--- |
| **Skewness** (Độ lệch của return) | -0.121 | -0.147 | -0.008 | Skewness âm báo hiệu "đuôi trái cực dài". Khoảng thời gian rớt đáy hoảng loạn tàn khốc của thị trường chứng khoán Việt rơi gắt hơn tiến trình bò lên từ từ. |
| **Kurtosis** (Độ nhọn đuôi return) | 2.208 | 2.297 | **3.090** | Kurtosis khổng lồ > 0 khẳng định Đuôi mập (Leptokurtic). Xác suất khủng hoảng tài chính cao gấp hàng chục lần so với công thức normal distribution dự kiến. VCB mập nhất. |
| **Kurtosis của Volume** | **29.25** | 4.435 | **48.15** | Đột biến cực điểm. Khối lượng đôi lúc bị giật lên gấp 5-10 lần bình quân tạo ra thanh khoản khủng do tâm lý tháo chạy hoặc bắt đáy dồn tiền của dòng Bigboy. |

### Diện Mạo Rủi Ro (Risk DNA) Của Từng Mã Ngân Hàng

Nhờ vào bộ 15 features đã được chọn lọc (qua SHAP analysis) và Skewness / Kurtosis, nghiên cứu đã mã hóa được hệ gen (DNA) riêng biệt của 3 tổ chức:
1.  **BID – Ngân hàng "Momentum Bán lẻ":** 
    Ít biến cố về giá nhưng hệ số giật cục của Volume cực cao (Kurtosis Volume = 29.2). Thể hiện cổ phiếu bị chi phối bởi dòng tiền "bay nhảy Fomo" của nhà đầu tư cá nhân nhưng không đủ uy lực xé rách được trục của Market Maker.
2.  **CTG – Ngân hàng "Tổn thương Vĩ mô":**
    Mang chỉ số Skewness lệch âm chí mạng (-0.147). Khi có các tin tức lạm phát toàn cầu hoặc hạ điểm xếp hạng tín nhiệm, CTG có quán tính rơi tự do thảm khốc nhất nhóm.
3.  **VCB – Biểu tượng "Blue-chip Dẫn dắt":**
    Xác suất lên và xuống được kiểm soát cân xứng tuyệt đối (Skew -0.008). Nhưng nó mang mã gen rủi ro nhọn nhất (Kurtosis return = 3.09 và Volume = 48). Khi VCB bung thanh khoản, nó không phải cá nhân hành động, mà đó là việc các quỹ bự tháo cống luân chuyển tỷ USD tái định vị xu hướng chung của toàn sàn HOSE.

## TÓM TẮT CHƯƠNG 3
Bao trùm toàn bộ cấu trúc và sự tương đương với khối lượng 3 chương báo cáo, Chương 3 đã tái thiết lập một bộ dữ liệu hoàn chỉnh thông qua việc: Định nghĩa siêu chi tiết **17 cột gốc và 25 features có thể tạo ra sau Feature Engineering (trong đó LK1 chọn sử dụng 15 features: 11 Technical + 4 Macro)** trên 3 file Dataset được ly khai biệt lập; Trừng phạt nhiễu loạn Missing Values và tận dụng Outliers như tín hiệu cảnh báo Khủng hoảng; Khẳng định tính Dừng (Stationarity) của dữ liệu lợi suất; Và đào bới cấu trúc "Fat tails" bằng EDA nâng cao. Móng đã xây vô ngần vững chãi chờ các đại diện Model bùng nổ ở Chương 4 phía sau.
