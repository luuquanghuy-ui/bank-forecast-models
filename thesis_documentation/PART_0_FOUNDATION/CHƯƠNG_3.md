# CHƯƠNG 3: BỘ DỮ LIỆU, BIẾN SỐ VÀ QUY TRÌNH TIỀN XỬ LÝ

Chương này trình bày chi tiết toàn bộ quy trình từ khâu thu thập nguồn dữ liệu thô đến khi xây dựng hoàn chỉnh bộ dữ liệu phục vụ cho các mô hình dự báo. Do đặc thù của bài toán dự báo tài chính, dữ liệu không chỉ bao gồm thông tin nội tại của cổ phiếu mà còn chứa các chỉ báo kinh tế vĩ mô. Nội dung chương này tương đương với quy trình thiết lập, tiền xử lý và phân tích khám phá dữ liệu (EDA) nhằm đảm bảo tính hợp lệ, tính dừng và độ tin cậy của đầu vào trước khi tiến hành mô hình hóa.

---

## 3.1. Nguồn Dữ Liệu Và Cấu Trúc Bộ Dữ Liệu

### 3.1.1. Tổng Quan Nguồn Dữ Liệu

Để đảm bảo phản ánh đầy đủ bối cảnh kinh tế vĩ mô Việt Nam cũng như hành vi riêng biệt của từng cổ phiếu ngân hàng, nghiên cứu đã tiến hành thu thập dữ liệu từ ba nguồn độc lập uy tín. Khoảng thời gian khảo sát kéo dài 10 năm, từ **04/02/2016 đến 27/02/2026**, tương đương khoảng **2,495 - 2,510 ngày giao dịch** cho mỗi mã cổ phiếu.

Nguồn dữ liệu thứ nhất là **Yahoo Finance**, cung cấp thông tin giao dịch vi mô hàng ngày của 3 mã ngân hàng bao gồm giá OHLCV (Open - giá mở cửa, High - giá cao nhất, Low - giá thấp nhất, Close - giá đóng cửa, Volume - khối lượng). Dữ liệu tại đây đã được điều chỉnh (adjusted) cho các sự kiện chia tách cổ phiếu hoặc trả cổ tức, đảm bảo chuỗi giá phản ánh chính xác giá trị thực tế của doanh nghiệp theo thời gian. Việc sử dụng dữ liệu đã điều chỉnh này đặc biệt quan trọng đối với nghiên cứu dài hạn như nghiên cứu này, bởi nếu không điều chỉnh, các sự kiện doanh nghiệp như chia tách cổ phiếu sẽ tạo ra các "bước nhảy" giả tạo trong chuỗi giá, làm sai lệch kết quả phân tích.

Nguồn thứ hai là **Investing.com**, cung cấp dữ liệu vĩ mô về chỉ số thị trường chứng khoán Việt Nam, cụ thể là VNIndex và VN30. VNIndex đại diện cho toàn bộ thị trường chứng khoán Việt Nam với hơn 300 cổ phiếu niêm yết, trong khi VN30 chỉ tập trung vào 30 công ty vốn hóa lớn nhất. VN30 thường được xem là proxy cho nhóm nhà đầu tư tổ chức (institutional investors) bởi các quỹ đầu tư lớn thường tập trung vào các blue-chips. Chỉ số này đóng vai trò là "trọng lực" ảnh hưởng lên giá cổ phiếu ngân hàng - khi VNIndex tăng hay giảm, các cổ phiếu ngân hàng thường có xu hướng đi cùng chiều do tâm lý đồng thuận thị trường.

Nguồn thứ ba là **Ngân Hàng Nhà Nước Việt Nam (NHNN)**, cung cấp dữ liệu vĩ mô về chi phí vốn và tỷ giá, bao gồm Tỷ giá trung tâm USD/VND và Lãi suất liên ngân hàng qua đêm. Đây là hai biến số vĩ mô có tầm ảnh hưởng quyết định đến hoạt động kinh doanh của các ngân hàng thương mại. Tỷ giá USD/VND ảnh hưởng trực tiếp đến giá trị tài sản ngoại hối và khả năng cạnh tranh trong hoạt động tín dụng xuất nhập khẩu, trong khi lãi suất liên ngân hàng phản ánh chi phí vay mượn qua đêm giữa các ngân hàng - một chỉ báo cho điều kiện thanh khoản và chi phí vốn trong toàn hệ thống tài chính.

Giai đoạn 10 năm này bao trùm nhiều chu kỳ kinh tế và các sự kiện lớn như: đại dịch COVID-19 (gây ra biến động chưa từng có trên toàn cầu vào tháng 3/2020), chu kỳ thắt chặt/nới lỏng tiền tệ của FED (đỉnh điểm là giai đoạn 2022-2024 với lạm phát cao nhất trong 40 năm), và những biến động mạnh của tỷ giá USD/VND trong giai đoạn 2022-2023 khi VND chịu áp lực giảm giá lớn. Sự đa dạng về điều kiện thị trường từ giai đoạn thịnh vượng đến khủng hoảng giúp mô hình dự báo học được các mẫu hình (patterns) thay đổi linh hoạt, từ trạng thái bình ổn đến các cú sốc cấp tính.

### 3.1.2. Ba Mã Cổ Phiếu Ngân Hàng Đại Diện

Nghiên cứu tập trung vào ba ngân hàng thương mại niêm yết lớn nhất Việt Nam. Việc lựa chọn không mang tính ngẫu nhiên mà đại diện cho ba phân khúc khác biệt trên thị trường, giúp kiểm chứng năng lực tổng quát hóa của các mô hình dự báo trên các loại cổ phiếu ngân hàng có đặc điểm kinh doanh và cơ cấu cổ đông khác nhau.

**BID (Ngân hàng TMCP Đầu tư và Phát triển Việt Nam - BIDV)** là ngân hàng quốc doanh có quy mô tài sản lớn nhất hệ thống ngân hàng Việt Nam. Cổ phiếu BID có mức giá tương đối thấp so với cùng kỳ (trung bình khoảng 24,000 VND), thanh khoản ở mức trung bình và chịu ảnh hưởng đáng kể từ dòng tiền của nhà đầu tư cá nhân. Đặc điểm này khiến BID thường xuyên trải qua các đợt "FOMO" (Fear Of Missing Out) - hiệu ứng tâm lý khi nhà đầu tư sợ bỏ lỡ cơ hội lợi nhuận và đổ xô mua vào một cách ồ ạt, tạo ra các đợt tăng giá bất thường. Khối lượng giao dịch của BID có kurtosis cực cao (29.257), phản ánh hiện tượng "bùng nổ" thanh khoản khi có thông tin tích cực hoặc tiêu cực đột ngột.

**CTG (Ngân hàng TMCP Công thương Việt Nam - VietinBank)** cũng là một ngân hàng quốc doanh với mức thanh khoản cực kỳ cao (trung bình 6.2 triệu cổ phiếu/ngày - cao nhất trong 3 mã). Cổ phiếu CTG rất nhạy cảm với các yếu tố vĩ mô, đặc biệt là sự thay đổi chính sách tiền tệ và biến động tỷ giá USD/VND. Phân tích skewness cho thấy CTG có downside skewness mạnh nhất (-0.147), nghĩa là khi thị trường giảm, CTG có xu hướng giảm mạnh hơn mức trung bình - đây là đặc điểm rủi ro quan trọng cho quản trị danh mục đầu tư.

**VCB (Ngân hàng TMCP Ngoại thương Việt Nam - Vietcombank)** là ngân hàng thương mại cổ phần tư nhân lớn nhất Việt Nam tính theo vốn hóa thị trường. Đây là cổ phiếu "blue-chip" mang tính dẫn dắt thị trường, thường có xu hướng biến động sát với chỉ số VNIndex và được nhiều quỹ đầu tư tổ chức theo dõi. Giá VCB cao nhất trong 3 mã (trung bình khoảng 39,000 VND), và đáng chú ý là VCB có fat tails mạnh nhất (excess kurtosis của |log_return| = 5.059) - nghĩa là những ngày biến động cực đoan xảy ra thường xuyên hơn so với hai mã còn lại. Điều này phản ánh đặc điểm của một blue-chip lớn: khi có tin tức tốt hoặc xấu, phản ứng của thị trường đối với VCB thường mạnh mẽ và nhanh chóng hơn.

Sự đa dạng trong lựa chọn ba mã từ ba phân khúc khác nhau (quốc doanh lớn nhất, quốc doanh thanh khoản cao, tư nhân blue-chip) cho phép đánh giá liệu mô hình dự báo có hoạt động tốt nhất quán trên các loại cổ phiếu ngân hàng khác nhau hay không - đây là câu hỏi có ý nghĩa thực tiễn quan trọng cho các nhà đầu tư và quản trị rủi ro.

### 3.1.3. Cấu Trúc Bộ Dữ Liệu

Thay vì gộp chung vào một file master khổng lồ, dữ liệu được tiền xử lý và tách lập thành ba bộ dữ liệu độc lập tương ứng với ba mã cổ phiếu: `banks_BID_dataset.csv`, `banks_CTG_dataset.csv`, và `banks_VCB_dataset.csv`. Việc tổ chức như vậy đáp ứng trực tiếp yêu cầu huấn luyện mô hình dự báo chuỗi thời gian riêng biệt cho từng mã, cung cấp cho các thuật toán một luồng thông tin biệt lập để tránh hiện tượng rò rỉ dữ liệu (data leakage) - một vấn đề nghiêm trọng trong machine learning có thể dẫn đến đánh giá quá lạc quan về năng lực dự báo thực tế của mô hình.

Sau khi hoàn tất Feature Engineering (trích xuất biến nội tại và tạo các biến độ trễ), mỗi bộ dữ liệu có kích thước dao động khoảng 2,450 dòng (sau khi drop các dòng khuyết thiếu do trượt thời gian). Từ 17 cột gốc, quá trình Feature Engineering tạo ra tổng cộng 25 features có thể sử dụng (bao gồm các biến lag, ratio, và derived indicators). Tuy nhiên, **Lăng Kính 1 (XGBoost) chỉ sử dụng 15 features** (11 Technical và 4 Macro) — các features được chọn dựa trên domain knowledge và availability tại thời điểm dự báo.

Bảng dưới thể hiện toàn bộ 25 features có thể tạo ra từ 17 cột gốc, được phân loại theo nhóm và loại:

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

## 3.2. Tiền Xử Lý Dữ Liệu

Chất lượng của dữ liệu quyết định đến 80% độ chính xác của các mô hình Machine Learning. Bộ dữ liệu sau khi thu thập thô đã được đưa qua quy trình làm sạch và feature engineering chặt chẽ để đảm bảo tính toàn vẹn và phù hợp cho việc huấn luyện mô hình.

### 3.2.1. Xử Lý Giá Trị Thiếu

Bản chất dữ liệu tài chính (cột 1 đến 5 và 20 đến 23 trong bảng trên) có độ toàn vẹn 100% - không có bất kỳ giá trị NaN nào trong dữ liệu thô thu thập từ Yahoo Finance và các nguồn chính thức khác. Tuy nhiên, khi hệ thống trích xuất **10 biến Feature Engineering liên quan đến chu kỳ (Lagged Features & Moving Averages)**, hàng loạt ô dữ liệu (NaN) xuất hiện tự nhiên tại các dòng đầu tiên do không đủ dữ liệu lịch sử để tính toán. Ví dụ, `ma50` cần 50 ngày đầu để lấy trung bình, do đó 49 ngày đầu tiên sẽ có giá trị NaN; tương tự, `volatility_20d` cần 20 ngày nên 19 dòng đầu tiên sẽ bị thiếu.

Nghiên cứu đã cân nhắc bốn chiến lược xử lý NaN từ lag operations. Phương pháp **Drop Rows** (xóa các dòng có NaN) được lựa chọn vì số dòng bị mất chỉ là 20-50 dòng đầu tiên - nhỏ hơn 2% tổng số dòng, hoàn toàn không gây ảnh hưởng đáng kể đến cấu trúc phân phối của tập dữ liệu. Các phương pháp thay thế như Forward Fill (điền giá trị gần nhất phía trước) hay Interpolation (nội suy) đều có nhược điểm là tạo ra các "bước nhảy" giả tạo trong dữ liệu, đặc biệt nghiêm trọng đối với các biến như volatility_20d where việc nội suy giá trị volatility cho ngày đầu tiên là hoàn toàn vô nghĩa về mặt kinh tế.

### 3.2.2. Xử Lý Ngày Nghỉ Lễ

Đặc thù của thị trường chứng khoán Việt Nam là kỳ nghỉ lễ Âm lịch kéo dài (Nghỉ Tết từ 7 đến 10 ngày liên tục). Đây là hiện tượng đặc thù của thị trường Việt Nam, khác với thị trường phương Tây chỉ nghỉ 1-2 ngày. Khoảng trống lớn nhất trong dữ liệu là **10 ngày liên tục** tương ứng với kỳ nghỉ Tết Nguyên Đán hàng năm.

Nghiên cứu **không sử dụng Forward-Fill** cho các ngày nghỉ lễ vì hai lý do nghiêm trọng. Thứ nhất, nếu điền giá ngày cận Tết cho toàn bộ những ngày nghỉ, tỷ suất sinh lời của những ngày đó sẽ bị ép bằng 0 (vì ln(P/P) = 0 khi P không đổi). Điều này sai lệch nghiêm trọng về mặt kinh tế vì giá trị doanh nghiệp tiếp tục thay đổi trong dịp lễ - chúng ta đơn giản là không quan sát được sự thay đổi đó, không có nghĩa là nó không xảy ra. Thứ hai, trong phân tích chuỗi thời gian tài chính, missing data (không có dữ liệu) không đồng nghĩa với "unchanged" (không thay đổi).

Giải pháp thực hiện là **giữ nguyên các khoảng Gap thời gian** và áp dụng công thức gap-aware cho log_return: $$log\_return_t = \ln\left(\frac{close_t}{close_{t-k}}\right)$$ trong đó k là số ngày giao dịch thực tế kể từ lần giao dịch trước. Các khoảng gap lớn hơn 3 ngày được đánh dấu trong dataset để có thể lọc ra khi cần phân tích riêng.

### 3.2.3. Xử Lý Giá Trị Ngoại Lai

Theo lý thuyết phân phối chuẩn, tỷ lệ Outlier lớn hơn 3 độ lệch chuẩn chỉ khoảng 0.27% (tương đương 1 ngày trong 1 năm rưỡi). Tuy nhiên khi quét z-score trên thực tế, `log_return` có tới **3% - 5%** giá trị Outliers - cao gấp 10-20 lần so với kỳ vọng từ phân phối chuẩn. Sự chênh lệch kỳ dị này minh chứng cho hiện tượng **đuôi mập (fat tails)** - đặc điểm phổ biến của dữ liệu tài chính được Mandelbrot (1963) phát hiện và nghiên cứu chuyên sâu.

Nghiên cứu **quyết định giữ nguyên vẹn 100% Outliers** vì đây không phải sai số kỹ thuật hay nhiễu cần loại bỏ, mà là "tín hiệu vàng" chứa đựng thông tin về các biến cố thị trường cực đoan như COVID-19 crash tháng 3/2020, các đợt phục hồi mạnh mẽ sau dịch, hay biến động tỷ giá USD/VND. Việc loại bỏ outliers sẽ khiến mô hình mất đi khả năng học cách phản ứng với những cú sốc thị trường - một kỹ năng then chốt trong quản trị rủi ro tài chính. Đặc biệt, mô hình GARCH phụ thuộc rất nhiều vào các giá trị cực đoan để ước tính volatility - loại bỏ outliers sẽ khiến GARCH đánh giá thấp rủi ro thực tế.

---

## 3.3. Xây Dựng Và Giải Thích Chi Tiết Các Biến Số Đầu Vào

### 3.3.1. Nhóm Biến Phân Tích Kỹ Thuật (Technical Indicators)

Các biến Technical phản ánh "hành vi nội tại của giá" - nơi phản chiếu sự kỳ vọng, lòng tham và nỗi sợ hãi của nhà đầu tư thông qua các mẫu hình giá và khối lượng giao dịch.

**Nhóm Giá Thô (Raw Data - 5 biến):** Bộ ngũ `open`, `high`, `low`, `close` tạo thành mô hình nến Nhật (candlestick) - công cụ phân tích kỹ thuật đã được sử dụng hơn một thế kỷ để nhận diện các mẫu hình giá. Trong đó, `close` (giá đóng cửa) là quan trọng nhất vì nó đại diện cho mức giá cuối cùng mà thị trường chấp nhận trong ngày giao dịch - điểm giá mà tất cả các phân tích và dự báo thường dựa vào. Khối lượng giao dịch (`volume`) đo lường cường độ tham gia của thị trường - một sự phá vỡ giá (breakout) kèm theo Volume tăng vọt có độ tin cậy vượt trội so với breakout với khối lượng nhỏ rò rỉ.

**Lợi suất sinh lời Logarit (`log_return`):** Thay vì đo lường bằng lợi suất đơn giản (P_t/P_{t-1} - 1), nghiên cứu sử dụng logarit tự nhiên vì hai lý do quan trọng. Thứ nhất, log-return có tính chất cộng dồn liên tục: log_return_t + log_return_{t+1} = ln(P_{t+1}/P_{t-1}), trong khi lợi suất đơn giản thì không có tính chất này. Thứ hai, log-return xử lý được tính xu hướng phình to (inflation effect) của giá theo thời gian, giúp biến số đạt chuẩn "tính dừng" (stationarity) - điều kiện tiên quyết cho hầu hết các mô hình chuỗi thời gian. Công thức: $$log\_return_t = \ln\left(\frac{close_t}{close_{t-1}}\right)$$

**Độ biến động 20 ngày (`volatility_20d`):** Tính bằng độ lệch chuẩn của chuỗi `log_return` trong cửa sổ 20 ngày giao dịch: $$volatility\_20d_t = std(log\_return_{t-19:t})$$ Đây là thước đo chuẩn mực nhất để đại diện cho Rủi ro (Risk) trong tài chính lượng, và cũng là biến mục tiêu (Target) trung tâm cho bài toán dự báo biến động bằng GARCH và các mô hình học máy.

**Đường Trung bình Động (`ma20`, `ma50`) và Tỷ lệ MA (`ma_ratio`):** MA20 phản ánh xu hướng ngắn hạn (khoảng một tháng giao dịch), trong khi MA50 phản ánh xu hướng trung hạn (khoảng một quý). Tỷ số `ma_ratio` = close / ma20 ứng dụng quy luật mean-reversion (hồi quy về mức trung bình): khi `ma_ratio` vọt lên quá cao so với 1, chứng tỏ cổ phiếu đang tăng quá dốc rời xa MA20 và có áp lực tự nhiên hút ngược trở về đường trung tâm.

**Chỉ số RSI và Tỷ lệ Volume (`rsi`, `volume_ratio`):** RSI (Relative Strength Index) với chu kỳ 14 ngày dao động từ 0 đến 100, đo lường tốc độ và sự thay đổi của giá. RSI > 70 cảnh báo "Quá mua" (overbought) - thị trường đang ở trạng thái căng cứng, dễ sập; RSI < 30 cảnh báo "Quá bán" (oversold) - cổ phiếu có thể đang bị định giá thấp hơn giá trị thực. Volume Ratio (= volume / trung bình volume 20 ngày) định lượng trực tiếp những pha "Nổ Volume" với giá trị > 1.5 lần mức bình quân - đây thường là dấu hiệu của dòng tiền lớn "Bigboy" xâm nhập hoặc tháo chạy.

**Nhóm Biến Trễ (Lagged Variables):** Bao gồm `return_lag1`, `return_lag2`, `return_lag3`, `return_lag5`, `volatility_lag1`, `volatility_lag2`, `rsi_lag1`. Việc tạo biến trễ là thiết kế bắt buộc trong dữ liệu chuỗi thời gian - nó ép mô hình phải nhìn vào thông tin của "ngày hôm qua" (t-1) để tiên lượng "ngày hôm nay" (t), xóa bỏ vĩnh viễn hành vi look-ahead (nhìn trộm tương lai). Việc lấy lag từ 1 đến 5 ngày (tương đương một tuần giao dịch) giúp mô hình đo được quán tính rơi rớt giá trị.

### 3.3.2. Nhóm Biến Kinh Tế Vĩ Mô (Macro Indicators)

Cổ phiếu ngân hàng "nhạy cảm Vĩ mô" ở mức cực độ. VNIndex và VN30 đã được chuẩn hóa về thang điểm 0.5-2.1 thay vì giá trị thực (ví dụ: 855-1576 điểm) để các mô hình hội tụ nhanh hơn và không bị ảnh hưởng bởi quy mô tuyệt đối khác nhau giữa các chỉ số.

**Chỉ số thị trường (`vnindex_close`, `vn30_close` và các biến trễ):** VNIndex đại diện cho "trọng lực" ảnh hưởng lên tất cả cổ phiếu ngân hàng. Đặc biệt, VN30 chứa các quỹ ETF khối ngoại với hành vi xác định rõ: sự tháo chạy hoặc mua gom của nhóm VN30 sẽ đánh trực tiếp vào hệ trục giá trị của BID, CTG, VCB. Các biến trễ `vnindex_lag1` và `vn30_lag1` được tạo để tránh hiện tượng đồng thời (contemporaneous correlation) có thể gây ra spurious correlation trong mô hình.

**Tỷ giá `usd_vnd`:** Đối với các đại ngân hàng nắm giữ tài sản ngoại hối khổng lồ và tham gia tín dụng xuất nhập khẩu, biến động USD/VND có tác động hai mặt: USD vọt tăng sẽ đúc kết lợi nhuận kinh doanh ngoại hối nhưng đồng thời đè bẹp khả năng phòng thủ lạm phát và tác động trực tiếp vào bảng cân đối kế toán của các ngân hàng có tỷ trọng cho vay ngoại tệ cao.

**Lãi suất liên ngân hàng (`interest_rate`):** Đây là định lượng sinh tử cho "Thanh khoản liên thông" của hệ thống. Khi NHNN hút tiền ròng đưa Interest rate lên cao, lập tức chi phí vốn (Cost of Funds) nhảy vọt, nợ xấu (NPL) phình to, kỳ vọng NIM sụt gãy - đánh gục giá niêm yết cổ phiếu trên sàn trong vòng vài phiên. Trong giai đoạn nghiên cứu (2016-2026), lãi suất chỉ có **7 giá trị duy nhất** (4.0%, 4.5%, 5.0%, 5.5%, 6.0%, 6.25%, 6.5%) - gần như một hằng số theo từng đoạn (piecewise constant), cho thấy NHNN điều chỉnh theo đợt, không phải liên tục.

---

## 3.4. Trực Quan Hóa Dữ Liệu Và Phân Tích Thống Kê Mô Tả

### 3.4.1. Biểu Đồ Giá Đóng Cửa (Close Price)

Biểu đồ đường (line chart) thể hiện diễn biến giá đóng cửa của ba mã ngân hàng trong giai đoạn 2016-2026 cho thấy rõ nét ba giai đoạn kinh tế hoàn toàn khác nhau.

Trong **giai đoạn 2016-2019** (trước COVID), cả ba mã đều cho thấy xu hướng tăng ổn định. BID tăng từ mức giá thấp nhất quanh 13-14,000 VND lên khoảng 30,000-35,000 VND. CTG giao động trong khoảng 16,000-22,000 VND với biên độ hẹp hơn. VCB dẫn đầu với mức tăng ấn tượng từ 30,000 VND lên đến vùng 80,000-90,000 VND trước dịch - phản ánh đà tăng trưởng của blue-chip hàng đầu thu hút dòng vốn ngoại. Giai đoạn này đặc trưng bởi thanh khoản dồi dào toàn cầu và tăng trưởng tín dụng mạnh mẽ của hệ thống ngân hàng Việt Nam.

**Giai đoạn COVID (tháng 3/2020)** tạo ra cú sốc ngắn nhưng cực kỳ mạnh. VNIndex giảm hơn 30% chỉ trong vài tuần - thị trường chứng khoán Việt Nam chứng kiến hiệu ứng "Fear Of Missing Out" đảo chiều thành hiệu ứng "Panic Selling" khi nhà đầu tư đồng loạt bán tháo để giữ tiền mặt. BID và VCB cũng không tránh khỏi xu hướng này, với mức giảm tương ứng khoảng 25-30% từ đỉnh. CTG với đặc tính "nhạy cảm vĩ mô" của mình có xu hướng phản ứng mạnh hơn trước các tin xấu.

**Giai đoạn 2020-2022 (phục hồi và bong bóng)** chứng kiến sự phục hồi vượt bậc sau COVID. Dòng tiền từ lãi suất thấp toàn cầu đổ mạnh vào chứng khoán Việt Nam, đẩy VNIndex lên đỉnh mọi thời đại. VCB dẫn đầu với mức giá vượt 100,000 VND vào đầu 2022 - một mức giá chưa từng có trong lịch sử cổ phiếu ngân hàng Việt Nam. BID cũng hưởng lợi với mức giá vượt 50,000 VND. Đây là giai đoạn "FOMO" điển hình - nhà đầu tư sợ bỏ lỡ cơ hội và đổ xô mua vào bất chấp valuation đã trở nên quá đắt đỏ.

**Giai đoạn 2022-2024 (thắt chặt tiền tệ)** chứng kiến sự điều chỉnh mạnh khi FED tăng lãi suất mạnh nhất trong 40 năm để chống lạm phát. Dòng tiền ngoại rút khỏi các thị trường mới nổi, đè nặng lên VNIndex. Tỷ giá USD/VND tăng mạnh từ mức 23,000 lên gần 26,000 VND/USD, tạo áp lực lên các ngân hàng có tỷ trọng cho vay ngoại tệ cao. CTG với đặc tính "macro-sensitive" của mình chứng kiến mức giá giảm về vùng 15,000-18,000 VND.

### 3.4.2. Biểu Đồ Khối Lượng Giao Dịch (Volume)

Biểu đồ cột (bar chart) thể hiện khối lượng giao dịch hàng ngày cho thấy đặc điểm thanh khoản hoàn toàn khác nhau giữa ba mã.

**CTG có thanh khoản cao nhất** với trung bình 6.2 triệu cổ phiếu/ngày - cao hơn gấp đôi so với BID (2.4 triệu) và VCB (1.82 triệu). Điều này phản ánh đặc điểm cổ phiếu quốc doanh được nhiều nhà đầu tư giao dịch, đặc biệt trong các giai đoạn biến động. Thanh khoản cao của CTG khiến nó trở thành "proxy" cho xu hướng của nhóm cổ phiếu ngân hàng - khi CTG tăng hay giảm, các mã khác thường có xu hướng đi cùng chiều.

**BID và VCB có thanh khoản thấp hơn nhưng "extremeness" cao hơn nhiều.** Kurtosis của volume BID = 29.257 và VCB = 48.158 - những con số "khủng khiếp" về mặt thống kê. Điều này có nghĩa là: mặc dù trung bình thanh khoản chỉ ở mức trung bình, nhưng có những ngày "bùng nổ" với khối lượng gấp 10-15 lần bình thường. Những ngày này thường tương ứng với các sự kiện tin tức lớn:

- **Ngày FOMO:** Khi thị trường đang lên, nhà đầu tư cá nhân đổ xô mua vào, đẩy khối lượng tăng vọt. Hiện tượng này đặc biệt rõ ở BID với tỷ lệ nhà đầu tư cá nhân cao trong cơ cấu cổ đông.

- **Ngày Panic Selling:** Khi có tin xấu đột ngột (ví dụ: thông tin về nợ xấu ngành ngân hàng, hoặc khủng hoảng tài chính toàn cầu), nhà đầu tư đồng loạt bán tháo, tạo ra "thanh khoản khủng hoảng" dù giá đang giảm mạnh.

- **Ngày rebalancing quỹ:** Các quỹ đầu tư lớn thường tái cân bằng danh mục vào cuối quý hoặc cuối năm, tạo ra các đợt volume cao bất thường có thể dự đoán được.

### 3.4.3. Biểu Đồ Phân Phối Lợi Suất (Histogram)

![Histogram Log Return vs Normal](thesis_charts/ch3_histogram_log_return.png)

*Hình 3.1: Phân phối log_return hàng ngày của 3 mã ngân hàng (đường đỏ = phân phối chuẩn lý thuyết). Đuôi mập (fat tails) thể hiện ở các cột cao hơn đường chuẩn ở hai đầu — đây là bằng chứng trực quan cho thấy outliers xảy ra thường xuyên hơn phân phối chuẩn dự đoán.*

Biểu đồ histogram so sánh phân phối log_return thực tế với phân phối chuẩn lý thuyết cho thấy ba đặc điểm quan trọng.

**Thứ nhất, đỉnh phân phối (peak) nhọn hơn phân phối chuẩn.** Điều này phản ánh hiện tượng "leptokurtic" - tức là phần lớn các ngày (khoảng 95-97%) có log_return dao động trong một vùng hẹp quanh 0, với độ lệch chuẩn "bình thường". Các ngày này thể hiện trạng thái "bình thường" của thị trường - không có tin tức đặc biệt, nhà đầu tư hành động theo thói quen, và biến động giá chỉ mang tính "noise" ngẫu nhiên.

**Thứ hai, hai đuôi (tails) mập hơn phân phối chuẩn.** Đây là phát hiện quan trọng nhất: các cột ở hai đầu (tương ứng với log_return < -3% hoặc > +3%) cao hơn đường cong chuẩn đáng kể. Điều này nghĩa là: xác suất xảy ra một ngày với biến động cực đoan (ví dụ: |log_return| > 5%) cao hơn nhiều so với dự đoán của lý thuyết tài chính cổ điển. Theo phân phối chuẩn, xác suất này chỉ khoảng 0.27% (1 ngày trong hơn 1 năm), nhưng thực tế trong dữ liệu, chúng ta thấy khoảng 3-5% (tức 10-20 lần cao hơn).

**Thứ ba, phân phối hơi lệch sang trái (left-skewed).** Cả ba mã đều có skewness âm nhẹ (-0.121 đến -0.008), nghĩa là đuôi bên trái (downside extreme) dài hơn đuôi bên phải một chút. Về mặt tâm lý thị trường, điều này phản ánh hiện tượng "crashes are faster than rallies" - khi thị trường giảm, tốc độ giảm thường nhanh và dữ dội hơn so với tốc độ tăng tương ứng. Đây là đặc điểm tâm lý phổ biến của thị trường tài chính toàn cầu: nỗi sợ hãi (fear) thường mạnh hơn lòng tham (greed).

![Boxplot |Log Return|](thesis_charts/ch3_boxplot_abs_return.png)

*Hình 3.2: Boxplot so sánh |log_return| giữa 3 mã ngân hàng. VCB có biến động trung bình thấp nhất (0.0118) nhưng có outliers nhiều nhất (excess kurtosis = 5.059). Orange dots = outliers.*

Boxplot so sánh |log_return| (biến động tuyệt đối) giữa ba mã cho thấy VCB có biến động trung bình thấp nhất (median và mean đều thấp hơn) nhưng lại có outliers nhiều và "extreme" nhất. Đây là profile của một blue-chip "ổn định nhưng khi bùng nổ thì rất mạnh" - phù hợp với đặc điểm cổ phiếu được nhiều quỹ tổ chức theo dõi, khi có tin tức thì phản ứng nhanh và mạnh hơn các mã khác.

---

## 3.5. Kiểm Định Tính Dừng (Stationarity Test)

### 3.5.1. Lý Thuyết Kiểm Định ADF

Trong phân tích dự báo chuỗi thời gian, thuật toán không thể phân tích một chuỗi số liệu "bay bổng" trôi dạt vô định. Dữ liệu phải có tính "Hồi quy dừng" (Stationary) - tức là trung bình (mean) và phương sai (variance) không tự do phình to theo thời gian. Nghiên cứu sử dụng **Augmented Dickey-Fuller (ADF) Test** - kiểm định thống kê phổ biến nhất để kiểm tra tính dừng.

ADF test kiểm định hai giả thuyết: **H₀ (Null Hypothesis)** rằng chuỗi tồn tại nghiệm đơn vị (Unit root) có nghĩa là chuỗi không dừng, và **H₁ (Alternative Hypothesis)** rằng chuỗi không có nghiệm đơn vị và do đó là dừng. Nếu p-value < 0.05, chúng ta bác bỏ H₀ và kết luận chuỗi dừng.

Tính dừng quan trọng vì ba lý do. Thứ nhất, nếu chuỗi không dừng (ví dụ: giá cổ phiếu có xu hướng tăng theo thời gian), thì "mean" không có ý nghĩa vì nó thay đổi liên tục - không thể dùng mean để đại diện cho chuỗi. Thứ hai, các mô hình ARIMA/GARCH giả định rằng phần dư (residuals) có tính dừng; nếu dữ liệu gốc không dừng, các ước lượng tham số sẽ không đáng tin cậy (spurious regression). Thứ ba, hai chuỗi không dừng có thể "có vẻ" tương quan cao chỉ vì cùng tăng theo thời gian - đây là correlation giả (spurious correlation) không có ý nghĩa kinh tế.

### 3.5.2. Kết Quả ADF Test Chi Tiết

**ADF Test Cho Close Price (Giá Đóng Cửa)**

| Mã | ADF Statistic | p-value | Lags Used | Critical Value (5%) | Kết Luận |
|----|--------------|---------|-----------|-------------------|----------|
| BID | -0.8696 | 0.7978 | 0 | -2.862 | **Không Dừng** |
| CTG | 0.3163 | 0.9781 | 0 | -2.862 | **Không Dừng** |
| VCB | -1.1495 | 0.6949 | 3 | -2.862 | **Không Dừng** |

Kết quả này hoàn toàn phù hợp với lý thuyết tài chính. Giá cổ phiếu trong dài hạn có xu hướng tăng theo thời gian (martingale process): E[Sₜ₊₁ | Iₜ] = Sₜ, nghĩa là giá hôm nay là best predictor cho giá ngày mai nhưng không có nghĩa là giá đứng yên - nó vẫn có thể tăng hoặc giảm ngẫu nhiên xung quanh một xu hướng tăng dài hạn. Không có lý do kinh tế nào để giá "quay về" một mức trung bình cố định vì giá phản ánh kỳ vọng của thị trường về triển vọng tương lai của doanh nghiệp - và triển vọng này thay đổi theo thời gian.

**ADF Test Cho Log_Return (Lợi Suất Log)**

| Mã | ADF Statistic | p-value | Lags Used | Critical Value (5%) | Kết Luận |
|----|--------------|---------|-----------|-------------------|----------|
| BID | -50.0254 | 0.0000 | 0 | -2.862 | **Dừng** ✓ |
| CTG | -34.7624 | 0.0000 | 1 | -2.862 | **Dừng** ✓ |
| VCB | -17.4754 | 0.0000 | 8 | -2.862 | **Dừng** ✓ |

Tất cả p-values = 0.0000 (nhỏ hơn rất nhiều so với 0.05), và ADF statistics cực kỳ âm (từ -17 đến -50), cho thấy chuỗi log_return dừng mạnh. Lý do rất đơn giản: log_return = ln(Pₜ/Pₜ₋₁) loại bỏ được xu hướng (detrending) vì nó đo lường tỷ lệ thay đổi, không phải mức giá tuyệt đối. Phần lớn các ngày, log_return dao động quanh 0 - có ngày tăng, có ngày giảm, nhưng trung bình ≈ 0, và quan trọng nhất, không có xu hướng tăng hay giảm theo thời gian.

**ADF Test Cho Volatility_20d (Độ Biến Động 20 Ngày)**

| Mã | ADF Statistic | p-value | Lags Used | Critical Value (5%) | Kết Luận |
|----|--------------|---------|-----------|-------------------|----------|
| BID | -4.2897 | 0.0005 | 23 | -2.862 | **Dừng** ✓ |
| CTG | -4.7207 | 0.0001 | 24 | -2.862 | **Dừng** ✓ |
| VCB | -4.7386 | 0.0001 | 25 | -2.862 | **Dừng** ✓ |

Tất cả p-values < 0.001 (rất significant), và ADF statistics đều âm hơn Critical Value 5%. Chuỗi volatility_20d dừng. Mặc dù volatility có tính clustering (biến động mạnh có xu hướng kéo dài), nhưng nó không tăng mãi mãi mà có **mean-reversion yếu** - nó có thể cao hoặc thấp tạm thời, nhưng cuối cùng sẽ quay về mức trung bình dài hạn. Đây là điều kiện phù hợp tuyệt đối cho GARCH model vì GARCH giả định conditional variance là stationary.

### 3.5.3. Hệ Quả Cho Việc Xây Dựng Mô Hình

Từ kết quả ADF, chúng tôi rút ra các hệ quả quan trọng cho việc lựa chọn và thiết kế mô hình.

**Không nên dùng giá close trực tiếp cho ARIMA/GARCH** vì close không dừng, các ước lượng tham số sẽ không đáng tin cậy (spurious). Thay vào đó, nên dùng log_return cho dự báo lợi suất.

**Log_return là dừng, phù hợp cho tất cả các mô hình** - ARIMA, GARCH, Neural Networks, và các học máy khác đều hoạt động tốt hơn với dữ liệu dừng vì các giả định thống kê của chúng được thỏa mãn.

**Volatility dừng → GARCH model có cơ sở lý thuyết vững.** GARCH giả định conditional variance là stationary, và ADF test xác nhận giả định này hợp lệ cho dữ liệu của chúng ta. Điều này có nghĩa là ước lượng volatility từ GARCH sẽ đáng tin cậy và có ý nghĩa kinh tế.

**Hai chiến lược dự báo cho hai target khác nhau:** Dự báo **giá** cần mô hình có khả năng xử lý non-stationarity - NeuralProphet (với trend component), TFT (với attention mechanism). Dự báo **biến động** có thể dùng trực tiếp log_return hoặc |log_return| với GARCH.

---

## 3.6. Phân Tích Thống Kê Mô Tả Nâng Cao

### 3.6.1. Thống Kê Mô Tả Giá Đóng Cửa

Bảng tổng hợp các đặc trưng thống kê của giá đóng cửa cho 3 mã ngân hàng trong giai đoạn 2016-2026:

| Thống Kê | BID | CTG | VCB |
|----------|-----|-----|-----|
| **Mean (VND)** | 24.76 | 16.27 | 39.38 |
| **Std (VND)** | 10.54 | 7.69 | 16.46 |
| **Min (VND)** | 7.51 | 6.20 | 11.96 |
| **25%** | 17.57 | 9.62 | 25.01 |
| **50% (Median)** | 24.04 | 15.84 | 41.02 |
| **75%** | 33.22 | 20.55 | 56.25 |
| **Max (VND)** | 55.00 | 41.50 | 76.00 |
| **Skewness** | 0.115 | 0.887 | -0.064 |
| **Kurtosis (Excess)** | -0.828 | 0.276 | -1.218 |

**BID** có giá trung bình ~24,760 VND với độ lệch chuẩn 10,540 VND cho thấy mức độ biến động tương đối lớn (biên độ ~45% quanh mean). Phân phối gần đối xứng (skewness = 0.115 gần 0). Giá trị kurtosis âm (-0.828) cho thấy phân phối giá "flatter" (phẳng hơn) so với phân phối chuẩn, nghĩa là ít có outliers cực đoan ở giá - điều này phù hợp với đặc điểm cổ phiếu bị chi phối bởi nhà đầu tư cá nhân có xu hướng giao dịch trong một vùng giá nhất định.

**CTG** có giá thấp nhất trong 3 mã (mean ~16,270 VND). Skewness dương mạnh (0.887) cho thấy có các giai đoạn giá tăng bất thường tạo đuôi phải dài - có thể là các đợt "pump" giá trong một số giai đoạn thị trường nóng. Đặc điểm này phản ánh tâm lý nhà đầu tư khi CTG có tin tức tích cực đột ngột.

**VCB** có giá cao nhất (mean ~39,380 VND) và là blue-chip có giá trị vốn hóa lớn nhất. Kurtosis âm (-1.218) cho thấy phân phối phẳng hơn phân phối chuẩn, ít outliers cực đoan ở mức giá - phản ánh sự ổn định tương đối của cổ phiếu blue-chip được nhiều quỹ nắm giữ dài hạn.

### 3.6.2. Thống Kê Mô Tả Log Return

| Thống Kê | BID | CTG | VCB |
|----------|-----|-----|-----|
| **Mean** | 0.000702 | 0.000648 | 0.000676 |
| **Std** | 0.021703 | 0.021148 | 0.017159 |
| **Min** | -0.072428 | -0.072807 | -0.072546 |
| **25%** | -0.009094 | -0.009867 | -0.007844 |
| **50% (Median)** | 0.000000 | 0.000000 | 0.000000 |
| **75%** | 0.011238 | 0.010483 | 0.008395 |
| **Max** | 0.068039 | 0.068053 | 0.067329 |
| **Skewness** | -0.121 | -0.146 | -0.008 |
| **Kurtosis (Excess)** | 2.208 | 2.297 | 3.090 |

**Mean dương nhưng rất nhỏ** (0.065-0.070% mỗi ngày): Thị trường ngân hàng Việt Nam có xu hướng tăng dài hạn nhưng với tốc độ khiêm tốn - phù hợp với mức tăng trưởng GDP danh nghĩa của Việt Nam trong giai đoạn này. Không có "miếng bánh ngon" nào cho nhà đầu tư ngắn hạn.

**Độ lệch chuẩn tương đương nhau** (1.7-2.2%): Cả 3 mã có mức biến động tương đương nhau, phù hợp với đặc điểm cùng ngành ngân hàng và cùng chịu ảnh hưởng của các yếu tố vĩ mô chung.

**Skewness âm nhẹ ở cả 3 mã**: Downside tail dài hơn upside - những ngày giảm mạnh tuy ít xảy ra hơn nhưng cường độ mạnh hơn những ngày tăng mạnh tương ứng. Đây là đặc điểm chung của dữ liệu tài chính: "crashes are faster than rallies." Tâm lý thị trường phản ánh ở đây là khi nhà đầu tư hoảng sợ, họ có xu hướng bán tháo ngay lập tức, trong khi khi lạc quan, họ lại do dự và mua từ từ.

**Kurtosis cao (2.2-3.1)**: Bằng chứng mạnh cho hiện tượng "fat tails" - outliers xảy ra thường xuyên hơn phân phối chuẩn dự đoán. Điều này có ý nghĩa cực kỳ quan trọng cho quản trị rủi ro: bất kỳ mô hình nào giả định phân phối chuẩn để ước tính rủi ro (như Value-at-Risk truyền thống) sẽ **systematically under-estimate** khả năng xảy ra sự kiện cực đoan.

### 3.6.3. Thống Kê Mô Tả Khối Lượng Giao Dịch

| Thống Kê | BID | CTG | VCB |
|----------|-----|-----|-----|
| **Mean** | 2.40 | 6.20 | 1.82 |
| **Std** | 2.39 | 5.24 | 2.40 |
| **Min** | 0.16 | 0.10 | 0.09 |
| **25%** | 1.12 | 2.53 | 0.79 |
| **50% (Median)** | 1.79 | 4.98 | 1.21 |
| **75%** | 2.82 | 8.31 | 1.90 |
| **Max** | 28.54 | 38.92 | 32.48 |
| **Skewness** | 4.353 | 1.777 | 5.848 |
| **Kurtosis (Excess)** | 29.257 | 4.435 | 48.158 |

**CTG có thanh khoản cao nhất** (mean ~6.2 triệu cp/ngày): Phù hợp với đặc điểm cổ phiếu được nhiều nhà đầu tư giao dịch, đặc biệt trong các giai đoạn biến động.

**BID và VCB thanh khoản thấp hơn** (1.8-2.4 triệu cp/ngày) nhưng **Skewness cực cao** (1.8-5.8): Phân phối volume rất lệch phải - trung bình bị kéo lên bởi các ngày volume cực cao bất thường. Điều này phản ánh hiện tượng "bùng nổ thanh khoản" đặc trưng của thị trường Việt Nam.

**Kurtosis extreme (4.4-48.2)**: Có những ngày "bùng nổ" thanh khoản với volume gấp nhiều lần bình thường, thường tương ứng với các sự kiện tin tức lớn hoặc hiệu ứng FOMO/Panic Selling. Con số 48.158 của VCB có nghĩa là có những ngày thanh khoản tăng vọt gấp hàng chục lần so với bình thường - một con số "kinh hoàng" về mặt thống kê nhưng hoàn toàn có thực trên thị trường Việt Nam.

### 3.6.4. Thống Kê Mô Tả Biến Vĩ Mô

| Thống Kê | VNIndex | VN30 | USD/VND | Interest Rate |
|----------|---------|------|---------|---------------|
| **Mean** | 1.08 | 1.09 | 23,546 | 5.23 |
| **Std** | 0.28 | 0.34 | 1,215 | 0.99 |
| **Min** | 0.54 | 0.56 | 21,699 | 4.00 |
| **25%** | 0.91 | 0.86 | 22,750 | 4.00 |
| **50% (Median)** | 1.05 | 1.05 | 23,203 | 5.50 |
| **75%** | 1.26 | 1.31 | 24,258 | 6.00 |
| **Max** | 1.90 | 2.10 | 26,425 | 6.50 |
| **Skewness** | 0.358 | 0.674 | 0.924 | 0.072 |
| **Kurtosis (Excess)** | -0.100 | 0.150 | -0.144 | -1.761 |

**VNIndex và VN30** đã được chuẩn hóa (normalized) trong dataset - giá trị trong khoảng 0.54-2.10, không phải giá trị thực của chỉ số (855-1576 điểm). Cả hai đều có skewness dương, cho thấy thị trường chứng khoán Việt Nam có xu hướng tăng nhanh và giảm chậm - đặc điểm của thị trường mới nổi với nhà đầu tư cá nhân chi phối.

**USD/VND** có skewness dương (0.924): Tỷ giá có xu hướng tăng (VND mất giá) mạnh hơn so với giảm, phản ánh áp lực lạm phát và nhập siêu đặc trưng của nền kinh tế Việt Nam. Giai đoạn 2022-2023 với áp lực USD mạnh toàn cầu đã đẩy USD/VND lên mức cao nhất trong nhiều năm.

**Interest Rate** chỉ có 7 giá trị duy nhất (4.0, 4.5, 5.0, 5.5, 6.0, 6.25, 6.5), cho thấy Ngân hàng Nhà nước điều chỉnh lãi suất theo đợt, không liên tục. Đây là biến gần như hằng số theo từng đoạn (piecewise constant), phản ánh cơ chế điều hành chính sách tiền tệ của Việt Nam.

---

## 3.7. Phân Tích ADN Rủi Ro Của Từng Mã Ngân Hàng

Kết hợp Skewness và Kurtosis, chúng ta có thể mô tả "ADN rủi ro" - tức là bản chất rủi ro đặc trưng của từng mã ngân hàng, giúp nhà đầu tư và quản trị rủi ro hiểu rõ hơn về profile của từng cổ phiếu.

### 3.7.1. BID — "Ngân hàng Retail Momentum"

| Đặc Điểm | Giá Trị | Ý Nghĩa |
|---------|---------|---------|
| Log_return Skewness | -0.121 (nhẹ âm) | Downside tail dài hơn upside |
| Log_return Kurtosis | 2.208 (cao) | Fat tails, nhiều outliers |
| Volume Kurtosis | 29.257 (rất cao) | Extreme spikes trong thanh khoản |
| Close Kurtosis | -0.828 (platykurtic) | Ít outliers ở giá |

**ADN Rủi Ro BID** mang đặc điểm của cổ phiếu bị chi phối bởi nhà đầu tư cá nhân (retail-dominated): thanh khoản có thể "bùng nổ" bất ngờ (volume spikes), nhưng giá cổ phiếu ít khi có những outlier values cực đoan. Các nhà đầu tư cá nhân thường giao dịch theo hiệu ứng FOMO - tạo ra những đợt tăng/giảm mạnh bất thường trong volume nhưng không nhất thiết tạo ra outliers trong giá. Họ mua đuổi khi thị trường lên, bán tháo khi thị trường xuống - tạo ra hiệu ứng "momentum" đặc trưng này.

### 3.7.2. CTG — "Ngân hàng Macro-Sensitive"

| Đặc Điểm | Giá Trị | Ý Nghĩa |
|---------|---------|---------|
| Log_return Skewness | -0.147 (âm nhất) | Downside tail dài nhất trong 3 mã |
| Log_return Kurtosis | 2.297 (cao) | Fat tails |
| Volume Kurtosis | 4.435 (thấp nhất) | Volume tương đối ổn định |
| Close Kurtosis | 0.276 (gần normal) | Phân phối giá gần chuẩn |

**ADN Rủi Ro CTG** có downside skewness âm mạnh nhất (-0.147) - khi thị trường giảm, CTG có xu hướng giảm mạnh hơn mức trung bình. Đây là đặc điểm rủi ro quan trọng cho quản trị danh mục: CTG không chỉ biến động nhiều (volatility cao) mà còn có xu hướng "rơi tự do" nhanh hơn trong các giai đoạn xấu. Điều này phản ánh sự nhạy cảm của CTG với các yếu tố vĩ mô - khi có tin xấu về kinh tế vĩ mô hay chính sách tiền tệ, CTG thường phản ứng mạnh và nhanh hơn các mã khác.

Volume tương đối ổn định (kurtosis thấp nhất = 4.435), cho thấy CTG ít bị ảnh hưởng bởi hiện tượng "retail stampede" - các đợt mua/bán hối hả của nhà đầu tư cá nhân. Điều này có thể là do CTG là cổ phiếu quốc doanh với tỷ lệ nhà đầu tư tổ chức cao hơn, nên hành vi của họ ổn định hơn.

### 3.7.3. VCB — "Blue-chip Dẫn dắt Thị trường"

| Đặc Điểm | Giá Trị | Ý Nghĩa |
|---------|---------|---------|
| Log_return Skewness | -0.008 (gần 0) | Gần như đối xứng hoàn hảo |
| Log_return Kurtosis | 3.090 (cao nhất) | Fat tails mạnh nhất |
| Volume Kurtosis | 48.158 (cao nhất) | Extreme volume spikes |
| Close Kurtosis | -1.218 (platykurtic) | Ít outliers ở giá |

**ADN Rủi Ro VCB** có log_return gần đối xứng nhất (skewness ≈ 0) - upside và downside equally likely, không biased về một hướng. Nhưng điểm đáng chú ý là fat tails mạnh nhất (kurtosis = 3.090), nghĩa là: mặc dù xác suất tăng và giảm như nhau, nhưng **cường độ của các movement cực đoan là lớn nhất** trong 3 mã. Volume kurtosis cực cao (48.158) cho thấy VCB thu hút sự chú ý thị trường cực đoan - khi VCB có tin tức, thanh khoản biến động mạnh hơn các mã khác vì nhiều quỹ đang theo dõi và hành động đồng thời.

Đây là profile của một cổ phiếu "volatile but symmetric" - biến động mạnh nhưng theo cả hai hướng. Khi thị trường tốt, VCB tăng mạnh; khi thị trường xấu, VCB cũng giảm mạnh. Đây là blue-chip có tính "beta" cao nhất - tức là biến động của VCB thường lớn hơn biến động của thị trường chung.

---

## 3.8. Tóm Tắt Chương 3

Chương 3 đã trình bày chi tiết toàn bộ quy trình từ thu thập dữ liệu thô đến khi xây dựng bộ dữ liệu hoàn chỉnh phục vụ cho các mô hình dự báo.

**Về Nguồn Dữ Liệu:** Nghiên cứu thu thập dữ liệu từ ba nguồn độc lập và uy tín - Yahoo Finance cho dữ liệu OHLCV, Investing.com cho chỉ số thị trường VNIndex và VN30, và Ngân hàng Nhà nước Việt Nam cho tỷ giá USD/VND và lãi suất liên ngân hàng. Khoảng thời gian 10 năm (2016-2026) bao trùm nhiều chu kỳ kinh tế từ thịnh vượng đến khủng hoảng, đảm bảo mô hình được huấn luyện và đánh giá trong điều kiện thị trường đa dạng.

**Về Cấu Trúc Dữ Liệu:** Ba bộ dữ liệu độc lập cho BID, CTG, VCB với 17 cột gốc và 25 features sau feature engineering (trong đó Lăng Kính 1 chọn sử dụng 15 features: 11 Technical + 4 Macro). Việc tách biệt theo từng mã giúp tránh hiện tượng data leakage giữa các mô hình.

**Về Tiền Xử Lý:** Dataset gốc không có NaN values. Ngày nghỉ Tết (gap 10 ngày) được xử lý bằng cách giữ nguyên gap thay vì forward-fill để tránh tạo ra log_return = 0 giả tạo. Outliers (3-5% dữ liệu) được giữ lại vì chúng mang thông tin kinh tế thật về các sự kiện thị trường cực đoan.

**Về Kiểm Định ADF:** Giá close không dừng (phù hợp với lý thuyết martingale), trong khi log_return và volatility_20d đều dừng ở cả 3 mã (p-value < 0.001). Điều này đảm bảo cơ sở lý thuyết vững cho GARCH và các mô hình chuỗi thời gian khác.

**Về Phân Tích EDA:** Tất cả các mã đều có |log_return| với fat tails (excess kurtosis = 3.0-5.1), nghĩa là outliers xảy ra thường xuyên hơn đáng kể so với dự đoán của phân phối chuẩn. CTG có downside skewness mạnh nhất (-0.147), trong khi VCB có fat tails mạnh nhất (excess kurtosis = 5.059). Mỗi mã có "ADN rủi ro" riêng: BID = retail momentum với hiệu ứng FOMO, CTG = macro-sensitive với phản ứng mạnh khi thị trường xấu, VCB = blue-chip dẫn dắt với biến động cực đoan nhưng đối xứng.

Những phát hiện từ chương này tạo nền tảng cho việc lựa chọn và thiết kế các mô hình dự báo trong các chương tiếp theo. Đặc biệt, fat tails của dữ liệu đòi hỏi các mô hình có khả năng xử lý volatility clustering và extreme movements - GARCH và TFT được kỳ vọng sẽ phù hợp hơn so với các mô hình tuyến tính truyền thống.

---

## Tài Liệu Liên Quan

- [0.1 Data Description](0.1_data_description.md) — Mô tả tổng quan dataset
- [0.2 Problem Formulation](0.2_problem_formulation.md) — Hai mục tiêu dự báo: Price vs Volatility
- [0.3 Fair Comparison Framework](0.3_fair_comparison_framework.md) — Phương pháp so sánh công bằng

## Nguồn Tham Khảo

1. Dickey, D.A. & Fuller, W.A. (1979). "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association*, 74(366), 427-431.
2. Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica*, 50(4), 987-1007.
3. Mandelbrot, B. (1963). "The Variation of Certain Speculative Prices." *The Journal of Business*, 36(4), 394-419.
4. Ngân hàng Nhà nước Việt Nam. "Tỷ giá USD/VND và lãi suất liên ngân hàng." sbv.gov.vn.
5. Yahoo Finance. Historical price data for BID, CTG, VCB.
