# BÁO CÁO THỰC NGHIỆM CHUYỂN GIAO TRI THỨC & MA TRẬN 7 CẤU HÌNH ĐỐI CHỨNG (TUẦN 02)
## Đề tài: Hệ Thống Xác Thực Chữ Ký Ngân Hàng Tự Động (AIO2026 — Module 05)
**Học viên thực hiện:** Trần Phương Bình  
**Chủ đề trọng tâm:** Reusing Pretrained Models  
**Bộ dữ liệu:** CEDAR Signature Dataset (55 chủ tài khoản, 30,360 cặp đối chiếu chuẩn Writer-Independent)  

---

### 1. Tiếp Thu Ý Kiến Chuyên Môn & Đính Chính Phân Tích Tuần 1

#### 1.1. Phản hồi từ Trợ giảng về Hiện tượng Mật độ Nét Mực
Ở báo cáo sơ bộ Tuần 1, số liệu thống kê thô cho thấy chữ ký giả mạo có tỷ lệ điểm ảnh mực trên toàn khung scan đạt **5.28%** so với **4.30%** của chữ ký thật (chênh lệch bề ngoài +23%). Nhận định ban đầu suy đoán rằng "người giả mạo thường ký chậm và tì mạnh ngòi bút".

#### 1.2. Tính toán & Kiểm chứng Khoa học
Qua quá trình tái thẩm định độc lập bằng thuật toán Bounding Box (cắt sát viền nét mực):
- **Diện tích khung quét gốc:** Ảnh scan thô của chữ ký giả bị thu hẹp đáng kể về chiều cao ($617.1 \times 296.2\text{ px} \approx 182.8\text{k px}$) so với chữ ký thật ($601.4 \times 365.5\text{ px} \approx 219.8\text{k px}$). Mẫu số của chữ ký giả nhỏ hơn ~17% đã tạo ra hiện tượng **ảo giác mật độ quang học** (Canvas Size Illusion).
- **Số lượng điểm ảnh mực tuyệt đối (Absolute Ink Pixels):**
  - Chữ ký thật: trung bình **9,450 – 10,076 px**.
  - Chữ ký giả: trung bình **9,650 – 10,226 px**.
  - Độ lệch thực tế chỉ **~1.5% đến 2.0%** (hoàn toàn tương đương về mặt vật lý).
- **Mật độ mực trong Bounding Box sát nét chữ:** Sau khi cắt bỏ toàn bộ lề giấy thừa, mật độ nét bút của chữ ký thật là $6.86\%$, chữ ký giả là $6.99\%$ (chênh lệch chỉ **+1.87%**).

#### 1.3. Ý nghĩa Học thuật Rút Ra
Đặc trưng phân biệt chữ ký giả mạo tinh vi (*Skilled Forgery*) **không nằm ở tổng lượng mực hay lực tì ngòi bút**, mà nằm ở **cấu trúc hình học không gian, độ cong nét bút, sự liền mạch và nhịp điệu vận động cơ bắp**. Điều này khẳng định sự cần thiết phải tận dụng năng lực trích xuất đặc trưng không gian sâu của các mạng nơ-ron tích chập và Attention chuyển giao tri thức (**Pretrained CNN & Vision Transformer**).

---

### 2. Tổng Quan Kết Quả Hoàn Thành Trong Tuần 2

| Hạng Mục | Nội Dung Kỹ Thuật Đã Thực Hiện | Tệp Minh Chứng | Trạng Thái |
|:---|:---|:---|:---:|
| **Bộ nạp dữ liệu** | Khởi tạo DataLoader phân tầng chuẩn Writer-Independent cho 20,976 cặp Train, 4,416 cặp Val, 4,968 cặp Test. | `src/dataset.py` | **ĐẠT 100%** |
| **Ma trận 7 mô hình** | Lập trình và đóng gói 7 cấu hình đối chứng từ MLP Baseline, Custom CNN đến ResNet-50 và ViT-Base. | `src/models.py` | **ĐẠT 100%** |
| **Kỹ thuật chuyển giao** | Cài đặt kỹ thuật đóng băng tầng (Freezing) và tốc độ học phân tầng (Discriminative LR). | `notebooks/02_pretrained_transfer_learning.ipynb` | **ĐẠT 100%** |
| **Quy trình huấn luyện** | Xây dựng công cụ huấn luyện `SignatureTrainer` với kiểm soát dừng sớm (Early Stopping) và Focal Loss. | `src/trainer.py` | **ĐẠT 100%** |
| **Hàm mất mát chuyên dụng** | Cài đặt BCEWithLogits, Focal Loss ($\gamma = 2.0$) và Label Smoothing BCE. | `src/losses.py` | **ĐẠT 100%** |

---

### 3. Ma Trận Thiết Kế & Thống Kê 7 Cấu Hình Thực Nghiệm Đối Chứng

Để giải quyết triệt để hiện tượng lệch miền dữ liệu (**Domain Shift**) khi chuyển từ miền ảnh màu tự nhiên (ImageNet-1k) sang miền ảnh chữ ký đơn sắc nhị phân ghép 3 kênh, nhóm nghiên cứu thiết lập ma trận 7 cấu hình thực nghiệm:

| STT | Cấu Hình Thử Nghiệm | Tổng Tham Số | Tham Số Huấn Luyện | Tỷ Lệ Trainable | Chiến Lược Tốc Độ Học & Hàm Loss | Ý Nghĩa Thực Nghiệm |
|:---:|:---|:---:|:---:|:---:|:---|:---|
| **1** | **MLP Baseline** (Duỗi phẳng 2D) | 100,664,833 | 100,664,833 | 100.00% | Cố định $10^{-3}$ \| BCE Loss | Chứng minh duỗi phẳng ảnh làm mất cấu trúc không gian 2D, bùng nổ tham số |
| **2** | **Custom 2D-CNN** (4 Blocks) | 3,842,433 | 3,842,433 | 100.00% | Cosine Annealing $10^{-3}$ \| BCE Loss | Đánh giá giới hạn của CNN tự huấn luyện từ đầu khi thiếu dữ liệu lớn |
| **3** | **ResNet-50 Feature Extraction** | 23,510,081 | 2,049 | 0.01% | Khóa 100% Backbone \| Train Head $10^{-3}$ | Đo lường tác động của Extreme Domain Shift khi giữ nguyên bộ lọc ImageNet |
| **4** | **ResNet-50 Partial Freezing** | 23,510,081 | 21,348,353 | 90.81% | Khóa Stage 1–2, Mở Stage 3–4 \| $10^{-4}$ | Giữ bộ lọc cạnh tầng nông, tinh chỉnh tầng sâu thích nghi hình thái chữ ký |
| **5** | **ResNet-50 Discriminative FT** | 23,510,081 | 23,510,081 | 100.00% | Phân tầng 3 nhóm: $10^{-5} \to 5\cdot 10^{-5} \to 10^{-3}$ \| Focal Loss ($\gamma=2.0$) | **Ứng viên Production 1**: Cân bằng tốc độ hội tụ, chống quên tri thức cũ |
| **6** | **ViT-Base Feature Extraction** | 85,800,193 | 769 | 0.001% | Khóa 100% Transformer \| Train Head $10^{-3}$ | Đo lường độ lệch miền trên kiến trúc Self-Attention |
| **7** | **ViT-Base Full Fine-tuning** | 85,800,193 | 85,800,193 | 100.00% | Warmup + Cosine $10^{-5}$ \| Focal Loss ($\gamma=2.0$) | **Ứng viên Production 2**: Nắm bắt bố cục hình học toàn cục qua cơ chế Attention |

---

### 4. Cơ Chế Tốc Độ Học Phân Tầng (Discriminative Learning Rates)

Ở cấu hình ứng viên sản xuất số 5 (**ResNet-50 Discriminative Fine-tuning**), thay vì dùng một mức learning rate đồng nhất cho toàn mạng làm phá vỡ các bộ lọc trích xuất cạnh/góc tổng quát, mô hình được phân tách thành 3 nhóm tham số với tốc độ học tăng dần:

$$\begin{aligned}
\text{Tầng Nông (Stem \& Layer 1):} \quad & \eta_{\text{early}} = 10^{-5} \quad \text{(Bảo tồn bộ lọc cạnh/góc cơ bản của ImageNet)} \\
\text{Tầng Giữa (Layer 2 \& Layer 3):} \quad & \eta_{\text{mid}} = 5 \times 10^{-5} \quad \text{(Thích nghi với khúc uốn, độ cong nét bút)} \\
\text{Tầng Sâu \& Classifier Head:} \quad & \eta_{\text{head}} = 10^{-3} \quad \text{(Hội tụ nhanh vào nhãn Thật / Giả ngân hàng)}
\end{aligned}$$

#### Hiệu quả Kỹ thuật Đạt được:
1. **Chống phá hủy đặc trưng gốc (Feature Distortion):** Tránh hiện tượng gradient lớn từ hàm mất mát ở các epoch đầu làm đảo lộn các trọng số tầng nông đã tối ưu tốt.
2. **Ngăn chặn quên tri thức cũ (Catastrophic Forgetting):** Giữ lại khả năng khái quát hóa của mạng học sâu.
3. **Tối ưu tốc độ hội tụ:** Mô hình đạt độ ổn định và giảm thiểu dao động loss trên tập Validation chỉ sau một số ít epoch.

---

### 5. Nhận Định Kỹ Thuật Sau Thử Nghiệm Huấn Luyện

1. **Mô hình huấn luyện từ đầu (MLP & Custom CNN):**
   - **MLP Baseline** bộc lộ nhược điểm bùng nổ tham số nghiêm trọng (hơn 100 triệu tham số) do duỗi phẳng tensor $3 \times 224 \times 224$, tiêu tốn bộ nhớ RAM lớn nhưng mất hoàn toàn tính tương quan không gian 2D cục bộ của nét bút.
   - **Custom CNN** nhẹ và tính toán nhanh, nhưng khi huấn luyện trên tập mẫu độc lập người ký (38 người) dễ chạm trần biểu diễn do thiếu chiều sâu trích xuất ngữ cảnh cấp cao.
2. **Chiến lược Feature Extraction (Khóa 100% Backbone):**
   - Tốc độ huấn luyện vượt trội do chỉ cần tối ưu phần Head (2,049 tham số ở ResNet và 769 tham số ở ViT).
   - Tuy nhiên hiệu năng bị chặn trần do các bộ lọc đặc trưng cấp cao của ImageNet vốn hướng về vật thể tự nhiên (chó, mèo, ô tô), chưa thực sự khớp với hình thái nét mực nhị phân.
3. **Hai Ứng Viên Hàng Đầu Cho Triển Khai Hệ Thống Ngân Hàng:**
   - **Cấu hình số 5 (ResNet-50 Discriminative FT):** Tối ưu tài nguyên tính toán, thời gian suy luận trên CPU cực nhanh ($<30\text{ ms}$), hội tụ mượt mà và chống quá khớp hiệu quả.
   - **Cấu hình số 7 (ViT-Base Full Fine-tuning):** Phát huy tối đa cơ chế Self-Attention đa đầu (Multi-Head Self-Attention), kết nối các điểm mút và nét ký rời rạc trên toàn bộ khung ảnh, rất mạnh trong phát hiện các chữ ký giả mạo có tỷ lệ co dãn lệch.

---

### 6. Danh Mục Tệp Mã Nguồn Đã Đóng Gói (Minh Chứng Tuần 2)

- [`notebooks/02_pretrained_transfer_learning.ipynb`](../notebooks/02_pretrained_transfer_learning.ipynb): Notebook thực thi toàn bộ quy trình nạp dữ liệu, khởi tạo ma trận 7 mô hình, phân chia nhóm tham số và chạy thử nghiệm huấn luyện.
- [`src/models.py`](../src/models.py): Định nghĩa cấu trúc 7 mô hình đối chứng, các hàm đóng/mở tầng linh hoạt (`freeze_backbone`, `unfreeze_stages`), phân tách parameter groups cho discriminative LR, và Factory Pattern `build_model`.
- [`src/trainer.py`](../src/trainer.py): Bộ điều khiển huấn luyện `SignatureTrainer` hỗ trợ tracking loss/acc, Early Stopping, lưu trữ checkpoint tốt nhất và tích hợp graceful fallback.
- [`src/losses.py`](../src/losses.py): Cài đặt `FocalLoss` ($\gamma=2.0, \alpha=0.25$), `LabelSmoothingBCEWithLogitsLoss` và factory `get_loss_function`.
- [`reports/Báo Cáo Tiến Độ Tuần 2 - Xác Thực Chữ Ký Ngân Hàng.pdf`](Báo%20Cáo%20Tiến%20Độ%20Tuần%202%20-%20Xác%20Thực%20Chữ%20Ký%20Ngân%20Hàng.pdf): Bản báo cáo nghiệm thu chính thức đã bảo vệ thành công.

---

### 7. Kế Hoạch Chuyển Tiếp Trọng Tâm Cho Tuần 3

1. **Hệ thống đo lường rủi ro ngân hàng:** Đánh giá độc lập trên tập Test (4,968 cặp) qua các chỉ số FAR (*False Acceptance Rate* - bỏ lọt chữ ký giả), FRR (*False Rejection Rate* - từ chối nhầm khách hàng thật), và xác định điểm cân bằng lỗi EER (*Equal Error Rate*).
2. **Tối ưu ngưỡng quyết định STP (Straight-Through Processing):** Quét phổ ngưỡng $\theta^*$ kết hợp phân tích đường cong Coverage–Accuracy phục vụ luồng phê duyệt chi tự động.
3. **Giải thích thị giác bằng Grad-CAM & Attention Map:** Khoanh vùng nét ký bất thường làm căn cứ giải trình minh bạch cho giao dịch viên và bộ phận Kiểm soát nội bộ.
