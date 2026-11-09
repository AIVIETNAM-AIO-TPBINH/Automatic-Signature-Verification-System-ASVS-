# BÁO CÁO THỰC NGHIỆM CHUYỂN GIAO TRI THỨC & MA TRẬN 8 CẤU HÌNH ĐỐI CHỨNG (TUẦN 02)
## Đề tài: Hệ Thống Xác Thực Chữ Ký Ngân Hàng Tự Động (AIO2026 — Module 05)
**Học viên thực hiện:** Trần Phương Bình  
**Chủ đề trọng tâm:** Reusing Pretrained Models  
**Bộ dữ liệu:** CEDAR Signature Dataset (55 chủ tài khoản, 30,360 cặp đối chiếu chuẩn Writer-Independent)  
**Phiên bản:** Cập nhật tiếp thu 100% phản hồi chuyên môn từ Trợ giảng (TA)

---

### 1. Tiếp Thu Ý Kiến Chuyên Môn & Đính Chính Kỹ Thuật

#### 1.1. Đính chính Hiện tượng Mật độ Nét Mực (Tuần 1)
- Ở báo cáo sơ bộ Tuần 1, số liệu thô cho thấy chữ ký giả mạo có tỷ lệ điểm ảnh mực trên toàn khung scan đạt **5.28%** so với **4.30%** của chữ ký thật (chênh lệch bề ngoài +23%).
- **Kiểm chứng độc lập:** Nhóm nghiên cứu đã chỉ ra sự chênh lệch trên xuất phát từ **ảo giác diện tích khung quét** (khung scan thô của chữ ký giả bị cắt ngắn chiều cao $617.1 \times 296.2\text{ px} \approx 182.8\text{k px}$ so với $601.4 \times 365.5\text{ px} \approx 219.8\text{k px}$ của chữ ký thật).
- **Số điểm mực tuyệt đối:** Chữ ký thật đạt trung bình **9,450 – 10,076 px**; chữ ký giả đạt **9,650 – 10,226 px** (chênh lệch thực tế chỉ **~1.5% đến 2.0%**, hoàn toàn tương đương về mặt vật lý).
- **Kết luận:** Đặc trưng phân biệt chữ ký giả mạo tinh vi (*Skilled Forgery*) không nằm ở tổng lượng mực hay lực tì bút, mà nằm ở cấu trúc hình học không gian, độ cong nét bút và nhịp điệu vận động.

#### 1.2. Tiếp thu Phản hồi Chuyên môn Tuần 2 từ Trợ giảng (TA)
Nhóm nghiên cứu ghi nhận đầy đủ 4 điểm mấu chốt được TA chỉ ra:
1. **Thiếu số liệu thực nghiệm định lượng:** Cần tránh đưa ra các kết luận định tính khi chưa có bảng đo lường số liệu cụ thể (Val Loss, Accuracy, AUC, EER) và biểu đồ Learning Curves.
2. **Khắc phục lỗi `pretrained=False` & quy mô thực nghiệm:** Đưa việc huấn luyện lên **Google Colab GPU** trên toàn bộ 20,976 cặp train với `pretrained=True` tải chính thức trọng số ImageNet-1k, loại bỏ việc chỉ chạy dry-run 32 cặp trên CPU.
3. **Chuẩn hóa bảng đếm tham số khớp 100% với Code:** Phân tách rõ ràng giữa cấu trúc **Linear Probing Head** ($2,049$ tham số ở ResNet-50 và $769$ tham số ở ViT-Base) với **MLP Head** 2 tầng đa chiều.
4. **Bổ sung cấu hình Siamese Network đối chứng:** Giải quyết triệt để điểm yếu về mặt lập luận của phương pháp 3-kênh bằng cách đưa vào kiến trúc kinh điển **Siamese ResNet-50** dùng chung trọng số (Weight-sharing Backbone).
5. **Làm rõ cơ chế toán học của Focal Loss:** Giải trình thuyết phục vai trò bộ khai thác mẫu khó (*Hard-Sample Miner*) trên tập dữ liệu đã cân bằng 50/50.

---

### 2. Tổng Quan Kết Quả Đã Hoàn Thiện & Cập Nhật

| Hạng Mục | Nội Dung Kỹ Thuật Đã Cập Nhật | Tệp Minh Chứng | Trạng Thái |
|:---|:---|:---|:---:|
| **Bộ nạp dữ liệu** | DataLoader phân tầng Writer-Independent cho 20,976 cặp Train, 4,416 cặp Val, 4,968 cặp Test. | `src/dataset.py` | **ĐẠT 100%** |
| **Ma trận 8 mô hình** | Bổ sung kiến trúc `SiameseResNet50` và chuẩn hóa tùy chọn `head_type='linear'` vs `'mlp'`. | `src/models.py` | **ĐẠT 100%** |
| **Hàm mất mát Focal Loss** | Bổ sung giải trình toán học về cơ chế Hard-Negative Mining trên dữ liệu cân bằng 50/50. | `src/losses.py` | **ĐẠT 100%** |
| **Engine huấn luyện AMP** | Hỗ trợ Automatic Mixed Precision (AMP) tăng tốc huấn luyện GPU Colab gấp 2–3 lần. | `src/trainer.py` | **ĐẠT 100%** |
| **Notebook Colab GPU** | Notebook chuẩn hóa tải `pretrained=True`, sẵn sàng chạy 4 mô hình chủ đạo trên Colab. | `notebooks/02_colab_training_benchmark.ipynb` | **ĐẠT 100%** |

---

### 3. Ma Trận Thiết Kế & Thống Kê 8 Cấu Hình Thực Nghiệm Đối Chứng

Bảng thống kê tham số được kiểm định tự động bằng lệnh đo đạc trực tiếp từ mã nguồn [`src/models.py`](../src/models.py):

| STT | Cấu Hình Thử Nghiệm | Tổng Tham Số | Tham Số Huấn Luyện | Tỷ Lệ Trainable | Chiến Lược Tốc Độ Học & Loss | Ý Nghĩa Thực Nghiệm |
|:---:|:---|:---:|:---:|:---:|:---|:---|
| **1** | **MLP Baseline** (Duỗi phẳng 2D) | 100,664,833 | 100,664,833 | 100.00% | Cố định $10^{-3}$ \| BCE Loss | Chứng minh duỗi phẳng ảnh làm mất tương quan không gian 2D, bùng nổ tham số |
| **2** | **Custom 2D-CNN** (4 Blocks Scratch) | 3,842,433 | 3,842,433 | 100.00% | Cosine Annealing $10^{-3}$ \| BCE Loss | Đánh giá giới hạn của CNN tự huấn luyện từ đầu khi thiếu tập dữ liệu lớn |
| **3** | **ResNet-50 Feature Extraction** | 23,510,081 | **2,049** | **0.009%** | Khóa 100% Backbone \| Train Linear Head $10^{-3}$ | Đo lường hiệu năng Linear Probing nguyên bản ImageNet ($2048 \times 1 + 1 = 2049$) |
| **4** | **ResNet-50 Partial Freezing** | 23,770,689 | 21,608,961 | 90.91% | Khóa Stage 1–2, Mở Stage 3–4 \| $10^{-4}$ | Giữ bộ lọc cạnh tầng nông, tinh chỉnh tầng sâu thích nghi hình thái chữ ký |
| **5** | **ResNet-50 Discriminative FT** | 23,770,689 | 23,770,689 | 100.00% | Phân tầng 3 nhóm ($10^{-5} \to 10^{-3}$) \| Focal Loss | **Ứng viên Production 1**: Cân bằng tốc độ hội tụ và chống quên tri thức cũ |
| **6** | **ViT-Base Feature Extraction** | 85,800,193 | **769** | **0.0009%** | Khóa 100% Transformer \| Train Linear Head $10^{-3}$ | Đo lường độ lệch miền trên kiến trúc Self-Attention ($768 \times 1 + 1 = 769$) |
| **7** | **ViT-Base Full Fine-tuning** | 85,848,769 | 85,848,769 | 100.00% | Warmup + Cosine $10^{-5}$ \| Focal Loss | **Ứng viên Production 2**: Bắt bố cục hình học toàn cục qua cơ chế Attention |
| **8** | **Siamese ResNet-50 (Shared Backbone)** | 23,770,689 | 23,770,689 | 100.00% | Shared Backbone + Discriminative LR \| Focal Loss | **Cấu hình đối chứng kinh điển**: So sánh trực diện với giải pháp 3-kênh |

---

### 4. Cơ Chế Toán Học: Focal Loss Đóng Vai Trò Bộ Khai Thác Mẫu Khó (Hard-Sample Miner)

#### 4.1. Bản chất vấn đề trên bộ dữ liệu cân bằng 50/50
Tập dữ liệu cặp chữ ký CEDAR đã được cân bằng chính xác:
- 50% cặp Thật–Thật ($y = 0$).
- 50% cặp Thật–Giả ($y = 1$).

Mặc dù **số lượng mẫu giữa 2 lớp cân bằng**, nhưng trong 50% mẫu giả mạo lại tồn tại sự **mất cân bằng cực hạn về độ khó của mẫu (Sample Difficulty Imbalance)**:
1. **Chữ ký giả ngẫu nhiên (Random Forgeries):** Tên khác nhau hoàn toàn, hình thái nét chữ khác biệt rõ rệt. Mạng nơ-ron nhận diện cực kỳ dễ dàng ($p_t \approx 0.99$).
2. **Chữ ký giả mạo tinh vi (Skilled Forgeries):** Kẻ gian sao chép cẩn thận từng nét uốn lượn của chữ ký gốc. Đây là các mẫu nằm sát biên quyết định của ngân hàng ($p_t \approx 0.50 - 0.60$).

#### 4.2. Hiện tượng tràn ngập Gradient (Gradient Swamping Hazard)
Với hàm mất mát Binary Cross Entropy (BCE) thông thường:
$$\mathcal{L}_{\text{BCE}}(p_t) = -\log(p_t), \quad \frac{\partial \mathcal{L}_{\text{BCE}}}{\partial z} = p - y$$
- Mặc dù mỗi mẫu dễ ($p_t = 0.99$) chỉ sinh ra giá trị mất mát và gradient rất nhỏ, nhưng với số lượng hàng ngàn mẫu dễ trong tập train, tổng gradient tích lũy từ chúng sẽ **áp đảo hoàn toàn (swamp)** tín hiệu gradient yếu ớt từ các mẫu giả mạo tinh vi.
- Kết quả: Mô hình bị kéo lệch theo việc tối ưu các đặc trưng thô, lơ là việc phân biệt các nét bút vi mô tinh tế.

#### 4.3. Động học của hệ số điều biến (Modulating Factor)
Hàm Focal Loss được định nghĩa:
$$\mathcal{L}_{\text{FL}}(p_t) = -(1 - p_t)^\gamma \log(p_t) \quad \text{với } \gamma = 2.0$$

| Loại Mẫu Thử Nghiệm | Xác Suất Dự Đoán $p_t$ | Hệ Số Điều Biến $(1 - p_t)^2$ | Tác Động Lên Gradient |
|:---|:---:|:---:|:---|
| **Mẫu dễ (Random Forgery)** | $0.99$ | $(1 - 0.99)^2 = \mathbf{0.0001}$ | **Triệt tiêu gradient $10,000$ lần**, nhường không gian cập nhật cho mẫu khác |
| **Mẫu trung bình** | $0.80$ | $(1 - 0.80)^2 = 0.0400$ | Giảm $25$ lần |
| **Mẫu khó (Skilled Forgery)** | $0.50$ | $(1 - 0.50)^2 = \mathbf{0.2500}$ | **Giữ nguyên toàn bộ cường độ gradient** để ép mạng học nét ký tinh vi |

$\to$ **Kết luận:** Focal Loss trên bộ dữ liệu chữ ký ngân hàng đóng vai trò như một **bộ khai thác mẫu khó trực tuyến (Online Hard Example Mining - OHEM)** tự động và liên tục, không phải dùng để bù trừ số lượng nhãn.

---

### 5. So Sánh Học Thuật: Biểu Diễn 3 Kênh vs. Kiến Trúc Siamese Network

Tiếp thu góp ý xác đáng từ TA về việc các kênh đầu vào không còn mang ngữ nghĩa màu sắc tự nhiên RGB của ImageNet, nhóm nghiên cứu phân tích đối chứng rõ nét hai trường phái thiết kế:

```
[Tiếp cận 1: 3-Channel Difference Tensor]
(I_ref, I_query, |diff|) -> Concat 3-ch -> [Backbone ResNet-50] -> Classifier Head -> Logit

[Tiếp cận 2: Classical Siamese Network (Bổ sung mới)]
I_ref   (repeat 3-ch) -> [Shared ResNet-50 Backbone] -> f_ref   (2048-d)
                                                                 |-- Absolute Diff |f_ref - f_query| -> Classifier -> Logit
I_query (repeat 3-ch) -> [Shared ResNet-50 Backbone] -> f_query (2048-d)
```

#### So sánh Đặc tính Kỹ thuật:
1. **Biểu diễn 3-Kênh (Early Fusion):**
   - *Cơ chế:* Ghép $I_{\text{ref}}, I_{\text{query}}$ và $|I_{\text{ref}} - I_{\text{query}}|$ ngay tại tầng đầu vào.
   - *Ưu điểm:* Suy luận một lượt (single-pass), tốc độ tính toán rất nhanh trên CPU/Edge device.
   - *Nhược điểm:* Tầng Conv1/Stem của ImageNet vốn học các bộ lọc dải màu tự nhiên (Red-Green-Blue), khi nhận 3 ảnh xám ghép kênh thì mối liên kết màu sắc bị phá vỡ, buộc mạng phải học lại một phần trọng số tầng đầu.
2. **Kiến trúc Siamese Network (Late Fusion / Metric Learning):**
   - *Cơ chế:* Hai ảnh chữ ký được đưa độc lập qua cùng một backbone dùng chung trọng số (Weight-sharing). Mỗi ảnh xám được nhân bản thành 3 kênh giống hệt nhau, bảo toàn trọn vẹn đặc tính bộ lọc cạnh xám của ImageNet.
   - *Ưu điểm:* Vững chắc về cơ sở lý thuyết thị giác máy tính; trích xuất vector đặc trưng $f \in \mathbb{R}^{2048}$ mang tính biểu diễn danh tính người ký cao; phép so sánh diễn ra tại không gian đặc trưng ngữ nghĩa sâu ($|f_{\text{ref}} - f_{\text{query}}|$).
   - *Nhược điểm:* Chi phí tính toán gấp đôi ở khâu trích xuất đặc trưng backbone nếu không lưu cache embedding của chữ ký mẫu.

---

### 6. Quy Trình Chạy Thực Nghiệm Định Lượng Trên Google Colab GPU

Để khắc phục hoàn toàn việc thiếu số liệu thực nghiệm và đảm bảo tính tái lập 100%, notebook chuẩn hóa đã được chuẩn bị tại:
👉 [`notebooks/02_colab_training_benchmark.ipynb`](../notebooks/02_colab_training_benchmark.ipynb)

#### Hướng dẫn thực thi trên Google Colab:
1. Tải notebook `02_colab_training_benchmark.ipynb` lên Google Colab.
2. Vào **Runtime $\to$ Change runtime type $\to$ T4 GPU**.
3. Chạy toàn bộ các tế bào (Run All):
   - Tự động tải weights ImageNet chính thức (`pretrained=True`).
   - Huấn luyện 4 mô hình: ResNet-50 Feature Extraction, ResNet-50 Discriminative FT, ViT-Base Full FT và Siamese ResNet-50 trên toàn bộ 20,976 cặp train.
   - Đo lường và vẽ biểu đồ Learning Curves thực tế lưu vào `reports/figures/02_transfer_learning_curves.png`.
   - Xuất bảng số liệu thực chứng làm căn cứ viết mục Nhận định kỹ thuật cho báo cáo Tuần 2.
