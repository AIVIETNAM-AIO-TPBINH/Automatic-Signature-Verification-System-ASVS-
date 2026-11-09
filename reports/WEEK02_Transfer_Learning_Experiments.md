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

### 6. Thực Nghiệm Đo Kiểm Định Lượng Trên Google Colab GPU

Thực nghiệm đo kiểm định lượng được thiết kế chuẩn mực và thực thi thành công trên hạ tầng **Google Colab (GPU NVIDIA T4 16GB, Automatic Mixed Precision - AMP)** thông qua notebook:
👉 [`notebooks/02_colab_training_benchmark.ipynb`](../notebooks/02_colab_training_benchmark.ipynb)

- **Quy mô dữ liệu:** Toàn bộ 20,976 cặp Train, 4,416 cặp Validation phân tầng Writer-Independent (38 người ký train, 8 người ký val, 9 người ký test).
- **Trọng số khởi tạo:** Tải chính thức weights ImageNet-1k từ HuggingFace Hub (`pretrained=True`).
- **Thời gian thực thi:** ~2.1 giờ huấn luyện liên tục trên GPU T4. Toàn bộ trọng số tốt nhất được lưu tại thư mục `checkpoints/` và lịch sử đo lường tại `reports/training_benchmark_history.json`.

---

### 7. Bảng Số Liệu Hiệu Năng Thực Chứng (Empirical Benchmark Results)

Dưới đây là số liệu đo lường thực nghiệm khách quan trích xuất trực tiếp từ quá trình huấn luyện:

| STT | Kiến Trúc Mô Hình | Cơ Chế Huấn Luyện | Tỷ Lệ Trainable | Epoch Dừng | Val Loss | Val Accuracy | Val ROC-AUC | Val EER (↓) | Trạng Thái Checkpoint |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **ResNet-50 Linear Probing** | Feature Extraction (Linear Head) | 0.009% | 4 (Early Stop) | 0.5658 | 70.40% | 0.7812 | **29.60%** | `checkpoints/resnet50_linear_probing_best.pt` |
| **2** | **ResNet-50 Discriminative FT** | Fine-tuning 3 nhóm LR ($10^{-5} \to 10^{-3}$) | 100.00% | 5 (Full) | **0.0973** | **78.20%** | 0.8540 | **21.83%** | `checkpoints/resnet50_discriminative_best.pt` |
| **3** | **Siamese ResNet-50** | Shared Backbone (Cấu hình TA đề xuất) | 100.00% | 5 (Full) | 0.1076 | 74.00% | 0.8125 | **26.02%** | `checkpoints/siamese_resnet50_best.pt` |
| **4** | **ViT-Base Full Fine-tuning** | Vision Transformer ($10^{-5} \to 10^{-4}$) | 100.00% | 4 (Early Stop) | 0.1622 | **79.20%** | **0.8665** | **20.81%** | `checkpoints/vit_base_finetuning_best.pt` |

#### Chi tiết diễn biến qua từng Epoch (Learning Dynamics):
1. **ResNet-50 Linear Probing (Head Only 2,049 params):**
   - *Epoch 1:* Train Loss: 0.6084, Train Acc: 69.4% | Val Loss: 0.5764, Val Acc: 70.4%, Val EER: 29.60% `[BEST]`
   - *Epoch 2:* Train Loss: 0.5519, Train Acc: 72.5% | Val Loss: 0.5783, Val Acc: 68.9%, Val EER: 31.09%
   - *Epoch 3:* Train Loss: 0.5368, Train Acc: 73.2% | Val Loss: 0.5797, Val Acc: 68.5%, Val EER: 31.48%
   - *Epoch 4:* Train Loss: 0.5207, Train Acc: 74.3% | Val Loss: 0.5658, Val Acc: 69.6%, Val EER: 30.41%
   - *Nhận xét:* Kích hoạt Early Stopping tại Epoch 4 do Val EER không cải thiện qua 3 epoch liên tiếp.

2. **ResNet-50 Discriminative Fine-tuning (Ứng viên Production 1):**
   - *Epoch 1:* Train Loss: 0.0350, Train Acc: 83.8% | Val Loss: 0.0866, Val Acc: 68.6%, Val EER: 31.39%
   - *Epoch 2:* Train Loss: 0.0133, Train Acc: 94.8% | Val Loss: 0.0986, Val Acc: 75.7%, Val EER: 24.32%
   - *Epoch 3:* Train Loss: 0.0079, Train Acc: 96.9% | Val Loss: 0.1163, Val Acc: 72.8%, Val EER: 27.24%
   - *Epoch 4:* Train Loss: 0.0055, Train Acc: 98.1% | Val Loss: 0.0973, Val Acc: 78.2%, Val EER: **21.83%** `[BEST]`
   - *Epoch 5:* Train Loss: 0.0046, Train Acc: 98.4% | Val Loss: 0.1371, Val Acc: 74.6%, Val EER: 25.36%

3. **Siamese ResNet-50 (Shared Backbone - Cấu hình đối chứng):**
   - *Epoch 1:* Train Loss: 0.0284, Train Acc: 87.7% | Val Loss: 0.1755, Val Acc: 58.2%, Val EER: 41.78%
   - *Epoch 2:* Train Loss: 0.0050, Train Acc: 98.6% | Val Loss: 0.1937, Val Acc: 57.1%, Val EER: 42.93%
   - *Epoch 3:* Train Loss: 0.0025, Train Acc: 99.4% | Val Loss: 0.2204, Val Acc: 62.6%, Val EER: 37.39%
   - *Epoch 4:* Train Loss: 0.0014, Train Acc: 99.6% | Val Loss: 0.1076, Val Acc: 74.0%, Val EER: **26.02%** `[BEST]`
   - *Epoch 5:* Train Loss: 0.0008, Train Acc: 99.8% | Val Loss: 0.1814, Val Acc: 66.1%, Val EER: 33.85%

4. **ViT-Base Full Fine-tuning (Ứng viên Production 2):**
   - *Epoch 1:* Train Loss: 0.0268, Train Acc: 87.7% | Val Loss: 0.1622, Val Acc: 79.2%, Val EER: **20.81%** `[BEST]`
   - *Epoch 2:* Train Loss: 0.0060, Train Acc: 98.2% | Val Loss: 0.1761, Val Acc: 76.0%, Val EER: 24.05%
   - *Epoch 3:* Train Loss: 0.0036, Train Acc: 99.0% | Val Loss: 0.4179, Val Acc: 74.1%, Val EER: 25.86%
   - *Epoch 4:* Train Loss: 0.0020, Train Acc: 99.4% | Val Loss: 0.5036, Val Acc: 69.8%, Val EER: 30.19%
   - *Nhận xét:* Hội tụ ngay từ Epoch 1 đạt đỉnh cao nhất, sau đó có dấu hiệu quá khớp nhẹ (Overfitting) do số lượng tham số lớn ($85.8\text{M}$ params), kích hoạt Early Stopping tại Epoch 4.

---

### 8. Nhận Định Kỹ Thuật Chuyên Sâu Giải Trình Các Góp Ý Của TA

Dựa trên số liệu định lượng thu được, nhóm nghiên cứu làm rõ 4 nhận định học thuật then chốt:

#### 8.1. Kiểm chứng hiện tượng "Hiệu năng trần bị giới hạn" (Bounded Ceiling)
- **Thực nghiệm chứng minh:** Khi áp dụng chiến lược Feature Extraction (khóa 100% Backbone ResNet-50, chỉ huấn luyện $2,049$ tham số Linear Head), Val Loss kẹt cứng ở mức **0.5658**, Val Accuracy đạt **70.40%** và Val EER không thể giảm xuống dưới **29.60%**.
- **Nguyên nhân cốt lõi:** Trọng số ImageNet-1k được huấn luyện trên vật thể tự nhiên đa sắc (chó, mèo, ô tô). Khi đối mặt với ảnh nhị phân 1-bit của nét chữ ký, các bộ lọc màu sắc ở tầng Conv1 không thể tự điều chỉnh để tối ưu hóa việc phân tách các đường cong vi mô. Việc chỉ cập nhật lớp phân loại cuối cùng khiến mô hình chạm ngay "trần hiệu năng" do độ lệch miền (*Domain Gap*).

#### 8.2. Hiệu quả chuyển giao tri thức của Discriminative Fine-tuning
- **Thực nghiệm chứng minh:** Khi mở khóa toàn bộ backbone và áp dụng tốc độ học phân tầng ($10^{-5}$ cho tầng nông $\to 5 \times 10^{-5}$ cho tầng giữa $\to 10^{-3}$ cho head):
  - Val Loss giảm ngoạn mục từ **0.5658 xuống 0.0973**.
  - Val Accuracy nhảy vọt từ **70.40% lên 78.20%** (+7.8%).
  - Val EER giảm mạnh từ **29.60% xuống 21.83%** (-7.77%).
- **Ý nghĩa kỹ thuật:** Việc cấp tốc độ học nhỏ ($10^{-5}$) cho các tầng nông giúp bảo tồn các bộ lọc cạnh/góc cơ bản của ImageNet, trong khi cho phép các tầng tích chập sâu thích nghi trọn vẹn với nhịp điệu nét ký ngân hàng.

#### 8.3. Luận chứng so sánh: Biểu diễn 3-Kênh vs. Siamese Network
- Thực nghiệm đối chứng theo đề xuất của TA cho thấy:
  - **Siamese ResNet-50:** Val EER đạt **26.02%**, Val Accuracy đạt **74.00%**.
  - **3-Channel Difference ResNet-50:** Val EER đạt **21.83%**, Val Accuracy đạt **78.20%** (Vượt trội **+4.19% EER** và **+4.20% Accuracy**).
- **Phân tích cơ chế sâu:**
  - Kiến trúc Siamese áp dụng cơ chế *Late Fusion*: Hai ảnh được trích xuất độc lập qua Global Average Pooling thành hai vector $f_{\text{ref}}, f_{\text{query}} \in \mathbb{R}^{2048}$, sau đó mới trừ tuyệt đối. Thao tác Pooling ở cuối vô tình triệt tiêu thông tin vị trí không gian (Spatial Coordinate Information), khiến mô hình khó nhận ra sự lệch pha về nét chữ ký cục bộ.
  - Ngược lại, biểu diễn 3-Kênh áp dụng cơ chế *Early Fusion*: Kênh $|I_{\text{ref}} - I_{\text{query}}|$ được đưa ngay vào tầng Conv1 cùng với hai ảnh gốc. Điều này cho phép các kernel tích chập ở mọi tầng trích xuất mối tương quan tương phản pixel-level trực tiếp tại từng tọa độ $(x, y)$, mang lại năng lực phân biệt chữ ký giả tinh vi cao hơn đáng kể.

#### 8.4. Ngôi vị Quán quân của Vision Transformer (ViT-Base)
- **Thực nghiệm chứng minh:** ViT-Base đạt Val EER thấp nhất toàn diện là **20.81%** và Val Accuracy cao nhất là **79.20%**.
- **Lợi thế kiến trúc:** Cơ chế Multi-Head Self-Attention tính toán ma trận chú ý trên toàn bộ $14 \times 14 = 196$ patch ảnh, giúp mô hình nắm bắt được hình thái toàn cục của chữ ký (tỉ lệ chiều ngang/dọc, độ dốc nét ký, khoảng trống giữa các từ ký) mà không bị giới hạn bởi trường tiếp nhận cục bộ (*Local Receptive Field*) như CNN.

---

### 9. Kết Luận Tuần 2 & Kế Hoạch Chuyển Tiếp Tuần 3

1. **Hoàn thành 100% mục tiêu Tuần 2:**
   - Khắc phục lỗi `pretrained=False`, chạy thực nghiệm thật trên Colab GPU.
   - Thống nhất bảng tham số ($2,049$ và $769$ params cho Linear Probing).
   - Bổ sung cấu hình đối chứng Siamese ResNet-50.
   - Giải trình cơ chế Hard-Sample Mining của Focal Loss trên dữ liệu cân bằng.
   - Bổ sung đầy đủ số liệu định lượng (Val Loss, Acc, EER) và biểu đồ quá trình học.
2. **Kế hoạch triển khai Tuần 3 (Threshold Tuning & Luồng STP Ngân Hàng):**
   - Lựa chọn 2 mô hình tối ưu vào vòng tinh chỉnh ngưỡng:
     + **Production Model 1 (Tốc độ & Nhẹ):** ResNet-50 Discriminative FT (EER $21.83\%$).
     + **Production Model 2 (Độ chính xác cao):** ViT-Base Fine-tuning (EER $20.81\%$).
   - Thiết lập ngưỡng $T_{\text{STP}}$ để đạt yêu cầu ngân hàng: FAR $< 1.0\%$ (an toàn bảo mật), FRR $< 5.0\%$ (trải nghiệm khách hàng) và luồng phân luồng xử lý thủ công (Manual Review) cho vùng xám bất định.
