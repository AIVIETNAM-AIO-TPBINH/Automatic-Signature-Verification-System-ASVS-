# Automated Banking Signature Verification System

Nghiên cứu thực nghiệm đối chứng về chuyển giao tri thức và thích nghi lệch miền (**Domain Adaptation**) sử dụng **Pretrained CNN (ResNet-50)**, **Vision Transformer (ViT-Base)** và **Siamese Network** trên bộ dữ liệu chuẩn quốc tế CEDAR Signature Dataset (55 chủ tài khoản, 30,360 cặp đối chiếu).

---

## 📌 Tổng Quan
- **Subtopic:** Deep Learning — Reusing Pretrained Models (Module 05 — AIO2026).
- **Bộ dữ liệu:** CEDAR Signature Dataset (55 chủ tài khoản; mỗi người gồm 24 chữ ký thật `full_org` và 24 chữ ký giả mạo tinh vi `full_forg` $\to$ 2,640 ảnh gốc).
- **Bài toán:** Đối chiếu và xác thực chữ ký tự động trên chứng từ ngân hàng (Ủy nhiệm chi/Séc) theo chuẩn phân tách độc lập người ký (**Writer-Independent Split**: 38 Train / 8 Val / 9 Test).

---

## 📅 Tiến Độ Thực Hiện (Project Roadmap)

- [x] **Tuần 1: Khám Phá Dữ Liệu Ảnh (Image EDA), Kiểm Chứng Rò Rỉ & Pipeline Tiền Xử Lý 3 Kênh**
  - [x] Phân tích hình thái nét chữ, tỷ lệ khung hình & kiểm chứng lượng mực thực tế (*Ink Density Bias & BBox Verification* đính chính theo phản hồi từ trợ giảng).
  - [x] Kiểm toán rò rỉ danh tính (**Zero Identity Leakage**): 38 người Train (20,976 cặp), 8 người Val (4,416 cặp), 9 người Test (4,968 cặp).
  - [x] Xây dựng bộ lá chắn tiền xử lý (*Preprocessing Shield*): Khử nền giấy Otsu $\to$ Auto-crop Bounding Box $\to$ Resize đệm viền $224 \times 224$.
  - [x] Biểu diễn hiệu số 3 kênh: $\mathbf{X}_{\text{input}} = \text{Concat}(I_{\text{ref}}, I_{\text{query}}, |I_{\text{ref}} - I_{\text{query}}|)$.
  - [x] Tăng cường dữ liệu `RandomAffine` nhẹ (bảo toàn hướng nét chữ, không lật).
  - [x] Sản phẩm bàn giao Tuần 1: `reports/WEEK01_System_Strategy.md`, `reports/Báo Cáo Tiến Độ Tuần 1 - Xác Thực Chữ Ký Ngân Hàng.pdf`, `src/preprocessing.py`, `src/dataset.py`, `notebooks/01_eda_and_image_pipeline.ipynb`.
- [x] **Tuần 2: Thực Nghiệm Chuyển Giao Tri Thức (Reusing Pretrained Models) & Ma Trận 8 Cấu Hình**
  - [x] Thiết lập và đóng gói toàn bộ ma trận 8 cấu hình thực nghiệm đối chứng từ MLP Baseline, Custom CNN đến Pretrained CNN (ResNet-50), Vision Transformer (ViT-Base) và Siamese Network (đối chứng kinh điển theo góp ý của TA).
  - [x] Cài đặt chiến lược thích nghi lệch miền: Đóng băng tầng (Freezing) và Tốc độ học phân tầng (Discriminative Learning Rates: $10^{-5} \to 10^{-3}$).
  - [x] Xây dựng bộ điều khiển huấn luyện `SignatureTrainer` tích hợp AMP (Mixed Precision) và Focal Loss ($\gamma = 2.0$) đóng vai trò bộ khai thác mẫu khó (*Hard-Sample Miner*).
  - [x] Chuẩn hóa notebook chạy GPU Colab với `pretrained=True`: `notebooks/02_colab_training_benchmark.ipynb`.
  - [x] Thu thập số liệu định lượng thực chứng (Empirical Benchmark) trên GPU Colab: Val Loss, Accuracy, ROC-AUC, EER.
  - [x] Sản phẩm bàn giao Tuần 2: `notebooks/02_pretrained_transfer_learning.ipynb`, `notebooks/02_colab_training_benchmark.ipynb`, `src/models.py`, `src/trainer.py`, `src/losses.py`, `reports/WEEK02_Transfer_Learning_Experiments.md`, `reports/Báo Cáo Tiến Độ Tuần 2 - Xác Thực Chữ Ký Ngân Hàng.pdf`.
- [ ] **Tuần 3: Tối Ưu Losses & Metrics, Ngưỡng STP & Giải Thích Grad-CAM** *(Đang thực hiện trên máy local)*
- [ ] **Tuần 4: Đo Kiểm Độ Trễ Suy Luận & Báo Cáo Kỹ Thuật Tổng Kết** *(Kế hoạch trên máy local)*

---

## 🔬 Ma Trận 8 Cấu Hình Đối Chứng (Tuần 2)

| STT | Cấu Hình Thử Nghiệm | Tổng Tham Số | Tham Số Huấn Luyện | Tỷ Lệ Trainable | Chiến Lược Tốc Độ Học & Hàm Loss |
|:---:|:---|:---:|:---:|:---:|:---|
| **1** | **MLP Baseline** (Duỗi phẳng 2D) | 100,664,833 | 100,664,833 | 100.00% | Cố định $10^{-3}$ \| BCE Loss |
| **2** | **Custom 2D-CNN** (4 Blocks) | 3,842,433 | 3,842,433 | 100.00% | Cosine Annealing $10^{-3}$ \| BCE Loss |
| **3** | **ResNet-50 Feature Extraction** | 23,510,081 | **2,049** | **0.009%** | Khóa 100% Backbone \| Train Linear Head $10^{-3}$ |
| **4** | **ResNet-50 Partial Freezing** | 23,770,689 | 21,608,961 | 90.91% | Khóa Stage 1–2, Mở Stage 3–4 \| $10^{-4}$ |
| **5** | **ResNet-50 Discriminative FT** | 23,770,689 | 23,770,689 | 100.00% | Phân tầng 3 nhóm ($10^{-5} \to 10^{-3}$) \| Focal Loss ($\gamma=2.0$) |
| **6** | **ViT-Base Feature Extraction** | 85,800,193 | **769** | **0.0009%** | Khóa 100% Transformer \| Train Linear Head $10^{-3}$ |
| **7** | **ViT-Base Full Fine-tuning** | 85,848,769 | 85,848,769 | 100.00% | Warmup + Cosine $10^{-5}$ \| Focal Loss ($\gamma=2.0$) |
| **8** | **Siamese ResNet-50 (Shared Backbone)** | 23,770,689 | 23,770,689 | 100.00% | Shared Backbone + Discriminative LR \| Focal Loss ($\gamma=2.0$) |

---

## 📊 Kết Quả Thực Nghiệm Định Lượng (Google Colab GPU T4 Benchmark)

Số liệu đo kiểm thực tế thu được khi huấn luyện trên toàn bộ tập dữ liệu (20,976 cặp train, 4,416 cặp val) với trọng số ImageNet-1k chính thức (`pretrained=True`):

| STT | Kiến Trúc Mô Hình | Phương Pháp Huấn Luyện | Val Accuracy | Val ROC-AUC | Val EER (↓) | Checkpoint Tốt Nhất |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| **1** | **ResNet-50 Linear Probing** | Feature Extraction (Head $2,049$ params) | 70.40% | 0.7812 | 29.60% | `resnet50_linear_probing_best.pt` |
| **2** | **ResNet-50 Discriminative FT** | Fine-tuning 3 nhóm LR ($10^{-5} \to 10^{-3}$) | **78.20%** | 0.8540 | **21.83%** | `resnet50_discriminative_best.pt` |
| **3** | **Siamese ResNet-50** | Shared Backbone (Cấu hình đối chứng TA) | 74.00% | 0.8125 | 26.02% | `siamese_resnet50_best.pt` |
| **4** | **ViT-Base Full Fine-tuning** | Vision Transformer ($10^{-5} \to 10^{-4}$) | **79.20%** | **0.8665** | **20.81%** | `vit_base_finetuning_best.pt` |

> 📌 **Báo cáo chuyên sâu:** Xem toàn bộ phân tích cơ chế toán học, so sánh 3-Kênh vs. Siamese và giải trình học thuật tại [`reports/WEEK02_Transfer_Learning_Experiments.md`](reports/WEEK02_Transfer_Learning_Experiments.md).

---

## 🚀 Hướng Dẫn Chạy & Tái Lập Thực Nghiệm

```bash
# 1. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 2. Khởi chạy Jupyter Lab / Notebook trên máy local:
jupyter lab

# Tuần 1: EDA, Kiểm toán rò rỉ và Preprocessing Shield
notebooks/01_eda_and_image_pipeline.ipynb

# Tuần 2: Thẩm định kiến trúc & tham số mô hình
notebooks/02_pretrained_transfer_learning.ipynb

# 3. Huấn luyện toàn bộ thực nghiệm định lượng trên Google Colab GPU (T4):
# Mở file notebooks/02_colab_training_benchmark.ipynb trên Colab và bấm "Run All".
```

---

## 📂 Cấu Trúc Thư Mục (Tiến Độ Nghiệm Thu Tuần 2)

```
.
├── data/
│   ├── raw/signatures/                      # Hướng dẫn cấu trúc dữ liệu ảnh gốc CEDAR (Readme.txt)
│   └── processed/                           # 3 tệp cặp ảnh đối chiếu chuẩn Writer-Independent
│       ├── train_pairs.csv                  # 20,976 cặp (ID 1 -> 38)
│       ├── val_pairs.csv                    # 4,416 cặp (ID 39 -> 46)
│       └── test_pairs.csv                   # 4,968 cặp (ID 47 -> 55)
├── notebooks/
│   ├── 01_eda_and_image_pipeline.ipynb      # Phân tích EDA, kiểm toán rò rỉ & Preprocessing Shield (Tuần 1)
│   ├── 02_pretrained_transfer_learning.ipynb # Ma trận 8 mô hình & kiểm định tham số (Tuần 2)
│   └── 02_colab_training_benchmark.ipynb    # Huấn luyện GPU Colab (Pretrained=True, Learning Curves)
├── plans/
│   └── KẾ HOẠCH NGHIÊN CỨU.docx             # Kế hoạch nghiên cứu 4 tuần chính thức
├── reports/
│   ├── figures/                             # Nơi lưu đồ thị phân tích & learning curves
│   ├── Báo Cáo Tiến Độ Tuần 1 - Xác Thực Chữ Ký Ngân Hàng.pdf # Báo cáo nghiệm thu chính thức Tuần 1
│   ├── Báo Cáo Tiến Độ Tuần 2 - Xác Thực Chữ Ký Ngân Hàng.pdf # Báo cáo nghiệm thu chính thức Tuần 2
│   ├── training_benchmark_history.json      # Kết quả đo lường định lượng 4 mô hình trên GPU Colab
│   ├── WEEK01_System_Strategy.md             # Báo cáo kỹ thuật chi tiết Tuần 1
│   └── WEEK02_Transfer_Learning_Experiments.md # Báo cáo kỹ thuật chi tiết Tuần 2 (Tiếp thu góp ý TA)
├── src/
│   ├── __init__.py
│   ├── dataset.py                           # PyTorch Dataset & DataLoader phân tách Writer-Independent
│   ├── preprocessing.py                     # Otsu binarization, bbox crop, padding resize, 3-ch stack
│   ├── models.py                            # 8 cấu hình mô hình (Linear/MLP heads, Siamese ResNet-50)
│   ├── trainer.py                           # SignatureTrainer tích hợp AMP & Early Stopping
│   ├── losses.py                            # Focal Loss (Hard-Sample Miner), Label Smoothing & BCE
│   └── utils.py                             # Tiện ích cố định seed, device auto-detect & I/O
├── .gitignore                               # Cấu hình bảo toàn mã nguồn Tuần 3 & 4 trên máy local
├── requirements.txt
└── README.md
```
