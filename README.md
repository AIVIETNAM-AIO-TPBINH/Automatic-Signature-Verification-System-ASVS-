# Automated Banking Signature Verification System

Nghiên cứu thực nghiệm đối chứng về chuyển giao tri thức và thích nghi lệch miền (**Domain Adaptation**) sử dụng **Pretrained CNN (ResNet-50)** và **Vision Transformer (ViT-Base)** trên bộ dữ liệu chuẩn quốc tế CEDAR Signature Dataset (55 chủ tài khoản, 30,360 cặp đối chiếu).

---

## 📌 Tổng Quan
- **Subtopic:** Deep Learning — Reusing Pretrained Models (Module 05 — AIO2026).
- **Bộ dữ liệu:** CEDAR Signature Dataset (55 chủ tài khoản; mỗi người gồm 24 chữ ký thật `full_org` và 24 chữ ký giả mạo tinh vi `full_forg` $\to$ 2,640 ảnh gốc).
- **Bài toán:** Đối chiếu và xác thực chữ ký tự động trên chứng từ ngân hàng (Ủy nhiệm chi/Séc) theo chuẩn phân tách độc lập người ký (**Writer-Independent Split**: 38 Train / 8 Val / 9 Test).

---

## 📅 Tiến Độ Thực Hiện (Project Roadmap)

- [x] **Tuần 1: Khám Phá Dữ Liệu Ảnh (Image EDA), Kiểm Chứng Rò Rỉ & Pipeline Tiền Xử Lý 3 Kênh**
  - [x] Phân tích hình thái nét chữ, tỷ lệ khung hình & kiểm chứng lượng mực thực tế (*Ink Density Bias & BBox Verification*).
  - [x] Kiểm toán rò rỉ danh tính (**Zero Identity Leakage**): 38 người Train (20,976 cặp), 8 người Val (4,416 cặp), 9 người Test (4,968 cặp).
  - [x] Xây dựng bộ lá chắn tiền xử lý (*Preprocessing Shield*): Khử nền giấy Otsu $\to$ Auto-crop Bounding Box $\to$ Resize đệm viền $224 \times 224$.
  - [x] Biểu diễn hiệu số 3 kênh: $\mathbf{X}_{\text{input}} = \text{Concat}(I_{\text{ref}}, I_{\text{query}}, |I_{\text{ref}} - I_{\text{query}}|)$.
  - [x] Tăng cường dữ liệu `RandomAffine` nhẹ (bảo toàn hướng nét chữ, không lật).
  - [x] Sản phẩm bàn giao Tuần 1: `reports/WEEK01_System_Strategy.md`, `src/preprocessing.py`, `src/dataset.py`, `notebooks/01_eda_and_image_pipeline.ipynb`.
- [ ] **Tuần 2: Thực Nghiệm Chuyển Giao Tri Thức & Ma Trận 7 Cấu Hình** *(Đang thực hiện)*
- [ ] **Tuần 3: Tối Ưu Losses & Metrics, Ngưỡng STP & Giải Thích Grad-CAM** *(Kế hoạch)*
- [ ] **Tuần 4: Đo Kiểm Độ Trễ Suy Luận & Báo Cáo Kỹ Thuật Tổng Kết** *(Kế hoạch)*

---

## 🚀 Hướng Dẫn Chạy Tuần 1
```bash
# 1. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 2. Khởi chạy Jupyter Lab
jupyter lab

# 3. Mở và chạy notebook Tuần 1:
# notebooks/01_eda_and_image_pipeline.ipynb
```

---

## 📂 Cấu Trúc Thư Mục (Tiến Độ Tuần 1)
```
.
├── data/
│   ├── raw/signatures/      # Hướng dẫn cấu trúc dữ liệu ảnh gốc CEDAR (Readme.txt)
│   └── processed/           # 3 tệp cặp ảnh đối chiếu chuẩn Writer-Independent
│       ├── train_pairs.csv  # 20,976 cặp (ID 1 -> 38)
│       ├── val_pairs.csv    # 4,416 cặp (ID 39 -> 46)
│       └── test_pairs.csv   # 4,968 cặp (ID 47 -> 55)
├── notebooks/
│   └── 01_eda_and_image_pipeline.ipynb # Notebook nghiên cứu hoàn chỉnh Tuần 1
├── plans/
│   └── KẾ HOẠCH NGHIÊN CỨU.docx        # Kế hoạch nghiên cứu 4 tuần chính thức
├── reports/
│   ├── figures/             # Nơi lưu đồ thị phân tích
│   └── WEEK01_System_Strategy.md # Báo cáo chiến lược hệ thống Tuần 1
├── src/
│   ├── __init__.py
│   ├── dataset.py           # PyTorch Dataset & DataLoader phân tách Writer-Independent
│   ├── preprocessing.py     # Otsu binarization, bbox crop, padding resize, 3-ch stack
│   └── utils.py             # Tiện ích cố định seed, device auto-detect & I/O
├── .gitignore
├── requirements.txt
└── README.md
```
