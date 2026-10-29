# BÁO CÁO CHIẾN LƯỢC HỆ THỐNG & KIỂM CHỨNG DỮ LIỆU (TUẦN 01)
## Đề tài: Hệ Thống Xác Thực Chữ Ký Ngân Hàng Tự Động (AIO2026 — Module 05)

---

### 1. Phân Tích Hiện Trạng & Nghiệp Vụ Ngân Hàng
- **Thách thức:** Tại quầy giao dịch và phòng thanh toán trong nước, việc đối chiếu chữ ký trên Ủy nhiệm chi (Payment Order) và Séc hiện được thực hiện thủ công bằng mắt thường. Thời gian thẩm định trung bình mất từ 30–60 giây/chứng từ.
- **Rủi ro chữ ký giả tinh vi (Skilled Forgeries):** Kẻ gian có tập luyện sao chép cấu trúc tổng thể của chữ ký thật, rất khó nhận biết bằng mắt thường, gây rủi ro thất thoát hàng tỷ đồng.
- **Sự khan hiếm mẫu chữ ký:** Mỗi chủ tài khoản chỉ đăng ký 1–3 mẫu chữ ký gốc. Do đó, việc huấn luyện mô hình sâu từ đầu (train from scratch) cho từng khách hàng sẽ bị hiện tượng học vẹt (overfitting) nghiêm trọng.

---

### 2. Kiểm Chứng Bẫy Rò Rỉ Danh Tính (Identity Leakage Audit)
- **Bẫy Random Split:** Nếu trộn lẫn 30,360 cặp chữ ký rồi chia ngẫu nhiên thành Train/Val/Test, mẫu chữ ký của cùng một người sẽ xuất hiện ở cả tập Train lẫn tập Test. Mô hình sẽ học thuộc lòng hình dạng của chữ ký cụ thể thay vì học quy luật tổng quát để phân biệt nét bút thật và nét bút giả.
- **Chiến lược Phân tách Độc lập theo Người ký (Writer-Independent Split):**
  - **Tập Train (38 chủ tài khoản, ID 01 – 38):** 20,976 cặp ảnh (10,488 cặp Thật–Thật, 10,488 cặp Thật–Giả).
  - **Tập Validation (8 chủ tài khoản, ID 39 – 46):** 4,416 cặp ảnh (2,208 cặp Thật–Thật, 2,208 cặp Thật–Giả).
  - **Tập Test (9 chủ tài khoản, ID 47 – 55):** 4,968 cặp ảnh (2,484 cặp Thật–Thật, 2,484 cặp Thật–Giả).
- **Kết quả Kiểm toán:** Giao giữa các tập $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$. Độ rò rỉ danh tính là **0.00%**.

---

### 3. Quy Trình Tiền Xử Lý Hình Thái & Biểu Diễn 3 Kênh (3-Channel Stacking)
Để loại bỏ nhiễu nền giấy và tái sử dụng trọng số ImageNet Pretrained mà không phá vỡ cấu trúc input 3 kênh RGB:
1. **Khử nền giấy bằng thuật toán Otsu:** Xác định ngưỡng nhị phân tự động tách mực viết khỏi sợi giấy và nền màu.
2. **Auto-crop Bounding Box:** Tự động cắt sát biên các nét chữ có mực, loại bỏ lề giấy dư thừa.
3. **Resize Bảo toàn Tỷ lệ & Đệm viền:** Đưa về kích thước chuẩn $224 \times 224$ với màu đệm trắng (255).
4. **Xếp chồng Hiệu số 3 Kênh:**
   $$\mathbf{X}_{\text{input}} = \text{Concat}\Big( I_{\text{ref}}, \; I_{\text{query}}, \; \vert{}I_{\text{ref}} - I_{\text{query}}\vert{} \Big) \in \mathbb{R}^{3 \times 224 \times 224}$$
   - Kênh 1 ($I_{\text{ref}}$): Chữ ký mẫu gốc đăng ký tại ngân hàng.
   - Kênh 2 ($I_{\text{query}}$): Chữ ký trên chứng từ giao dịch thực tế.
   - Kênh 3 ($\vert{}I_{\text{ref}} - I_{\text{query}}\vert{}$): Bản đồ sai lệch không gian nét bút, trực tiếp kích hoạt các bộ lọc phát hiện cạnh sai khác của mạng Pretrained.

---

### 4. Tăng Cường Dữ Liệu Hợp Lệ (Data Augmentation)
- Sử dụng phép biến đổi Afin ngẫu nhiên nhẹ (`RandomAffine`):
  - Xoay nhẹ nét bút $\pm 5^\circ$
  - Co giãn nhẹ $0.95 \to 1.05$
  - Dịch chuyển nhẹ $\pm 3\%$
- **Nghiêm cấm lật ngang (Horizontal Flip) hoặc lật dọc (Vertical Flip)** vì làm phá vỡ chiều viết tự nhiên và ngữ nghĩa của nét chữ ký.

