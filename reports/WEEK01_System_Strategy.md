# BÁO CÁO CHIẾN LƯỢC HỆ THỐNG & KIỂM CHỨNG DỮ LIỆU (TUẦN 01)
## Đề tài: Hệ Thống Xác Thực Chữ Ký Ngân Hàng Tự Động (AIO2026 — Module 05)

---

### 1. Phân Tích Hiện Trạng & Nghiệp Vụ Ngân Hàng
- **Thách thức:** Tại quầy giao dịch và phòng thanh toán trong nước, việc đối chiếu chữ ký trên Ủy nhiệm chi (Payment Order) và Séc hiện được thực hiện thủ công bằng mắt thường. Thời gian thẩm định trung bình mất từ 30–60 giây/chứng từ.
- **Rủi ro chữ ký giả tinh vi (Skilled Forgeries):** Kẻ gian có tập luyện sao chép cấu trúc tổng thể của chữ ký thật, rất khó nhận biết bằng mắt thường, gây rủi ro thất thoát hàng tỷ đồng.
- **Sự khan hiếm mẫu chữ ký:** Mỗi chủ tài khoản chỉ đăng ký 1–3 mẫu chữ ký gốc. Do đó, việc huấn luyện mô hình sâu từ đầu (train from scratch) cho từng khách hàng sẽ bị hiện tượng học vẹt (overfitting) nghiêm trọng.

---

### 2. Phân Tích Hình Thái Nét Chữ & Giải Mã Ảo Giác Mật Độ Mực (Ink Density Bias)
- **Thực nghiệm đo đạc & Hiện tượng ảo giác:**
  - *Trên mẫu đối chứng 100 ảnh đầu tiên:* Nếu chỉ tính mật độ mực thô trên toàn khung canvas ($\text{Density}_{\text{raw}} = \frac{\text{Số pixel mực}}{W \times H}$), chữ ký giả mạo cho kết quả trung bình $5.50\%$ ($626.4 \times 303.3$ px) so với $4.36\%$ ($604.5 \times 383.3$ px) của chữ ký thật (chênh lệch bề ngoài lên đến $+26.05\%$).
  - **Bản chất kỹ thuật:** Đây là **ảo giác đo lường do khung quét ảnh** (Canvas Size Optical Illusion). Trong bộ dữ liệu CEDAR, các ảnh giả mạo có chiều cao trung bình bị cắt ngắn hơn hẳn so với ảnh thật, dẫn đến diện tích khung mẫu số nhỏ hơn đáng kể.
- **Kiểm chứng Định lượng Khách quan trên toàn bộ 2,640 ảnh CEDAR (1,320 Thật vs 1,320 Giả):**
  - **Kích thước khung thô:** Chữ ký thật đạt trung bình $535.7 \times 355.7$ px; chữ ký giả đạt $552.0 \times 345.2$ px.
  - **Mật độ mực thô trên canvas:** Chữ ký thật là $3.71\%$; chữ ký giả là $3.56\%$ (chênh lệch chỉ **-3.83%**).
  - **Số điểm mực tuyệt đối (Absolute Ink Pixels):** Chữ ký thật đạt trung bình $6,911.2$ px; chữ ký giả đạt $6,618.4$ px. Chênh lệch thực tế chỉ vỏn vẹn **-4.24%** (hoàn toàn tương đồng).
  - **Mật độ mực trong Bounding Box sát nét chữ:** Sau khi cắt bỏ toàn bộ lề giấy thừa, mật độ nét bút của chữ ký thật là $6.70\%$, chữ ký giả là $6.21\%$ (chênh lệch **-7.41%**).
  - **Kết luận khoa học:** Bác bỏ giả thuyết suy diễn cho rằng "người giả mạo tì mạnh ngòi bút làm nét đọng mực nhiều hơn". Nét mực thực tế của chữ ký thật và chữ ký giả mạo có cùng đặc tính vật lý.
- **Minh chứng cho tính cấp thiết của Preprocessing Shield:**
  - Sai lệch viền giấy chứng minh rằng nếu nạp trực tiếp ảnh thô vào mạng nơ-ron, mô hình sẽ bị bẫy *Shortcut Learning* (học đặc trưng lề giấy thay vì nét chữ).
  - Do đó, bước **Auto-crop Bounding Box**, **Binarization Otsu** và **Resize đệm trắng đối xứng $224 \times 224$** trong Preprocessing Shield là chốt chặn tiên quyết để chuẩn hóa dữ liệu đầu vào.

---

### 3. Kiểm Chứng Bẫy Rò Rỉ Danh Tính (Identity Leakage Audit)
- **Bẫy Random Split:** Nếu trộn lẫn 30,360 cặp chữ ký rồi chia ngẫu nhiên thành Train/Val/Test, mẫu chữ ký của cùng một người sẽ xuất hiện ở cả tập Train lẫn tập Test. Mô hình sẽ học thuộc lòng hình dạng của chữ ký cụ thể thay vì học quy luật tổng quát để phân biệt nét bút thật và nét bút giả.
- **Chiến lược Phân tách Độc lập theo Người ký (Writer-Independent Split):**
  - **Tập Train (38 chủ tài khoản, ID 01 – 38):** 20,976 cặp ảnh (10,488 cặp Thật–Thật, 10,488 cặp Thật–Giả).
  - **Tập Validation (8 chủ tài khoản, ID 39 – 46):** 4,416 cặp ảnh (2,208 cặp Thật–Thật, 2,208 cặp Thật–Giả).
  - **Tập Test (9 chủ tài khoản, ID 47 – 55):** 4,968 cặp ảnh (2,484 cặp Thật–Thật, 2,484 cặp Thật–Giả).
- **Kết quả Kiểm toán:** Giao giữa các tập $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$. Độ rò rỉ danh tính là **0.00%**.

---

### 4. Quy Trình Tiền Xử Lý Hình Thái & Biểu Diễn 3 Kênh (3-Channel Stacking)
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

### 5. Tăng Cường Dữ Liệu Hợp Lệ (Data Augmentation)
- Sử dụng phép biến đổi Afin ngẫu nhiên nhẹ (`RandomAffine`):
  - Xoay nhẹ nét bút $\pm 5^\circ$
  - Co giãn nhẹ $0.95 \to 1.05$
  - Dịch chuyển nhẹ $\pm 3\%$
- **Nghiêm cấm lật ngang (Horizontal Flip) hoặc lật dọc (Vertical Flip)** vì làm phá vỡ chiều viết tự nhiên và ngữ nghĩa của nét chữ ký.
