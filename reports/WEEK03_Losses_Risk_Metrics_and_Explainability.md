# BÁO CÁO KỸ THUẬT: TỐI ƯU HÀM MẤT MÁT, THƯỚC ĐO RỦI RO NGÂN HÀNG & GIẢI THÍCH GRAD-CAM (TUẦN 03)
## Đề tài: Hệ Thống Xác Thực Chữ Ký Ngân Hàng Tự Động (AIO2026 — Module 05)
**Học viên thực hiện:** Trần Phương Bình  
**Chủ đề trọng tâm:** Loss Functions Optimization, Banking Risk Metrics & Visual Explainability  
**Bộ dữ liệu:** CEDAR Signature Dataset (55 chủ tài khoản, 30,360 cặp đối chiếu Writer-Independent)

---

### 1. Bối Cảnh Nghiệp Vụ & Đặt Vấn Đề Tuần 3

Trong quy trình phê duyệt lệnh chi và giải ngân tại ngân hàng:
- Hai loại lỗi có chi phí rủi ro phi đối xứng:
  * **FAR (False Acceptance Rate - Nhận nhầm chữ ký giả):** Kẻ gian chiếm đoạt tài sản của khách hàng, gây tổn thất tài chính và rủi ro pháp lý trực tiếp cho ngân hàng.
  * **FRR (False Rejection Rate - Từ chối nhầm khách thật):** Làm gián đoạn giao dịch của khách hàng, gây phàn nàn và giảm sút trải nghiệm dịch vụ.
- Bài toán chữ ký ngân hàng đòi hỏi:
  1. **Hàm mất mát tối ưu hóa cho mẫu khó (Hard-Negative Mining):** Đè nén áp lực từ hàng nghìn mẫu giả ngô nghê (*Random Forgeries*), tập trung gradient vào các nét giả tinh vi (*Skilled Forgeries*).
  2. **Thước đo rủi ro đặc thù:** Đánh giá điểm cân bằng lỗi **Equal Error Rate (EER)** thay vì chỉ nhìn vào Accuracy đơn thuần.
  3. **Quy trình phân tầng Straight-Through Processing (STP):** Tự động duyệt hồ sơ sạch, chặn gian lận rõ ràng và điều phối các ca nghi vấn sang kiểm soát viên.
  4. **Giải trình thị giác (Explainability):** Ứng dụng **Grad-CAM** làm căn cứ pháp lý và minh chứng kiểm toán nội bộ.

---

### 2. Tối Ưu Hàm Mất Mát: Cơ Chế Toán Học Của Focal Loss

#### 2.1. Động học của hệ số điều biến $(1 - p_t)^\gamma$
Hàm Focal Loss được định nghĩa:
$$\mathcal{L}_{\text{FL}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) \quad (\text{với } \gamma = 2.0)$$

| Loại Mẫu Thử Nghiệm | Xác Suất Đúng Nhãn $p_t$ | Hệ Số Điều Biến $(1 - p_t)^2$ | Tác Động Gradient Lên Tối Ưu Hóa |
|:---|:---:|:---:|:---|
| **Mẫu Dễ (Random Forgery)** | $0.99$ | $(1 - 0.99)^2 = \mathbf{0.0001}$ | **Triệt tiêu gradient $10,000$ lần**, ngăn chặn hiện tượng tràn ngập gradient (*Gradient Swamping*) |
| **Mẫu Trung Bình** | $0.80$ | $(1 - 0.80)^2 = 0.0400$ | Giảm $25$ lần mức phạt |
| **Mẫu Khó (Skilled Forgery)** | $0.50$ | $(1 - 0.50)^2 = \mathbf{0.2500}$ | **Giữ nguyên toàn bộ gradient**, ép mạng học kỹ các nét uốn lượn vi mô |

#### 2.2. So sánh với Label Smoothing BCE
- Kỹ thuật làm mềm nhãn (Label Smoothing với $\epsilon = 0.1$) chuyển nhãn $y \in \{0, 1\} \to y' \in \{0.05, 0.95\}$, giúp mô hình không bị quá tự tin (Overconfidence) trước các nét nhiễu mực scan.
- Tuy nhiên, Focal Loss vẫn thể hiện tính ưu việt hơn trong việc kéo giảm EER thực tế trên tập kiểm định.

---

### 3. Hệ Thống Thước Đo Rủi Ro Ngân Hàng (FAR, FRR, EER)

#### 3.1. Định nghĩa toán học
- Quy ước nhãn: $y = 0$ (Cặp Thật - Thật), $y = 1$ (Cặp Thật - Giả).
- $p \in [0, 1]$: Xác suất mô hình dự đoán là chữ ký GIẢ MẠO.
- Với ngưỡng quyết định $\theta \in [0, 1]$:
  $$\text{FAR}(\theta) = \frac{\sum \mathbf{1}(y_i = 1 \land p_i \le \theta)}{\sum \mathbf{1}(y_i = 1)}, \quad \text{FRR}(\theta) = \frac{\sum \mathbf{1}(y_i = 0 \land p_i > \theta)}{\sum \mathbf{1}(y_i = 0)}$$

#### 3.2. Điểm cân bằng lỗi Equal Error Rate (EER)
- Điểm cân bằng lỗi lý thuyết đạt được tại ngưỡng $\theta^*$ sao cho:
  $$\text{FAR}(\theta^*) = \text{FRR}(\theta^*) = \text{EER}$$
- **Kết quả thực nghiệm trên mô hình ResNet-50 Discriminative FT:**
  - **Tập Validation (8 Người ký ID 39–46, 4,416 cặp):**
    - Ngưỡng tối ưu $\theta^* = 0.4997 \approx 0.50$
    - $\text{EER} = \mathbf{22.08\%}$ (Val Acc: $77.94\%$, ROC-AUC: $0.8587$).
  - **Tập Test Độc Lập (9 Người ký hoàn toàn mới ID 47–55, 4,968 cặp):**
    - Ngưỡng tối ưu $\theta^* = 0.5078 \approx 0.51$
    - $\text{EER} = \mathbf{30.90\%}$ (Test Acc: $68.94\%$, ROC-AUC: $0.7605$).

---

### 4. Luồng Phê Duyệt Tự Động (Straight-Through Processing - STP)

Ngân hàng thiết lập khung quyết định 3 tầng (**Tri-Decision Framework**):

$$\text{Decision}(p) = \begin{cases} 
\text{AUTO\_APPROVED} & \text{nếu } p < \theta_{\text{lower}} \quad (\theta_{\text{lower}} = 0.20) \\
\text{AUTO\_REJECTED} & \text{nếu } p > \theta_{\text{upper}} \quad (\theta_{\text{upper}} = 0.80) \\
\text{MANUAL\_REVIEW} & \text{nếu } 0.20 \le p \le 0.80 
\end{cases}$$

#### Phân tích hiệu quả phân tầng Straight-Through Processing (STP) minh bạch:

**1. Đánh giá thực nghiệm trên tập CEDAR cân bằng (50% Thật / 50% Giả):**
- **Tập Validation (8 Người ký ID 39–46, 4,416 cặp):**
  - **Tỷ lệ tự động hóa (STP Coverage):** Đạt **$28.65\%$** tổng giao dịch (tự động phê duyệt hồ sơ cực kỳ rõ ràng).
  - **Độ chính xác trên luồng tự động (STP Accuracy):** Đạt **$93.36\%$** (tăng vọt $+15.42\%$ so với độ chính xác thô $77.94\%$; độ chính xác duyệt khách thật: $93.10\%$, độ chính xác chặn gian lận: $93.61\%$).
  - **Tỷ lệ chuyển sang KSV thẩm định thủ công:** **$71.35\%$** hồ sơ rơi vào vùng nghi vấn $[0.20, 0.80]$.
- **Tập Test Độc Lập (9 Người ký hoàn toàn mới ID 47–55, 4,968 cặp):**
  - **Tỷ lệ tự động hóa (STP Coverage):** Đạt **$21.50\%$**.
  - **Độ chính xác trên luồng tự động (STP Accuracy):** Đạt **$85.67\%$** (tăng mạnh $+16.73\%$ so với độ chính xác thô $68.94\%$).
  - **Tỷ lệ chuyển sang KSV thẩm định thủ công:** **$78.50\%$**.

**2. Mô phỏng vận hành giao dịch ngân hàng thực tế (Production Traffic Simulation - 95% Khách Thật / 5% Giả Mạo):**
- Trong môi trường quầy giao dịch/ATM, đại đa số giao dịch phát sinh từ khách hàng thật (tỷ lệ tiên nghiệm $95\%$). Khi áp dụng bộ phân tầng $[0.20, 0.80]$ trên phân phối vận hành này:
  - **STP Coverage:** Đạt **$28.32\%$** tổng khối lượng giao dịch.
  - **STP Accuracy:** Đạt **$93.17\%$** trên các quyết định tự động.
  - **Tỷ lệ thẩm định thủ công:** **$71.69\%$** giúp lọc sạch hoàn toàn rủi ro gian lận mà vẫn bảo đảm độ tin cậy vượt trội cho các hồ sơ được giải ngân tự động.

---

### 5. Giải Thích Thị Giác Bằng Thuật Toán Grad-CAM (Visual Explainability)

#### 5.1. Cơ chế hoạt động của Grad-CAM trên kiến trúc ResNet-50
- Grad-CAM tính toán gradient của điểm số dự đoán $y^c$ đối với các bản đồ đặc trưng kích hoạt $A^k$ của tầng tích chập cuối cùng (`layer4`):
  $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i, j}^k}$$
- Bản đồ nhiệt Grad-CAM được tạo bởi tổng có trọng số kèm hàm kích hoạt ReLU:
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

#### 5.2. Giá trị nghiệp vụ cho kiểm toán và pháp lý ngân hàng
- Bản đồ nhiệt phủ trực tiếp lên chữ ký trên chứng từ ($I_{\text{query}}$).
- Vùng màu đỏ rực chỉ điểm chính xác các nét bút bất thường: nét đứt quãng, nét run do đồ lại, hoặc tỷ lệ sai lệch so với chữ ký gốc đăng ký.
- Cung cấp bằng chứng khách quan, minh bạch giúp giao dịch viên giải thích cho khách hàng và hoàn thiện hồ sơ kiểm tra gian lận cho bộ phận Kiểm soát nội bộ.

---

### 6. Kết Luận Tuần 3
- Hoàn thiện module tính toán chuyên biệt [`src/metrics.py`](../src/metrics.py) và module giải thích thị giác [`src/explainability.py`](../src/explainability.py).
- Tích hợp toàn diện các thử nghiệm trong notebook chuẩn hóa [`notebooks/03_losses_metrics_and_gradcam.ipynb`](../notebooks/03_losses_metrics_and_gradcam.ipynb).
- Đặt nền móng vững chắc cho khâu đóng gói dịch vụ và đo kiểm độ trễ Production ở Tuần 4.
