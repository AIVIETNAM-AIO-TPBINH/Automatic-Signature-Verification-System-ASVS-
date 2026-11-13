"""
Automated Banking Signature Verification System (ASVS)
Streamlit Enterprise Application - High Contrast & Clean UI
Author: Tran Phuong Binh (Module 05: Deep Learning — Reusing Pretrained Models)
"""

import os
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import streamlit as st

# Ensure root workspace is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Setup environment
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / "scratch" / "matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from src.models import build_model
from src.preprocessing import (
    preprocess_single_signature,
    build_3channel_difference_tensor,
)
from src.explainability import GradCAM
from src.utils import get_device

# ==============================================================================
# Page Configuration & Clean High-Contrast Enterprise Styling
# ==============================================================================
st.set_page_config(
    page_title="ASVS — Hệ Thống Xác Thực Chữ Ký Ngân Hàng",
    page_icon="🖋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container clean padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Top Banking Header */
    .banking-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    }
    .banking-header h1 {
        color: #ffffff !important;
        font-size: 26px;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .banking-header p {
        color: #94a3b8 !important;
        font-size: 14px;
        margin: 0;
        line-height: 1.5;
    }

    /* Clean Card Containers */
    .card-box {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }
    .card-title {
        color: #0f172a !important;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
    }
    .card-subtitle {
        color: #64748b !important;
        font-size: 13px;
        margin-bottom: 14px;
    }

    /* Decision Banners */
    .banner-approved {
        background: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 22px 26px;
        color: #065f46;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
    }
    .banner-rejected {
        background: #fff1f2;
        border: 2px solid #f43f5e;
        border-radius: 14px;
        padding: 22px 26px;
        color: #9f1239;
        box-shadow: 0 4px 12px rgba(244, 63, 94, 0.1);
    }
    .banner-manual {
        background: #fffbeb;
        border: 2px solid #f59e0b;
        border-radius: 14px;
        padding: 22px 26px;
        color: #92400e;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);
    }

    /* Button CTA */
    div.stButton > button {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 14px 28px !important;
        width: 100% !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
        transform: translateY(-1px);
    }

    /* Metric values clarity */
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }

    /* Image frame */
    .stImage img {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Model Loading & Caching (Supports Local Checkpoint & Cloud Deployment)
# ==============================================================================
@st.cache_resource
def load_verification_model():
    device = get_device()
    ckpt_path = BASE_DIR / "checkpoints" / "resnet50_discriminative_best.pt"
    head_path = BASE_DIR / "assets" / "weights" / "classifier_head_best.pt"
    
    if ckpt_path.exists():
        model = build_model("resnet50_discriminative_finetuning", pretrained=False, head_type="mlp")
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
    elif head_path.exists():
        # Cloud deployment: Pretrained ImageNet backbone + fine-tuned classifier head
        model = build_model("resnet50_discriminative_finetuning", pretrained=True, head_type="mlp")
        head_sd = torch.load(head_path, map_location=device)
        model.classifier.load_state_dict(head_sd)
    else:
        model = build_model("resnet50_discriminative_finetuning", pretrained=True, head_type="mlp")

    model.eval()
    model.to(device)
    cam = GradCAM(model)
    return model, cam, device

model, cam_extractor, device = load_verification_model()

# ==============================================================================
# Sample Scenarios Catalog (11 Scenarios - Cloud & Local Compatible)
# ==============================================================================
def resolve_sample(name: str, folder: str = "full_org") -> str:
    p_asset = BASE_DIR / "assets" / "demo_samples" / name
    if p_asset.exists():
        return str(p_asset)
    p_cedar = BASE_DIR / "data" / "raw" / "signatures" / folder / name
    return str(p_cedar)

SAMPLES = {
    "1. [Hợp Lệ] Chủ TK 01 — Nét chuẩn mực": {
        "ref": resolve_sample("original_1_1.png", "full_org"),
        "query": resolve_sample("original_1_2.png", "full_org"),
        "type": "genuine",
        "desc": "Cùng chủ tài khoản ID 01 ký trên mẫu đăng ký và ủy nhiệm chi."
    },
    "2. [Hợp Lệ] Chủ TK 04 — Nét uốn lượn": {
        "ref": resolve_sample("original_4_1.png", "full_org"),
        "query": resolve_sample("original_4_2.png", "full_org"),
        "type": "genuine",
        "desc": "Chữ ký hoa mỹ của chủ TK ID 04 trên 2 hợp đồng tín dụng."
    },
    "3. [Hợp Lệ] Chủ TK 08 — Nét thanh nét đậm": {
        "ref": resolve_sample("original_8_1.png", "full_org"),
        "query": resolve_sample("original_8_2.png", "full_org"),
        "type": "genuine",
        "desc": "Chữ ký dài nhiều nét móc phức tạp của chủ TK ID 08."
    },
    "4. [Hợp Lệ] Chủ TK 10 — Viết tay họ tên": {
        "ref": resolve_sample("original_10_1.png", "full_org"),
        "query": resolve_sample("original_10_2.png", "full_org"),
        "type": "genuine",
        "desc": "Chữ ký viết họ tên đầy đủ, độ ổn định đường nét cao."
    },
    "5. [Giả Tinh Vi] Kẻ gian đồ nét TK 01": {
        "ref": resolve_sample("original_1_1.png", "full_org"),
        "query": resolve_sample("forgeries_1_1.png", "full_forg"),
        "type": "skilled",
        "desc": "Kẻ gian sao chép tỉ mỉ từng nét uốn của tài khoản ID 01."
    },
    "6. [Giả Tinh Vi] Kẻ gian đồ nét TK 03": {
        "ref": resolve_sample("original_3_1.png", "full_org"),
        "query": resolve_sample("forgeries_3_1.png", "full_forg"),
        "type": "skilled",
        "desc": "Nét đồ uốn lượn bị run rẩy và đứt đoạn ở các góc cua nét móc."
    },
    "7. [Giả Tinh Vi] Kẻ gian đồ nét TK 05": {
        "ref": resolve_sample("original_5_1.png", "full_org"),
        "query": resolve_sample("forgeries_5_1.png", "full_forg"),
        "type": "skilled",
        "desc": "Giả mạo cấu trúc chung nhưng sai lệch góc nghiêng ngòi bút."
    },
    "8. [Giả Tinh Vi] Kẻ gian đồ nét TK 08": {
        "ref": resolve_sample("original_8_1.png", "full_org"),
        "query": resolve_sample("forgeries_8_1.png", "full_forg"),
        "type": "skilled",
        "desc": "Kẻ gian đồ lại chữ ký dài nhưng lộ sai phân áp lực mực."
    },
    "9. [Nghi Vấn KSV] Biến thiên nét ký TK 02": {
        "ref": resolve_sample("original_2_1.png", "full_org"),
        "query": resolve_sample("original_2_2.png", "full_org"),
        "type": "manual",
        "desc": "Chữ ký thật cùng chủ TK nhưng ký vội biến thiên nét (cần KSV thẩm định)."
    },
    "10. [Giả Ngẫu Nhiên] Nhầm lẫn TK 01 vs TK 02": {
        "ref": resolve_sample("original_1_1.png", "full_org"),
        "query": resolve_sample("original_2_1.png", "full_org"),
        "type": "random",
        "desc": "Chữ ký của tài khoản ID 02 ký trên hồ sơ ủy nhiệm chi của ID 01."
    },
    "11. [Giả Ngẫu Nhiên] Nhầm lẫn TK 03 vs TK 05": {
        "ref": resolve_sample("original_3_1.png", "full_org"),
        "query": resolve_sample("original_5_1.png", "full_org"),
        "type": "random",
        "desc": "Hai chữ ký hoàn toàn độc lập khác tác giả và bố cục nét."
    }
}

# ==============================================================================
# Sidebar: Settings & Framework Info
# ==============================================================================
with st.sidebar:
    st.markdown("### 🏛️ ASVS Core Banking")
    st.caption("Automated Signature Verification System v2.0")
    
    st.markdown("---")
    st.markdown("#### ⚙️ Cấu Hình Ngưỡng STP")
    lower_th = st.slider(
        "Ngưỡng phê duyệt tự động (Lower Threshold):",
        min_value=0.05, max_value=0.40, value=0.20, step=0.05,
        help="Xác suất giả mạo dưới ngưỡng này được ngân hàng duyệt chi tức thì."
    )
    upper_th = st.slider(
        "Ngưỡng từ chối tự động (Upper Threshold):",
        min_value=0.60, max_value=0.95, value=0.80, step=0.05,
        help="Xác suất giả mạo trên ngưỡng này bị khóa giao dịch ngay lập tức."
    )
    
    st.markdown("---")
    st.markdown("#### 📑 Thông Số Hệ Thống")
    st.markdown(f"""
    - **Kiến trúc:** ResNet-50 Discriminative FT
    - **Tri thức:** Pretrained ImageNet-1k
    - **Đầu vào:** Concat(I_ref, I_query, |Diff|)
    - **Tiền xử lý:** Otsu Binarization + Auto-crop
    - **Thiết bị:** `{str(device).upper()}`
    - **Đồ án:** Module 05 Deep Learning (AIO2026)
    - **Học viên:** Trần Phương Bình
    """)

# ==============================================================================
# Header
# ==============================================================================
st.markdown("""
<div class="banking-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <h1>🖋️ Hệ Thống Xác Thực Chữ Ký Ngân Hàng Tự Động (ASVS)</h1>
            <p>Giải pháp thẩm định nét ký sinh trắc học theo khung quy trình Straight-Through Processing (STP)</p>
        </div>
        <div style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.5); padding: 6px 14px; border-radius: 30px; font-size: 13px; font-weight: 700; color: #34d399;">
            ● MÔ HÌNH RESNET-50 SẴN SÀNG
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# Preset Scenario Selection Card
# ==============================================================================
with st.container():
    st.markdown("""
    <div class="card-box">
        <div class="card-title">⚡ Thử Nghiệm Nhanh Với Bộ 11 Kịch Bản Chuẩn Ngân Hàng</div>
        <div class="card-subtitle">Chọn bất kỳ kịch bản nào dưới đây để hệ thống tự động nạp mẫu và chạy đối chiếu tức thì:</div>
    </div>
    """, unsafe_allow_html=True)
    
    preset_name = st.selectbox(
        "Danh mục 11 kịch bản mẫu CEDAR Benchmark:",
        options=["-- Tùy chọn tải ảnh bên ngoài --"] + list(SAMPLES.keys()),
        index=0,
        label_visibility="collapsed"
    )

# Session state image holders
if "ref_img" not in st.session_state:
    st.session_state.ref_img = None
if "query_img" not in st.session_state:
    st.session_state.query_img = None

if preset_name != "-- Tùy chọn tải ảnh bên ngoài --":
    p_info = SAMPLES[preset_name]
    st.session_state.ref_img = Image.open(p_info["ref"])
    st.session_state.query_img = Image.open(p_info["query"])
    st.info(f"📌 **Tình huống nghiệp vụ:** {p_info['desc']}")

# ==============================================================================
# Dual Signature Inputs (Left: Ref, Right: Query)
# ==============================================================================
col_ref, col_query = st.columns(2)

with col_ref:
    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 12px; padding: 16px; margin-bottom: 12px;">
        <b style="color: #0f172a; font-size: 15px;">1. Chữ Ký Mẫu Đăng Ký (I_ref)</b><br>
        <span style="color: #64748b; font-size: 12px;">Chữ ký lưu trong hệ thống Core Banking khi mở tài khoản.</span>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_ref = st.file_uploader(
        "Tải lên ảnh chữ ký mẫu gốc (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"],
        key="upload_ref"
    )
    if uploaded_ref is not None:
        st.session_state.ref_img = Image.open(uploaded_ref)
    
    if st.session_state.ref_img is not None:
        st.image(st.session_state.ref_img, caption="Ảnh mẫu đăng ký (I_ref)", use_container_width=True)
    else:
        st.warning("Vui lòng tải lên ảnh mẫu hoặc chọn 1 kịch bản từ danh mục.")

with col_query:
    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 12px; padding: 16px; margin-bottom: 12px;">
        <b style="color: #0f172a; font-size: 15px;">2. Chữ Ký Trên Chứng Từ (I_query)</b><br>
        <span style="color: #64748b; font-size: 12px;">Chữ ký cần xác thực trên ủy nhiệm chi, séc hoặc hợp đồng.</span>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_query = st.file_uploader(
        "Tải lên ảnh chữ ký trên chứng từ (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"],
        key="upload_query"
    )
    if uploaded_query is not None:
        st.session_state.query_img = Image.open(uploaded_query)
        
    if st.session_state.query_img is not None:
        st.image(st.session_state.query_img, caption="Ảnh chứng từ đối chiếu (I_query)", use_container_width=True)
    else:
        st.warning("Vui lòng tải lên ảnh chứng từ hoặc chọn 1 kịch bản từ danh mục.")

# ==============================================================================
# Verification Action
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)
btn_verify = st.button("🚀 TIẾN HÀNH ĐỐI CHIẾU & PHÂN TÍCH RỦI RO NGÂN HÀNG", use_container_width=True)

if btn_verify or (preset_name != "-- Tùy chọn tải ảnh bên ngoài --"):
    if st.session_state.ref_img is None or st.session_state.query_img is None:
        st.error("⚠️ Vui lòng cung cấp đầy đủ cả 2 ảnh: Chữ ký mẫu gốc và Chữ ký trên chứng từ để đối chiếu!")
    else:
        with st.spinner("Đang kích hoạt lá chắn tiền xử lý và trích xuất đặc trưng mạng ResNet-50..."):
            t_start = time.perf_counter()
            
            # 1. Preprocessing Shield (Otsu binarization, Auto-crop, Padding)
            ref_proc = preprocess_single_signature(st.session_state.ref_img, target_size=(224, 224))
            query_proc = preprocess_single_signature(st.session_state.query_img, target_size=(224, 224))
            
            # 2. 3-Channel Difference Stacking
            input_tensor = build_3channel_difference_tensor(ref_proc, query_proc).unsqueeze(0).to(device)
            
            # 3. Model Inference
            with torch.no_grad():
                logit = model(input_tensor)
                p_forged = float(torch.sigmoid(logit).cpu().item())
                similarity = 1.0 - p_forged
                
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            
            # 4. Difference Map
            ref_arr = np.array(ref_proc.convert("L"), dtype=np.float32)
            query_arr = np.array(query_proc.convert("L"), dtype=np.float32)
            diff_arr = np.abs(ref_arr - query_arr).astype(np.uint8)
            diff_img = Image.fromarray(diff_arr, mode="L")
            
            # 5. Grad-CAM Explainability
            overlay_img = None
            try:
                heatmap = cam_extractor.generate_heatmap(input_tensor, target_class=1)
                overlay_img = cam_extractor.overlay_on_image(query_proc, heatmap, alpha=0.45)
            except Exception:
                overlay_img = query_proc
                
        # 6. Decision & Verdict
        if p_forged < lower_th:
            verdict_vi = "HỢP LỆ — PHÊ DUYỆT TỰ ĐỘNG"
            stp_status = "AUTO_APPROVED"
            risk_level = "THẤP (AN TOÀN)"
            box_class = "banner-approved"
            desc_text = "Chữ ký chứng từ khớp chặt chẽ với chữ ký gốc đăng ký. Giao dịch được phê duyệt chi tự động qua cổng STP."
        elif p_forged > upper_th:
            verdict_vi = "GIẢ MẠO — TỪ CHỐI GIAO DỊCH"
            stp_status = "AUTO_REJECTED"
            risk_level = "CỰC KỲ NGUY HIỂM"
            box_class = "banner-rejected"
            desc_text = "Chữ ký có dấu hiệu sai lệch cấu trúc nghiêm trọng. Hệ thống tự động khóa giao dịch cảnh báo gian lận ngân hàng."
        else:
            verdict_vi = "NGHI VẤN — THẨM ĐỊNH THỦ CÔNG"
            stp_status = "ESCALATE_AUDITOR"
            risk_level = "TRUNG BÌNH (CẦN KIỂM SOÁT)"
            box_class = "banner-manual"
            desc_text = f"Xác suất rủi ro nằm trong vùng nghi vấn [{lower_th:.2f}, {upper_th:.2f}]. Hồ sơ được chuyển sang kiểm soát viên thẩm định kèm bản đồ nhiệt Grad-CAM."

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 KẾT QUẢ ĐỐI CHIẾU & QUYẾT ĐỊNH NGHIỆP VỤ")
        
        # Decision Banner (Crystal clear high contrast)
        st.markdown(f"""
        <div class="{box_class}">
            <div style="font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.9; margin-bottom: 6px;">
                QUY TRÌNH: {stp_status} | MỨC ĐỘ RỦI RO: {risk_level}
            </div>
            <div style="font-size: 24px; font-weight: 800; margin-bottom: 8px;">
                {verdict_vi}
            </div>
            <div style="font-size: 14px; font-weight: 500; line-height: 1.5;">
                {desc_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 4 Metric KPI Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Điểm tương đồng (Similarity)", f"{similarity * 100:.2f} %")
            st.progress(similarity)
        with m2:
            st.metric("Xác suất giả mạo (Risk Prob)", f"{p_forged * 100:.2f} %")
            st.progress(p_forged)
        with m3:
            st.metric("Quy trình ngân hàng (STP)", stp_status)
        with m4:
            st.metric("Độ trễ suy luận (SLA)", f"{latency_ms:.1f} ms")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 👁️ BẢN ĐỒ ĐỐI CHIẾU THỊ GIÁC 4 TẦNG & GRAD-CAM")
        
        # 4-Panel Visual Inspection
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.image(ref_proc, caption="1. Mẫu gốc tiền xử lý (Otsu & Crop)", use_container_width=True)
        with c2:
            st.image(query_proc, caption="2. Chứng từ tiền xử lý (Otsu & Crop)", use_container_width=True)
        with c3:
            st.image(diff_img, caption="3. Kênh sai phân vi mô |Diff|", use_container_width=True)
        with c4:
            st.image(overlay_img, caption="4. Bản đồ nhiệt Grad-CAM phủ màu", use_container_width=True)
            
        st.info("📌 **Giải thích bản đồ Grad-CAM:** Các vùng nhiệt đỏ chỉ điểm những đoạn nét uốn cong hoặc sai lệch vi mô có mức độ nghi ngờ cao nhất phục vụ công tác kiểm toán nội bộ.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 13px; color: #64748b; padding: 10px 0;">
    Đồ Án Module 05: Deep Learning — Reusing Pretrained Models (AIO2026)<br>
    Học viên thực hiện: <b>Trần Phương Bình</b> | Kiến trúc: <b>ResNet-50 Discriminative Fine-Tuning</b>
</div>
""", unsafe_allow_html=True)
