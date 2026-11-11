"""
Automated Banking Signature Verification System (ASVS) - Web Demo Application
Framework: Flask + Tailwind CSS + PyTorch
Author: Tran Phuong Binh (Module 05 - Deep Learning Reusing Pretrained Models)
"""

import os
import io
import time
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import torch

# Ensure local imports work reliably
BASE_DIR = Path(__file__).resolve().parent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / "scratch" / "matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from src.models import build_model
from src.preprocessing import (
    preprocess_single_signature,
    build_3channel_difference_tensor,
    encode_image_base64,
    decode_base64_image
)
from src.explainability import GradCAM
from src.utils import get_device

app = Flask(__name__)
device = get_device()
print(f"[*] Khởi động ASVS Web Demo trên thiết bị: {device}")

# Khởi tạo mô hình và nạp checkpoint tốt nhất
model = build_model("resnet50_discriminative_finetuning", pretrained=False, head_type="mlp")
checkpoint_path = BASE_DIR / "checkpoints" / "resnet50_discriminative_best.pt"
if checkpoint_path.exists():
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        print(f"[+] Đã nạp thành công checkpoint thực nghiệm: {checkpoint_path}")
    except Exception as e:
        print(f"[!] Cảnh báo nạp checkpoint: {e}")
else:
    print("[*] Sử dụng mô hình khởi tạo (chưa nạp checkpoint).")

model.eval()
model.to(device)
cam_extractor = GradCAM(model)

# Đường dẫn bộ mẫu chữ ký CEDAR
CEDAR_DIR = BASE_DIR / "data" / "raw" / "signatures"

SAMPLES = {
    # 1. Nhóm Chữ Ký Thật (Genuine)
    "genuine_01": {
        "id": "genuine_01",
        "category": "genuine",
        "category_vi": "Hợp Lệ",
        "badge_color": "emerald",
        "title": "Mẫu 01: Chủ TK 01 (Nét chuẩn mực)",
        "desc": "Cùng chủ TK ID 01 ký đăng ký mẫu và ký trên ủy nhiệm chi.",
        "ref": str(CEDAR_DIR / "full_org" / "original_1_1.png"),
        "query": str(CEDAR_DIR / "full_org" / "original_1_2.png"),
        "expected": "HỢP LỆ — PHÊ DUYỆT TỰ ĐỘNG"
    },
    "genuine_04": {
        "id": "genuine_04",
        "category": "genuine",
        "category_vi": "Hợp Lệ",
        "badge_color": "emerald",
        "title": "Mẫu 02: Chủ TK 04 (Nét uốn lượn)",
        "desc": "Chữ ký hoa mỹ của chủ TK ID 04 trên 2 hợp đồng tín dụng.",
        "ref": str(CEDAR_DIR / "full_org" / "original_4_1.png"),
        "query": str(CEDAR_DIR / "full_org" / "original_4_2.png"),
        "expected": "HỢP LỆ — PHÊ DUYỆT TỰ ĐỘNG"
    },
    "genuine_08": {
        "id": "genuine_08",
        "category": "genuine",
        "category_vi": "Hợp Lệ",
        "badge_color": "emerald",
        "title": "Mẫu 03: Chủ TK 08 (Nét thanh nét đậm)",
        "desc": "Chữ ký dài nhiều nét móc phức tạp của chủ TK ID 08.",
        "ref": str(CEDAR_DIR / "full_org" / "original_8_1.png"),
        "query": str(CEDAR_DIR / "full_org" / "original_8_2.png"),
        "expected": "HỢP LỆ — PHÊ DUYỆT TỰ ĐỘNG"
    },
    "genuine_10": {
        "id": "genuine_10",
        "category": "genuine",
        "category_vi": "Hợp Lệ",
        "badge_color": "emerald",
        "title": "Mẫu 04: Chủ TK 10 (Phong cách viết tay)",
        "desc": "Chữ ký viết họ tên đầy đủ, ổn định cao của chủ TK ID 10.",
        "ref": str(CEDAR_DIR / "full_org" / "original_10_1.png"),
        "query": str(CEDAR_DIR / "full_org" / "original_10_2.png"),
        "expected": "HỢP LỆ — PHÊ DUYỆT TỰ ĐỘNG"
    },

    # 2. Nhóm Giả Mạo Tinh Vi (Skilled Forgery)
    "skilled_01": {
        "id": "skilled_01",
        "category": "skilled",
        "category_vi": "Giả Tinh Vi",
        "badge_color": "rose",
        "title": "Mẫu 05: Giả tinh vi TK 01",
        "desc": "Kẻ gian sao chép tỉ mỉ từng nét uốn của tài khoản ID 01.",
        "ref": str(CEDAR_DIR / "full_org" / "original_1_1.png"),
        "query": str(CEDAR_DIR / "full_forg" / "forgeries_1_1.png"),
        "expected": "GIẢ MẠO — TỪ CHỐI GIAO DỊCH"
    },
    "skilled_03": {
        "id": "skilled_03",
        "category": "skilled",
        "category_vi": "Giả Tinh Vi",
        "badge_color": "rose",
        "title": "Mẫu 06: Giả tinh vi TK 03",
        "desc": "Nét đồ uốn lượn bị run rẩy và đứt đoạn ở các góc cua nét móc.",
        "ref": str(CEDAR_DIR / "full_org" / "original_3_1.png"),
        "query": str(CEDAR_DIR / "full_forg" / "forgeries_3_1.png"),
        "expected": "GIẢ MẠO — TỪ CHỐI GIAO DỊCH"
    },
    "skilled_05": {
        "id": "skilled_05",
        "category": "skilled",
        "category_vi": "Giả Tinh Vi",
        "badge_color": "rose",
        "title": "Mẫu 07: Giả tinh vi TK 05",
        "desc": "Giả mạo cấu trúc chung nhưng sai lệch góc nghiêng ngòi bút.",
        "ref": str(CEDAR_DIR / "full_org" / "original_5_1.png"),
        "query": str(CEDAR_DIR / "full_forg" / "forgeries_5_1.png"),
        "expected": "GIẢ MẠO — TỪ CHỐI GIAO DỊCH"
    },
    "skilled_08": {
        "id": "skilled_08",
        "category": "skilled",
        "category_vi": "Giả Tinh Vi",
        "badge_color": "rose",
        "title": "Mẫu 08: Giả tinh vi TK 08",
        "desc": "Kẻ gian đồ lại chữ ký dài nhưng lộ sai phân áp lực mực.",
        "ref": str(CEDAR_DIR / "full_org" / "original_8_1.png"),
        "query": str(CEDAR_DIR / "full_forg" / "forgeries_8_1.png"),
        "expected": "GIẢ MẠO — TỪ CHỐI GIAO DỊCH"
    },

    # 3. Nhóm Nghi Vấn Cần Kiểm Soát (Manual Review)
    "manual_02": {
        "id": "manual_02",
        "category": "manual",
        "category_vi": "Nghi Vấn KSV",
        "badge_color": "amber",
        "title": "Mẫu 09: Biến thiên nét ký TK 02",
        "desc": "Cùng chủ TK ID 02 nhưng ký vội biến thiên nét (chuyển KSV kèm Grad-CAM).",
        "ref": str(CEDAR_DIR / "full_org" / "original_2_1.png"),
        "query": str(CEDAR_DIR / "full_org" / "original_2_2.png"),
        "expected": "NGHI VẤN — THẨM ĐỊNH THỦ CÔNG"
    },

    # 4. Nhóm Giả Ngẫu Nhiên / Nhầm Lẫn Hồ Sơ (Random Forgery)
    "random_01": {
        "id": "random_01",
        "category": "random",
        "category_vi": "Giả Ngẫu Nhiên",
        "badge_color": "purple",
        "title": "Mẫu 10: Nhầm lẫn TK 01 vs TK 02",
        "desc": "Chữ ký của tài khoản ID 02 ký trên hồ sơ ủy nhiệm chi của ID 01.",
        "ref": str(CEDAR_DIR / "full_org" / "original_1_1.png"),
        "query": str(CEDAR_DIR / "full_org" / "original_2_1.png"),
        "expected": "GIẢ MẠO — TỪ CHỐI GIAO DỊCH"
    },
    "random_02": {
        "id": "random_02",
        "category": "random",
        "category_vi": "Giả Ngẫu Nhiên",
        "badge_color": "purple",
        "title": "Mẫu 11: Nhầm lẫn TK 03 vs TK 05",
        "desc": "Hai chữ ký hoàn toàn độc lập khác tác giả và bố cục nét.",
        "ref": str(CEDAR_DIR / "full_org" / "original_3_1.png"),
        "query": str(CEDAR_DIR / "full_org" / "original_5_1.png"),
        "expected": "GIẢ MẠO — TỪ CHỐI GIAO DỊCH"
    }
}

# Aliases for backward compatibility
SAMPLES["genuine"] = SAMPLES["genuine_01"]
SAMPLES["skilled_forgery"] = SAMPLES["skilled_01"]
SAMPLES["random_forgery"] = SAMPLES["random_01"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def list_samples():
    """Danh sách tổng hợp 11 kịch bản mẫu phục vụ kiểm thử."""
    catalog = []
    for k in [
        "genuine_01", "genuine_04", "genuine_08", "genuine_10",
        "skilled_01", "skilled_03", "skilled_05", "skilled_08",
        "manual_02", "random_01", "random_02"
    ]:
        v = SAMPLES[k]
        catalog.append({
            "id": v["id"],
            "category": v["category"],
            "category_vi": v["category_vi"],
            "badge_color": v["badge_color"],
            "title": v["title"],
            "desc": v["desc"],
            "expected": v["expected"]
        })
    return jsonify(catalog)


@app.route("/api/sample/<sample_type>", methods=["GET"])
def get_sample(sample_type):
    if sample_type not in SAMPLES:
        return jsonify({"error": f"Mẫu không hợp lệ: {sample_type}"}), 400

    info = SAMPLES[sample_type]
    if not (os.path.exists(info["ref"]) and os.path.exists(info["query"])):
        return jsonify({"error": "Không tìm thấy tệp ảnh mẫu CEDAR cục bộ."}), 404

    try:
        ref_img = Image.open(info["ref"])
        query_img = Image.open(info["query"])

        return jsonify({
            "status": "success",
            "sample_type": sample_type,
            "title": info["title"],
            "desc": info["desc"],
            "expected": info["expected"],
            "ref_base64": f"data:image/png;base64,{encode_image_base64(ref_img)}",
            "query_base64": f"data:image/png;base64,{encode_image_base64(query_img)}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/verify", methods=["POST"])
def verify_signatures():
    t_start = time.perf_counter()
    try:
        ref_img = None
        query_img = None

        # 1. Đọc dữ liệu từ multipart form hoặc JSON base64
        if "ref_file" in request.files and "query_file" in request.files:
            ref_file = request.files["ref_file"]
            query_file = request.files["query_file"]
            if ref_file.filename and query_file.filename:
                ref_img = Image.open(ref_file.stream)
                query_img = Image.open(query_file.stream)

        if ref_img is None or query_img is None:
            data = request.get_json(silent=True) or {}
            ref_b64 = data.get("ref_base64")
            query_b64 = data.get("query_base64")
            if ref_b64 and query_b64:
                ref_img = decode_base64_image(ref_b64)
                query_img = decode_base64_image(query_b64)

        if ref_img is None or query_img is None:
            return jsonify({"error": "Vui lòng cung cấp đủ cả 2 ảnh: Chữ ký mẫu gốc và Chữ ký trên chứng từ."}), 400

        # 2. Tiền xử lý Preprocessing Shield (Otsu, Auto-crop, Resize with padding)
        ref_processed = preprocess_single_signature(ref_img, target_size=(224, 224))
        query_processed = preprocess_single_signature(query_img, target_size=(224, 224))

        # 3. Tạo tensor 3 kênh: Concat(I_ref, I_query, |I_ref - I_query|)
        input_tensor = build_3channel_difference_tensor(ref_processed, query_processed).unsqueeze(0).to(device)

        # 4. Kênh chênh lệch hiển thị
        ref_arr = np.array(ref_processed.convert("L"), dtype=np.float32)
        query_arr = np.array(query_processed.convert("L"), dtype=np.float32)
        diff_arr = np.abs(ref_arr - query_arr).astype(np.uint8)
        diff_img = Image.fromarray(diff_arr, mode="L")

        # 5. Suy luận qua mô hình
        with torch.no_grad():
            logit = model(input_tensor)
            p_forged = float(torch.sigmoid(logit).cpu().item())
            similarity_score = 1.0 - p_forged

        # 6. Khung quyết định nghiệp vụ ngân hàng (Straight-Through Processing - STP)
        # Lower threshold: 0.20 | Upper threshold: 0.80
        if p_forged < 0.20:
            verdict = "GENUINE_APPROVED"
            verdict_vi = "HỢP LỆ — PHÊ DUYỆT TỰ ĐỘNG"
            stp_status = "AUTO_APPROVED"
            risk_level = "THẤP (AN TOÀN)"
            color = "green"
            badge = "bg-emerald-500"
            description = "Chữ ký chứng từ khớp chặt chẽ với chữ ký gốc đăng ký. Giao dịch được phê duyệt chi tự động."
        elif p_forged > 0.80:
            verdict = "FORGERY_REJECTED"
            verdict_vi = "GIẢ MẠO — TỪ CHỐI GIAO DỊCH"
            stp_status = "AUTO_REJECTED"
            risk_level = "CỰC KỲ NGUY HIỂM"
            color = "red"
            badge = "bg-rose-500"
            description = "Chữ ký có dấu hiệu sai lệch cấu trúc nghiêm trọng. Hệ thống tự động khóa giao dịch cảnh báo gian lận."
        else:
            verdict = "MANUAL_REVIEW"
            verdict_vi = "NGHI VẤN — THẨM ĐỊNH THỦ CÔNG"
            stp_status = "ESCALATE_AUDITOR"
            risk_level = "TRUNG BÌNH (CẦN KIỂM SOÁT)"
            color = "amber"
            badge = "bg-amber-500"
            description = "Xác suất nằm trong vùng nghi vấn rủi ro [0.20, 0.80]. Hồ sơ được chuyển sang kiểm soát viên thẩm định kèm bản đồ nhiệt Grad-CAM."

        # 7. Trích xuất bản đồ nhiệt Grad-CAM
        overlay_b64 = None
        try:
            heatmap = cam_extractor.generate_heatmap(input_tensor, target_class=1)
            overlay_pil = cam_extractor.overlay_on_image(query_processed, heatmap, alpha=0.45)
            overlay_b64 = f"data:image/png;base64,{encode_image_base64(overlay_pil)}"
        except Exception as e:
            print(f"[!] Lỗi trích xuất Grad-CAM: {e}")

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return jsonify({
            "status": "success",
            "verdict": verdict,
            "verdict_vi": verdict_vi,
            "stp_status": stp_status,
            "risk_level": risk_level,
            "color": color,
            "badge": badge,
            "description": description,
            "similarity_score": round(similarity_score * 100, 2),
            "forgery_probability": round(p_forged * 100, 2),
            "latency_ms": round(latency_ms, 2),
            "device": str(device).upper(),
            "ref_processed_base64": f"data:image/png;base64,{encode_image_base64(ref_processed)}",
            "query_processed_base64": f"data:image/png;base64,{encode_image_base64(query_processed)}",
            "diff_base64": f"data:image/png;base64,{encode_image_base64(diff_img)}",
            "gradcam_overlay_base64": overlay_b64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Lỗi hệ thống trong quá trình đối chiếu: {str(e)}"}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "HEALTHY",
        "service": "Banking Signature Verification System",
        "model": "ResNet-50 Discriminative Fine-tuning (Pretrained ImageNet-1k)",
        "device": str(device)
    })


if __name__ == "__main__":
    from threading import Timer
    import webbrowser

    port = int(os.environ.get("PORT", 5000))
    url = f"http://127.0.0.1:{port}"
    print(f"\n{'='*70}")
    print(f"[*] HỆ THỐNG XÁC THỰC CHỮ KÝ NGÂN HÀNG ĐÃ KHỞI CHẠY THÀNH CÔNG!")
    print(f"[*] Địa chỉ truy cập: {url}")
    print(f"[*] Nhấn Ctrl+C để dừng máy chủ")
    print(f"{'='*70}\n")

    # Tự động mở trình duyệt web sau khi server đã sẵn sàng nhận kết nối
    def open_browser():
        time.sleep(1.2)
        print(f"[*] Đang tự động mở trình duyệt tới {url}...")
        webbrowser.open_new(url)

    Timer(0.5, open_browser).start()
    app.run(host="127.0.0.1", port=port, debug=False)
