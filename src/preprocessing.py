"""
Preprocessing Shield & 3-Channel Difference Stacking Module.
Implements Otsu binarization, bounding-box auto-cropping, aspect-ratio preserved padding,
and 3-channel difference tensor composition [I_ref, I_query, |I_ref - I_query|].
Ensures strict Train-Serve consistency between training pipeline and FastAPI inference.
"""

import io
import base64
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image
import torch


def otsu_threshold(gray_img: np.ndarray) -> int:
    """
    Computes Otsu's optimal binarization threshold using pure NumPy.
    Works for grayscale images with intensity in [0, 255].
    """
    hist, _ = np.histogram(gray_img.ravel(), bins=256, range=(0, 256))
    total_pixels = gray_img.size
    current_max = 0.0
    threshold = 128

    sum_total = np.dot(np.arange(256), hist)
    weight_background = 0.0
    sum_background = 0.0

    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        # Between-class variance
        var_between = (
            weight_background
            * weight_foreground
            * ((mean_background - mean_foreground) ** 2)
        )

        if var_between > current_max:
            current_max = var_between
            threshold = t

    return int(threshold)


def clean_paper_background(
    img: Union[Image.Image, np.ndarray],
    use_otsu: bool = True
) -> Image.Image:
    """
    Converts image to grayscale and removes background noise.
    In signature verification, paper is white (~255) and ink strokes are dark (< threshold).
    """
    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            pil_img = Image.fromarray(img).convert("L")
        else:
            pil_img = Image.fromarray(img, mode="L")
    else:
        pil_img = img.convert("L")

    arr = np.array(pil_img, dtype=np.uint8)

    if use_otsu:
        t = otsu_threshold(arr)
        # Binarize: stroke pixels keep intensity or become 0, paper background becomes pure 255
        cleaned = np.where(arr > t, 255, arr)
    else:
        cleaned = arr

    return Image.fromarray(cleaned.astype(np.uint8), mode="L")


def auto_crop_signature(
    pil_img: Image.Image,
    threshold: int = 240,
    margin: int = 8
) -> Image.Image:
    """
    Cuts the bounding box tightly around the ink strokes, removing redundant margins.
    Ink pixels are defined as pixels with intensity < threshold.
    """
    arr = np.array(pil_img.convert("L"))
    stroke_mask = arr < threshold

    if not np.any(stroke_mask):
        return pil_img

    rows = np.any(stroke_mask, axis=1)
    cols = np.any(stroke_mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Expand margin slightly without exceeding boundaries
    ymin = max(0, ymin - margin)
    ymax = min(arr.shape[0] - 1, ymax + margin)
    xmin = max(0, xmin - margin)
    xmax = min(arr.shape[1] - 1, xmax + margin)

    return pil_img.crop((xmin, ymin, xmax + 1, ymax + 1))


def resize_with_padding(
    pil_img: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    pad_color: int = 255
) -> Image.Image:
    """
    Resizes image to target_size (width, height) preserving aspect ratio
    and padding remaining space with white paper color.
    """
    w_target, h_target = target_size
    w_orig, h_orig = pil_img.size

    if w_orig == 0 or h_orig == 0:
        return Image.new("L", target_size, color=pad_color)

    scale = min(w_target / w_orig, h_target / h_orig)
    w_new = int(round(w_orig * scale))
    h_new = int(round(h_orig * scale))

    w_new = max(1, min(w_target, w_new))
    h_new = max(1, min(h_target, h_new))

    resized = pil_img.resize((w_new, h_new), Image.Resampling.BILINEAR)

    # Paste into centered canvas
    new_canvas = Image.new("L", target_size, color=pad_color)
    paste_x = (w_target - w_new) // 2
    paste_y = (h_target - h_new) // 2
    new_canvas.paste(resized, (paste_x, paste_y))

    return new_canvas


def preprocess_single_signature(
    image_input: Union[str, bytes, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    use_otsu: bool = True,
    auto_crop: bool = True
) -> Image.Image:
    """
    Full single-signature processing shield:
    1. Read input (path, bytes, or Image)
    2. Background cleaning via Otsu threshold
    3. Auto-crop around bounding box
    4. Aspect-ratio preserved resize with white padding to target_size
    """
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    cleaned = clean_paper_background(img, use_otsu=use_otsu)
    if auto_crop:
        cleaned = auto_crop_signature(cleaned)
    final_img = resize_with_padding(cleaned, target_size=target_size)
    return final_img


def build_3channel_difference_tensor(
    ref_img: Image.Image,
    query_img: Image.Image,
    normalize_imagenet: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    Constructs the 3-channel input representation:
        X = Concat(I_ref, I_query, |I_ref - I_query|) in R^(3 x 224 x 224)
    Normalized by ImageNet statistics to match Pretrained model weights.
    """
    ref_arr = np.array(ref_img.convert("L"), dtype=np.float32) / 255.0
    query_arr = np.array(query_img.convert("L"), dtype=np.float32) / 255.0

    diff_arr = np.abs(ref_arr - query_arr)

    # Shape: (3, H, W)
    stacked = np.stack([ref_arr, query_arr, diff_arr], axis=0)
    tensor = torch.from_numpy(stacked)

    if normalize_imagenet:
        mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean_t) / std_t

    return tensor


def preprocess_pair(
    ref_input: Union[str, bytes, Image.Image],
    query_input: Union[str, bytes, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
    normalize_imagenet: bool = True
) -> torch.Tensor:
    """
    High-level entry point used in both training dataset and production FastAPI service.
    Guarantees strict train-serve consistency.
    """
    ref_processed = preprocess_single_signature(ref_input, target_size=target_size)
    query_processed = preprocess_single_signature(query_input, target_size=target_size)
    return build_3channel_difference_tensor(
        ref_processed, query_processed, normalize_imagenet=normalize_imagenet
    )


def decode_base64_image(b64_str: str) -> Image.Image:
    """Decodes a base64 encoded string to PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    image_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_data))


def encode_image_base64(img: Image.Image) -> str:
    """Encodes PIL Image to base64 PNG string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

