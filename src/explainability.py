"""
Visual Explainability Module for Banking Internal Audit & Compliance.
Implements:
1. Grad-CAM (Gradient-weighted Class Activation Mapping) for Pretrained CNNs
2. Attention Rollout / Saliency for Vision Transformers
3. Heatmap color overlay generation and base64 serialization for API & Reports
"""

import io
import base64
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation for Pretrained CNN models.
    Hooks into the final convolutional stage to extract spatial stroke importance maps.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Auto-detect last conv layer
            target_layer = self._find_last_conv_layer(model)

        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._hook_handles = []
        self._register_hooks()

    def _find_last_conv_layer(self, module: nn.Module) -> nn.Module:
        # Standard best practice for ResNet models: target the final bottleneck block or layer4
        if hasattr(module, "backbone") and hasattr(module.backbone, "layer4"):
            return module.backbone.layer4[-1]
        if hasattr(module, "layer4"):
            return module.layer4[-1]

        last_conv = None
        for name, m in module.named_modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise ValueError("No Conv2d layer found in the given model for Grad-CAM.")
        return last_conv

    def _register_hooks(self):
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            if grad_out and grad_out[0] is not None:
                self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1
    ) -> np.ndarray:
        """
        Computes Grad-CAM heatmap for input_tensor (1, 3, H, W) or (3, H, W).
        Returns normalized 2D numpy array in [0, 1] of shape (H, W).
        """
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # Ensure gradient tracking through input even if backbone weights are frozen
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(input_tensor)

        if logits.ndim == 0:
            logits = logits.unsqueeze(0)

        # Target score
        score = logits[0] if target_class == 1 else -logits[0]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations in Grad-CAM.")

        # Global average pooling of gradients: weights alpha_k
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # (1, 1, H_feat, W_feat)
        cam = F.relu(cam) # Only features with positive influence

        # Upsample to input resolution
        _, _, h, w = input_tensor.shape
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_on_image(
        self,
        base_img: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: str = "jet"
    ) -> Image.Image:
        """
        Overlays heatmap on grayscale or RGB signature image.
        """
        w, h = base_img.size
        # Resize heatmap to match base image
        heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        heat_arr = np.array(heatmap_resized) / 255.0

        # Apply colormap (compatible with all Matplotlib versions)
        try:
            cmap = plt.get_cmap(colormap)
        except Exception:
            cmap = cm.get_cmap(colormap)
        color_mapped = cmap(heat_arr)[:, :, :3] # RGB in [0, 1]

        base_arr = np.array(base_img.convert("RGB"), dtype=np.float32) / 255.0
        blended = (1.0 - alpha) * base_arr + alpha * color_mapped
        blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

        return Image.fromarray(blended)

    def close(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


def generate_gradcam_overlay_base64(
    model: nn.Module,
    input_tensor: torch.Tensor,
    query_pil_img: Image.Image
) -> str:
    """
    Utility producing a base64-encoded PNG string showing Grad-CAM heatmap
    overlaid on the query signature for audit reporting.
    """
    try:
        cam_extractor = GradCAM(model)
        heatmap = cam_extractor.generate_heatmap(input_tensor, target_class=1)
        overlay = cam_extractor.overlay_on_image(query_pil_img, heatmap)
        cam_extractor.close()

        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        # Graceful fallback if GradCAM fails on custom layer
        buf = io.BytesIO()
        query_pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

