"""
Loss Functions for Signature Verification & Hard-Example Mining.
Implements:
1. Binary Cross Entropy with Logits
2. Focal Loss (gamma=2.0) for down-weighting easy samples and penalizing skilled forgeries
3. Label Smoothing BCE to combat model overconfidence
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    When gamma = 2.0, correctly classified easy examples (p_t ~ 0.99) receive near-zero loss,
    forcing the gradient update to concentrate on ambiguous and skilled forgeries.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        reduction: str = "mean",
        eps: float = 1e-8
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B,), targets: (B,) with values in {0, 1}
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # p_t: probability of true class
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Binary cross entropy per element
        bce = -torch.log(p_t)

        loss = modulating_factor * bce

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingBCE(nn.Module):
    """
    Binary Cross Entropy with Label Smoothing:
    Smooths target y -> y * (1 - eps) + 0.5 * eps.
    Prevents the model from driving logit magnitudes to extreme infinity.
    """
    def __init__(self, eps: float = 0.05, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed_targets = targets * (1.0 - self.eps) + 0.5 * self.eps
        return F.binary_cross_entropy_with_logits(logits, smoothed_targets, reduction=self.reduction)


def get_loss_function(name: str = "focal", **kwargs) -> nn.Module:
    """
    Factory function returning the configured loss function.
    """
    name_lower = name.lower()
    if name_lower in ["focal", "focal_loss"]:
        gamma = kwargs.get("gamma", 2.0)
        alpha = kwargs.get("alpha", 0.25)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif name_lower in ["label_smoothing", "ls_bce"]:
        eps = kwargs.get("eps", 0.05)
        return LabelSmoothingBCE(eps=eps)
    elif name_lower in ["bce", "bce_with_logits"]:
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss function name: {name}")

