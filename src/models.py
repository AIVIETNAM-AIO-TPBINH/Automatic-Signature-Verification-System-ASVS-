"""
Model Architectures & Transfer Learning Module.
Implements:
1. MLP Baseline (Flat 2D representation benchmark)
2. Custom 2D-CNN (4 convolutional blocks trained from scratch)
3. Pretrained CNN (ResNet-50 / ConvNeXt with flexible layer freezing and discriminative LR groups)
4. Pretrained Vision Transformer (ViT-Base / Swin-Tiny with attention block unfreezing)
5. Unified Model Factory for all 7 research configurations.
"""

from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing timm and torchvision; provide lightweight standalone fallback if unavailable
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class MLPBaseline(nn.Module):
    """
    Configuration 1: MLP Baseline.
    Flattens 3x224x224 into 150,528 dimensions.
    Used to demonstrate that destroying 2D spatial connectivity harms stroke feature learning.
    """
    def __init__(
        self,
        input_dim: int = 3 * 224 * 224,
        hidden_dims: List[int] = [512, 256, 64],
        dropout: float = 0.3
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) -> (B, 150528)
        flat = x.view(x.size(0), -1)
        logits = self.net(flat).squeeze(-1)
        return logits


class Custom2DCNN(nn.Module):
    """
    Configuration 2: Custom 4-Stage 2D CNN trained from scratch.
    Benchmarking deep learning without pretrained ImageNet weight initialization.
    """
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [32, 64, 128, 256],
        dropout: float = 0.3
    ):
        super().__init__()
        blocks = []
        prev_c = in_channels

        for c in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_c, c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout2d(dropout * 0.5)
                )
            )
            prev_c = c

        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.global_pool(feat).flatten(1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


class StandaloneResNet50Fallback(nn.Module):
    """
    A self-contained ResNet-50 implementation that runs without external torchvision/timm
    dependencies if needed, supporting exact block structure and named stages.
    """
    class Bottleneck(nn.Module):
        expansion = 4
        def __init__(self, in_planes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return self.relu(out)

    def __init__(self):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers = [self.Bottleneck(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * 4
        for _ in range(1, blocks):
            layers.append(self.Bottleneck(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class PretrainedCNNModel(nn.Module):
    """
    Configurations 3, 4, 5: Pretrained CNN (ResNet-50 / ConvNeXt).
    Supports:
    - Feature Extraction (100% frozen backbone)
    - Partial Freezing (freeze early layers 1-2, fine-tune stages 3-4)
    - Full Discriminative Fine-tuning (layer-wise learning rates)
    """
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.is_custom_fallback = False

        if HAS_TIMM:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
        elif HAS_TORCHVISION and hasattr(tv_models, backbone_name):
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            full_model = getattr(tv_models, backbone_name)(weights=weights)
            in_features = full_model.fc.in_features
            full_model.fc = nn.Identity()
            self.backbone = full_model
        else:
            self.backbone = StandaloneResNet50Fallback()
            in_features = 2048
            self.is_custom_fallback = True

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) if self.is_custom_fallback else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if self.is_custom_fallback:
            feat = self.global_pool(feat).flatten(1)
        elif feat.ndim > 2:
            feat = feat.flatten(1)
        logits = self.classifier(feat).squeeze(-1)
        return logits

    def freeze_backbone(self):
        """Freezes 100% of the backbone parameters (Linear Probing)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_last_n_stages(self, n: int = 2):
        """
        Partial freezing strategy:
        Freezes early layers (stem and early stages),
        unfreezes the last n stages and classifier head.
        """
        # First freeze all backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        if hasattr(self.backbone, "layer3") and hasattr(self.backbone, "layer4"):
            if n >= 1:
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True
            if n >= 2:
                for param in self.backbone.layer3.parameters():
                    param.requires_grad = True
        else:
            # Fallback by parameter list slicing
            params = list(self.backbone.parameters())
            cutoff = int(len(params) * (1.0 - (n * 0.25)))
            for p in params[cutoff:]:
                p.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreezes all parameters across the network."""
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_groups(
        self,
        lr_early: float = 1e-5,
        lr_mid: float = 5e-5,
        lr_head: float = 1e-3,
        weight_decay: float = 1e-2
    ) -> List[Dict[str, Any]]:
        """
        Discriminative Learning Rates:
        Assigns smaller learning rates to early general edge/corner layers,
        moderate rates to mid stages, and high rates to task-specific classifier head.
        """
        early_params = []
        mid_params = []

        if hasattr(self.backbone, "layer1"):
            early_params += list(self.backbone.conv1.parameters()) + list(self.backbone.bn1.parameters())
            early_params += list(self.backbone.layer1.parameters())
            mid_params += list(self.backbone.layer2.parameters()) + list(self.backbone.layer3.parameters())
            head_params = list(self.backbone.layer4.parameters()) + list(self.classifier.parameters())
        else:
            all_bb = list(self.backbone.parameters())
            p1 = len(all_bb) // 3
            p2 = 2 * len(all_bb) // 3
            early_params = all_bb[:p1]
            mid_params = all_bb[p1:p2]
            head_params = all_bb[p2:] + list(self.classifier.parameters())

        return [
            {"params": early_params, "lr": lr_early, "weight_decay": weight_decay},
            {"params": mid_params, "lr": lr_mid, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ]


class StandaloneViTFallback(nn.Module):
    """
    Lightweight self-contained Vision Transformer module for demonstration/fallback
    when external timm is not installed.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 768, depth: int = 6, num_heads: int = 8):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        p = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, p), dim=1) + self.pos_embed[:, :p.size(1) + 1]
        x = self.transformer(x)
        return x[:, 0] # Return CLS token embedding


class PretrainedViTModel(nn.Module):
    """
    Configurations 6, 7: Pretrained Vision Transformer (ViT-Base / Swin-Tiny).
    Captures long-range spatial stroke dependencies using global self-attention.
    """
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_name = model_name

        if HAS_TIMM:
            self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            in_features = self.encoder.num_features
        else:
            self.encoder = StandaloneViTFallback(embed_dim=768, depth=6, num_heads=8)
            in_features = 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        if feat.ndim > 2:
            feat = feat.mean(dim=1)
        logits = self.classifier(feat).squeeze(-1)
        return logits

    def freeze_backbone(self):
        """Freezes all ViT encoder attention blocks (Feature Extraction)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreezes all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_groups(
        self,
        lr_backbone: float = 1e-5,
        lr_head: float = 1e-4,
        weight_decay: float = 1e-2
    ) -> List[Dict[str, Any]]:
        return [
            {"params": list(self.encoder.parameters()), "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": list(self.classifier.parameters()), "lr": lr_head, "weight_decay": weight_decay},
        ]


def build_model(config_key: str, **kwargs) -> nn.Module:
    """
    Factory function instantiating one of the 7 research configurations.
    Keys:
        - "mlp_baseline"
        - "custom_cnn"
        - "resnet50_feature_extraction"
        - "resnet50_partial_freezing"
        - "resnet50_discriminative_finetuning"
        - "vit_base_feature_extraction"
        - "vit_base_full_finetuning"
    """
    pretrained = kwargs.pop("pretrained", True)

    if config_key == "mlp_baseline":
        return MLPBaseline(**kwargs)

    elif config_key == "custom_cnn":
        return Custom2DCNN(**kwargs)

    elif config_key in [
        "resnet50_feature_extraction",
        "resnet50_partial_freezing",
        "resnet50_discriminative_finetuning",
    ]:
        model = PretrainedCNNModel(backbone_name="resnet50", pretrained=pretrained, **kwargs)
        if config_key == "resnet50_feature_extraction":
            model.freeze_backbone()
        elif config_key == "resnet50_partial_freezing":
            model.unfreeze_last_n_stages(n=2)
        elif config_key == "resnet50_discriminative_finetuning":
            model.unfreeze_all()
        return model

    elif config_key in [
        "vit_base_feature_extraction",
        "vit_base_full_finetuning",
    ]:
        model = PretrainedViTModel(model_name="vit_base_patch16_224", pretrained=pretrained, **kwargs)
        if config_key == "vit_base_feature_extraction":
            model.freeze_backbone()
        elif config_key == "vit_base_full_finetuning":
            model.unfreeze_all()
        return model

    else:
        raise ValueError(f"Unknown configuration key: {config_key}")
