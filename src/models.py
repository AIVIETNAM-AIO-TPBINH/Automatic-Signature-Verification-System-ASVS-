"""
Model Architectures & Transfer Learning Module.
Implements:
1. MLP Baseline (Flat 2D representation benchmark)
2. Custom 2D-CNN (4 convolutional blocks trained from scratch)
3. Pretrained CNN (ResNet-50 / ConvNeXt with flexible layer freezing and discriminative LR groups)
4. Pretrained Vision Transformer (ViT-Base / Swin-Tiny with attention block unfreezing)
5. Siamese ResNet-50 (Classical weight-sharing metric learning benchmark)
6. Unified Model Factory for all research configurations.
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
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        logits = self.network(flat).squeeze(-1)
        return logits


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, pool: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return self.pool(x)


class Custom2DCNN(nn.Module):
    """
    Configuration 2: Custom 2D-CNN (4 convolutional blocks trained from scratch).
    Evaluates representation capacity limits when trained without ImageNet priors.
    """
    def __init__(self, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),    # 224 -> 112
            ConvBlock(32, 64),             # 112 -> 56
            ConvBlock(64, 128),            # 56 -> 28
            ConvBlock(128, 256),           # 28 -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class StandaloneResNet50Fallback(nn.Module):
    """Clean ResNet-50 fallback when torchvision/timm weights are unavailable locally."""
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.num_features = 512 * Bottleneck.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class PretrainedCNNModel(nn.Module):
    """
    Configurations 3, 4, 5: Pretrained CNN (ResNet-50 / ConvNeXt).
    Supports:
    - Feature Extraction (100% frozen backbone)
    - Partial Freezing (freeze early layers 1-2, fine-tune stages 3-4)
    - Full Discriminative Fine-tuning (layer-wise learning rates)
    - Configurable classifier head: 'linear' (1 layer = 2,049 params) or 'mlp' (2 layers = 262k params).
    """
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
        head_type: str = "linear"
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.head_type = head_type
        self.is_custom_fallback = False

        loaded_backbone = False
        if HAS_TIMM:
            try:
                self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
                in_features = self.backbone.num_features
                loaded_backbone = True
            except Exception as e:
                print(f"[!] Cảnh báo timm: {e}. Đang chuyển sang torchvision fallback...")

        if not loaded_backbone and HAS_TORCHVISION and hasattr(tv_models, backbone_name):
            try:
                weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
                full_model = getattr(tv_models, backbone_name)(weights=weights)
                in_features = full_model.fc.in_features
                full_model.fc = nn.Identity()
                self.backbone = full_model
                loaded_backbone = True
            except Exception as e:
                print(f"[!] Cảnh báo torchvision: {e}")

        if not loaded_backbone:
            self.backbone = StandaloneResNet50Fallback()
            in_features = 2048
            self.is_custom_fallback = True

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) if self.is_custom_fallback else nn.Identity()

        if head_type == "linear":
            # Exact Linear Probing head: Linear(2048, 1) -> 2,049 parameters
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 1)
            )
        else:
            # Multi-layer Perceptron head: Linear(2048, 128) -> BN -> ReLU -> Dropout -> Linear(128, 1)
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
        Discriminative Learning Rate Groups:
        Assigns lower learning rates to early feature extractors,
        moderate rates to mid stages, and high rates to task-specific classifier head.
        """
        if hasattr(self.backbone, "layer1") and hasattr(self.backbone, "layer4"):
            early_params = (
                list(self.backbone.conv1.parameters()) +
                list(self.backbone.bn1.parameters()) +
                list(self.backbone.layer1.parameters())
            )
            mid_params = (
                list(self.backbone.layer2.parameters()) +
                list(self.backbone.layer3.parameters())
            )
            head_params = list(self.backbone.layer4.parameters()) + list(self.classifier.parameters())
        else:
            all_bb = list(self.backbone.parameters())
            p1 = int(len(all_bb) * 0.33)
            p2 = int(len(all_bb) * 0.66)
            early_params = all_bb[:p1]
            mid_params = all_bb[p1:p2]
            head_params = all_bb[p2:] + list(self.classifier.parameters())

        return [
            {"params": early_params, "lr": lr_early, "weight_decay": weight_decay},
            {"params": mid_params, "lr": lr_mid, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ]


class StandaloneViTFallback(nn.Module):
    """Compact Vision Transformer fallback for local CPU testing."""
    def __init__(self, embed_dim: int = 768, depth: int = 6, num_heads: int = 8):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.num_features = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_embed(x).flatten(2).transpose(1, 2)
        out = self.encoder(p)
        return out.mean(dim=1)


class PretrainedViTModel(nn.Module):
    """
    Configurations 6, 7: Pretrained Vision Transformer (ViT-Base / Swin-Tiny).
    Captures long-range spatial stroke dependencies using global self-attention.
    Supports 'linear' head (769 params) or 'mlp' head (49k params).
    """
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.1,
        head_type: str = "linear"
    ):
        super().__init__()
        self.model_name = model_name
        self.head_type = head_type

        if HAS_TIMM:
            self.encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            in_features = self.encoder.num_features
        else:
            self.encoder = StandaloneViTFallback(embed_dim=768, depth=6, num_heads=8)
            in_features = 768

        if head_type == "linear":
            # Exact Linear Probing head: Linear(768, 1) -> 769 parameters
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 1)
            )
        else:
            # Multi-layer MLP head
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


class SiameseResNet50(nn.Module):
    """
    Configuration 8 (Classical Benchmark): Weight-Sharing Siamese ResNet-50.
    Direct implementation responding to TA feedback:
    Compares the 3-channel difference tensor heuristic against the classical
    twin-branch architecture where each signature is embedded independently
    through a shared ImageNet-pretrained backbone.

    Workflow:
    1. Channel 0 (I_ref) and Channel 1 (I_query) are extracted and fed through
       the SAME shared ResNet-50 backbone:
       f_ref   = Backbone(I_ref repeated to 3-ch)   in R^2048
       f_query = Backbone(I_query repeated to 3-ch) in R^2048
    2. Computes absolute semantic distance vector: h = |f_ref - f_query| in R^2048
    3. Classification Head: maps embedding difference to verification logit.
    """
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        head_type: str = "mlp",
        combine_mode: str = "diff"
    ):
        super().__init__()
        self.head_type = head_type
        self.combine_mode = combine_mode
        self.is_custom_fallback = False

        if HAS_TIMM:
            self.backbone = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
        elif HAS_TORCHVISION and hasattr(tv_models, "resnet50"):
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            full_model = tv_models.resnet50(weights=weights)
            in_features = full_model.fc.in_features
            full_model.fc = nn.Identity()
            self.backbone = full_model
        else:
            self.backbone = StandaloneResNet50Fallback()
            in_features = 2048
            self.is_custom_fallback = True

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) if self.is_custom_fallback else nn.Identity()

        # In enhanced Siamese mode ('diff_mult'), we concatenate absolute difference and Hadamard product
        clf_in_features = in_features * 2 if combine_mode == "diff_mult" else in_features

        if head_type == "linear":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(clf_in_features, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(clf_in_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(128, 1)
            )

    def extract_features(self, img_1ch: torch.Tensor) -> torch.Tensor:
        # Replicates single channel grayscale to 3 channels to preserve ImageNet natural edge filters
        if img_1ch.shape[1] == 1:
            img = img_1ch.repeat(1, 3, 1, 1)
        else:
            img = img_1ch

        feat = self.backbone(img)
        if self.is_custom_fallback:
            feat = self.global_pool(feat).flatten(1)
        elif feat.ndim > 2:
            feat = feat.flatten(1)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard input [B, 3, 224, 224]:
        # Channel 0: I_ref
        # Channel 1: I_query
        if x.shape[1] >= 2:
            x_ref = x[:, 0:1, :, :]
            x_query = x[:, 1:2, :, :]
        else:
            raise ValueError(f"Siamese network requires at least 2 channels, received shape {x.shape}")

        f_ref = self.extract_features(x_ref)
        f_query = self.extract_features(x_query)

        # Compute feature similarity representation
        diff = torch.abs(f_ref - f_query)
        if self.combine_mode == "diff_mult":
            mult = f_ref * f_query
            combined = torch.cat([diff, mult], dim=1)
        else:
            combined = diff
        logits = self.classifier(combined).squeeze(-1)
        return logits

    def freeze_backbone(self):
        """Freezes shared backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreezes shared backbone and classifier head."""
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_groups(
        self,
        lr_early: float = 1e-5,
        lr_mid: float = 5e-5,
        lr_head: float = 1e-3,
        weight_decay: float = 1e-2
    ) -> List[Dict[str, Any]]:
        if hasattr(self.backbone, "layer1") and hasattr(self.backbone, "layer4"):
            early_params = (
                list(self.backbone.conv1.parameters()) +
                list(self.backbone.bn1.parameters()) +
                list(self.backbone.layer1.parameters())
            )
            mid_params = (
                list(self.backbone.layer2.parameters()) +
                list(self.backbone.layer3.parameters())
            )
            head_params = list(self.backbone.layer4.parameters()) + list(self.classifier.parameters())
        else:
            all_bb = list(self.backbone.parameters())
            p1 = int(len(all_bb) * 0.33)
            p2 = int(len(all_bb) * 0.66)
            early_params = all_bb[:p1]
            mid_params = all_bb[p1:p2]
            head_params = all_bb[p2:] + list(self.classifier.parameters())

        return [
            {"params": early_params, "lr": lr_early, "weight_decay": weight_decay},
            {"params": mid_params, "lr": lr_mid, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ]


def build_model(config_key: str, **kwargs) -> nn.Module:
    """
    Factory function instantiating one of the research configurations.
    Keys:
        - "mlp_baseline": MLP Baseline (Duỗi phẳng 2D)
        - "custom_cnn": Custom 2D-CNN (4 Blocks)
        - "resnet50_feature_extraction": Pretrained ResNet-50 (Khóa 100% Backbone)
        - "resnet50_partial_freezing": Pretrained ResNet-50 (Khóa Stage 1-2, Mở Stage 3-4)
        - "resnet50_discriminative_finetuning": Pretrained ResNet-50 (Phân tầng LR)
        - "vit_base_feature_extraction": Pretrained ViT-Base (Khóa 100% Transformer)
        - "vit_base_full_finetuning": Pretrained ViT-Base (Full Fine-tuning)
        - "siamese_resnet50": Siamese ResNet-50 (Shared Backbone Benchmark)
    """
    pretrained = kwargs.pop("pretrained", True)
    default_head = "linear" if "feature_extraction" in config_key else "mlp"
    head_type = kwargs.pop("head_type", default_head)

    if config_key == "mlp_baseline":
        return MLPBaseline(**kwargs)

    elif config_key == "custom_cnn":
        return Custom2DCNN(**kwargs)

    elif config_key in [
        "resnet50_feature_extraction",
        "resnet50_partial_freezing",
        "resnet50_discriminative_finetuning",
    ]:
        model = PretrainedCNNModel(
            backbone_name="resnet50",
            pretrained=pretrained,
            head_type=head_type,
            **kwargs
        )
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
        model = PretrainedViTModel(
            model_name="vit_base_patch16_224",
            pretrained=pretrained,
            head_type=head_type,
            **kwargs
        )
        if config_key == "vit_base_feature_extraction":
            model.freeze_backbone()
        elif config_key == "vit_base_full_finetuning":
            model.unfreeze_all()
        return model

    elif config_key == "siamese_resnet50":
        model = SiameseResNet50(
            pretrained=pretrained,
            head_type=head_type,
            **kwargs
        )
        model.unfreeze_all()
        return model

    else:
        raise ValueError(f"Unknown configuration key: {config_key}")
