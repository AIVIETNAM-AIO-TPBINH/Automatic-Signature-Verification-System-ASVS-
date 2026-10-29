"""
CEDAR Dataset & DataLoader Module.
Handles Writer-Independent pairs loading, affine-only data augmentation,
and batch generation for signature verification models.
"""

import os
from typing import Optional, Callable, Dict, Any, Tuple
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import (
    preprocess_single_signature,
    build_3channel_difference_tensor,
)


class RandomAffineAugmentation:
    """
    Applies gentle affine transformations (rotation +/-5 deg, slight scale, translation)
    to emulate natural human writing variations.
    Strictly avoids horizontal or vertical flips to preserve stroke direction semantics.
    """
    def __init__(
        self,
        degrees: float = 5.0,
        scale: Tuple[float, float] = (0.95, 1.05),
        translate: Tuple[float, float] = (0.03, 0.03),
        fill_color: int = 255
    ):
        self.degrees = degrees
        self.scale = scale
        self.translate = translate
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = float(np.random.uniform(-self.degrees, self.degrees))
        scale_factor = float(np.random.uniform(self.scale[0], self.scale[1]))
        max_dx = self.translate[0] * img.width
        max_dy = self.translate[1] * img.height
        dx = float(np.random.uniform(-max_dx, max_dx))
        dy = float(np.random.uniform(-max_dy, max_dy))

        # Rotate around center with white background
        rotated = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=self.fill_color)

        # Scale and pad/crop back to original size
        w_new = int(round(img.width * scale_factor))
        h_new = int(round(img.height * scale_factor))
        scaled = rotated.resize((w_new, h_new), Image.Resampling.BILINEAR)

        canvas = Image.new("L", (img.width, img.height), color=self.fill_color)
        paste_x = (img.width - w_new) // 2 + int(dx)
        paste_y = (img.height - h_new) // 2 + int(dy)
        canvas.paste(scaled, (paste_x, paste_y))

        return canvas


class CEDARSignatureDataset(Dataset):
    """
    CEDAR Signature Verification Dataset.
    Loads paired genuine-genuine and genuine-forged samples.
    Label convention:
        0: Genuine-Genuine (Match)
        1: Genuine-Forged (Mismatch / Forgery)
    """
    def __init__(
        self,
        csv_file: str,
        data_root: Optional[str] = None,
        is_train: bool = False,
        target_size: Tuple[int, int] = (224, 224),
        use_augmentation: bool = True,
        normalize_imagenet: bool = True
    ):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.data_root = data_root or ""
        self.is_train = is_train
        self.target_size = target_size
        self.normalize_imagenet = normalize_imagenet

        if is_train and use_augmentation:
            self.augmentor = RandomAffineAugmentation(degrees=5.0)
        else:
            self.augmentor = None

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, path_str: str) -> str:
        # Normalize Windows / POSIX slashes
        norm_path = path_str.replace("\\", os.sep).replace("/", os.sep)
        if os.path.exists(norm_path):
            return norm_path
        if self.data_root:
            combined = os.path.join(self.data_root, norm_path)
            if os.path.exists(combined):
                return combined
        return norm_path

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        ref_path = self._resolve_path(row["ref_path"])
        query_path = self._resolve_path(row["query_path"])
        label = float(row["label"])
        writer_id = int(row["writer_id"])

        ref_img = preprocess_single_signature(ref_path, target_size=self.target_size)
        query_img = preprocess_single_signature(query_path, target_size=self.target_size)

        if self.augmentor is not None:
            # Augment each signature independently to simulate physical signing differences
            ref_img = self.augmentor(ref_img)
            query_img = self.augmentor(query_img)

        tensor = build_3channel_difference_tensor(
            ref_img, query_img, normalize_imagenet=self.normalize_imagenet
        )

        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "writer_id": torch.tensor(writer_id, dtype=torch.long),
            "ref_path": ref_path,
            "query_path": query_path,
        }


def get_dataloaders(
    processed_dir: str = "data/processed",
    batch_size: int = 32,
    num_workers: int = 0,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoaders for train, val, and test splits.
    """
    train_csv = os.path.join(processed_dir, "train_pairs.csv")
    val_csv = os.path.join(processed_dir, "val_pairs.csv")
    test_csv = os.path.join(processed_dir, "test_pairs.csv")

    train_ds = CEDARSignatureDataset(train_csv, is_train=True, target_size=target_size)
    val_ds = CEDARSignatureDataset(val_csv, is_train=False, target_size=target_size)
    test_ds = CEDARSignatureDataset(test_csv, is_train=False, target_size=target_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader

