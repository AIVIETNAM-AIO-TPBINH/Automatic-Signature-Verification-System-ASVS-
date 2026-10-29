"""
Utility Functions for Banking Signature Verification Project.
Provides random seed fixing, device management, checkpoint I/O, and metrics visualization.
"""

import os
import random
import json
from typing import Dict, Any, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """Sets random seed across all libraries for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detects CUDA GPU or fallback to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Saves dictionary to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """Loads dictionary from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str
) -> None:
    """Saves model and training checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Loads checkpoint weights into model and optimizer."""
    dev = device or get_device()
    checkpoint = torch.load(filepath, map_location=dev)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint

