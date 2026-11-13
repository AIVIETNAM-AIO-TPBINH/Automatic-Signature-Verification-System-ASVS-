"""
Training Engine & Experiment Runner.
Manages epoch iteration, discriminative optimization, learning rate warmup/annealing,
validation evaluation, early stopping, and checkpoint management.
Supports Mixed Precision (AMP) for accelerated GPU training on Colab.
"""

import os
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.losses import get_loss_function

try:
    from src.metrics import evaluate_banking_metrics, find_equal_error_rate
except ImportError:
    def evaluate_banking_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """Fallback metric evaluator calculating continuous ranking metrics even without sklearn."""
        y_true = np.asarray(y_true).ravel()
        y_scores = np.asarray(y_scores).ravel()
        if len(y_true) == 0:
            return {"accuracy": 0.0, "loss": 0.0, "eer": 0.0, "macro_f1": 0.0, "roc_auc": 0.5}
        y_pred = (y_scores > threshold).astype(int)
        acc = float(np.mean(y_pred == y_true))
        
        # Continuous rank-based AUC calculation
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        if n_pos > 0 and n_neg > 0:
            order = np.argsort(y_scores)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(len(y_scores)) + 1
            u_stat = np.sum(ranks[y_true == 1]) - (n_pos * (n_pos + 1)) / 2.0
            auc = float(u_stat / (n_pos * n_neg))
        else:
            auc = 0.5

        # Compute empirical FAR/FRR grid
        ths = np.linspace(0.0, 1.0, 200)
        fars = [float(np.mean(y_scores[y_true == 1] <= th)) if n_pos > 0 else 0.0 for th in ths]
        frrs = [float(np.mean(y_scores[y_true == 0] > th)) if n_neg > 0 else 0.0 for th in ths]
        idx = int(np.argmin(np.abs(np.array(fars) - np.array(frrs))))
        eer = float((fars[idx] + frrs[idx]) / 2.0)

        return {
            "accuracy": acc,
            "loss": 0.0,
            "eer": eer,
            "macro_f1": acc,
            "roc_auc": auc,
        }


class SignatureTrainer:
    """
    Unified Trainer for Signature Verification experiments.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "signature_model",
        use_amp: bool = True
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or get_loss_function("focal")
        self.optimizer = optimizer or AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Automatic Mixed Precision for Colab GPU speedup
        self.use_amp = use_amp and (self.device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_eer": [],
            "val_f1": [],
            "val_auc": [],
        }
        self.best_val_eer = float("inf")
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt")

    def _get_autocast_context(self):
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=self.use_amp)
        return torch.cuda.amp.autocast(enabled=self.use_amp)

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            with self._get_autocast_context():
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(labels.cpu().numpy())

        epoch_loss = total_loss / len(self.train_loader.dataset)
        epoch_acc = float(np.mean((np.array(all_preds) > 0.5) == np.array(all_targets)))
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        eval_loader = loader or self.val_loader
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in eval_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with self._get_autocast_context():
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(labels.cpu().numpy())

        eval_loss = total_loss / len(eval_loader.dataset)
        metrics = evaluate_banking_metrics(np.array(all_targets), np.array(all_preds))
        metrics["loss"] = eval_loss
        return metrics

    def fit(
        self,
        epochs: int = 25,
        early_stopping_patience: int = 7,
        scheduler: Optional[Any] = None
    ) -> Dict[str, List[float]]:
        print(f"[*] Bắt đầu huấn luyện {self.model_name} trên thiết bị: {self.device} (AMP={self.use_amp})")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)

            if scheduler is not None:
                scheduler.step()

            # Record metrics
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_eer"].append(val_metrics["eer"])
            self.history["val_f1"].append(val_metrics["macro_f1"])
            self.history["val_auc"].append(val_metrics["roc_auc"])

            val_eer = val_metrics["eer"]
            time_spent = time.time() - t0

            # Checkpoint best model based on Equal Error Rate (EER) or lowest loss
            if val_eer < self.best_val_eer:
                self.best_val_eer = val_eer
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "history": self.history,
                    },
                    self.best_checkpoint_path
                )
                ckpt_str = f"[SAVED BEST EER: {val_eer:.4f}]"
            else:
                patience_counter += 1
                ckpt_str = f"(Patience: {patience_counter}/{early_stopping_patience})"

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({time_spent:.1f}s) | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.3f} "
                f"EER: {val_eer:.4f} F1: {val_metrics['macro_f1']:.3f} | {ckpt_str}"
            )

            if patience_counter >= early_stopping_patience:
                print(f"[!] Dừng sớm (Early stopping) tại epoch {epoch} do Val EER không cải thiện.")
                break

        print(f"[+] Huấn luyện hoàn tất. Checkpoint tốt nhất tại: {self.best_checkpoint_path}")
        if os.path.exists(self.best_checkpoint_path):
            print(f"[*] Tự động tái nạp trọng số epoch tốt nhất từ: {self.best_checkpoint_path}")
            ckpt = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
        return self.history

    def evaluate_test_set(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Đánh giá mô hình tốt nhất trên tập Test độc lập (Unseen Writers, ID 47-55).
        Tự động nạp lại checkpoint có EER thấp nhất trước khi suy luận.
        """
        if os.path.exists(self.best_checkpoint_path):
            ckpt = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[*] Đã nạp best checkpoint (Epoch {ckpt.get('epoch', '?')}) để thẩm định trên Test Set.")
        return self.evaluate(test_loader)
