"""
Banking Signature Verification Metrics & STP Decision Optimizer.
Implements:
1. Standard classification metrics: Accuracy, Precision, Recall, Macro-F1, ROC-AUC
2. Banking Risk Metrics:
   - FAR (False Acceptance Rate - Forgery accepted as genuine)
   - FRR (False Rejection Rate - Genuine customer signature rejected)
   - EER (Equal Error Rate where FAR == FRR)
3. Straight-Through Processing (STP) Threshold Optimizer & Coverage-Accuracy Trade-off
"""

from typing import Dict, Any, Tuple, List
import numpy as np
try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve
    )
except (ImportError, Exception):
    def accuracy_score(y_true, y_pred):
        y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
        return float(np.mean(y_true == y_pred))

    def precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0):
        y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        fn = float(np.sum((y_true == 1) & (y_pred == 0)))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1, None

    def roc_auc_score(y_true, y_scores):
        y_true, y_scores = np.asarray(y_true).ravel(), np.asarray(y_scores).ravel()
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        if n_pos == 0 or n_neg == 0:
            return 0.5
        order = np.argsort(y_scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(y_scores)) + 1
        u_stat = np.sum(ranks[y_true == 1]) - (n_pos * (n_pos + 1)) / 2.0
        return float(u_stat / (n_pos * n_neg))

    def roc_curve(y_true, y_scores):
        thresholds = np.linspace(1.0, 0.0, 100)
        tpr = [float(np.mean((y_scores >= th)[y_true == 1])) for th in thresholds]
        fpr = [float(np.mean((y_scores >= th)[y_true == 0])) for th in thresholds]
        return np.array(fpr), np.array(tpr), thresholds


def compute_far_frr_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    num_thresholds: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes FAR and FRR curves across a grid of thresholds in [0, 1].
    Convention:
        y_true == 0: Genuine-Genuine pair
        y_true == 1: Genuine-Forged pair
        y_scores in [0, 1]: Probability that the signature is FORGED (class 1).
        If score > threshold -> Predicted FORGED (Rejected).
        If score <= threshold -> Predicted GENUINE (Accepted).

    FAR: True label is 1 (Forged), but score <= threshold (Accepted).
    FRR: True label is 0 (Genuine), but score > threshold (Rejected).
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores).ravel()

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    far_list = []
    frr_list = []

    total_forged = int(np.sum(y_true == 1))
    total_genuine = int(np.sum(y_true == 0))

    if total_forged == 0 or total_genuine == 0:
        # Graceful fallback when evaluating a single-class batch
        return thresholds, np.zeros_like(thresholds), np.zeros_like(thresholds)

    for th in thresholds:
        # False Accept: actual forged (1), predicted genuine (score <= th)
        fa = np.sum((y_true == 1) & (y_scores <= th))
        # False Reject: actual genuine (0), predicted forged (score > th)
        fr = np.sum((y_true == 0) & (y_scores > th))

        far_list.append(fa / total_forged)
        frr_list.append(fr / total_genuine)

    return thresholds, np.array(far_list), np.array(frr_list)


def find_equal_error_rate(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[float, float, float]:
    """
    Finds the Equal Error Rate (EER) and the optimal operational threshold theta*.
    Returns:
        (eer, optimal_threshold, far_at_threshold)
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores).ravel()

    total_forged = int(np.sum(y_true == 1))
    total_genuine = int(np.sum(y_true == 0))
    if total_forged == 0 or total_genuine == 0:
        return 0.5, 0.5, 0.5

    thresholds, far, frr = compute_far_frr_curve(y_true, y_scores, num_thresholds=2000)
    diff = np.abs(far - frr)
    best_idx = int(np.argmin(diff))

    optimal_threshold = float(thresholds[best_idx])
    eer = float((far[best_idx] + frr[best_idx]) / 2.0)
    far_at_th = float(far[best_idx])

    return eer, optimal_threshold, far_at_th


def evaluate_banking_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculates full comprehensive metrics for research report and benchmarking.
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores).ravel()

    y_pred = (y_scores > threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = 0.5

    total_forged = np.sum(y_true == 1)
    total_genuine = np.sum(y_true == 0)

    fa = np.sum((y_true == 1) & (y_scores <= threshold))
    fr = np.sum((y_true == 0) & (y_scores > threshold))

    far = float(fa / total_forged) if total_forged > 0 else 0.0
    frr = float(fr / total_genuine) if total_genuine > 0 else 0.0

    eer, optimal_th, _ = find_equal_error_rate(y_true, y_scores)

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(auc),
        "far": float(far),
        "frr": float(frr),
        "eer": float(eer),
        "optimal_threshold": float(optimal_th),
        "eval_threshold": float(threshold),
    }


def compute_stp_coverage_accuracy(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    lower_th: float = 0.20,
    upper_th: float = 0.80
) -> Dict[str, float]:
    """
    Simulates Straight-Through Processing (STP) in banking operations:
    - Auto-Approved (Genuine): score < lower_th -> pred = 0
    - Auto-Rejected (Fraud): score > upper_th  -> pred = 1
    - Manual Review: lower_th <= score <= upper_th (Queued for Teller/Officer verification)
    """
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores).ravel()
    n_total = len(y_true)

    if n_total == 0:
        return {
            "stp_coverage": 0.0,
            "stp_accuracy": 1.0,
            "auto_approve_rate": 0.0,
            "auto_reject_rate": 0.0,
            "manual_review_rate": 1.0,
            "auto_approve_accuracy": 1.0,
            "auto_reject_accuracy": 1.0,
            "lower_th": lower_th,
            "upper_th": upper_th,
        }

    approve_mask = (y_scores < lower_th)
    reject_mask = (y_scores > upper_th)
    auto_mask = approve_mask | reject_mask

    coverage = float(np.mean(auto_mask))
    manual_review_rate = float(1.0 - coverage)
    auto_approve_rate = float(np.mean(approve_mask))
    auto_reject_rate = float(np.mean(reject_mask))

    # Auto-approve accuracy: of those auto-approved (pred=0), fraction where y_true == 0
    n_approved = int(np.sum(approve_mask))
    if n_approved > 0:
        auto_approve_acc = float(np.mean(y_true[approve_mask] == 0))
    else:
        auto_approve_acc = 1.0

    # Auto-reject accuracy: of those auto-rejected (pred=1), fraction where y_true == 1
    n_rejected = int(np.sum(reject_mask))
    if n_rejected > 0:
        auto_reject_acc = float(np.mean(y_true[reject_mask] == 1))
    else:
        auto_reject_acc = 1.0

    if np.sum(auto_mask) > 0:
        auto_true = y_true[auto_mask]
        auto_scores = y_scores[auto_mask]
        auto_pred = (auto_scores > 0.5).astype(int)
        auto_accuracy = float(accuracy_score(auto_true, auto_pred))
    else:
        auto_accuracy = 1.0

    return {
        "stp_coverage": coverage,
        "stp_accuracy": auto_accuracy,
        "auto_approve_rate": auto_approve_rate,
        "auto_reject_rate": auto_reject_rate,
        "manual_review_rate": manual_review_rate,
        "auto_approve_accuracy": auto_approve_acc,
        "auto_reject_accuracy": auto_reject_acc,
        "lower_th": lower_th,
        "upper_th": upper_th,
    }


def simulate_production_traffic_stp(
    genuine_scores: np.ndarray,
    forged_scores: np.ndarray,
    genuine_prior: float = 0.95,
    lower_th: float = 0.20,
    upper_th: float = 0.80,
    num_samples: int = 20000,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Simulates production banking operational traffic where genuine customer transactions
    dominate (e.g., 95% genuine, 5% forgery attempts).
    Draws bootstrap samples from actual model prediction score pools.
    """
    rng = np.random.RandomState(random_seed)
    n_genuine = int(num_samples * genuine_prior)
    n_forged = num_samples - n_genuine

    sample_gen_scores = rng.choice(genuine_scores, size=n_genuine, replace=True)
    sample_forg_scores = rng.choice(forged_scores, size=n_forged, replace=True)

    sim_y_true = np.concatenate([np.zeros(n_genuine, dtype=int), np.ones(n_forged, dtype=int)])
    sim_scores = np.concatenate([sample_gen_scores, sample_forg_scores])

    # Shuffle
    perm = rng.permutation(num_samples)
    sim_y_true = sim_y_true[perm]
    sim_scores = sim_scores[perm]

    metrics = compute_stp_coverage_accuracy(sim_y_true, sim_scores, lower_th, upper_th)
    metrics["genuine_prior"] = genuine_prior
    return metrics

