"""
Reusable metric set for Lead-Scoring.

• ROC / PR AUC
• KS
• Brier & Expected-Calibration-Error
• Lift@k  (absolute or top-% cut)
• Segment lift table  (Low / Medium / High)

All functions are NumPy-vektörize -- büyük setlerde çok hızlıdır.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import metrics
from omegaconf import OmegaConf
from pathlib import Path
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# classical metrics
# -------------------------------------------------------------------------
def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return metrics.roc_auc_score(y_true, y_prob)


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return metrics.average_precision_score(y_true, y_prob)


def ks_stat(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    return np.max(np.abs(tpr - fpr))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return np.mean((y_true - y_prob) ** 2).item()


# -------------------------------------------------------------------------
# calibration – equal-frequency binning  (aka histogram-bin ECE)
# -------------------------------------------------------------------------
def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = None
) -> float:
    """
    ECE (lower is better).
    
    Args:
        y_true: Gerçek değerler
        y_prob: Olasılık tahminleri
        n_bins: Bin sayısı. None ise split.yaml'dan bin_edges kullanılır.
        
    Returns:
        float: Expected Calibration Error değeri
    """
    # Varsayılan n_bins değeri
    if n_bins is None:
        n_bins = 10
    
    # split.yaml'dan bin_edges değerlerini al
    try:
        config_path = Path("configs/split.yaml")
        if config_path.exists() and n_bins == 10:  # Değiştirilmemişse varsayılan değeri kullan
            config = OmegaConf.load(config_path)
            bin_edges = config.get("bin_edges", None)
            if bin_edges:
                df = pd.DataFrame({"y": y_true, "p": y_prob})
                df["bin"] = pd.cut(df["p"], bins=bin_edges, include_lowest=True)
                grouped = df.groupby("bin")
                avg_p = grouped["p"].mean()
                avg_y = grouped["y"].mean()
                ece = np.sum(grouped.size() * np.abs(avg_p - avg_y)) / len(df)
                return ece.item()
    except Exception as e:
        print(f"ECE hesaplanırken hata: {e}")
    
    # Varsayılan hesaplama (split.yaml yoksa veya bin_edges yoksa veya bir hata oluştuysa)
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    grouped = df.groupby("bin")
    avg_p = grouped["p"].mean()
    avg_y = grouped["y"].mean()
    ece = np.sum(grouped.size() * np.abs(avg_p - avg_y)) / len(df)
    return ece.item()


# -------------------------------------------------------------------------
# Lift utilities
# -------------------------------------------------------------------------
def lift_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: float | int = 0.1,
) -> float:
    """
    Lift at top-k.

    Parameters
    ----------
    k : float in (0,1]  => top-percentage
        int  >= 1        => absolute number of rows
    """
    assert 0 < k <= 1 or isinstance(k, int)
    n = len(y_true)
    top_n = int(k * n) if isinstance(k, float) else int(k)
    idx = np.argsort(y_prob)[::-1][:top_n]
    cr_top = y_true[idx].mean()
    cr_base = y_true.mean()
    return (cr_top / cr_base).item() if cr_base > 0 else np.nan


def lift_by_segment(
    y_true: np.ndarray,
    segment: np.ndarray,  # Low/Medium/High
) -> pd.DataFrame:
    """Return a table with conversion-rate and lift vs. base per segment."""
    base = y_true.mean()
    df = pd.DataFrame({"y": y_true, "seg": segment})
    tbl = (
        df.groupby("seg")["y"]
        .agg(["count", "mean"])
        .rename(columns={"count": "lead_cnt", "mean": "conv_rate"})
    )
    tbl["lift"] = tbl["conv_rate"] / base
    return tbl.reset_index()


# -------------------------------------------------------------------------
# full evaluator (dict)  – for MLflow logging
# -------------------------------------------------------------------------
def evaluate_all(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    segment: np.ndarray | None = None,
) -> dict:
    res = {
        "roc_auc": roc_auc(y_true, y_prob),
        "pr_auc": pr_auc(y_true, y_prob),
        "ks": ks_stat(y_true, y_prob),
        "brier": brier(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "lift_at_10pct": lift_at_k(y_true, y_prob, 0.10),
        "lift_at_5pct": lift_at_k(y_true, y_prob, 0.05),
    }
    if segment is not None:
        seg_tbl = lift_by_segment(y_true, segment)
        for _, row in seg_tbl.iterrows():
            seg = row["seg"].lower()
            res[f"lift_{seg}"] = row["lift"]
            res[f"cr_{seg}"] = row["conv_rate"]
    return {k: float(v) for k, v in res.items()}


# -------------------------------------------------------------------------
# Classification Metrics
# -------------------------------------------------------------------------
def calculate_classification_metrics(y_true, y_prob, threshold=0.5):
    """
    Çeşitli sınıflandırma metriklerini hesaplar.
    
    Args:
        y_true: Gerçek değerler (0/1)
        y_prob: Olasılık tahminleri (0-1 arası)
        threshold: Sınıflandırma eşiği (varsayılan: 0.5)
        
    Returns:
        dict: Metrik değerleri
    """
    # Tahmin
    y_pred = (y_prob >= threshold).astype(int)
    
    # Temel metrikler
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC ve PR-AUC
    roc_auc_value = metrics.roc_auc_score(y_true, y_prob)
    pr_auc_value = metrics.average_precision_score(y_true, y_prob)
    
    # Confusion matrix
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Sonuçları sözlük olarak döndür
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,  # Sensitivity, TPR
        'specificity': specificity,  # TNR
        'f1': f1,
        'roc_auc': roc_auc_value,
        'pr_auc': pr_auc_value,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


# -------------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------------
def plot_roc_curve(y_true, y_prob, title="ROC Curve"):
    """
    ROC eğrisini çizer.
    
    Args:
        y_true: Gerçek değerler (0/1)
        y_prob: Olasılık tahminleri (0-1 arası)
        title: Grafik başlığı
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    roc_auc = metrics.roc_auc_score(y_true, y_prob)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    return fpr, tpr, roc_auc


def plot_precision_recall_curve(y_true, y_prob, title="Precision-Recall Curve"):
    """
    Precision-Recall eğrisini çizer.
    
    Args:
        y_true: Gerçek değerler (0/1)
        y_prob: Olasılık tahminleri (0-1 arası)
        title: Grafik başlığı
    """
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)
    pr_auc = metrics.average_precision_score(y_true, y_prob)
    
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'{title} (AP = {pr_auc:.3f})')
    
    return precision, recall, pr_auc
