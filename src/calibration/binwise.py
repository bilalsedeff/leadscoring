"""
Segment bazlı (Low/Medium/High) metrik hesaplama ve görselleştirme modülü.
Data leak'i önlemek için sadece validation veya test seti üzerinde kullanılmalıdır.
"""
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from omegaconf import OmegaConf
import os

def load_segment_config():
    """
    split.yaml'dan segment bilgilerini yükler
    
    Returns:
        tuple: (bin_edges, bin_labels)
    """
    config_path = Path("configs/split.yaml")
    if config_path.exists():
        config = OmegaConf.load(config_path)
        bin_edges = config.get("bin_edges", [0.0, 0.25, 0.75, 1.0])
        bin_labels = config.get("bin_labels", ["Low", "Medium", "High"])
        return bin_edges, bin_labels
    else:
        return [0.0, 0.25, 0.75, 1.0], ["Low", "Medium", "High"]

def assign_segments(probabilities, bin_edges=None, bin_labels=None):
    """
    Olasılık değerlerini segmentlere atar
    
    Args:
        probabilities: Model olasılık tahminleri
        bin_edges: Segment sınırları (None ise split.yaml'dan alınır)
        bin_labels: Segment etiketleri (None ise split.yaml'dan alınır)
        
    Returns:
        np.ndarray: Segment etiketleri
    """
    if bin_edges is None or bin_labels is None:
        bin_edges, bin_labels = load_segment_config()
        
    # Segmentlere ayır
    indices = np.digitize(probabilities, bin_edges) - 1
    return np.array(bin_labels)[np.clip(indices, 0, len(bin_labels)-1)]

def compute_metrics(y_true, p_prob):
    """
    Temel sınıflandırma metriklerini hesaplar
    
    Args:
        y_true: Gerçek değerler (0/1)
        p_prob: Model olasılık tahminleri
        
    Returns:
        dict: Metrik değerleri
    """
    return {
        "roc_auc": roc_auc_score(y_true, p_prob),
        "pr_auc": average_precision_score(y_true, p_prob),
        "brier": brier_score_loss(y_true, p_prob),
        # Ek metrikler
        "accuracy": np.mean((p_prob >= 0.5) == y_true),
        "avg_prob": np.mean(p_prob),
        "avg_true": np.mean(y_true)
    }

def lift_table(y_true, segments):
    """
    Segment bazlı lift tablosu oluşturur
    
    Args:
        y_true: Gerçek değerler (0/1)
        segments: Segment etiketleri (Low/Medium/High)
        
    Returns:
        pd.DataFrame: Segment bazlı dönüşüm oranları ve lift değerleri
    """
    df = pd.DataFrame({"y": y_true, "seg": segments})
    base = df["y"].mean()
    tbl = (df.groupby("seg")["y"]
             .agg(["count", "mean"])
             .rename(columns={"mean": "conv_rate"}))
    tbl["lift"] = tbl["conv_rate"] / base
    tbl["pct_total"] = tbl["count"] / tbl["count"].sum() * 100
    return tbl.reset_index()

def calculate_binwise_metrics(y_true, y_prob, binner=None):
    """
    Olasılık tahminlerini binner ile segmentlere ayırır ve her segment için metrikler hesaplar
    
    Args:
        y_true: Gerçek değerler (0/1)
        y_prob: Model olasılık tahminleri
        binner: ProbabilityBinner nesnesi (None ise varsayılan yapılandırma kullanılır)
        
    Returns:
        dict: Segment bazlı metrikler
    """
    # Binner oluştur (eğer verilmemişse)
    if binner is None:
        from .binner import ProbabilityBinner
        binner = ProbabilityBinner()
    
    # Segmentlere ayır
    segments = binner.assign_bins(y_prob)
    
    # Bilinen segment sırası (Low, Medium, High) - split.yaml'dan alınır
    expected_order = list(binner.labels)
    
    # Her segment için metrikler
    unique_segments = np.unique(segments)
    
    # Beklenen tüm segmentler var mı kontrol et
    # Eğer yoksa, sıfır değerlerle başlat
    bin_counts = {bin_name: 0 for bin_name in expected_order}
    conversion_rates = {bin_name: 0.0 for bin_name in expected_order}
    avg_proba = {bin_name: 0.0 for bin_name in expected_order}
    
    # Mevcut segmentler için gerçek değerleri hesapla
    for bin_name in unique_segments:
        if bin_name in expected_order:
            mask = segments == bin_name
            bin_counts[bin_name] = np.sum(mask)
            if bin_counts[bin_name] > 0:
                conversion_rates[bin_name] = np.mean(y_true[mask])
                avg_proba[bin_name] = np.mean(y_prob[mask])
    
    # Sonuçları bir sözlük olarak döndür - sabit sırayla
    return {
        'bins': np.array(expected_order),
        'bin_counts': bin_counts,
        'conversion_rates': [conversion_rates[bin_name] for bin_name in expected_order],
        'avg_proba': avg_proba
    }

def plot_lift_chart(lift_df, output_dir=None):
    """
    Lift tablosunu görselleştirir
    
    Args:
        lift_df: lift_table fonksiyonundan dönen DataFrame
        output_dir: Çıktı dizini (None ise kaydedilmez, sadece görüntülenir)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sıralama: Low, Medium, High şeklinde
    order = ["Low", "Medium", "High"] if all(x in lift_df["seg"].values for x in ["Low", "Medium", "High"]) else None
    
    # Bar plot
    sns.barplot(x="seg", y="lift", data=lift_df, order=order, ax=ax)
    
    # Dönüşüm oranlarını ekle
    for i, row in lift_df.iterrows():
        ax.text(i, row["lift"] + 0.05, f"CR: {row['conv_rate']:.2%}\n({row['count']} leads)", 
                ha='center', va='bottom')
    
    # Düz çizgi (baseline)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    
    ax.set_title("Segment Bazlı Lift Değerleri")
    ax.set_ylabel("Lift")
    ax.set_xlabel("Segment")
    
    plt.tight_layout()
    
    # Kaydet
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "segment_lift_chart.png"), dpi=300)
        plt.close()
    else:
        plt.show()

def evaluate_segments(y_true, y_prob, output_dir=None):
    """
    Segment bazlı değerlendirme yapar ve sonuçları görselleştirir
    
    Args:
        y_true: Gerçek değerler (0/1)
        y_prob: Model olasılık tahminleri
        output_dir: Çıktı dizini (None ise kaydedilmez)
        
    Returns:
        tuple: (metrics_dict, lift_df, segments)
    """
    # Genel metrikler
    metrics = compute_metrics(y_true, y_prob)
    
    # Segmentlere ayır
    segments = assign_segments(y_prob)
    
    # Lift tablosu
    lift_df = lift_table(y_true, segments)
    
    # Segment bazlı metrikler
    segment_metrics = {}
    for segment in np.unique(segments):
        mask = segments == segment
        if sum(mask) > 0:  # Eğer segmentte örnek varsa
            segment_metrics[segment] = compute_metrics(y_true[mask], y_prob[mask])
    
    # Birleştir
    metrics["segments"] = segment_metrics
    
    # Segment dağılımı
    segment_counts = pd.Series(segments).value_counts().to_dict()
    metrics["segment_counts"] = segment_counts
    
    # Görselleştirme
    if output_dir:
        # Lift tablosu
        plot_lift_chart(lift_df, output_dir)
        
        # Kalibrasyon grafiği
        plt.figure(figsize=(10, 6))
        
        # Bin sayısı
        bin_count = min(10, len(y_true) // 50)  # En az bin başına 50 örnek
        
        # Olasılık aralıklarını oluştur
        bins = np.linspace(0, 1, bin_count + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
        
        # Her bin için gerçek oranları hesapla
        bin_true = np.array([y_true[bin_indices == i].mean() if sum(bin_indices == i) > 0 else np.nan for i in range(len(bins) - 1)])
        bin_pred = np.array([(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)])
        bin_count = np.array([sum(bin_indices == i) for i in range(len(bins) - 1)])
        
        # Kalibrasyon grafiği
        plt.scatter(bin_pred, bin_true, s=bin_count/sum(bin_count)*500, alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Tahmin Edilen Olasılık')
        plt.ylabel('Gerçek Olasılık')
        plt.title('Kalibrasyon Grafiği')
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "calibration_plot.png"), dpi=300)
            plt.close()
        
        # Metrikleri kaydet
        pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, "segment_metrics.csv"), index=False)
        lift_df.to_csv(os.path.join(output_dir, "segment_lift.csv"), index=False)
    
    return metrics, lift_df, segments
