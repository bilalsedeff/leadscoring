import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from ..calibration.binwise import evaluate_segments, lift_table, plot_lift_chart

def build_pdf(y_true, p_cal, segments, out_path):
    """
    Model performans raporunu oluşturur ve kaydeder
    
    Args:
        y_true: Gerçek değerler (0/1)
        p_cal: Kalibre edilmiş olasılık tahminleri
        segments: Segment etiketleri (Low/Medium/High)
        out_path: Çıktı dizini
    """
    # Çıktı dizinini oluştur
    out_dir = Path(out_path).parent
    os.makedirs(out_dir, exist_ok=True)
    
    # Segment bazlı değerlendirme yap
    metrics, lift_df, _ = evaluate_segments(y_true, p_cal, out_dir)
    
    # Ana metrikleri kaydet
    metrics_df = pd.DataFrame({
        'Metric': ['ROC-AUC', 'PR-AUC', 'Brier Score', 'Accuracy'],
        'Value': [
            metrics['roc_auc'],
            metrics['pr_auc'],
            metrics['brier'],
            metrics['accuracy']
        ]
    })
    metrics_df.to_csv(out_dir / "model_metrics.csv", index=False)
    
    # Segment tablosunu kaydet
    lift_table(y_true, segments).to_csv(out_dir / "segment_metrics.csv", index=False)
    
    # Görselleştirme
    plot_lift_chart(lift_df, out_dir)
    
    # Bir PDF özeti de eklenebilir (reportlab veya pdfkit ile)
    # TODO: pdfkit / reportlab ile tek PDF'e göm
    
    return metrics
