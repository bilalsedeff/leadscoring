import pandas as pd
import numpy as np
import hydra
from omegaconf import OmegaConf
from pathlib import Path
from typing import List, Union, Optional

class ProbabilityBinner:
    def __init__(self, edges=None, labels=None):
        """
        Model olasılıklarını Low/Medium/High segmentlere ayıran sınıf.
        
        Args:
            edges: Segment eşik değerleri. None ise split.yaml'dan alınır.
            labels: Segment etiketleri. None ise split.yaml'dan alınır.
        """
        if edges is None or labels is None:
            # split.yaml'dan değerleri al
            config_path = Path("configs/split.yaml")
            if config_path.exists():
                config = OmegaConf.load(config_path)
                self.edges = config.get("bin_edges", [0.0, 0.25, 0.75, 1.0])
                self.labels = config.get("bin_labels", ["Low", "Medium", "High"])
            else:
                # Varsayılan değerler
                self.edges = [0.0, 0.25, 0.75, 1.0]
                self.labels = ["Low", "Medium", "High"]
        else:
            self.edges = edges
            self.labels = labels
            
    def transform(self, p):
        """
        Olasılık değerlerini segment etiketlerine dönüştürür.
        
        Args:
            p: Olasılık değerleri (array veya liste)
            
        Returns:
            Kategorik segment etiketleri
        """
        idx = np.digitize(p, self.edges, right=False) - 1
        return pd.Categorical.from_codes(idx, categories=self.labels)
        
    def assign_bins(self, p):
        """
        Olasılık değerlerini segment etiketlerine dönüştürür (transform metodunun alternatifi).
        
        Args:
            p: Olasılık değerleri (array veya liste)
            
        Returns:
            Kategorik segment etiketleri (numpy array olarak)
        """
        # NumPy array'e dönüştür
        p_array = np.asarray(p)
        
        # Segment indekslerini bul
        indices = np.digitize(p_array, self.edges) - 1
        
        # Segment indekslerini sınırlandır (edges dışında değer olabilir)
        indices = np.clip(indices, 0, len(self.labels) - 1)
        
        # İndeksleri etiketlere dönüştür - OmegaConf indeksleme sorunuyla başa çıkmak için
        # OmegaConf'dan yüklenen listeler için int64 gibi indeksler sorun çıkarabilir
        # Önce labels'ı normal Python listesine dönüştürelim
        labels_list = list(self.labels)
        
        # Şimdi indeksleri kullanarak etiketleri alalım
        return np.array([labels_list[int(i)] for i in indices])

# Geriye dönük uyumluluk için PotentialBinner takma adı
PotentialBinner = ProbabilityBinner

def probability_binning(
    y_proba: Union[np.ndarray, pd.Series], 
    bins: List[float] = [0.0, 0.25, 0.75, 1.0], 
    labels: List[str] = ['Low', 'Medium', 'High']
) -> np.ndarray:
    """
    Olasılık tahminlerini belirtilen eşik değerlerine göre kategorik sınıflara ayırır.
    
    Args:
        y_proba: Olasılık tahminleri, 0-1 aralığında
        bins: Eşik değerleri (sınır noktaları)
        labels: Her bin için etiketler (bins uzunluğundan 1 eksik olmalı)
        
    Returns:
        np.ndarray: Kategorik sınıf etiketleri
    """
    if len(bins) - 1 != len(labels):
        raise ValueError(f"Bin sayısı ({len(bins)-1}) ve etiket sayısı ({len(labels)}) uyuşmuyor")
    
    # Olasılıkları kategorilere dönüştür
    return pd.cut(y_proba, bins=bins, labels=labels)

def get_bin_stats(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    bins: List[float] = [0.0, 0.25, 0.75, 1.0],
    labels: List[str] = ['Low', 'Medium', 'High']
) -> pd.DataFrame:
    """
    Her olasılık aralığı (bin) için istatistikler hesaplar.
    
    Args:
        y_true: Gerçek sınıf değerleri (0-1)
        y_pred: Olasılık tahminleri (0-1 aralığında)
        bins: Eşik değerleri
        labels: Bin etiketleri
        
    Returns:
        pd.DataFrame: Bin bazında istatistikler (count, mean_true, mean_pred vs.)
    """
    # Verileri DataFrame'e dönüştür
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    # Tahminleri kategorilere ayır
    df['bin'] = probability_binning(y_pred, bins, labels)
    
    # Bin bazında istatistikler
    stats = df.groupby('bin').agg(
        count=('y_true', 'count'),
        mean_true=('y_true', 'mean'),
        mean_pred=('y_pred', 'mean'),
        min_pred=('y_pred', 'min'),
        max_pred=('y_pred', 'max')
    ).reset_index()
    
    # Yüzde dağılımı
    stats['percent'] = stats['count'] / stats['count'].sum() * 100
    
    # Kalibrasyon hatası (gerçek değer - tahmin)
    stats['calibration_error'] = stats['mean_true'] - stats['mean_pred']
    
    return stats
