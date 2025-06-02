"""
Üst düzey etkileşim & oran özellikleri
"""
import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Optional

def add_pairwise_ratios(df: pd.DataFrame, num_cols: Optional[List[str]] = None, max_pairs: int = 20):
    """
    Sayısal kolonlar arasındaki oranları (özellik bölme işlemleri) hesaplar.
    
    Args:
        df: Veri çerçevesi
        num_cols: Oran hesaplanacak sayısal kolonlar listesi (None ise otomatik tespit edilir)
        max_pairs: Eklenecek maksimum oran özelliği sayısı
        
    Returns:
        DataFrame: Oran özellikleri eklenmiş veri çerçevesi
    """
    df = df.copy()
    
    # Sayısal kolonlar belirtilmemişse otomatik tespit et
    if num_cols is None:
        # Sadece sayısal kolonları seç (int ve float)
        num_cols = df.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()
        
        # ID ve tarih kolonlarını hariç tut
        exclude_patterns = ['id', 'Id', 'ID', 'date', 'Date', 'time', 'Time', 'year', 'month', 'day']
        num_cols = [col for col in num_cols if not any(pattern in col for pattern in exclude_patterns)]
    
    # max_pairs'ten fazla çift varsa, özellik limitini kontrol et
    total_pairs = len(list(combinations(num_cols, 2)))
    if total_pairs > max_pairs:
        # En önemli kolonları seçmek için bir strateji belirleyebiliriz
        # Şimdilik ilk max_pairs sayıda çifti kullanıyoruz
        all_pairs = list(combinations(num_cols[:int(np.sqrt(max_pairs * 2))], 2))
        selected_pairs = all_pairs[:max_pairs]
    else:
        selected_pairs = combinations(num_cols, 2)
    
    # Oranları hesapla
    for (a, b) in selected_pairs:
        if a in df.columns and b in df.columns:  # Kolonların varlığını kontrol et
            ratio_name = f"{a}_DIV_{b}"
            
            # 0 bölünme koruması - epsilon değeri ekleyerek daha güvenli
            epsilon = 1e-8  # Çok küçük bir değer
            df[ratio_name] = df[a] / (df[b].replace(0, epsilon) + epsilon)
            
            # İşlemin sonucunda oluşabilecek uç değerleri sınırla
            df[ratio_name] = df[ratio_name].clip(-1e6, 1e6)
            
            # NaN ve Inf değerlerini temizle
            df[ratio_name] = df[ratio_name].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df

def add_products(df: pd.DataFrame, num_cols: Optional[List[str]] = None, top_k: int = 10):
    """
    Sayısal kolonlar arasındaki çarpımları hesaplar.
    
    Args:
        df: Veri çerçevesi
        num_cols: Çarpım hesaplanacak sayısal kolonlar listesi (None ise otomatik tespit edilir)
        top_k: Çarpım için kullanılacak ilk top_k kolon
        
    Returns:
        DataFrame: Çarpım özellikleri eklenmiş veri çerçevesi
    """
    df = df.copy()
    
    # Sayısal kolonlar belirtilmemişse otomatik tespit et
    if num_cols is None:
        # Sadece sayısal kolonları seç (int ve float)
        num_cols = df.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()
        
        # ID ve tarih kolonlarını hariç tut
        exclude_patterns = ['id', 'Id', 'ID', 'date', 'Date', 'time', 'Time', 'year', 'month', 'day']
        num_cols = [col for col in num_cols if not any(pattern in col for pattern in exclude_patterns)]
    
    # Top_k sayıda kolon seç
    use_cols = num_cols[:min(top_k, len(num_cols))]
    
    # Çarpımları hesapla
    for a, b in combinations(use_cols, 2):
        if a in df.columns and b in df.columns:  # Kolonların varlığını kontrol et
            product_name = f"{a}_X_{b}"
            df[product_name] = df[a] * df[b]
            
            # Olası aşırı değerleri kırp
            df[product_name] = df[product_name].clip(-1e9, 1e9)
            
            # NaN değerlerini temizle
            df[product_name] = df[product_name].fillna(0)
    
    return df
