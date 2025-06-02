from imblearn.over_sampling import SMOTENC
import numpy as np
from src.utils.logger import get_logger

log = get_logger()

def balanced_resample(X, y, cat_idx):
    """
    SMOTENC ile sınıf dengesizliğini giderir.
    Hata durumunda orijinal veriyi döndürür.
    
    Args:
        X: Özellikler
        y: Hedef değişken
        cat_idx: Kategorik özellik indeksleri
        
    Returns:
        X_res, y_res: Dengelenmiş veri seti
    """
    try:
        # Tüm sınıflar için auto strateji kullan
        smote = SMOTENC(categorical_features=cat_idx,
                        sampling_strategy='auto', random_state=0)
        X_res, y_res = smote.fit_resample(X, y)
        log.info(f"SMOTE uygulandı: {len(X)} -> {len(X_res)} örnek")
        return X_res, y_res
    except Exception as e:
        # Hata durumunda orijinal veriyi döndür
        log.warning(f"SMOTE uygulanırken hata oluştu, orijinal veri kullanılıyor: {str(e)}")
        return X, y
