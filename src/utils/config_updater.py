import os
import yaml
from pathlib import Path
from omegaconf import OmegaConf
from typing import List, Optional
import pandas as pd
from src.utils.logger import get_logger
from src.utils.paths import PROJECT_ROOT, get_experiment_dir

log = get_logger()

def update_experiment_config(
    num_cols: Optional[List[str]] = None, 
    cat_cols: Optional[List[str]] = None,
    experiment_name: Optional[str] = None,
    # Geriye uyumluluk için eski parametre isimleri de kabul edilecek
    num_features: Optional[List[str]] = None,
    cat_features: Optional[List[str]] = None
) -> bool:
    """
    experiment.yaml dosyasını günceller, özellikle num_cols ve cat_cols listelerini.
    
    Args:
        num_cols: Sayısal özellikler listesi
        cat_cols: Kategorik özellikler listesi
        experiment_name: Deney adı (opsiyonel)
        num_features: num_cols ile aynı (geriye uyumluluk için)
        cat_features: cat_cols ile aynı (geriye uyumluluk için)
    
    Returns:
        bool: Güncelleme başarılıysa True, değilse False
    """
    # Eski parametre isimlerini yeni isimlere dönüştür
    if num_cols is None and num_features is not None:
        num_cols = num_features
    if cat_cols is None and cat_features is not None:
        cat_cols = cat_features
    
    # Mutlak yol kullanma
    config_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    
    if not config_path.exists():
        log.error(f"Config dosyası bulunamadı: {config_path}")
        return False
    
    try:
        # OmegaConf ile mevcut konfigürasyonu yükle
        cfg = OmegaConf.load(config_path)
        
        # Değiştirilecek alanları güncelle
        if num_cols is not None:
            cfg.num_cols = num_cols
        
        if cat_cols is not None:
            cfg.cat_cols = cat_cols
            
        if experiment_name is not None:
            cfg.experiment_name = experiment_name
        
        # YAML formatında yazma için YAML kütüphanesini kullan
        # OmegaConf'un yapısını standart dictionary'ye çevir
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # YAML formatına dönüştür ve dosyaya yaz
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        log.info(f"Experiment yapılandırması güncellendi: {config_path}")
        return True
    except Exception as e:
        log.error(f"Yapılandırma güncellenirken hata: {e}")
        return False

def update_from_feature_importance(top_k: int, output_dir: Optional[Path] = None) -> bool:
    """
    Feature importance sonuçlarını kullanarak experiment.yaml dosyasını günceller.
    
    Args:
        top_k: Kullanılacak özellik sayısı
        output_dir: Çıktı dizini (None ise varsayılan olarak get_experiment_dir() kullanılır)
        
    Returns:
        bool: Güncelleme başarılıysa True, değilse False
    """
    try:
        # Çıktı dizini belirleme
        if output_dir is None:
            # Aktif deney dizinini kullan
            from src.features.importance import get_importance_output_dir
            output_dir = get_importance_output_dir()
        
        # Feature importance sonuçlarını oku
        imp_path = output_dir / "importance.csv"
        feature_types_path = output_dir / "feature_types.csv"
        
        if not imp_path.exists() or not feature_types_path.exists():
            log.error(f"Feature importance dosyaları bulunamadı: {imp_path}, {feature_types_path}")
            
            # Eski yolları dene
            legacy_imp_path = PROJECT_ROOT / "outputs" / "feature_importance" / "importance.csv"
            legacy_types_path = PROJECT_ROOT / "outputs" / "feature_importance" / "feature_types.csv"
            
            if legacy_imp_path.exists() and legacy_types_path.exists():
                log.info(f"Eski feature importance dosyaları kullanılıyor: {legacy_imp_path}")
                imp_path = legacy_imp_path
                feature_types_path = legacy_types_path
            else:
                return False
            
        imp_df = pd.read_csv(imp_path)
        types_df = pd.read_csv(feature_types_path)
        
        # En önemli top_k özelliği seç
        selected_features = imp_df['feature'].head(top_k).tolist()
        
        # Seçilen özellikleri kategorik ve sayısal olarak ayır
        selected_types = types_df[types_df['feature'].isin(selected_features)]
        
        num_cols = selected_types[selected_types['type'] == 'numeric']['feature'].tolist()
        cat_cols = selected_types[selected_types['type'] == 'categorical']['feature'].tolist()
        
        # Experiment.yaml dosyasını güncelle
        return update_experiment_config(num_cols=num_cols, cat_cols=cat_cols)
    
    except Exception as e:
        log.error(f"Feature importance sonuçlarından yapılandırma güncellenirken hata: {e}")
        return False 