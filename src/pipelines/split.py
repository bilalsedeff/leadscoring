"""
Veri bölme pipeline'ı
"""
import click
import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
import yaml
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
from omegaconf import OmegaConf
import time

from src.ingestion.loader import read_datamart
from src.preprocessing.cleaning import BasicCleaner, load_cleaning_config
from src.preprocessing.splitters import time_group_split, stratified_group_split
from src.preprocessing.type_cast import apply_type_map
from src.utils.logger import get_logger
from src.utils.paths import get_experiment_dir
import joblib

log = get_logger()

def run_split(train_cutoff: str, val_cutoff: str, test_cutoff: str = None, 
              time_col: str = "LeadCreatedDate", group_col: str = "account_Id", 
              random_seed: int = 42, output_dir: str = None,
              force_balance: bool = False) -> Dict[str, Any]:
    """
    Veri hazırlama ve split işlemlerini çalıştırır.
    
    Args:
        train_cutoff: Eğitim seti için kesme tarihi (YYYY-MM-DD formatı)
        val_cutoff: Validasyon seti için kesme tarihi (YYYY-MM-DD formatı)
        test_cutoff: Test seti için kesme tarihi (YYYY-MM-DD formatı, opsiyonel)
        time_col: Zaman sütunu (LeadCreatedDate)
        group_col: Grup sütunu (account_Id)
        random_seed: Rastgele sayı üreteci tohum değeri
        output_dir: Çıktı dizini
        force_balance: Her sette mutlaka veri olmasını zorla
        
    Returns:
        Dict: İşlem sonuçları
    """
    # Çıktı dizini - sadece experiment directory kullan
    if output_dir is None:
        output_dir = str(get_experiment_dir())
    
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Adım 1: Ham veriyi yükle
    log.info("Ham veri yükleniyor...")
    raw_df = read_datamart()
    
    # Adım 2: Önce ham veriyi time-based split et (veri sızıntısı olmaması için)
    log.info(f"Ham veri time-based split ediliyor... (cutoff: train={train_cutoff}, val={val_cutoff}, test={test_cutoff})")
    train_raw, val_raw, test_raw = time_group_split(
        df=raw_df,
        cutoff=train_cutoff, 
        val_cutoff=val_cutoff,
        test_cutoff=test_cutoff,
        time_col=time_col,
        group_col=group_col,
        force_balance=force_balance,
        random_seed=random_seed
    )
    
    # Ham bölünmüş veri setlerinin boyutları
    log.info(f"Ham split sonuçları - Train: {train_raw.shape}, Validation: {val_raw.shape}, Test: {test_raw.shape}")
    
    # Adım 3: SADECE train verisi üzerinde temizleyici oluştur ve eğit
    log.info("Temizleyici SADECE train verisi üzerinde eğitiliyor...")
    cleaner = BasicCleaner()
    cleaner.fit(train_raw)
    
    # Adım 4: Eğitilmiş temizleyiciyi tüm ham setlere uygula
    log.info("Eğitilmiş temizleyici tüm veri setlerine uygulanıyor...")
    train_df = cleaner.transform(train_raw)
    val_df = cleaner.transform(val_raw)
    test_df = cleaner.transform(test_raw)
    
    # Temizlenmiş veri setlerinin boyutları
    log.info(f"Temizlenmiş split sonuçları - Train: {train_df.shape}, Validation: {val_df.shape}, Test: {test_df.shape}")
    
    # Ana dizinlere veri setlerini kaydet
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    log.info(f"Veri setleri experiment dizinine kaydedildi: {output_dir}")
    
    # Veri setlerini data/processed dizinine de kaydet
    try:
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(processed_dir / "train.csv", index=False)
        val_df.to_csv(processed_dir / "validation.csv", index=False)
        test_df.to_csv(processed_dir / "test.csv", index=False)
        
        log.info(f"Veri setleri data/processed dizinine de kaydedildi")
    except Exception as e:
        log.warning(f"Veri setleri data/processed dizinine kaydedilirken hata: {e}")
    
    # Çıktıları hazırla
    results = {
        "train_shape": train_df.shape,
        "val_shape": val_df.shape,
        "test_shape": test_df.shape,
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path
    }
    
    # UYARI: Test seti tamamen boşsa uyarı ver
    if test_df.shape[0] == 0:
        log.warning("TEST SETİ BOŞ! Lütfen cutoff değerlerini kontrol edin.")
    
    # Metadata.json dosyası oluştur
    import json
    metadata = {
        "timestamp": int(time.time()),
        "train_cutoff": train_cutoff,
        "val_cutoff": val_cutoff,
        "test_cutoff": test_cutoff,
        "time_col": time_col,
        "group_col": group_col,
        "random_seed": random_seed,
        "train_rows": train_df.shape[0],
        "val_rows": val_df.shape[0],
        "test_rows": test_df.shape[0],
        "columns": train_df.columns.tolist(),
        "force_balance": force_balance
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    return results

@click.command()
@click.option('--train-cutoff', required=True, type=str, help='Eğitim seti için kesme tarihi (YYYY-MM-DD formatı)')
@click.option('--val-cutoff', required=True, type=str, help='Validasyon seti için kesme tarihi (YYYY-MM-DD formatı)')
@click.option('--test-cutoff', type=str, help='Test seti için kesme tarihi (YYYY-MM-DD formatı, opsiyonel)')
@click.option('--time-col', default="LeadCreatedDate", help='Zaman sütunu')
@click.option('--group-col', default="account_Id", help='Grup sütunu')
@click.option('--random-seed', default=42, type=int, help='Rastgele sayı üreteci tohum değeri')
@click.option('--output-dir', help='Çıktı dizini')
@click.option('--force-balance/--no-force-balance', default=True, help='Her sette mutlaka veri olmasını zorla')
def main(train_cutoff, val_cutoff, test_cutoff, time_col, group_col, random_seed, output_dir, force_balance):
    """Veri hazırlama ve split pipeline'ı.
    
    Ham veriyi yükler, zaman bazlı böler, temizler ve eğitim/validasyon/test setlerini oluşturur.
    """
    try:
        results = run_split(
            train_cutoff=train_cutoff,
            val_cutoff=val_cutoff,
            test_cutoff=test_cutoff,
            time_col=time_col,
            group_col=group_col,
            random_seed=random_seed,
            output_dir=output_dir,
            force_balance=force_balance
        )
        
        log.info("Veri hazırlama ve split işlemi başarıyla tamamlandı.")
        log.info(f"Train: {results['train_shape'][0]} satır, {results['train_shape'][1]} sütun")
        log.info(f"Validation: {results['val_shape'][0]} satır, {results['val_shape'][1]} sütun")
        log.info(f"Test: {results['test_shape'][0]} satır, {results['test_shape'][1]} sütun")
        
    except Exception as e:
        log.exception("Veri hazırlama ve split işleminde hata")
        raise

if __name__ == "__main__":
    main() 