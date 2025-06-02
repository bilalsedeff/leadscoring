"""
Model metriklerini getirmek için yardımcı fonksiyonlar.
"""

import os
import json
import logging
from pathlib import Path
import glob
import mlflow
from typing import Dict, Optional, List
from src.utils.paths import PROJECT_ROOT, get_experiment_dir
from src.utils.logger import get_logger
import numpy as np
import pandas as pd

logger = get_logger(__name__)

def get_latest_metrics(experiment_name=None):
    """
    Belirtilen experiment için son çalıştırmanın metriklerini getirir.
    
    Args:
        experiment_name: Metriklerini getirmek istediğimiz experiment adı
            None ise, en son çalıştırmanın metrikleri getirilir.
    
    Returns:
        dict: Metrik ismi ve değerlerini içeren sözlük, veya hata durumunda None
    """
    try:
        # MLflow API üzerinden experiment bul
        if experiment_name:
            # İsme göre experiment'i bul
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            # Bulunamazsa id ile dene
            if not experiment:
                try:
                    experiment = mlflow.get_experiment(experiment_name)
                except:
                    pass
        else:
            # Tüm experimentleri getir ve en son çalıştırmanı bul
            experiments = mlflow.search_experiments()
            if not experiments:
                logger.warning("Hiç experiment bulunamadı.")
                return None
                
            experiment = experiments[-1]  # Son eklenen experiment
        
        if not experiment:
            logger.warning(f"'{experiment_name}' adlı experiment bulunamadı.")
            return None
            
        # Experiment için son çalıştırmayı bul
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], 
                                 order_by=["attribute.start_time DESC"])
        
        if runs.empty:
            logger.warning(f"'{experiment.name}' experiment'i için çalıştırma bulunamadı.")
            return None
            
        # İlk (en son) çalıştırmanın metriklerini al
        latest_run = runs.iloc[0]
        
        # Metrikler runs DataFrame'inde "metrics." önekiyle yer alır
        metrics = {}
        for col in runs.columns:
            if col.startswith("metrics."):
                metric_name = col.split(".", 1)[1]
                metrics[metric_name] = latest_run[col]
        
        return metrics
        
    except Exception as e:
        logger.exception(f"Metrikler getirilirken hata: {e}")
        return None

def get_all_metrics(limit=10):
    """
    Tüm experimentlerin son çalıştırmalarının metriklerini getirir.
    
    Args:
        limit: Getirilecek maksimum experiment sayısı
        
    Returns:
        list: Her bir elementin experiment adı ve metriklerini içerdiği liste
    """
    try:
        experiments = mlflow.search_experiments()
        results = []
        
        for exp in experiments[:limit]:
            metrics = get_latest_metrics(exp.experiment_id)
            if metrics:
                results.append({
                    "experiment_name": exp.name,
                    "experiment_id": exp.experiment_id,
                    "metrics": metrics
                })
                
        return results
        
    except Exception as e:
        logger.exception(f"Tüm metrikler getirilirken hata: {e}")
        return []

def get_metrics_from_json(model_dir: Optional[Path] = None) -> Dict[str, float]:
    """
    Model dizinindeki metrics.json dosyasından metrikleri okur.
    
    Args:
        model_dir: Model dizini. None ise en son modelin dizini kullanılır.
    
    Returns:
        Dict: Metrik adı → değer sözlüğü
    """
    if model_dir is None:
        # Önce aktif deney dizininden model dizinini bulmaya çalış
        try:
            model_dir = get_experiment_dir() / "models"
        except Exception:
            # Tüm outputs/run_* altındaki model dizinlerini bul, en yenisini al
            run_dirs = sorted(glob.glob(str(PROJECT_ROOT / "outputs" / "run_*")))
            if not run_dirs:
                logger.error("Hiçbir run dizini bulunamadı!")
                return {}
            
            # En son run_* dizinini kullan
            latest_run = run_dirs[-1]
            model_dir = Path(latest_run) / "models"
    
    metrics_path = model_dir / "metrics.json"
    
    if not metrics_path.exists():
        # Alt klasörlerde ara
        model_dirs = [d for d in model_dir.glob("*") if d.is_dir()]
        for d in model_dirs:
            alt_metrics_path = d / "metrics.json"
            if alt_metrics_path.exists():
                metrics_path = alt_metrics_path
                break
        else:
            logger.error(f"Metrik dosyası bulunamadı: {metrics_path}")
            return {}
    
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        logger.error(f"Metrik dosyası okunamadı: {e}")
        return {}

def save_metrics_to_json(metrics: Dict[str, float], output_dir: Optional[Path] = None) -> bool:
    """
    Model metriklerini metrics.json dosyasına kaydeder.
    
    Args:
        metrics: Metrik adı → değer sözlüğü
        output_dir: Çıktı dizini. None ise aktif deney dizini / models kullanılır.
    
    Returns:
        bool: Başarılı ise True, değilse False
    """
    if output_dir is None:
        # Aktif deney dizinini al
        try:
            output_dir = get_experiment_dir() / "models"
        except Exception as e:
            logger.error(f"Aktif deney dizini alınamadı: {e}")
            return False
    
    # Dizinin var olduğundan emin ol
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_dir / "metrics.json"
    
    try:
        # NumPy değerleri Python native tipine dönüştür
        cleaned_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (np.integer, np.floating, np.bool_)):
                cleaned_metrics[k] = v.item()
            else:
                cleaned_metrics[k] = v
        
        with open(metrics_path, "w") as f:
            json.dump(cleaned_metrics, f, indent=4)
        
        logger.info(f"Metrikler kaydedildi: {metrics_path}")
        return True
    except Exception as e:
        logger.error(f"Metrikler kaydedilemedi: {e}")
        return False

def get_metrics_from_mlflow(run_id: Optional[str] = None) -> Dict[str, float]:
    """
    MLflow'dan run metriklerini alır.
    
    Args:
        run_id: MLflow run ID. None ise en son aktif run kullanılır.
    
    Returns:
        Dict: Metrik adı → değer sözlüğü
    """
    if run_id is None:
        try:
            # En son active run'ı al
            active_run = mlflow.active_run()
            if active_run:
                run_id = active_run.info.run_id
            else:
                # MLflow API'sini kullanarak en son run'ı bul
                client = mlflow.tracking.MlflowClient()
                runs = client.search_runs(experiment_ids=["0"])
                if runs:
                    run_id = runs[0].info.run_id
                else:
                    logger.error("Hiçbir MLflow run bulunamadı!")
                    return {}
        except Exception as e:
            logger.error(f"MLflow run ID alınamadı: {e}")
            return {}
    
    try:
        # Metrikleri al
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        metrics = run.data.metrics
        
        return metrics
    except Exception as e:
        logger.error(f"MLflow metrikleri alınamadı: {e}")
        return {}

def format_metrics_table(metrics: Dict[str, float]) -> str:
    """
    Model metriklerini tablo formatında formatlar.
    
    Args:
        metrics: Metrik adı → değer sözlüğü
    
    Returns:
        str: Formatlanmış tablo metni
    """
    if not metrics:
        return "Metrik bulunamadı."
    
    # DataFrame oluştur
    df = pd.DataFrame(
        {"Metrik": list(metrics.keys()), "Değer": list(metrics.values())}
    )
    
    # Değerleri formatla
    df["Değer"] = df["Değer"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x)
    )
    
    # Tabloyu formatla
    table = "| Metrik | Değer |\n| ------ | ----- |\n"
    for _, row in df.iterrows():
        table += f"| {row['Metrik']} | {row['Değer']} |\n"
    
    return table

if __name__ == "__main__":
    # Test
    metrics = get_latest_metrics()
    if metrics:
        print("Son çalıştırmanın metrikleri:")
        for name, value in metrics.items():
            print(f"{name}: {value}")
    else:
        print("Metrikler getirilemedi.") 