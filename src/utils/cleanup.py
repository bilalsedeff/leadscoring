import os
import shutil
from pathlib import Path
import time
import yaml
from src.utils.paths import PROJECT_ROOT, get_experiment_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Yapılandırmadan arşiv günü sayısını alma
def _load_config():
    """app_config.yaml'dan yapılandırmayı yükler."""
    config_path = PROJECT_ROOT / "configs" / "app_config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Yapılandırma dosyası yüklenemedi: {e}")
    
    # Varsayılan değerler
    return {"archive_days": 30}

def cleanup_old_dirs(verbose=False):
    """
    Eski legacy klasörleri (artifacts/mlruns) temizler.
    Bu fonksiyon sadece eskiden kök dizinde bulunan artifacts ve mlruns klasörlerini temizler.
    Outputs/run_* klasörlerini temizlemek için archive_outputs() ve purge_archives() fonksiyonlarını kullanın.
    
    Args:
        verbose: Ayrıntılı log çıktısı isteniyor mu?
    """
    
    # Outputs klasörü
    outputs_dir = PROJECT_ROOT / "outputs"
    if not outputs_dir.exists():
        logger.warning(f"{outputs_dir} klasörü bulunamadı, oluşturuluyor...")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        return
    
    # Eski artifacts ve mlruns klasörlerini temizle (eski sürümlerde kullanılıyordu)
    legacy_dirs = [PROJECT_ROOT / 'artifacts', PROJECT_ROOT / 'mlruns']
    
    for dir_path in legacy_dirs:
        if dir_path.exists():
            try:
                if dir_path.is_dir():
                    if verbose:
                        logger.info(f"{dir_path} klasörü temizleniyor (eski format)...")
                    shutil.rmtree(dir_path)
                else:
                    if verbose:
                        logger.info(f"{dir_path} bir klasör değil, atlanıyor.")
            except Exception as e:
                logger.error(f"{dir_path} temizlenirken hata: {e}")
    
    # MLflow SQLITE DB için doğru klasör yapısını oluştur
    if not outputs_dir.exists():
        outputs_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        logger.info("Temizlik tamamlandı. Tüm yeni çıktılar outputs/run_timestamp altında saklanacak.")

def archive_outputs(days_old=None, verbose=False):
    """
    Eski çıktı klasörlerini arşivler veya siler.
    
    Args:
        days_old: Bu günden kaç gün önce oluşturulmuş klasörleri arşivlemeli
                 None ise app_config.yaml'dan alınır
        verbose: Ayrıntılı log çıktısı isteniyor mu?
    """
    # Yapılandırmadan arşiv gün sayısını al
    if days_old is None:
        config = _load_config()
        days_old = config.get("archive_days", 30)
    
    outputs_dir = PROJECT_ROOT / 'outputs'
    if not outputs_dir.exists():
        logger.warning("outputs klasörü bulunamadı, atlanıyor.")
        return
    
    # Arşiv klasörü
    archive_dir = outputs_dir / 'archive'
    archive_dir.mkdir(exist_ok=True)
    
    # Şu anki zaman
    now = time.time()
    
    # outputs altındaki run_* klasörlerini kontrol et
    archived_count = 0
    for run_dir in outputs_dir.glob('run_*'):
        if not run_dir.is_dir() or run_dir.name == 'archive':
            continue
        
        # Klasör oluşturma zamanını al
        try:
            # Unix timestamp'i çıkar (run_1633509284 gibi)
            timestamp = int(run_dir.name.split('_')[1])
            dir_age_days = (now - timestamp) / (60 * 60 * 24)
            
            # Belirlenen günden eski mi?
            if dir_age_days > days_old:
                if verbose:
                    logger.info(f"{run_dir.name} ({dir_age_days:.1f} gün önce) arşivleniyor...")
                # Arşive taşı
                shutil.move(str(run_dir), str(archive_dir / run_dir.name))
                archived_count += 1
        except Exception as e:
            logger.error(f"{run_dir} için yaş belirlenirken hata: {e}")
    
    if verbose:
        logger.info(f"Arşivleme tamamlandı. {archived_count} klasör arşive taşındı.")
    return archived_count

def purge_archives(keep_last_n=5, verbose=False):
    """
    Arşiv klasöründeki en eski çalıştırmaları tamamen siler.
    Yalnızca en son N çalıştırmayı tutar.
    
    Args:
        keep_last_n: Saklanacak son N çalıştırma sayısı
        verbose: Ayrıntılı log çıktısı isteniyor mu?
    """
    outputs_dir = PROJECT_ROOT / 'outputs'
    archive_dir = outputs_dir / 'archive'
    
    if not archive_dir.exists():
        if verbose:
            logger.info("Arşiv klasörü bulunamadı, atlanıyor.")
        return 0
    
    # Arşivdeki klasörleri oluşturma zamanına göre sırala
    run_dirs = []
    for run_dir in archive_dir.glob('run_*'):
        if not run_dir.is_dir():
            continue
        
        try:
            # Unix timestamp'i çıkar (run_1633509284 gibi)
            timestamp = int(run_dir.name.split('_')[1])
            run_dirs.append((timestamp, run_dir))
        except Exception:
            # İsimlendirme farklıysa atla
            continue
    
    # Zamanlamaya göre sırala (eski → yeni)
    run_dirs.sort()
    
    # En eski klasörleri sil (keep_last_n kadarını tut)
    purged_count = 0
    if len(run_dirs) > keep_last_n:
        for _, dir_path in run_dirs[:-keep_last_n]:
            if verbose:
                logger.info(f"{dir_path.name} kalıcı olarak siliniyor...")
            try:
                shutil.rmtree(dir_path)
                purged_count += 1
            except Exception as e:
                logger.error(f"{dir_path} silinirken hata: {e}")
    
    if verbose:
        logger.info(f"Temizleme tamamlandı. {purged_count} arşiv klasörü kalıcı olarak silindi.")
    return purged_count

if __name__ == "__main__":
    # Komut satırından çağrıldığında temizliği yap
    cleanup_old_dirs(verbose=True)
    archive_outputs(verbose=True)
    purge_archives(verbose=True) 