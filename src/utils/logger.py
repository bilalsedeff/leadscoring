import logging, sys, os
from pathlib import Path
import time
from typing import Optional

# Log formatı
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

def get_logger(name: str = "lead_scoring", log_to_file: bool = True, 
               log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Logger oluşturur. Hem konsola hem de dosyaya log yazar.
    
    Args:
        name: Logger adı
        log_to_file: Dosyaya log yazılsın mı?
        log_dir: Log dosyalarının yazılacağı dizin (None ise outputs/logs kullanılır)
    
    Returns:
        Logger nesnesi
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # tekrar ekleme
        return logger
    
    logger.setLevel(LOG_LEVEL)
    
    # Konsola yazdırma (UTF-8 desteği ekleniyor)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(_FMT))
    # Python 3.7+ için:
    try:
        sh.stream.reconfigure(encoding='utf-8')
    except Exception as e:
        logger.warning(f"UTF-8 desteği ayarlanamadı: {e}")
    logger.addHandler(sh)
    # Dosyaya log yazma (isteğe bağlı)
    if log_to_file:
        try:
            # Log dizini belirleme
            if log_dir is None:
                # Önce PROJECT_ROOT'u bulmaya çalış
                try:
                    from src.utils.paths import PROJECT_ROOT, get_experiment_dir
                    # Aktif deney dizinini veya varsayılan çıktı dizinini kullan
                    try:
                        # Öncelikle mevcut deneyden log dizinini al
                        experiment_dir = get_experiment_dir()
                        log_dir = experiment_dir / "logs"
                    except Exception as e:
                        # Experiment oluşturulamadıysa, os.environ["OUTPUT_DIR"] kullan
                        output_dir = os.environ.get("OUTPUT_DIR", "outputs")
                        log_dir = Path(output_dir) / "logs"
                        logger.debug(f"Aktif deney dizini alınamadı: {e}, {output_dir}/logs kullanılıyor")
                except ImportError:
                    # Path modülü yardımıyla proje kökü tahmin et
                    base_dir = Path(__file__).resolve().parents[2]  # src/utils/logger.py → 3 üst = proje kökü
                    # os.environ["OUTPUT_DIR"] kullan, yoksa outputs kullan
                    output_dir = os.environ.get("OUTPUT_DIR", "outputs")
                    log_dir = Path(output_dir) / "logs"
            
            # Log dizini oluştur
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Log dosyası yolu
            log_file = log_dir / f"{name}.log"
            
            # FileHandler ekle
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(logging.Formatter(_FMT))
            logger.addHandler(fh)
            
            logger.debug(f"Dosya loglama aktif: {log_file}")
        except Exception as e:
            logger.warning(f"Dosya loglama etkinleştirilemiyor: {e}")

    return logger

def get_experiment_logger(name: str = "lead_scoring", experiment_dir: Optional[Path] = None) -> logging.Logger:
    """
    Belirli bir deney için logger oluşturur. Bu logger, o deneyin dizinine log yazar.
    
    Args:
        name: Logger adı
        experiment_dir: Deney dizini (None ise aktif deney dizini kullanılır)
    
    Returns:
        Logger nesnesi
    """
    try:
        # Deney dizini belirleme
        if experiment_dir is None:
            try:
                from src.utils.paths import get_experiment_dir
                experiment_dir = get_experiment_dir()
            except Exception as e:
                # Deney dizini alınamazsa uyarı ver ve normal logger'a dön
                logger = get_logger(name, log_to_file=False)
                logger.warning(f"Deney dizini alınamadı: {e}, standart logger kullanılıyor.")
                return logger
        
        # Log dizini
        log_dir = experiment_dir / "logs"
        
        return get_logger(name, log_to_file=True, log_dir=log_dir)
    except Exception as e:
        # Hata durumunda standart logger'a dön
        logger = get_logger(name, log_to_file=False)
        logger.warning(f"Deney logger'ı oluşturulamadı, standart logger kullanılıyor: {e}")
        return logger
