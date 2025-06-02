from pathlib import Path
import os
import time
import sys
from typing import Union, Optional

def find_project_root():
    """
    Proje kökünü bulur. Önce ortam değişkenini kontrol eder,
    bulamazsa marker dosyalara bakarak yukarı doğru çıkar.
    
    Returns:
        Path: Proje kökü dizini
    """
    # 1. Ortam değişkenini kontrol et
    if "PROJECT_ROOT" in os.environ:
        root = Path(os.environ["PROJECT_ROOT"])
        if root.exists():
            return root.resolve()
    
    # 2. Marker dosyalara bakarak yukarı doğru çık
    marker_files = [".git", "pyproject.toml", "requirements.txt", "README.md"]
    
    # Başlangıç noktası - bu dosyanın dizini
    current_dir = Path(__file__).resolve().parent
    
    # Ana sürücü kökü olana kadar yukarı çık
    while current_dir != current_dir.parent:
        # Marker dosyalardan herhangi biri var mı?
        for marker in marker_files:
            if (current_dir / marker).exists():
                return current_dir
        
        # Bir üst dizine geç
        current_dir = current_dir.parent
    
    # 3. Hiçbir marker bulunamazsa, bu dosyanın 3 üst dizinini kullan
    # (src/utils/paths.py --> src/utils --> src --> root)
    fallback_root = Path(__file__).resolve().parents[2]
    print(f"UYARI: Proje kökü marker dosyalarla belirlenemedi, varsayılan: {fallback_root}")
    return fallback_root

# Proje kök dizinini belirle
PROJECT_ROOT = find_project_root()

# Veri dizinleri
RAW_DATA      = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA  = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA= PROJECT_ROOT / "data" / "processed"

# Global değişken - aktif experiment dizini
_ACTIVE_EXPERIMENT_DIR = None

# Çıktı dizini belirleme (ortam değişkeninden veya varsayılan)
def get_output_base_dir():
    """
    Çıktı ana dizinini belirler.
    Önce OUTPUT_DIR ortam değişkenine bakar, yoksa varsayılan outputs/ kullanır.
    
    Returns:
        Path: Çıktı ana dizini
    """
    if "OUTPUT_DIR" in os.environ:
        output_dir = Path(os.environ["OUTPUT_DIR"])
    else:
        output_dir = PROJECT_ROOT / "outputs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# Experiment klasörleri için timestamp oluştur
def get_experiment_dir(create_new=False):
    """
    Aktif deneyler için bir timestamp klasörü döndürür.
    
    Args:
        create_new: True ise yeni bir experiment klasörü oluşturur,
                   False ise (varsayılan) aktif experiment klasörünü kullanır.
    
    Returns:
        Path: Experiment dizini (outputs/run_timestamp)
    """
    global _ACTIVE_EXPERIMENT_DIR
    
    output_base = get_output_base_dir()
    active_run_file = output_base / "active_run.txt"
    
    # Eğer create_new istendiyse, zorla yeni klasör oluştur
    if create_new:
        timestamp = int(time.time())
        _ACTIVE_EXPERIMENT_DIR = output_base / f"run_{timestamp}"
        _ACTIVE_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Aktif klasörü dosyaya yaz
        with open(active_run_file, "w") as f:
            f.write(str(_ACTIVE_EXPERIMENT_DIR))
            
        return _ACTIVE_EXPERIMENT_DIR
    
    # Eğer global değişken zaten ayarlandıysa, onu kullan
    if _ACTIVE_EXPERIMENT_DIR is not None:
        return _ACTIVE_EXPERIMENT_DIR
    
    # active_run.txt dosyasını kontrol et
    if active_run_file.exists():
        try:
            with open(active_run_file, "r") as f:
                run_dir = f.read().strip()
                if run_dir and Path(run_dir).exists():
                    _ACTIVE_EXPERIMENT_DIR = Path(run_dir)
                    return _ACTIVE_EXPERIMENT_DIR
        except Exception:
            pass  # Dosya okuma hatası olursa, yeni oluşturmaya devam et
    
    # Hiçbiri çalışmazsa yeni bir experiment klasörü oluştur
    timestamp = int(time.time())
    _ACTIVE_EXPERIMENT_DIR = output_base / f"run_{timestamp}"
    _ACTIVE_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Aktif klasörü dosyaya yaz
    with open(active_run_file, "w") as f:
        f.write(str(_ACTIVE_EXPERIMENT_DIR))
    
    return _ACTIVE_EXPERIMENT_DIR

def reset_experiment_dir():
    """
    Aktif experiment dizinini sıfırlar, bir sonraki get_experiment_dir() 
    çağrısında yeni bir klasör oluşturulur.
    """
    global _ACTIVE_EXPERIMENT_DIR
    _ACTIVE_EXPERIMENT_DIR = None

# MLflow tracking URI için fonksiyon
def get_mlflow_tracking_uri(exp_dir=None):
    """
    MLflow için tracking URI döndürür.
    
    Args:
        exp_dir: Experiment dizini (None ise aktif olan kullanılır)
        
    Returns:
        str: MLflow tracking URI
    """
    if exp_dir is None:
        exp_dir = get_experiment_dir()
    mlruns_dir = exp_dir / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    return f"file:{mlruns_dir.absolute()}"

def get_latest_experiment_dir():
    """
    En son oluşturulan experiment dizinini bulur.
    
    Returns:
        Path: En son experiment dizini veya None (bulunamazsa)
    """
    output_base = get_output_base_dir()
    run_dirs = list(output_base.glob("run_*"))
    
    if not run_dirs:
        return None
    
    # En son değiştirilen run_* klasörünü bul
    return max(run_dirs, key=lambda d: d.stat().st_mtime if d.is_dir() else 0)

def get_project_root() -> Path:
    """Proje kök dizinini döndürür"""
    return PROJECT_ROOT

def update_experiment_dir(new_dir: Union[str, Path]) -> Path:
    """
    Mevcut deney dizinini günceller
    
    Args:
        new_dir: Yeni deney dizini (string veya Path nesnesi)
        
    Returns:
        Path: Güncellenen deney dizini
    """
    global _ACTIVE_EXPERIMENT_DIR
    
    # String ise Path'e dönüştür
    if isinstance(new_dir, str):
        new_dir = Path(new_dir)
    
    # Dizinin var olduğundan emin ol
    new_dir.mkdir(parents=True, exist_ok=True)
    
    # Global değişkeni güncelle
    _ACTIVE_EXPERIMENT_DIR = new_dir
    
    return _ACTIVE_EXPERIMENT_DIR

def get_data_dir() -> Path:
    """
    Veri dizinini döndürür
    
    Returns:
        Path: Veri dizini
    """
    return PROJECT_ROOT / "data"

def get_raw_data_path() -> Path:
    """
    Ham veri dosyasının yolunu döndürür
    
    Returns:
        Path: Ham veri dosyası yolu
    """
    data_dir = get_data_dir()
    return data_dir / "raw" / "Conversion_Datamart.csv"

def get_config_dir() -> Path:
    """
    Yapılandırma dizinini döndürür
    
    Returns:
        Path: Yapılandırma dizini
    """
    return PROJECT_ROOT / "configs"

def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Dizinin var olduğundan emin olur, yoksa oluşturur
    
    Args:
        directory: Oluşturulacak dizin
        
    Returns:
        Path: Oluşturulan dizin
    """
    if isinstance(directory, str):
        directory = Path(directory)
    
    directory.mkdir(parents=True, exist_ok=True)
    return directory

# Yeni fonksiyon ekleyelim - data dizinini kontrol eden
def check_data_directories():
    """
    Gerekli data dizinlerinin varlığını kontrol eder ve yoksa oluşturur.
    
    Returns:
        dict: Dizin bilgileri
    """
    from pathlib import Path
    
    # Ana dizinler
    data_dir = Path(PROJECT_ROOT) / "data"
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"
    
    # Dizinleri oluştur
    data_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    interim_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    
    # Split verileri var mı kontrol et
    split_files = {
        "train": processed_dir / "train.csv",
        "validation": processed_dir / "validation.csv",
        "test": processed_dir / "test.csv"
    }
    
    exists = {name: path.exists() for name, path in split_files.items()}
    
    # Split dizini ve durumu
    results = {
        "data_dir": str(data_dir),
        "raw_dir": str(raw_dir),
        "interim_dir": str(interim_dir),
        "processed_dir": str(processed_dir),
        "split_exists": all(exists.values()),
        "split_files": {name: str(path) for name, path in split_files.items()},
        "split_status": exists
    }
    
    return results
