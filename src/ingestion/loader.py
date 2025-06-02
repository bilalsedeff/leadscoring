import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.paths import RAW_DATA, INTERIM_DATA
from omegaconf import OmegaConf
import logging

logger = logging.getLogger(__name__)

def read_datamart() -> pd.DataFrame:
    cfg = OmegaConf.load("configs/data.yaml")
    
    # OmegaConf nesnelerini Python dict'e dönüştür
    dtype_map = dict(cfg.dtypes) if hasattr(cfg, "dtypes") else {}
    dt_cols = cfg.datetime_cols
    
    # dt_cols'un bir liste olduğundan emin olalım
    if dt_cols and not isinstance(dt_cols, list):
        if isinstance(dt_cols, str):
            # Virgülle ayrılmış stringi listeye dönüştür ve boşlukları temizle
            dt_cols = [col.strip() for col in dt_cols.split(',')]
        else:
            dt_cols = list(dt_cols) if hasattr(dt_cols, '__iter__') else []

    # Integer sütunlarını Int64 tipine dönüştür (NA destekli)
    for col, dtype in dtype_map.items():
        if dtype in ['int8', 'int16', 'int32', 'int64', 'int']:
            # int8, int16 gibi tipleri pandas NA destekli Int64 olarak değiştir
            dtype_map[col] = 'Int64'  # Büyük I ile Int64 pandas'ın NA destekli tipi
            logger.debug(f"{col} sütunu {dtype} -> Int64 olarak değiştirildi (NA desteği için)")

    # Önce CSV dosyasının başlıklarını al (parse_dates olmadan)
    try:
        # Önce virgül ile dene
        try:
            headers = pd.read_csv(RAW_DATA / "Conversion_Datamart.csv", sep=';', nrows=0).columns.tolist()
        except:
            # Virgül başarısız olursa noktalı virgül ile dene
            headers = pd.read_csv(RAW_DATA / "Conversion_Datamart.csv", sep=',', nrows=0).columns.tolist()
            
        # Eğer tek bir kolon varsa, dosya yanlış ayrılmış olabilir
        if len(headers) == 1:
            # Noktalı virgül ile tekrar dene
            temp_headers = pd.read_csv(RAW_DATA / "Conversion_Datamart.csv", sep=';', nrows=0).columns.tolist()
            if len(temp_headers) > 1:
                headers = temp_headers
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV dosyası bulunamadı: {RAW_DATA / 'Conversion_Datamart.csv'}")
    # Sadece CSV'de var olan tarih sütunlarını kullan
    existing_dt_cols = [col for col in dt_cols if col in headers]
    
    if len(existing_dt_cols) < len(dt_cols):
        missing_cols = set(dt_cols) - set(existing_dt_cols)
        logger.warning(f"Uyarı: Bazı tarih sütunları CSV'de bulunamadı ve atlandı: {missing_cols}")
    
    # CSV'yi oku
    # Önce virgül ile dene, eğer tek kolon varsa noktalı virgül ile tekrar dene
    try:
        # İlk deneme - noktalı virgül ayırıcı ile
        try:
            df = pd.read_csv(RAW_DATA / "Conversion_Datamart.csv",
                            sep=';',
                            dtype=dtype_map,
                            parse_dates=existing_dt_cols if existing_dt_cols else None,
                            low_memory=False)
            
            if len(df.columns) == 1:
                # Dosya yanlış ayrılmış olabilir, virgül ile tekrar dene
                df = pd.read_csv(RAW_DATA / "Conversion_Datamart.csv",
                                sep=',',
                                dtype=dtype_map,
                                parse_dates=existing_dt_cols if existing_dt_cols else None,
                                low_memory=False)
        except ValueError as e:
            # Integer sütunlardaki NA değerleri için hata oluştuysa
            logger.warning(f"İlk CSV okuma denemesi başarısız oldu: {e}")
            logger.info("NA değerleri olan tam sayı sütunları için alternatif yöntem deneniyor...")
            
            # Veri tiplerini belirtmeden oku ve sonra dönüşüm yap
            df = pd.read_csv(RAW_DATA / "Conversion_Datamart.csv",
                             sep=';',
                             parse_dates=existing_dt_cols if existing_dt_cols else None,
                             low_memory=False)
            
            if len(df.columns) == 1:
                # Dosya yanlış ayrılmış olabilir, virgül ile tekrar dene
                df = pd.read_csv(RAW_DATA / "Conversion_Datamart.csv",
                                sep=',',
                                parse_dates=existing_dt_cols if existing_dt_cols else None,
                                low_memory=False)
            
            # Şimdi veri tiplerini dönüştür
            for col, dtype in dtype_map.items():
                if col in df.columns:
                    try:
                        if dtype in ['int8', 'int16', 'int32', 'int64', 'int']:
                            df[col] = df[col].astype('Int64')  # NA destekli integer
                        elif dtype in ['float32', 'float64', 'float']:
                            df[col] = df[col].astype(dtype)
                        elif dtype == 'category':
                            df[col] = df[col].astype(dtype)
                    except Exception as conv_err:
                        logger.warning(f"{col} sütunu {dtype} tipine dönüştürülürken hata: {conv_err}")
                        # Hatada devam et, tip dönüşümü olmadan bırak
    except Exception as e:
        raise RuntimeError(f"CSV dosyası okunurken hata oluştu: {e}")
    
    # Convert NULLs to NaNs
    df = df.replace({None: np.nan})
    # Convert 'NULL's to NaNs
    df = df.replace("NULL", np.nan)
    # optional cache
    df.to_parquet(INTERIM_DATA / "datamart_raw.parquet", index=False)
    return df
