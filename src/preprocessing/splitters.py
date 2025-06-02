from sklearn.model_selection import StratifiedGroupKFold
import numpy as np, pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime
import logging
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def load_split_config():
    """
    split.yaml yapılandırma dosyasını yükler
    
    Returns:
        dict: Split yapılandırması
    """
    config_path = Path("configs/split.yaml")
    if not config_path.exists():
        logger.warning("split.yaml bulunamadı, varsayılan yapılandırma kullanılacak.")
        return {
            "cutoff": "2024-07-01",
            "group_col": "account_Id",
            "target": "Target_IsConverted",
            "time_col": "LeadCreatedDate",
            "cv_folds": 5,
            "cv_strategy": "time_group",
            "random_seed": 42,
            "use_percentage_split": False,
            "test_size": 0.2,
            "validation_size": 0.15
        }
    
    return OmegaConf.load(config_path)

def time_group_split(df: pd.DataFrame, cutoff: str = None,
                     val_cutoff: str = None, test_cutoff: str = None, 
                     time_col: str = "LeadCreatedDate", group_col: str = "account_Id", 
                     target: str = "Target_IsConverted", force_balance: bool = False,
                     random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Zamansal bazlı train/validation/test bölünmesi yapar.
    
    Args:
        df: Bölünecek veri çerçevesi
        cutoff: Train seti için kesme tarihi (YYYY-MM-DD formatı, ör: "2024-06-30")
        val_cutoff: Validation seti için kesme tarihi (YYYY-MM-DD formatı, ör: "2024-11-30") 
        test_cutoff: Test seti için kesme tarihi (YYYY-MM-DD formatı, ör: "2025-04-30")
        time_col: Zaman kolonu adı (LeadCreatedDate)
        group_col: Gruplama için kolon adı (örn: account_Id)
        target: Hedef değişken adı
        force_balance: Her bir sette mutlaka veri olmasını zorla (True ise)
        random_seed: Rastgele sayı üreteci için tohum değeri
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train, validation, test) veri setleri
    """
    # Rastgele sayı üretecini başlat
    np.random.seed(random_seed)
    
    # Yapılandırmayı yükle ve varsayılan değerler ata - KULLANICININ İSTEDİĞİ ARALIKLAR
    config_path = Path("configs/split.yaml")
    if config_path.exists():
        config = OmegaConf.load(config_path)
        cutoff = cutoff or config.get('train_cutoff', "2024-06-30")
        val_cutoff = val_cutoff or config.get('val_cutoff', "2024-11-30")
        test_cutoff = test_cutoff or config.get('test_cutoff', "2025-04-30")
        time_col = time_col or config.get('time_col', "LeadCreatedDate")
        group_col = group_col or config.get('group_col', "account_Id")
        target = target or config.get('target_col', "Target_IsConverted")
    else:
        # Varsayılan değerler - kullanıcının istediği tarih aralıkları
        cutoff = cutoff or "2024-06-30"
        val_cutoff = val_cutoff or "2024-11-30" 
        test_cutoff = test_cutoff or "2025-04-30"
    
    logger.info(f"Split parametreleri: train_cutoff={cutoff}, val_cutoff={val_cutoff}, test_cutoff={test_cutoff}")
    
    # Zaman kolonunun varlığını kontrol et
    if time_col not in df.columns:
        # İlgili tarih kolonunu arama
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'Date' in col]
        if date_cols:
            time_col = date_cols[0]
            logger.info(f"Belirtilen zaman kolonu bulunamadı, {time_col} kullanılacak")
        else:
            raise ValueError(f"Zaman kolonu bulunamadı: {time_col}")
    
    # Zaman kolonunu datetime tipine dönüştür - SADECE YYYY-MM-DD formatı
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df = df.copy()
            # Direkt YYYY-MM-DD formatında datetime'a dönüştür
            df[time_col] = pd.to_datetime(df[time_col], format='%Y-%m-%d', errors='coerce')
            
            # Eğer format uyumsuzsa infer ile dene
            if df[time_col].isna().any():
                df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True, errors='coerce')
            
            logger.info(f"{time_col} kolonu datetime tipine dönüştürüldü")
        except Exception as e:
            raise ValueError(f"Zaman kolonu datetime'a dönüştürülemedi: {e}")
    
    # Cutoff değerlerini datetime'a dönüştür
    try:
        cutoff_dt = pd.to_datetime(cutoff)
        val_cutoff_dt = pd.to_datetime(val_cutoff)
        test_cutoff_dt = pd.to_datetime(test_cutoff)
    except Exception as e:
        raise ValueError(f"Cutoff tarihleri geçersiz format: {e}")
    
    # Tarih aralığını kontrol et
    time_min = df[time_col].min()
    time_max = df[time_col].max()
    logger.info(f"Veri tarih aralığı: {time_min} - {time_max}")
    logger.info(f"Split tarihleri: Train ≤ {cutoff_dt}, Val: {cutoff_dt} < x ≤ {val_cutoff_dt}, Test: {val_cutoff_dt} < x ≤ {test_cutoff_dt}")
    
    # Maskeler oluştur
    train_mask = (df[time_col] <= cutoff_dt)
    val_mask = (df[time_col] > cutoff_dt) & (df[time_col] <= val_cutoff_dt)
    test_mask = (df[time_col] > val_cutoff_dt) & (df[time_col] <= test_cutoff_dt)
    
    # Grup bütünlüğünü korumak için grup bazlı split yap
    if group_col and group_col in df.columns:
        logger.info(f"Grup bütünlüğü korunarak veri bölünüyor ({group_col})...")
        
        # Her grup için minimum tarihini bul
        group_min_dates = df.groupby(group_col)[time_col].min()
        
        # Grupları tarihe göre sınıflandır
        train_groups = group_min_dates[group_min_dates <= cutoff_dt].index
        val_groups = group_min_dates[(group_min_dates > cutoff_dt) & (group_min_dates <= val_cutoff_dt)].index
        test_groups = group_min_dates[group_min_dates > val_cutoff_dt].index
        
        # Veriyi gruplara göre böl
        train = df[df[group_col].isin(train_groups)].copy()
        validation = df[df[group_col].isin(val_groups)].copy()
        test = df[df[group_col].isin(test_groups)].copy()
        
        logger.info(f"Grup bazlı split: Train={len(train_groups)} grup, Val={len(val_groups)} grup, Test={len(test_groups)} grup")
        
        # Grup kesişimi kontrolü
        train_groups_set = set(train_groups)
        val_groups_set = set(val_groups) 
        test_groups_set = set(test_groups)
        
        if train_groups_set.intersection(val_groups_set):
            logger.warning("Train ve validation grupları arasında kesişim var!")
        if train_groups_set.intersection(test_groups_set):
            logger.warning("Train ve test grupları arasında kesişim var!")
        if val_groups_set.intersection(test_groups_set):
            logger.warning("Validation ve test grupları arasında kesişim var!")
    else:
        # Grup kolonu yoksa direkt tarih bazlı split
        train = df[train_mask].copy()
        validation = df[val_mask].copy()
        test = df[test_mask].copy()
        logger.info("Grup kolonu olmadan direkt tarih bazlı split yapıldı")
    
    # Split sonuçlarını kontrol et
    train_count = len(train)
    val_count = len(validation)
    test_count = len(test)
    
    logger.info(f"Split sonuçları - Train: {train_count}, Validation: {val_count}, Test: {test_count}")
    
    # Boş setler varsa uyarı ver
    if train_count == 0:
        logger.error("Train seti boş!")
    if val_count == 0:
        logger.error("Validation seti boş!")
    if test_count == 0:
        logger.error("Test seti boş!")
    
    # Hedef değişken dağılımını kontrol et
    if target and target in df.columns:
        train_pos = train[target].mean() if train_count > 0 else 0
        val_pos = validation[target].mean() if val_count > 0 else 0
        test_pos = test[target].mean() if test_count > 0 else 0
        logger.info(f"Hedef dağılımı - Train: {train_pos:.2%}, Val: {val_pos:.2%}, Test: {test_pos:.2%}")
    
    # Tarih aralıklarını kontrol et
    if train_count > 0:
        logger.info(f"Train tarih aralığı: {train[time_col].min()} - {train[time_col].max()}")
    if val_count > 0:
        logger.info(f"Val tarih aralığı: {validation[time_col].min()} - {validation[time_col].max()}")
    if test_count > 0:
        logger.info(f"Test tarih aralığı: {test[time_col].min()} - {test[time_col].max()}")
    
    return train, validation, test

def stratified_group_split(df: pd.DataFrame, group_col: str, target_col: str,
                           test_size: float = 0.2, val_size: float = 0.15, 
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified grup bazlı bölme yapar.
    
    Aynı gruba ait örnekler aynı split'e düşer ve hedef değişken oranları korunur.
    
    Args:
        df: Bölünecek veri çerçevesi
        group_col: Grup kolonu
        target_col: Hedef değişken
        test_size: Test setinin oranı
        val_size: Validation setinin oranı
        random_state: Rastgele durum
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train, validation, test) veri setleri
    """
    # Yapılandırmayı yükle
    config_path = Path("configs/split.yaml")
    if config_path.exists():
        config = OmegaConf.load(config_path)
        
        # Parametreler tanımlanmamışsa yapılandırmadan al
        group_col = group_col or config.get('group_col')
        target_col = target_col or config.get('target_col')
        test_size = test_size or config.get('test_size', 0.2)
        val_size = val_size or config.get('val_size', 0.15)
        random_state = random_state or config.get('random_state', 42)
    
    # Grup ve hedef kolonlarının varlığını kontrol et
    if group_col not in df.columns:
        raise ValueError(f"Grup kolonu bulunamadı: {group_col}")
    
    if target_col not in df.columns:
        raise ValueError(f"Hedef kolonu bulunamadı: {target_col}")
    
    # Grup bazlı özellikleri hesapla
    groups = df[group_col].unique()
    
    # Grup düzeyinde stratifikasyon için, her grup için hedef değişken ortalamasını hesapla
    group_targets = df.groupby(group_col)[target_col].mean()
    
    # Hedef ortalamasına göre grupları sınıflara ayır
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    group_targets_binned = pd.cut(group_targets, bins=bins, labels=False)
    group_strata = pd.DataFrame({
        group_col: group_targets.index,
        'strata': group_targets_binned
    })
    
    # Stratified split için grup listesi oluştur
    X = group_strata[[group_col]]
    y = group_strata['strata']
    
    # Önce test setini ayır
    temp_ratio = test_size + val_size
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size / temp_ratio, 
        stratify=y, random_state=random_state
    )
    
    # Sonra validation setini ayır
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size / (1 - test_size),
        stratify=y_temp, random_state=random_state
    )
    
    # Grup listelerini oluştur
    train_groups = X_train[group_col].tolist()
    val_groups = X_val[group_col].tolist()
    test_groups = X_test[group_col].tolist()
    
    # Orijinal veri setini gruplara göre böl
    train = df[df[group_col].isin(train_groups)].copy()
    validation = df[df[group_col].isin(val_groups)].copy()
    test = df[df[group_col].isin(test_groups)].copy()
    
    # Hedef dağılımını kontrol et
    train_pos = train[target_col].mean()
    val_pos = validation[target_col].mean()
    test_pos = test[target_col].mean()
    
    logger.info(f"Hedef dağılımı - Train: {train_pos:.2%}, Validation: {val_pos:.2%}, Test: {test_pos:.2%}")
    logger.info(f"Split sonuçları - Train: {train.shape}, Validation: {validation.shape}, Test: {test.shape}")
    
    return train, validation, test

def date_based_split(df, cutoff, config, group_col, target, time_col):
    """
    Tarih bazlı veri bölme
    
    Args:
        df: Veri çerçevesi
        cutoff: Eğitim/test ayrım tarihi
        config: Split yapılandırması
        group_col: Grup sütunu
        target: Hedef değişken
        time_col: Tarih sütunu
        
    Returns:
        tuple: (train, validation, test) veya (train, hold, folds)
    """
    # Cutoff değerini belirle
    if cutoff is None:
        # Eski format için train_end parametresini kontrol et (geriye uyumluluk)
        cutoff = config.get("cutoff", config.get("train_end", "2024-07-31"))
    
    # Detaylı tarih aralıkları
    train_end = config.get("train_end", cutoff)
    validation_start = config.get("validation_start", cutoff)
    validation_end = config.get("validation_end", None)
    test_start = config.get("test_start", validation_end)
    test_end = config.get("test_end", None)
    
    # Tarihleri datetime nesnelerine dönüştür
    train_end = pd.to_datetime(train_end)
    validation_start = pd.to_datetime(validation_start)
    validation_end = pd.to_datetime(validation_end) if validation_end else None
    test_start = pd.to_datetime(test_start) if test_start else validation_end
    test_end = pd.to_datetime(test_end) if test_end else None
    
    logger.info(f"Tarih bazlı bölme: Train <= {train_end}, Validation: {validation_start} - {validation_end}, Test: {test_start} - {test_end}")
    
    # Veriyi zaman bazlı olarak böl
    train = df[df[time_col] <= train_end]
    
    # Eski format uyumluluğu - sadece train ve hold_out
    if validation_end is None:
        hold = df[df[time_col] > train_end]
        
        # Folds oluştur (CV için)
        cv_folds = config.get("cv_folds", 5)
        sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=config.get("random_seed", 42))
        folds = list(sgkf.split(train, train[target], groups=train[group_col])) if group_col and target else None
        
        return train, hold, folds
    
    # Yeni format - train, validation ve test
    validation = df[(df[time_col] > validation_start) & (df[time_col] <= validation_end)]
    
    if test_end:
        test = df[(df[time_col] > test_start) & (df[time_col] <= test_end)]
    else:
        test = df[df[time_col] > test_start]
    
    # Grup bütünlüğü kontrolü
    if group_col:
        train_groups = set(train[group_col].unique())
        val_groups = set(validation[group_col].unique()) - train_groups
        test_groups = set(test[group_col].unique()) - train_groups - val_groups
        
        # Grupları ayrıştır
        train = df[df[group_col].isin(train_groups)]
        validation = df[df[group_col].isin(val_groups)]
        test = df[df[group_col].isin(test_groups)]
        
        logger.info(f"Grup bazlı bölme sonrası: Train: {len(train)} satır, {len(train_groups)} grup")
        logger.info(f"Validation: {len(validation)} satır, {len(val_groups)} grup")
        logger.info(f"Test: {len(test)} satır, {len(test_groups)} grup")
    
    # CV folds oluştur
    cv_folds = config.get("cv_folds", 5)
    sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=config.get("random_seed", 42))
    folds = list(sgkf.split(train, train[target], groups=train[group_col])) if group_col and target else None
    
    return train, validation, test, folds

def percentage_split(df, config, group_col, target, time_col):
    """
    Yüzdesel veri bölme (zaman sırasına göre)
    
    Args:
        df: Veri çerçevesi
        config: Split yapılandırması
        group_col: Grup sütunu
        target: Hedef değişken
        time_col: Tarih sütunu
        
    Returns:
        tuple: (train, validation, test, folds)
    """
    # Yüzde değerlerini al
    test_size = config.get("test_size", 0.2)
    validation_size = config.get("validation_size", 0.15)
    
    logger.info(f"Yüzdesel bölme: Train: %{100-(test_size+validation_size)*100:.1f}, "
                f"Validation: %{validation_size*100:.1f}, Test: %{test_size*100:.1f}")
    
    # Eğer zaman sütunu varsa, zamana göre sırala
    if time_col and time_col in df.columns:
        df = df.sort_values(by=time_col)
    
    # Eğer grup sütunu varsa, grupları koru
    if group_col and group_col in df.columns:
        # Zaman sırasına göre grupları sırala (her grubun son tarihine göre)
        if time_col and time_col in df.columns:
            group_last_time = df.groupby(group_col)[time_col].max().reset_index()
            group_last_time = group_last_time.sort_values(by=time_col)
            sorted_groups = group_last_time[group_col].tolist()
        else:
            # Zaman sütunu yoksa rastgele sırala
            sorted_groups = df[group_col].unique().tolist()
            np.random.shuffle(sorted_groups)
        
        # Toplam grup sayısını hesapla
        n_groups = len(sorted_groups)
        test_groups_n = int(n_groups * test_size)
        val_groups_n = int(n_groups * validation_size)
        
        # Grupları böl
        test_groups = set(sorted_groups[-test_groups_n:])
        val_groups = set(sorted_groups[-(test_groups_n+val_groups_n):-test_groups_n])
        train_groups = set(sorted_groups[:-(test_groups_n+val_groups_n)])
        
        # Veriyi böl
        train = df[df[group_col].isin(train_groups)]
        validation = df[df[group_col].isin(val_groups)]
        test = df[df[group_col].isin(test_groups)]
        
        logger.info(f"Grup bazlı yüzdesel bölme: Train: {len(train)} satır, {len(train_groups)} grup")
        logger.info(f"Validation: {len(validation)} satır, {len(val_groups)} grup")
        logger.info(f"Test: {len(test)} satır, {len(test_groups)} grup")
    else:
        # Grup sütunu yoksa indeksleri böl
        n_samples = len(df)
        test_n = int(n_samples * test_size)
        val_n = int(n_samples * validation_size)
        
        test = df.iloc[-test_n:]
        validation = df.iloc[-(test_n+val_n):-test_n]
        train = df.iloc[:-(test_n+val_n)]
        
        logger.info(f"Satır bazlı yüzdesel bölme: Train: {len(train)}, Validation: {len(validation)}, Test: {len(test)}")
    
    # CV folds oluştur
    cv_folds = config.get("cv_folds", 5)
    sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=config.get("random_seed", 42))
    folds = list(sgkf.split(train, train[target], groups=train[group_col])) if group_col and target else None
    
    return train, validation, test, folds
