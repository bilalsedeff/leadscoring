"""
python -m src.pipelines.statistical_tests --test_type=<test_tipi> (CLI'dan çağrılır)
"""
import os
import click
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import logging
from typing import List, Dict, Optional, Union, Any
import seaborn as sns

# Matplotlib ayarları - açık figür sayısını sınırla
import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 50  # Varsayılan 20 yerine 50'ye çıkar
plt.style.use('default')  # Daha temiz bir stil kullan

from src.ingestion.loader import read_datamart
from src.preprocessing.cleaning import BasicCleaner
from src.features.engineering import add_temporal_features, AdvancedFeatureEngineer
from src.preprocessing.splitters import time_group_split, load_split_config
from src.evaluation.statistical_tests import (
    chi_square_test,
    t_test_by_group,
    anova_test,
    conversion_rate_comparison,
    correlation_analysis
)
from src.utils.logger import get_logger
from src.utils.paths import get_experiment_dir

# Logger
logger = logging.getLogger("lead_scoring.statistical_tests")

# ID ve tarih kolonları - her zaman dışlanacak
EXCLUDE_COLUMNS = [
    # ID kolonları
    "LeadId", "account_Id", "Id", "Opportunityid", "Contactid", "AccountId",
    # Tarih kolonları
    "LeadCreatedDate", "ConvertedDate", "CloseDate", "LastActivityDate", 
    "LastModifiedDate", "SystemModstamp", "CreatedDate"
]

# Yüksek kardinaliteli branch name kolonları
HIGH_CARDINALITY_COLUMNS = [
    "Branch_Name__c", "BranchName"
]

def load_data(train_path, use_train_only, train_cutoff=None, val_cutoff=None):
    """
    Veri setini yükler. Eğitim seti yoksa ham veriyi yükleyip bölme yapar.
    
    Args:
        train_path: Eğitim seti yolu
        use_train_only: Sadece eğitim verisi kullanılsın mı?
        train_cutoff: Eğitim kesme noktası
        val_cutoff: Validasyon kesme noktası
        
    Returns:
        pd.DataFrame: Veri seti
    """
    # Birden fazla potansiyel yolu kontrol et
    potential_paths = [
        train_path,                                  # Belirtilen yol
        Path("data/processed/train.csv"),            # data/processed altındaki train.csv
    ]
    
    # En son run dizinindeki split'i bul
    try:
        from src.utils.paths import get_experiment_dir
        exp_dir = get_experiment_dir()
        potential_paths.append(exp_dir / "train.csv")
    except Exception as e:
        logger.warning(f"Experiment dizini belirlenirken hata: {e}")
    
    # Outputs dizinindeki tüm run klasörlerini kontrol et
    for run_dir in sorted(Path("outputs").glob("run_*"), reverse=True):
        potential_paths.append(run_dir / "train.csv")
    
    # Tüm potansiyel yolları kontrol et
    loaded_path = None
    for path in potential_paths:
        if isinstance(path, str):
            path = Path(path)
        
        if path.exists():
            loaded_path = path
            logger.info(f"Eğitim veri seti yükleniyor: {path}")
            try:
                df = pd.read_csv(path, low_memory=False)  # DtypeWarning'i önlemek için low_memory=False
                return df
            except Exception as e:
                logger.error(f"Veri yüklenirken hata: {e}")
                continue
    
    # Hiçbir veri seti bulunamadıysa
    if loaded_path is None:
        logger.warning(f"Hiçbir split verisi bulunamadı. Kontrol edilen yerler:")
        for path in potential_paths:
            logger.warning(f"  - {path}")
    else:
        logger.warning(f"Veri seti bulundu ama yüklenemedi: {loaded_path}")
    
    # Ham veriyi yüklemeyi dene
    logger.info("Ham veriyi yüklüyorum...")
    try:
        # Ham veriyi oku
        raw_df = read_datamart()
    except Exception as e:
        logger.error(f"Ham veri yüklenirken hata: {e}")
        raise ValueError(f"Ne split edilmiş veri ne de ham veri yüklenemedi: {e}")
    
    # Eğer train_cutoff veya val_cutoff belirtilmemişse, split.yaml'dan almaya çalış
    if not train_cutoff or not val_cutoff:
        try:
            from omegaconf import OmegaConf
            config_path = Path("configs/split.yaml")
            if config_path.exists():
                config = OmegaConf.load(config_path)
                train_cutoff = train_cutoff or config.get("train_cutoff")
                val_cutoff = val_cutoff or config.get("val_cutoff")
                logger.info(f"Split yapılandırmasından kesme noktaları alındı: train={train_cutoff}, val={val_cutoff}")
        except Exception as e:
            logger.warning(f"Split yapılandırması yüklenirken hata: {e}")
    
    # Hala cutoff yoksa varsayılan değerler kullan - KULLANICININ İSTEDİĞİ TARİHLER
    if not train_cutoff:
        train_cutoff = "2024-06-30"  # Train ≤ 2024-06-30
        logger.warning(f"Train cutoff belirtilmemiş, varsayılan değer kullanılıyor: {train_cutoff}")
    
    if not val_cutoff:
        val_cutoff = "2024-11-30"  # Validation: 2024-07-01 to 2024-11-30
        logger.warning(f"Validation cutoff belirtilmemiş, varsayılan değer kullanılıyor: {val_cutoff}")
    
    # Veriyi böl
    try:
        train_df, val_df, test_df = time_group_split(
            raw_df, 
            cutoff=train_cutoff, 
            val_cutoff=val_cutoff,
            test_cutoff="2025-04-30",  # Test: 2024-12-01 to 2025-04-30
            time_col="LeadCreatedDate" if "LeadCreatedDate" in raw_df.columns else None,
            group_col="account_Id" if "account_Id" in raw_df.columns else None
        )
        
        # Veriyi kaydet
        try:
            # Processed dizini varsa oraya kaydet
            processed_dir = Path("data/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            train_df.to_csv(processed_dir / "train.csv", index=False)
            val_df.to_csv(processed_dir / "validation.csv", index=False)
            test_df.to_csv(processed_dir / "test.csv", index=False)
            logger.info(f"Bölünmüş veri setleri processed dizinine kaydedildi: {processed_dir}")
        except Exception as e:
            logger.warning(f"Bölünmüş veri setleri kaydedilirken hata: {e}")
        
        # Sadece eğitim verisini mi kullanacağız?
        if use_train_only:
            logger.info("Sadece eğitim verisi kullanılıyor")
            return train_df
        else:
            logger.info("Tüm veri kullanılıyor (train + validation + test)")
            # Veri sızıntısı riski!
            return pd.concat([train_df, val_df, test_df], ignore_index=True)
    except Exception as e:
        logger.error(f"Veri bölme işlemi sırasında hata: {e}")
        logger.warning("Ham veri doğrudan kullanılıyor (veri sızıntısı riski!)")
        return raw_df

@click.command()
@click.option('--test_type', type=click.Choice([
    'chi_square', 't_test', 'anova', 'conversion_rate', 'correlation', 'all'
]), default='all', help='Yapılacak istatistiksel test tipi')
@click.option('--categorical_cols', '-c', multiple=True, help='Kategorik değişkenler (virgülle ayrılmış)')
@click.option('--numeric_cols', '-n', multiple=True, help='Sayısal değişkenler (virgülle ayrılmış)')
@click.option('--group_col', '-g', help='Grup değişkeni (t-test, ANOVA ve conversion rate için)')
@click.option('--target_col', '-t', default='Target_IsConverted', help='Hedef değişken')
@click.option('--corr_method', type=click.Choice(['pearson', 'spearman', 'kendall']), 
             default='pearson', help='Korelasyon yöntemi')
@click.option('--min_corr', type=float, default=0.05, help='Minimum korelasyon değeri (mutlak)')
@click.option('--output_subdir', help='Çıktı alt dizini (varsayılan: timestamp)')
@click.option('--train_path', default="data/processed/train.csv", help='Eğitim veri seti yolu')
@click.option('--use_train_only/--use_all_data', default=True, help='Sadece eğitim verisini mi kullanmak istiyorsunuz?')
@click.option('--exclude_id_cols/--include_id_cols', default=True, help='Kimlik kolonlarını analiz dışında bırak')
@click.option('--exclude_date_cols/--include_date_cols', default=True, help='Tarih kolonlarını analiz dışında bırak')
@click.option('--exclude_high_cardinality/--include_high_cardinality', default=True, help='Yüksek kardinaliteli kolonları analiz dışında bırak')
@click.option('--train_cutoff', help='Eğitim için kesim tarihi (YYYY-MM-DD formatında)')
@click.option('--val_cutoff', help='Validasyon için kesim tarihi (YYYY-MM-DD formatında)')
def main(test_type, categorical_cols, numeric_cols, group_col, target_col, 
        corr_method, min_corr, output_subdir, train_path, use_train_only, exclude_id_cols,
        exclude_date_cols, exclude_high_cardinality, train_cutoff, val_cutoff):
    """İstatistiksel testler pipeline'ı.
    
    Veri üzerinde çeşitli istatistiksel testler gerçekleştirir ve sonuçları görselleştirir.
    """
    # Log
    logger.info(f"İstatistiksel test pipeline'ı başlatılıyor. Test tipi: {test_type}")
            
    # Dışlanacak kolonlar listesi
    exclude_cols = []
    
    # ID ve tarih kolonlarını dışla
    if exclude_id_cols:
        id_cols = [col for col in EXCLUDE_COLUMNS if "Date" not in col]
        exclude_cols.extend(id_cols)
        logger.info(f"ID kolonları analizden çıkarılacak: {id_cols}")
        
    if exclude_date_cols:
        date_cols = [col for col in EXCLUDE_COLUMNS if "Date" in col]
        exclude_cols.extend(date_cols)
        # Ayrıca *Date ile biten tüm kolonları da hariç tut
        logger.info(f"Tarih kolonları analizden çıkarılacak: {date_cols}")
    
    if exclude_high_cardinality:
        exclude_cols.extend(HIGH_CARDINALITY_COLUMNS)
        logger.info(f"Yüksek kardinaliteli kolonlar analizden çıkarılacak: {HIGH_CARDINALITY_COLUMNS}")
    
    # Veri seti hazırlığı
    try:
        # Veriyi yükle (eğitim verisi yoksa ham veriyi bölecek)
        df = load_data(train_path, use_train_only, train_cutoff, val_cutoff)
    except Exception as e:
        logger.error(f"Veri yüklenirken hata: {e}")
        raise
    
    # Veri seti bilgisi
    logger.info(f"Veri seti boyutu: {df.shape}")
    
    # Çıktı dizini hazırlığı - experiment directory altında statistical_tests klasörü
    timestamp = int(time.time())
    data_source_label = 'train_only' if use_train_only else 'all_data'
    output_subdir = output_subdir or f"{test_type}_{timestamp}_{data_source_label}"
    
    # Experiment directory altında statistical_tests klasörü oluştur
    from src.utils.paths import get_experiment_dir
    exp_dir = get_experiment_dir()
    statistical_tests_dir = exp_dir / "statistical_tests"  # statistical_tests klasörü eklendi
    output_dir = statistical_tests_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Çıktılar kaydedilecek: {output_dir}")
            
    # Sayısal ve kategorik kolonları belirle
    if not categorical_cols:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Dışlanacak kolonları çıkar
        cat_cols = [col for col in cat_cols if col not in exclude_cols and col != target_col]
    else:
        cat_cols = list(categorical_cols)
    
    if not numeric_cols:
        num_cols = df.select_dtypes(include=['int', 'float', 'Int64']).columns.tolist()
        # Dışlanacak kolonları çıkar
        num_cols = [col for col in num_cols if col not in exclude_cols and col != target_col]
    else:
        num_cols = list(numeric_cols)
    
    # Tarih (Date) içeren tüm kolonları dışla (ek kontrol)
    if exclude_date_cols:
        cat_cols = [col for col in cat_cols if not (("date" in col.lower()) or ("Date" in col))]
        num_cols = [col for col in num_cols if not (("date" in col.lower()) or ("Date" in col))]
    
    logger.info(f"Analiz edilecek kategorik kolonlar: {len(cat_cols)} adet")
    logger.info(f"Analiz edilecek sayısal kolonlar: {len(num_cols)} adet")
    
    # İlgili test tipine göre analiz yapma
    if test_type in ['chi_square', 'all']:
        # Ki-kare testleri
        logger.info(f"Ki-kare testleri çalıştırılıyor ({len(cat_cols)} kategorik değişken)...")
        chi_results = []
        
        for col in cat_cols:
            try:
                # Ki-kare testini uygula
                result = chi_square_test(df, col, target_col, output_dir=output_dir)
                chi_results.append(result)
                
                # Sonuçları logla
                if 'p_value' in result:
                    sig_text = "ANLAMLI" if result.get('significant', False) else "anlamsız"
                    logger.info(f"{col}: chi2={result.get('chi2', 'N/A')}, p={result.get('p_value', 'N/A')}, {sig_text}")
            except Exception as e:
                logger.error(f"{col} için ki-kare testi sırasında hata: {e}")
    
    if test_type in ['t_test', 'all']:
        # T-test işlemleri
        if group_col in df.columns:
            logger.info(f"T-testleri çalıştırılıyor ({len(num_cols)} sayısal değişken)...")
        
            for col in num_cols:
                try:
                    result = t_test_by_group(df, col, group_col, output_dir=output_dir)
                    
                    # Sonuçları logla
                    if 'p_value' in result:
                        sig_text = "ANLAMLI" if result.get('significant', False) else "anlamsız"
                        logger.info(f"{col}: t={result.get('t_statistic', 'N/A')}, p={result.get('p_value', 'N/A')}, {sig_text}")
                except Exception as e:
                    logger.error(f"{col} için t-test sırasında hata: {e}")
        else:
            # Otomatik grup değişkeni bulma
            # Hedef değişken ikili sınıflandırma (0,1) için ideal bir grup değişkenidir
            if target_col in df.columns and df[target_col].nunique() <= 2:
                auto_group_col = target_col
                logger.info(f"T-test için otomatik olarak hedef değişken '{target_col}' grup değişkeni olarak kullanılıyor")
                
                for col in num_cols:
                    try:
                        result = t_test_by_group(df, col, auto_group_col, output_dir=output_dir)
                        
                        # Sonuçları logla
                        if 'p_value' in result:
                            sig_text = "ANLAMLI" if result.get('significant', False) else "anlamsız"
                            logger.info(f"{col}: t={result.get('t_statistic', 'N/A')}, p={result.get('p_value', 'N/A')}, {sig_text}")
                    except Exception as e:
                        logger.error(f"{col} için t-test sırasında hata: {e}")
            else:
                # Diğer uygun ikili değişkenleri bul
                binary_cols = [col for col in cat_cols if df[col].nunique() <= 2 and df[col].nunique() > 1]
                if binary_cols:
                    auto_group_col = binary_cols[0]
                    logger.info(f"T-test için otomatik olarak '{auto_group_col}' grup değişkeni olarak kullanılıyor")
                    
                    for col in num_cols:
                        try:
                            result = t_test_by_group(df, col, auto_group_col, output_dir=output_dir)
                            
                            # Sonuçları logla
                            if 'p_value' in result:
                                sig_text = "ANLAMLI" if result.get('significant', False) else "anlamsız"
                                logger.info(f"{col}: t={result.get('t_statistic', 'N/A')}, p={result.get('p_value', 'N/A')}, {sig_text}")
                        except Exception as e:
                            logger.error(f"{col} için t-test sırasında hata: {e}")
                else:
                    logger.warning(f"T-test için gerekli grup değişkeni '{group_col}' bulunamadı ve uygun otomatik değişken de bulunamadı, test atlanıyor.")
    
    if test_type in ['anova', 'all']:
        # ANOVA testleri
        if group_col in df.columns:
            logger.info(f"ANOVA testleri çalıştırılıyor ({len(num_cols)} sayısal değişken)...")
        
            for col in num_cols:
                try:
                    result = anova_test(df, col, group_col, output_dir=output_dir)
                    
                    # Sonuçları logla
                    if 'p_value' in result:
                        sig_text = "ANLAMLI" if result.get('significant', False) else "anlamsız"
                        logger.info(f"{col}: F={result.get('f_statistic', 'N/A')}, p={result.get('p_value', 'N/A')}, {sig_text}")
                except Exception as e:
                    logger.error(f"{col} için ANOVA testi sırasında hata: {e}")
        else:
            # Otomatik grup değişkeni bulma
            # ANOVA için kategorik değişkenlerden uygun birini seç
            categorical_cols = [col for col in cat_cols if df[col].nunique() > 2 and df[col].nunique() <= 10]
            if categorical_cols:
                auto_group_col = categorical_cols[0]
                logger.info(f"ANOVA için otomatik olarak '{auto_group_col}' grup değişkeni olarak kullanılıyor")
                
                for col in num_cols:
                    try:
                        result = anova_test(df, col, auto_group_col, output_dir=output_dir)
                        
                        # Sonuçları logla
                        if 'p_value' in result:
                            sig_text = "ANLAMLI" if result.get('significant', False) else "anlamsız"
                            logger.info(f"{col}: F={result.get('f_statistic', 'N/A')}, p={result.get('p_value', 'N/A')}, {sig_text}")
                    except Exception as e:
                        logger.error(f"{col} için ANOVA testi sırasında hata: {e}")
            else:
                logger.warning(f"ANOVA için gerekli grup değişkeni '{group_col}' bulunamadı ve uygun otomatik değişken de bulunamadı, test atlanıyor.")
    
    if test_type in ['conversion_rate', 'all']:
        # Dönüşüm oranı analizleri
        logger.info(f"Dönüşüm oranı analizleri çalıştırılıyor ({len(cat_cols)} kategorik değişken)...")
        
        for col in cat_cols:
            try:
                result = conversion_rate_comparison(df, col, target_col, output_dir=output_dir)
                
                # Sonuçları logla
                sig_count = len(result.get('significant_segments', []))
                logger.info(f"{col}: Anlamlı segment sayısı={sig_count}, Genel oran={result.get('overall_rate', 'N/A')}")
            except Exception as e:
                logger.error(f"{col} için dönüşüm oranı analizi sırasında hata: {e}")
    
    if test_type in ['correlation', 'all']:
        # Korelasyon analizi
        logger.info(f"Korelasyon analizi çalıştırılıyor ({len(num_cols)} sayısal değişken)...")
        
        try:
            result = correlation_analysis(df, num_cols, target_col, method=corr_method, 
                                        min_abs_corr=min_corr, output_dir=output_dir)
            
            # Hedef değişken ile korelasyonları logla
            if 'target_correlations' in result and result['target_correlations']:
                sorted_corrs = sorted(result['target_correlations'].items(), 
                                    key=lambda x: abs(x[1]['correlation']), reverse=True)
                
                logger.info(f"Hedef değişkenle en yüksek korelasyonlu 5 özellik:")
                for i, (feat, data) in enumerate(sorted_corrs[:5], 1):
                    corr = data['correlation']
                    logger.info(f"{i}. {feat}: {corr:.4f} ({'pozitif' if corr > 0 else 'negatif'})")
        except Exception as e:
            logger.error(f"Korelasyon analizi sırasında hata: {e}")
    
    # Özet dosyası oluştur
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"İstatistiksel Test Özeti\n")
        f.write(f"=====================\n\n")
        f.write(f"Test Tipi: {test_type}\n")
        f.write(f"Veri Seti: {'Sadece Train' if use_train_only else 'Tüm Veri'}\n")
        f.write(f"Veri Boyutu: {df.shape}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\nDışlanan Kolonlar:\n")
        for col in exclude_cols:
            f.write(f"- {col}\n")
        f.write(f"\nAnalize Dahil Edilen Kolonlar:\n")
        f.write(f"Kategorik ({len(cat_cols)}):\n")
        for col in cat_cols:
            f.write(f"- {col}\n")
        f.write(f"\nSayısal ({len(num_cols)}):\n")
        for col in num_cols:
            f.write(f"- {col}\n")
    
    logger.info(f"İstatistiksel testler tamamlandı. Sonuçlar: {output_dir}")
    return output_dir

if __name__ == "__main__":
    main() 