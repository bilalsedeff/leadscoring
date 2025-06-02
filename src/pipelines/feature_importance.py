"""
python -m src.pipelines.feature_importance  (CLI'dan çağrılır)
"""
import os
import click
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time

from src.ingestion.loader import read_datamart
from src.preprocessing.cleaning import BasicCleaner, SmartFeatureSelector, stable_feature_subset, load_cleaning_config
from src.features.engineering import add_temporal_features
from src.preprocessing.type_cast import apply_type_map
from src.features.importance import (
    shap_importance, 
    permutation_feature_importance,
    cross_validated_importance,
    feature_importance_report,
    select_features,
    get_importance_output_dir
)
from src.utils.logger import get_logger

log = get_logger()

@click.command()
@click.option('--method', type=click.Choice(['shap', 'permutation', 'both']), default='shap',
              help='Feature importance hesaplama metodu')
@click.option('--cv/--no-cv', default=True, help='Cross-validation kullan')
@click.option('--n-folds', default=5, help='Cross-validation fold sayısı')
@click.option('--n-estimators', default=300, help='Ağaç sayısı')
@click.option('--top-k', default=None, type=int, help='Seçilecek özellik sayısı')
@click.option('--stability-threshold', default=0.6, help='Kararlılık eşiği')
@click.option('--train-path', default="outputs/split/train.csv", help='Eğitim veri seti yolu')
@click.option('--output-dir', default=None, help='Çıktı dizini (varsayılan: active experiment dir)')
@click.option('--preprocess/--no-preprocess', default=True, help='Veriyi ön işleme uygula')
@click.option('--filter-features/--no-filter-features', default=True, 
              help='Özellik seçiminde akıllı filtreleme kullan')
@click.option('--interactive/--no-interactive', default=False, 
              help='Etkileşimli mod (figürleri ekranda göster)')
def main(method, cv, n_folds, n_estimators, top_k, stability_threshold, train_path, 
         output_dir, preprocess, filter_features, interactive):
    """Feature importance hesaplar ve sonuçları görselleştirir"""
    log.info(f"Feature importance hesaplanıyor...")
    log.info(f"Parametre bilgileri:")
    log.info(f"  - Metod: {method}")
    log.info(f"  - Cross-validation: {cv}")
    log.info(f"  - Fold sayısı: {n_folds}")
    log.info(f"  - Ağaç sayısı: {n_estimators}")
    log.info(f"  - Kararlılık eşiği: {stability_threshold}")
    log.info(f"  - Eğitim verisi: {train_path}")
    
    # Çıktı dizinini belirle
    output_dir = get_importance_output_dir(output_dir)
    log.info(f"Sonuçlar {output_dir} dizinine kaydedilecek")
    
    # Eğitim verisi var mı kontrol et
    train_file = Path(train_path)
    if not train_file.exists():
        log.error(f"{train_path} bulunamadı! Önce split adımını çalıştırın.")
        log.error("Örnek: python -m src.pipelines.split")
        return None

    # Eğitim verisini yükle
    try:
        df = pd.read_csv(train_file, low_memory=False)
        log.info(f"Eğitim verisi başarıyla yüklendi. Boyut: {df.shape} ({df.shape[0]} satır, {df.shape[1]} sütun)")
    except Exception as e:
        log.error(f"Eğitim verisi yüklenirken hata: {e}")
        log.info("Ham veriden yükleme deneniyor...")
        
        # Alternatif olarak ham veriyi yükleyip bölme
        df = read_datamart()
        
        if preprocess:
            log.info("Veri temizleniyor...")
            cleaner = BasicCleaner()
            df = cleaner.fit_transform(df)
            
            log.info("Tip dönüşümleri uygulanıyor...")
            df = apply_type_map(df)
        
        # Tarih sütununa göre sırala ve train seti seç (veri sızıntısından kaçınmak için)
        from src.preprocessing.splitters import time_group_split
        train_df, _, _ = time_group_split(df)
        df = train_df
        log.info(f"Ham veriden train seti oluşturuldu. Boyut: {df.shape}")
    
    # Veri analizi
    null_cols = df.columns[df.isnull().sum() > 0]
    log.info(f"Eksik değer içeren sütun sayısı: {len(null_cols)} / {len(df.columns)}")
    if len(null_cols) > 0:
        log.info(f"En çok eksik değere sahip 5 sütun:")
        for col in df.isnull().sum().sort_values(ascending=False).head(5).index:
            null_pct = df[col].isnull().mean() * 100
            log.info(f"  - {col}: {df[col].isnull().sum()} eksik değer (%{null_pct:.2f})")
    
    # Temporal özellikler ekle
    log.info("Temporal özellikler ekleniyor...")
    try:
        df_with_temporal = add_temporal_features(df)
        log.info(f"Temporal özellikler eklendi. Yeni boyut: {df_with_temporal.shape} (+{df_with_temporal.shape[1] - df.shape[1]} yeni sütun)")
        df = df_with_temporal
    except Exception as e:
        log.warning(f"Temporal özellikler eklenirken hata oluştu: {e}")
        log.warning("Temporal özellikler eklemeden devam ediliyor.")
    
    # Hedef değişkeni ve özellikleri ayır
    log.info("Feature importance hesaplanıyor...")
    target_col = "Target_IsConverted"
    
    if target_col not in df.columns:
        log.error(f"Hedef sütun '{target_col}' veri setinde bulunamadı!")
        available_cols = [col for col in df.columns if 'target' in col.lower() or 'convert' in col.lower()]
        if available_cols:
            log.info(f"Benzer sütunlar: {available_cols}")
            target_col = available_cols[0]
            log.info(f"'{target_col}' hedef değişken olarak kullanılacak")
        else:
            log.error("Uygun hedef değişken bulunamadı. İşlem iptal ediliyor.")
            return None
    
    # Kategorik ve sayısal özellikleri belirle
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_col in num_cols:
        num_cols.remove(target_col)
    
    log.info(f"Sayısal sütun sayısı: {len(num_cols)}")
    log.info(f"Kategorik sütun sayısı: {len(cat_cols)}")
    
    # İstatistikleri kaydet
    feature_stats = pd.DataFrame({
        'feature': num_cols + cat_cols,
        'type': ['numeric'] * len(num_cols) + ['categorical'] * len(cat_cols)
    })
    feature_stats.to_csv(output_dir / "feature_types.csv", index=False)
    log.info(f"Özellik tipleri {output_dir / 'feature_types.csv'} dosyasına kaydedildi")
    
    # Datetime kolonlarını da kaydet (isteğe bağlı)
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if date_cols:
        # Date kolonlarını da feature stats'a ekle
        date_df = pd.DataFrame({
            'feature': date_cols,
            'type': ['datetime'] * len(date_cols)
        })
        feature_stats = pd.concat([feature_stats, date_df], ignore_index=True)
        feature_stats.to_csv(output_dir / "feature_types.csv", index=False)
        log.info(f"Tarih sütunları: {len(date_cols)}")
    
    # Akıllı özellik seçimi (isteğe bağlı)
    if filter_features:
        log.info("Akıllı özellik seçimi uygulanıyor...")
        try:
            # Cleaning config yükle
            cleaning_config = load_cleaning_config()
            
            # SmartFeatureSelector oluştur
            selector = SmartFeatureSelector(
                missing_thresh=cleaning_config.get('feature_selection', {}).get('missing_thresh', 0.3),
                near_zero_var_thresh=cleaning_config.get('feature_selection', {}).get('near_zero_var_thresh', 0.01),
                correlation_thresh=cleaning_config.get('feature_selection', {}).get('correlation_thresh', 0.95),
                target_correlation_min=cleaning_config.get('feature_selection', {}).get('target_correlation_min', 0.02)
            )
            
            # One-hot encoding öncesi boyut
            pre_encoding_shape = df.shape
            
            # Seçiciyi eğit ve dönüştür
            selected_df = selector.fit_transform(df.copy())
            
            # Değişimleri logla
            selected_features = selected_df.columns.tolist()
            if target_col in selected_features:
                selected_features.remove(target_col)
                
            removed_features = set(df.columns) - set(selected_df.columns)
            if removed_features:
                log.info(f"Akıllı seçici tarafından {len(removed_features)} özellik elendi:")
                log.info(f"  - Düşük varyans: {len(selector.to_drop_)} özellik")
            
            # Seçilen özellikleri kullan
            df = selected_df
            
            # İstatistikleri güncelle
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if target_col in num_cols:
                num_cols.remove(target_col)
            
            log.info(f"Akıllı seçim sonrası sayısal sütun sayısı: {len(num_cols)}")
            log.info(f"Akıllı seçim sonrası kategorik sütun sayısı: {len(cat_cols)}")
        except Exception as e:
            log.error(f"Akıllı özellik seçimi sırasında hata: {e}")
            log.warning("Akıllı özellik seçimi atlanıyor...")
    
    # One-hot encoding uygula (kategorik değişkenler için)
    log.info("One-hot encoding uygulanıyor...")
    try:
        # Özellik isimlerini temizle (LightGBM için özel JSON karakterleri kaldır)
        def clean_feature_name(name):
            # JSON özel karakterlerini temizle
            if isinstance(name, str):
                # Köşeli parantezler, süslü parantezler, tırnak işaretleri vb. karakterleri temizle
                import re
                # Tüm özel JSON karakterlerini kaldır: {, }, [, ], ", &, %, +, ', \, /, <, >, *, ^, |, ?, !
                cleaned = re.sub(r'["\{\}\[\]\&\%\+\'\\\/\<\>\*\^\|\?\!]', '_', name)
                
                # Boşlukları alt çizgi ile değiştir
                cleaned = re.sub(r'\s+', '_', cleaned)
                
                # Başta ve sonda alt çizgileri temizle
                cleaned = cleaned.strip('_')
                
                # Ardışık alt çizgileri tek alt çizgiye dönüştür
                cleaned = re.sub(r'_+', '_', cleaned)
                
                # Eğer boş kaldıysa bir değer ata
                if not cleaned:
                    cleaned = "feature"
                
                return cleaned
            return str(name)
            
        # Önce kolon isimlerini temizle
        cleaned_cols = {col: clean_feature_name(col) for col in num_cols + cat_cols}
        df_cleaned = df[num_cols + cat_cols].rename(columns=cleaned_cols)
        
        # One-hot encoding uygula
        X_prep = pd.get_dummies(df_cleaned, columns=[cleaned_cols[col] for col in cat_cols], drop_first=False)
        y = df[target_col]
        
        # Oluşan tüm yeni sütun adlarını da temizle (one-hot encoding sonrası)
        X_prep_cleaned = X_prep.copy()
        all_columns_cleaned = {col: clean_feature_name(col) for col in X_prep.columns}
        X_prep_cleaned.rename(columns=all_columns_cleaned, inplace=True)
        
        log.info(f"One-hot encoding sonrası veri boyutları: {X_prep_cleaned.shape}")
        log.info(f"Toplam özellik sayısı: {X_prep_cleaned.shape[1]}")
        
        # Hedef sınıf dengesini kontrol et
        class_balance = y.value_counts(normalize=True) * 100
        log.info(f"Hedef değişken sınıf dağılımı:")
        for cls, pct in class_balance.items():
            log.info(f"  - Sınıf {cls}: %{pct:.2f} ({(y == cls).sum()} örnek)")
        
        # Hafıza kullanımını kontrol et
        memory_usage = X_prep_cleaned.memory_usage(deep=True).sum() / (1024 * 1024)  # MB cinsinden
        log.info(f"One-hot encoded veri bellek kullanımı: {memory_usage:.2f} MB")
        
        # Çok fazla özellik varsa uyarı ver
        if X_prep_cleaned.shape[1] > 500:
            log.warning(f"Özellik sayısı çok yüksek ({X_prep_cleaned.shape[1]}). Hesaplama süresi uzun olabilir.")
            log.warning("Özellik sayısını azaltmak için daha sıkı filtreler kullanmayı düşünün.")
            
        # X_prep yerine temizlenmiş X_prep_cleaned'ı kullanın
        X_prep = X_prep_cleaned
    except Exception as e:
        log.error(f"One-hot encoding sırasında hata: {e}")
        return None
    
    log.info(f"SADECE EĞİTİM VERİSİ ÜZERİNDE FEATURE IMPORTANCE HESAPLANIYOR! (Veri sızıntısını önlemek için)")
    
    # Feature importance hesapla
    start_time = time.time()
    if method == 'both':
        methods = ['shap', 'permutation']
        log.info(f"Her iki yöntem (SHAP ve Permutation) ile özellik önem analizi yapılıyor...")
        
        results = feature_importance_report(
            X_prep, y, methods=methods, n_folds=n_folds, n_estimators=n_estimators
        )
        
        # Her iki yöntemin sonuçlarını birleştir
        combined_results = pd.DataFrame()
        for m in methods:
            m_df = results[m][['feature', 'mean_importance', 'stability']]
            m_df = m_df.rename(columns={
                'mean_importance': f'{m}_importance',
                'stability': f'{m}_stability'
            })
            if combined_results.empty:
                combined_results = m_df
            else:
                combined_results = pd.merge(combined_results, m_df, on='feature', how='outer')
        
        # Ortalama önem skorunu hesapla
        combined_results['mean_importance'] = combined_results[[f'{m}_importance' for m in methods]].mean(axis=1)
        combined_results['mean_stability'] = combined_results[[f'{m}_stability' for m in methods]].mean(axis=1)
        
        # Önemine göre sırala
        combined_results = combined_results.sort_values('mean_importance', ascending=False)
        
        # Sonuçları kaydet
        combined_results.to_csv(output_dir / "combined_importance.csv", index=False)
        log.info(f"Kombine önem skorları {output_dir / 'combined_importance.csv'} dosyasına kaydedildi")
        
        # Seçilen özellikleri belirle
        selected = select_features(
            combined_results.rename(columns={'mean_stability': 'stability'}), 
            top_k=top_k, 
            stability_threshold=stability_threshold
        )
        
        # Sonuç olarak combined_results'ı döndür
        importance_df = combined_results[['feature', 'mean_importance']].rename(
            columns={'mean_importance': 'importance'})
        
    else:
        if cv:
            # Cross-validation ile importance hesapla
            log.info(f"{n_folds}-fold cross-validation ile {method.upper()} özellik önem analizi yapılıyor...")
            imp_df = cross_validated_importance(
                X_prep, y, n_folds=n_folds, method=method, 
                n_estimators=n_estimators, save_plot=True
            )
            # Seçilen özellikleri belirle
            selected = select_features(imp_df, top_k=top_k, stability_threshold=stability_threshold)
            importance_df = imp_df[['feature', 'mean_importance']].rename(
                columns={'mean_importance': 'importance'})
        else:
            # Basit importance hesapla
            log.info(f"Tek run üzerinde {method.upper()} özellik önem analizi yapılıyor...")
            if method == 'shap':
                imp = shap_importance(X_prep, y, n_estimators=n_estimators)
                importance_df = pd.DataFrame({
                    'feature': imp.index,
                    'importance': imp.values
                })
            else:  # permutation
                imp = permutation_feature_importance(X_prep, y, n_estimators=n_estimators)
                importance_df = imp[['feature', 'importance']]
            
            # İlk top_k özelliği seç
            if top_k is not None:
                selected = importance_df['feature'].iloc[:top_k].tolist()
            else:
                selected = importance_df['feature'].tolist()
    
    end_time = time.time()
    log.info(f"Özellik önem analizi {end_time - start_time:.2f} saniyede tamamlandı")
    
    # Sonuçları kaydet
    importance_df.to_csv(output_dir / "importance.csv", index=False)
    log.info(f"Önem skorları {output_dir / 'importance.csv'} dosyasına kaydedildi")
    
    with open(output_dir / "selected_features.txt", "w") as f:
        for feature in selected:
            f.write(f"{feature}\n")
    log.info(f"Seçilen özellikler {output_dir / 'selected_features.txt'} dosyasına kaydedildi")
    
    # Özetleme logu
    log.info(f"Toplam {len(selected)} özellik seçildi")
    
    # En önemli özellikleri göster
    top_n = min(20, len(importance_df))
    log.info(f"En önemli {top_n} özellik:")
    for i, (_, row) in enumerate(importance_df.head(top_n).iterrows()):
        stability_info = ""
        if 'stability' in row:
            stability_info = f" (stability: {row['stability']:.2f})"
        log.info(f"  {i+1}. {row['feature']} - {row['importance']:.6f}{stability_info}")
    
    # Stability eşiğinin altındaki özelliklerin uyarısını göster
    if stability_threshold > 0 and 'stability' in importance_df.columns:
        low_stability = importance_df[importance_df['stability'] < stability_threshold]
        if not low_stability.empty:
            log.warning(f"{len(low_stability)} özellik stability eşiğinin ({stability_threshold}) altında!")
            log.warning("Bu özellikler farklı train/test bölmelerinde tutarsız sonuç verebilir.")
            log.warning(f"Düşük kararlılıklı özellikler: {', '.join(low_stability['feature'].head(5).tolist())}...")
    
    # experiment.yaml güncelleme için config_updater'ı kullan
    try:
        from src.utils.config_updater import update_from_feature_importance
        if update_from_feature_importance(len(selected)):
            log.info("experiment.yaml başarıyla güncellendi.")
        else:
            log.warning("experiment.yaml güncellenemedi!")
    except Exception as e:
        log.error(f"experiment.yaml güncellenirken hata: {str(e)}")
    
    # Etkileşimli mod etkinse, figürleri göster
    if interactive:
        plt.ion()  # Interactive mode açık
        try:
            fig_files = list(output_dir.glob("*.png"))
            log.info(f"{len(fig_files)} görsel bulundu. İncelemek için figür pencerelerini kontrol edin.")
            for fig_file in fig_files:
                img = plt.imread(fig_file)
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.title(fig_file.name)
                plt.tight_layout()
                plt.show()
            
            input("Devam etmek için ENTER tuşuna basın...")
        except Exception as e:
            log.error(f"Görseller gösterilirken hata: {e}")
        finally:
            plt.ioff()  # Interactive mode kapalı
    
    return importance_df

if __name__ == "__main__":
    main()
