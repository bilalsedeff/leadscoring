import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import seaborn as sns
import os
import time
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from ..registry.mlflow_wrapper import configure_mlflow_output_dir
from ..utils.paths import get_experiment_dir
from src.preprocessing.cleaning import stable_feature_subset
from src.utils.logger import get_logger
import re

log = get_logger()

# Sabit OUTPUT_DIR yerine dinamik olarak hesapla
def get_importance_output_dir(run_name=None):
    """
    Feature importance için çıktı dizinini belirler.
    
    Args:
        run_name: Çalıştırma adı (None ise timestamp oluşturulur)
        
    Returns:
        Path: Çıktı dizini
    """
    # Önce get_experiment_dir() ile deney dizinini bul
    experiment_dir = get_experiment_dir()
    output_dir = experiment_dir / "feature_importance"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Legacy dizin kontrolünü parametreyle yönet
    use_legacy = False
    legacy_dir = Path("outputs/feature_importance")
    
    # Legacy dizini sadece experiment_dir yoksa ve legacy_dir varsa kullan
    if use_legacy and not experiment_dir.exists() and legacy_dir.exists():
        output_dir = legacy_dir
    
    return output_dir

def shap_importance(X, y, n_estimators=300, save_plot=True, run_name=None):
    """SHAP değerlerini kullanarak feature importance hesaplar"""
    output_dir = get_importance_output_dir(run_name)
    
    # Model eğitimi
    model = LGBMClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    
    try:
        # SHAP değerlerini hesapla
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # İkili sınıflandırma için, pozitif sınıfın SHAP değerlerini al
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # İkili sınıflandırma için, pozitif sınıfın SHAP değerlerini al
            shap_values_class1 = shap_values[1]
        else:
            # Tekli çıktı (regresyon veya tekli çıktılı sınıflandırma) için
            shap_values_class1 = shap_values
        
        # Ortalama mutlak SHAP değerlerini hesapla
        imp = pd.Series(np.abs(shap_values_class1).mean(0), index=X.columns).sort_values(ascending=False)
        
        # SHAP görselleştirmeleri
        if save_plot:
            # 1. Klasik bar plot (iyileştirilmiş)
            plt.figure(figsize=(14, 10))
            top_features = imp.head(20)
            plt.barh(range(len(top_features)), top_features.values[::-1], color='#1E88E5')
            plt.yticks(range(len(top_features)), top_features.index[::-1])
            plt.xlabel('Mean |SHAP Value|')
            plt.title('Top 20 Features by SHAP Value Impact', fontsize=14)
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(output_dir / "shap_importance_bar.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Summary Plot (SHAP Beeswarm)
            plt.figure(figsize=(12, 10))
            # En önemli 20 özelliği seç
            top_indices = [X.columns.get_loc(col) for col in imp.index[:20]]
            
            # SHAP summary plot
            try:
                shap.summary_plot(
                    shap_values_class1[:, top_indices],
                    X.iloc[:, top_indices],
                    feature_names=imp.index[:20].tolist(),
                    plot_type="dot",
                    plot_size=(12, 10),
                    color_bar_label="Feature Value",
                    show=False
                )
                plt.title("SHAP Summary Plot (Beeswarm)", fontsize=14)
                plt.tight_layout()
                plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"SHAP Summary plot oluşturulurken hata: {e}")
            
            # 3. Bar Summary
            plt.figure(figsize=(12, 10))
            try:
                shap.summary_plot(
                    shap_values_class1[:, top_indices],
                    X.iloc[:, top_indices],
                    feature_names=imp.index[:20].tolist(),
                    plot_type="bar",
                    plot_size=(12, 10),
                    show=False
                )
                plt.title("SHAP Feature Importance", fontsize=14)
                plt.tight_layout()
                plt.savefig(output_dir / "shap_summary_bar.png", dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"SHAP Bar plot oluşturulurken hata: {e}")
            
            # 4. Dependence Plot (En önemli 2 özellik için)
            try:
                for i, feature in enumerate(imp.index[:2]):
                    plt.figure(figsize=(10, 7))
                    feat_idx = X.columns.get_loc(feature)
                    
                    # İkinci önemli özelliği etkileşim olarak kullan (varsa)
                    if len(imp) > 1 and i == 0:
                        interaction_idx = X.columns.get_loc(imp.index[1])
                        interaction_feature = imp.index[1]
                    else:
                        # Korelasyonu en yüksek olan başka bir özelliği bul
                        corr = X.corr()[feature].abs().sort_values(ascending=False)
                        corr = corr[corr.index != feature]  # Kendisi hariç
                        if not corr.empty:
                            interaction_feature = corr.index[0]
                            interaction_idx = X.columns.get_loc(interaction_feature)
                        else:
                            interaction_feature = None
                            interaction_idx = None
                    
                    # Dependence plot
                    if interaction_feature:
                        shap.dependence_plot(
                            feat_idx, 
                            shap_values_class1, 
                            X,
                            interaction_index=interaction_idx,
                            feature_names=X.columns,
                            show=False
                        )
                        plt.title(f"SHAP Dependence Plot: {feature} (interaction: {interaction_feature})", fontsize=12)
                    else:
                        shap.dependence_plot(
                            feat_idx, 
                            shap_values_class1, 
                            X,
                            feature_names=X.columns,
                            show=False
                        )
                        plt.title(f"SHAP Dependence Plot: {feature}", fontsize=12)
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / f"shap_dependence_{feature.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                print(f"SHAP Dependence plot oluşturulurken hata: {e}")
            
            # 5. Force plot (birkaç örnek için)
            try:
                # Random seçilmiş 3 örnek
                sample_indices = np.random.choice(range(len(X)), 3, replace=False)
                
                for i, idx in enumerate(sample_indices):
                    plt.figure(figsize=(20, 3))
                    force_plot = shap.force_plot(
                        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                        shap_values_class1[idx, :],
                        X.iloc[idx, :],
                        feature_names=X.columns,
                        matplotlib=True,
                        show=False
                    )
                    plt.title(f"SHAP Force Plot: Sample {i+1}", fontsize=12)
                    plt.tight_layout()
                    plt.savefig(output_dir / f"shap_force_plot_sample{i+1}.png", dpi=300, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                print(f"SHAP Force plot oluşturulurken hata: {e}")
        
        return imp
    
    except Exception as e:
        print(f"SHAP hesaplaması sırasında hata: {e}")
        # Hata durumunda basit feature importance döndür
        return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

def permutation_feature_importance(X, y, n_estimators=300, n_repeats=10, save_plot=True, run_name=None):
    """Permutation importance kullanarak feature importance hesaplar"""
    # Model eğit
    model = LGBMClassifier(n_estimators=n_estimators, random_state=0)
    model.fit(X, y)
    
    # Permutation importance hesapla
    perm_importance = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=0)
    
    # Sonuçları DataFrame'e dönüştür
    perm_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    # Görselleştirme
    if save_plot:
        output_dir = get_importance_output_dir(run_name)
        plt.figure(figsize=(12, 10))
        plt.barh(perm_imp['feature'][:20], perm_imp['importance'][:20])
        plt.xlabel('Importance (Mean decrease in model score)')
        plt.title('Permutation Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig(output_dir / "permutation_importance.png")
        plt.close()
    
    return perm_imp

def clean_feature_name(name):
    """
    LightGBM için özellik adını temizler.
    LightGBM özel JSON karakterleri içeren özellik adlarını desteklemez.
    
    Args:
        name: Temizlenecek özellik adı
        
    Returns:
        str: Temizlenmiş özellik adı
    """
    if not isinstance(name, str):
        return str(name)
    
    # Tüm özel JSON karakterlerini kaldır: {, }, [, ], ", &, %, +, ', \, /, <, >
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

def cross_validated_importance(X, y, n_folds=5, method='shap', n_estimators=100, random_state=42, save_plot=False):
    """
    Cross-validation ile feature importance hesaplar
    
    Args:
        X (pd.DataFrame): Özellikler
        y (pd.Series): Hedef değişken
        n_folds (int): CV fold sayısı
        method (str): 'shap' veya 'permutation'
        n_estimators (int): Ağaç sayısı
        random_state (int): Random seed
        save_plot (bool): Grafikleri kaydet
        
    Returns:
        pd.DataFrame: Feature importance sonuçları
    """
    from sklearn.model_selection import KFold
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # Tüm özellik adlarını temizle - LightGBM özel karakterleri kabul etmiyor
    X_cleaned = X.copy()
    original_to_clean = {}
    
    # Özellik adlarını temizle ve orijinal-temiz eşleşmesini kaydet
    for col in X.columns:
        clean_col = clean_feature_name(col)
        original_to_clean[col] = clean_col
    
    # Tüm sütun adlarını temizlenmiş hallerle değiştir
    X_cleaned = X.rename(columns=original_to_clean)
    
    # X ve y numpy array'e çevir (LightGBM için)
    if isinstance(X_cleaned, pd.DataFrame):
        feature_names = X_cleaned.columns.tolist()
        X_values = X_cleaned.values
    else:
        X_values = X_cleaned
        feature_names = [f"feature_{i}" for i in range(X_values.shape[1])]
    
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y
    
    # Cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    importances = []
    
    fold_num = 1
    for train_idx, val_idx in kf.split(X_values):
        X_train, y_train = X_values[train_idx], y_values[train_idx]
        
        # Model ve feature importance hesaplama
        if method == 'shap':
            try:
                import lightgbm as lgb
                import shap
                
                # LightGBM modeli
                model = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=random_state)
                
                # Temizlenmiş özellik adlarını kullanarak eğit
                model.fit(X_train, y_train, feature_name=feature_names)
                
                # SHAP değerleri hesapla
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_values)
                
                # Sınıflandırma problemi için pozitif sınıfın SHAP değerlerini kullan
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Pozitif sınıf (binary classification)
                
                # SHAP değerlerinden feature importance hesapla
                fold_importance = np.abs(shap_values).mean(axis=0)
                
            except Exception as e:
                print(f"SHAP hesaplama hatası (fold {fold_num}): {e}")
                # Alternatif olarak permutation importance kullan
                from sklearn.inspection import permutation_importance
                
                model = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=random_state)
                model.fit(X_train, y_train)
                
                perm_importance = permutation_importance(model, X_train, y_train, 
                                                       n_repeats=5, random_state=random_state)
                fold_importance = perm_importance.importances_mean
                
        else:  # permutation
            from sklearn.inspection import permutation_importance
            import lightgbm as lgb
            
            model = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=random_state)
            model.fit(X_train, y_train, feature_name=feature_names)
            
            perm_importance = permutation_importance(model, X_train, y_train, 
                                                   n_repeats=5, random_state=random_state)
            fold_importance = perm_importance.importances_mean
        
        # Önem değerlerini DataFrame'e dönüştür
        df = pd.DataFrame({
            'feature': feature_names,
            f'importance_fold_{fold_num}': fold_importance
        })
        importances.append(df)
        
        fold_num += 1
    
    # Tüm fold'ları birleştir
    result = importances[0]
    for i in range(1, n_folds):
        result = result.merge(importances[i], on='feature')
    
    # Ortalama ve std hesapla
    cols = [f'importance_fold_{i+1}' for i in range(n_folds)]
    result['mean_importance'] = result[cols].mean(axis=1)
    result['std_importance'] = result[cols].std(axis=1)
    
    # Önem derecesine göre sırala
    result = result.sort_values('mean_importance', ascending=False)
    
    # Kararlılık skoru hesapla (std / mean)
    result['stability'] = 1 - (result['std_importance'] / (result['mean_importance'] + 1e-10))
    result['stability'] = result['stability'].clip(0, 1)  # 0-1 arasına sınırla
    
    # Temizlenmiş özellik adlarından orijinal isimlere geri dönüş yap
    clean_to_original = {v: k for k, v in original_to_clean.items()}
    result['original_feature'] = result['feature'].map(clean_to_original)
    
    # Grafik çiz
    if save_plot:
        try:
            output_dir = get_importance_output_dir()
            plt.figure(figsize=(12, 8))
            
            # Görselleştirme için top 30 özellik
            top_k = min(30, len(result))
            
            # Top özellikleri göster
            bars = plt.barh(range(top_k), result['mean_importance'][:top_k], 
                          xerr=result['std_importance'][:top_k], align='center', 
                          error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
            
            plt.yticks(range(top_k), result['feature'][:top_k])
            plt.xlabel('Importance')
            plt.title(f'Top {top_k} Feature Importance ({method.upper()})')
            plt.tight_layout()
            
            # Sonuçları kaydet
            plt.savefig(f"{output_dir}/feature_importance_{method}.png", dpi=300, bbox_inches='tight')
            plt.close()  # Figürü kapat
        except Exception as e:
            print(f"Grafik oluşturma hatası: {e}")
    
    return result

def select_features(importance_df, top_k=None, stability_threshold=0.6):
    """Önem ve kararlılık skorlarına göre feature seçimi yapar"""
    if top_k is None:
        # Stabilite eşiğine göre seçim yap
        selected = importance_df[importance_df['stability'] >= stability_threshold]
    else:
        # İlk top_k feature'ı seç
        selected = importance_df.iloc[:top_k]
    
    # Sonuçları kaydet
    selected[['feature', 'mean_importance', 'stability']].to_csv(
        get_importance_output_dir() / "selected_features.csv", index=False)
    
    return selected['feature'].tolist()

def feature_importance_report(X, y, methods=['shap', 'permutation'], n_folds=5, n_estimators=300):
    """Farklı feature importance yöntemlerini çalıştırır ve bir rapor oluşturur"""
    results = {}
    output_dir = get_importance_output_dir()
    
    print(f"Veri boyutları: {X.shape} - {len(X.columns)} özellik analiz ediliyor")
    print(f"İşleme başlanıyor... Yöntemler: {', '.join(methods)}")
    
    # Temizlik aşaması logları
    from src.preprocessing.cleaning import SmartFeatureSelector
    print("\n--- Temizlik Aşaması Öncesi İstatistikler ---")
    print(f"Toplam özellik sayısı: {X.shape[1]}")
    
    # Eksik değer analizi
    null_counts = X.isnull().sum()
    high_null_cols = null_counts[null_counts > 0.1 * len(X)]
    print(f"Yüksek oranda eksik değer içeren kolonlar: {len(high_null_cols)} "
          f"({len(high_null_cols)/X.shape[1]*100:.1f}%)")
    
    # Düşük varyans analizi
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    try:
        selector.fit(X)
        low_var_features = X.columns[~selector.get_support()].tolist()
        print(f"Düşük varyansa sahip kolonlar: {len(low_var_features)} "
              f"({len(low_var_features)/X.shape[1]*100:.1f}%)")
    except Exception as e:
        print(f"Varyans analizi sırasında hata: {e}")
    
    # Korelasyon analizi
    try:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        high_corr_features = [column for column in upper.columns if any(upper[column] > 0.8)]
        print(f"Yüksek korelasyonlu kolonlar: {len(high_corr_features)} "
              f"({len(high_corr_features)/X.shape[1]*100:.1f}%)")
    except Exception as e:
        print(f"Korelasyon analizi sırasında hata: {e}")
        
    print("-------------------------------------------\n")
    
    # Her bir metodu çalıştır
    for method in methods:
        print(f"{method.capitalize()} tabanlı feature importance hesaplanıyor...")
        start_time = time.time()
        imp = cross_validated_importance(
            X, y, n_folds=n_folds, method=method, 
            n_estimators=n_estimators, save_plot=True
        )
        results[method] = imp
        print(f"{method.capitalize()} analizi tamamlandı. Süre: {time.time() - start_time:.2f} saniye")
    
    # Farklı yöntemlerin karşılaştırmasını yap
    if len(methods) > 1:
        # Top 20 feature'ların kesişimini analiz et
        plt.figure(figsize=(14, 10))
        
        # Her yöntemin top-20 özelliklerini topla
        top_features = {}
        method_rankings = {}
        for method in methods:
            top_features[method] = set(results[method]['feature'][:20].tolist())
            
            # Her özelliğin rank'ini sakla
            method_rankings[method] = pd.Series(
                range(1, len(results[method]) + 1),
                index=results[method]['feature']
            )
        
        # Venn diyagramı için matplotlib-venn'in olup olmadığını kontrol et
        try:
            from matplotlib_venn import venn2, venn3
            if len(methods) == 2:
                venn2([top_features[methods[0]], top_features[methods[1]]], 
                    set_labels=[f"{methods[0].upper()}\nTop 20", f"{methods[1].upper()}\nTop 20"])
            elif len(methods) == 3:
                venn3([top_features[methods[0]], top_features[methods[1]], top_features[methods[2]]], 
                    set_labels=[f"{methods[0].upper()}\nTop 20", f"{methods[1].upper()}\nTop 20", 
                                f"{methods[2].upper()}\nTop 20"])
            
            plt.title('Top 20 Features - Method Comparison', fontsize=14)
            plt.savefig(output_dir / "method_venn_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        except ImportError:
            print("matplotlib-venn kütüphanesi bulunamadı. Venn diyagramı oluşturulamıyor.")
            
            # Alternatif olarak bar chart karşılaştırması yap
            plt.figure(figsize=(15, 12))
            method_colors = {'shap': '#1E88E5', 'permutation': '#D81B60', 'boruta': '#FFC107'}
            
            # Tüm yöntemlerin top 10 özelliklerini birleştir
            all_top_features = set()
            for method in methods:
                all_top_features.update(results[method]['feature'][:10].tolist())
            
            # Her metod için barları yan yana göster
            x = np.arange(len(all_top_features))
            width = 0.8 / len(methods)  # Yan yana bar genişliği
            
            feature_list = list(all_top_features)
            for i, method in enumerate(methods):
                # Her özelliğin sırasını belirle, eğer top 20'de yoksa NaN olacak
                method_data = []
                for feature in feature_list:
                    if feature in results[method]['feature'][:10].tolist():
                        rank = results[method]['feature'][:10].tolist().index(feature)
                        method_data.append(10 - rank)  # Ters çevir (10 en önemli)
                    else:
                        method_data.append(0)  # Top 10'da yoksa 0 değeri
                
                plt.bar(x + i*width - (len(methods)-1)*width/2, method_data, width, 
                        label=method.capitalize(), color=method_colors.get(method, f'C{i}'))
            
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Importance Rank (10=Most Important)', fontsize=12)
            plt.title('Top 10 Features Comparison Across Methods', fontsize=14)
            plt.xticks(x, feature_list, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(output_dir / "method_bar_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
        # Özellik önem sıralamaları arası korelasyon
        common_features = set()
        for method in methods:
            common_features.update(results[method]['feature'].tolist())
        
        rank_df = pd.DataFrame(index=list(common_features))
        
        for method in methods:
            method_ranks = pd.Series(
                range(1, len(results[method]) + 1),
                index=results[method]['feature']
            )
            rank_df[method] = method_ranks
        
        # NaN değerleri yüksek bir rank ile doldur (varsa)
        max_rank = max([len(results[method]) for method in methods])
        rank_df = rank_df.fillna(max_rank + 1)
        
        # Spearmans rank korelasyonu
        corr = rank_df.corr(method='spearman')
        
        # Korelasyon matrisi heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Üst üçgeni maskeleme
        
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   fmt='.2f', linewidths=1, square=True, mask=mask)
        plt.title('Spearman Rank Correlation Between Feature Importance Methods', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "method_rank_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Scatter plot: İki metod arasındaki karşılaştırma (daha fazla metod varsa ilk ikisi)
        if len(methods) >= 2:
            method1, method2 = methods[0], methods[1]
            plt.figure(figsize=(12, 10))
            
            # Ortak özellikleri bul
            common_feats = set(results[method1]['feature'][:30]) & set(results[method2]['feature'][:30])
            
            # Importance değerlerini al
            common_df = pd.DataFrame(index=common_feats)
            for method in [method1, method2]:
                filtered_imp = results[method][results[method]['feature'].isin(common_feats)]
                common_df[f"{method}_importance"] = filtered_imp.set_index('feature')['mean_importance']
                common_df[f"{method}_stability"] = filtered_imp.set_index('feature')['stability']
            
            # Stability değerlerine göre renk ve boyut
            avg_stability = (common_df[f"{method1}_stability"] + common_df[f"{method2}_stability"]) / 2
            sizes = avg_stability * 200 + 30
            colors = plt.cm.RdYlGn(avg_stability)
            
            # Scatter plot
            plt.scatter(common_df[f"{method1}_importance"], common_df[f"{method2}_importance"], 
                      s=sizes, c=colors, alpha=0.7, edgecolors='black')
            
            # Etiketler - sadece önemli olanlar
            for idx, row in common_df.iterrows():
                # Eğer stability yüksekse veya her iki metotta da önemli bir özellikse etiketle
                if avg_stability[idx] > 0.7 or (row[f"{method1}_importance"] > common_df[f"{method1}_importance"].median() and 
                                              row[f"{method2}_importance"] > common_df[f"{method2}_importance"].median()):
                    plt.annotate(idx, (row[f"{method1}_importance"], row[f"{method2}_importance"]), 
                               fontsize=9, alpha=0.8, xytext=(5, 5), textcoords='offset points')
            
            # Eşit ölçek referans çizgisi
            min_val = min(common_df[f"{method1}_importance"].min(), common_df[f"{method2}_importance"].min())
            max_val = max(common_df[f"{method1}_importance"].max(), common_df[f"{method2}_importance"].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            plt.xlabel(f'{method1.capitalize()} Importance', fontsize=12)
            plt.ylabel(f'{method2.capitalize()} Importance', fontsize=12)
            plt.title(f'Comparison of {method1.capitalize()} vs {method2.capitalize()} Importance Values', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Renk çubuğu ekle
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label('Average Stability Score', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"comparison_{method1}_vs_{method2}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    return results

def calculate_shap_importance(model, X, feature_names=None):
    """SHAP feature importance hesaplar.
    
    Args:
        model: Eğitilmiş model
        X: Özellik matrisi
        feature_names: Özellik isimleri
        
    Returns:
        shap_values: SHAP değerleri
        shap_importances: SHAP önem skorları
    """
    # SHAP explainer oluştur
    if hasattr(model, 'predict_proba'):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model)
    
    # SHAP değerlerini hesapla
    shap_values = explainer(X)
    
    # Özellik önemlerini hesapla
    shap_importances = np.abs(shap_values.values).mean(0)
    
    # Özellik isimlerini ekle
    if feature_names is not None:
        importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': shap_importances
        })
        importances_df = importances_df.sort_values('importance', ascending=False)
    else:
        importances_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(shap_importances))],
            'importance': shap_importances
        })
        importances_df = importances_df.sort_values('importance', ascending=False)
    
    return shap_values, importances_df

def calculate_permutation_importance(model, X, y, feature_names=None, n_repeats=10, random_state=42):
    """Permutation feature importance hesaplar.
    
    Args:
        model: Eğitilmiş model
        X: Özellik matrisi
        y: Hedef değişken
        feature_names: Özellik isimleri
        n_repeats: Tekrar sayısı
        random_state: Random seed
        
    Returns:
        perm_importances: Permutation önem skorları
    """
    # Permutation importance hesapla
    perm_importance = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state
    )
    
    # Özellik isimlerini ekle
    if feature_names is not None:
        importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        })
        importances_df = importances_df.sort_values('importance', ascending=False)
    else:
        importances_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(perm_importance.importances_mean))],
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        })
        importances_df = importances_df.sort_values('importance', ascending=False)
    
    return importances_df

def calculate_boruta_importance(X, y, feature_names=None, max_iter=100, random_state=42):
    """Boruta feature importance hesaplar.
    
    Args:
        X: Özellik matrisi
        y: Hedef değişken
        feature_names: Özellik isimleri
        max_iter: Maksimum iterasyon sayısı
        random_state: Random seed
        
    Returns:
        boruta_importances: Boruta önem skorları
    """
    # Random Forest sınıflandırıcısı oluştur
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10, random_state=random_state)
    
    # Boruta özellik seçici oluştur
    boruta = BorutaPy(
        rf, 
        n_estimators='auto',
        max_iter=max_iter,
        verbose=0,
        random_state=random_state
    )
    
    # Boruta ile özellik seçimi yap
    boruta.fit(X.values, y.values)
    
    # Özellik önemlerini al
    importances = boruta.importance_history_
    
    # Son iterasyondaki önemleri al
    last_importances = importances[-1] if len(importances) > 0 else boruta.importance_
    
    # Özellik isimlerini ekle
    if feature_names is not None:
        importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': last_importances,
            'selected': boruta.support_
        })
        importances_df = importances_df.sort_values('importance', ascending=False)
    else:
        importances_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(last_importances))],
            'importance': last_importances,
            'selected': boruta.support_
        })
        importances_df = importances_df.sort_values('importance', ascending=False)
    
    return importances_df

def save_importance_results(importances, method, shap_values=None, X=None, feature_names=None, run_name=None):
    """Feature importance sonuçlarını kaydeder.
    
    Args:
        importances: Feature importance DataFrame
        method: Kullanılan metod (shap, permutation, boruta)
        shap_values: SHAP değerleri (sadece SHAP için)
        X: Özellik matrisi (sadece SHAP için)
        feature_names: Özellik isimleri (sadece SHAP için)
        run_name: Çalıştırma adı (None ise timestamp oluşturulur)
        
    Returns:
        output_path: Sonuçların kaydedildiği dizin
    """
    # Çıktı dizinini yapılandır
    output_dir = get_importance_output_dir(run_name)
    
    # Metoda göre alt klasör oluştur
    output_path = os.path.join(output_dir, method)
    os.makedirs(output_path, exist_ok=True)
    
    # Sonuçları CSV olarak kaydet
    importances.to_csv(os.path.join(output_path, f"{method}_importance.csv"), index=False)
    
    # Görselleştirme
    plt.figure(figsize=(10, 8))
    
    # En önemli 20 özelliği göster
    top_importances = importances.head(20)
    
    if method == 'shap':
        # SHAP özet grafiği
        if shap_values is not None and X is not None and feature_names is not None:
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "shap_summary.png"))
            plt.close()
        
        # SHAP bar grafiği
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_importances)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "shap_bar.png"))
        plt.close()
    
    elif method == 'permutation':
        # Permutation bar grafiği (hata çubukları ile)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_importances, xerr=top_importances['std'])
        plt.title('Permutation Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "permutation_bar.png"))
        plt.close()
    
    elif method == 'boruta':
        # Boruta bar grafiği (seçilen özellikler vurgulanarak)
        plt.figure(figsize=(10, 8))
        selected = top_importances[top_importances['selected']].copy()
        rejected = top_importances[~top_importances['selected']].copy()
        
        # Seçilen özellikleri yeşil, reddedilenleri kırmızı göster
        if not selected.empty:
            sns.barplot(x='importance', y='feature', data=selected, color='green', label='Selected')
        if not rejected.empty:
            sns.barplot(x='importance', y='feature', data=rejected, color='red', label='Rejected')
        
        plt.title('Boruta Feature Importance')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "boruta_bar.png"))
        plt.close()
    
    else:
        # Genel bar grafiği
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_importances)
        plt.title(f'{method.capitalize()} Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{method}_bar.png"))
        plt.close()
    
    print(f"{method} feature importance sonuçları {output_path} dizinine kaydedildi.")
    return output_path

def run_feature_importance(X, y, feature_names=None, methods=None, run_name=None):
    """Çeşitli feature importance metodlarını çalıştırır ve sonuçları kaydeder.
    
    Args:
        X: Özellik matrisi
        y: Hedef değişken
        feature_names: Özellik isimleri
        methods: Kullanılacak metodlar (None ise tümü)
        run_name: Çalıştırma adı (None ise timestamp oluşturulur)
        
    Returns:
        results: Feature importance sonuçları sözlüğü
    """
    if methods is None:
        methods = ['shap', 'permutation', 'boruta']
    
    results = {}
    
    # Temel model oluştur (tüm metodlar için)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    for method in methods:
        print(f"{method} feature importance hesaplanıyor...")
        
        if method == 'shap':
            # SHAP importance hesapla
            shap_values, shap_importances = calculate_shap_importance(model, X, feature_names)
            save_importance_results(shap_importances, method, shap_values, X, feature_names, run_name)
            results[method] = shap_importances
        
        elif method == 'permutation':
            # Permutation importance hesapla
            perm_importances = calculate_permutation_importance(model, X, y, feature_names)
            save_importance_results(perm_importances, method, run_name=run_name)
            results[method] = perm_importances
        
        elif method == 'boruta':
            # Boruta importance hesapla
            boruta_importances = calculate_boruta_importance(X, y, feature_names)
            save_importance_results(boruta_importances, method, run_name=run_name)
            results[method] = boruta_importances
        
        else:
            print(f"UYARI: Bilinmeyen metod '{method}' atlanıyor.")
    
    return results
