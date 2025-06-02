"""
İstatistiksel testler modülü. Lead Conversion analizinde kullanılabilecek
çeşitli istatistiksel testleri ve analizleri içerir.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.formula.api import ols
from statsmodels.graphics.mosaicplot import mosaic
from pathlib import Path
import logging
import os
import time
from ..registry.mlflow_wrapper import configure_mlflow_output_dir
from typing import Dict, List, Any, Union, Optional, Tuple

# Matplotlib ayarları - açık figür sayısını sınırla
import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 50  # Varsayılan 20 yerine 50'ye çıkar
plt.style.use('default')  # Daha temiz bir stil kullan

logger = logging.getLogger("lead_scoring.stats")

def chi_square_test(df: pd.DataFrame, 
                   categorical_col: str, 
                   target_col: str = 'Target_IsConverted',
                   significance_level: float = 0.05,
                   output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Kategorik bir değişkenle hedef değişken arasındaki ilişkiyi analiz etmek için ki-kare testi
    
    Args:
        df: Veri çerçevesi
        categorical_col: Kategorik sütun
        target_col: Hedef sütun (ikili)
        significance_level: Anlamlılık seviyesi
        output_dir: Grafiklerin kaydedileceği dizin
    
    Returns:
        Dict: Ki-kare testi sonuçları
    """
    try:
        # Eksik değerleri filtrele
        df_clean = df[[categorical_col, target_col]].dropna()
        
        # Çapraz tablo oluştur
        cross_tab = pd.crosstab(df_clean[categorical_col], df_clean[target_col])
        
        # Ki-kare testi
        chi2, p, dof, expected = stats.chi2_contingency(cross_tab)
        
        # Sonucu yorumla
        if p < significance_level:
            interpretation = f"Anlamli iliski bulundu (p={p:.4f})"
            significant = True
        else:
            interpretation = f"Anlamli iliski bulunamadi (p={p:.4f})"
            significant = False
        
        # Grafikler
        if output_dir:
            # Eğer output_dir zaten Path nesnesi ise, dönüştürme
            if not isinstance(output_dir, Path):
                output_dir = Path(output_dir)
            
            # Alt klasör oluşturmaya çalışma - mevcut dizini kullan
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Conversion rate plot
            plt.figure(figsize=(10, 6))
            conv_rates = cross_tab.div(cross_tab.sum(axis=1), axis=0)[1]
            conv_rates.sort_values(ascending=False).plot(kind='bar')
            plt.axhline(y=df_clean[target_col].mean(), color='r', linestyle='--', 
                       label=f'Ortalama Donusum ({df_clean[target_col].mean():.4f})')
            plt.xlabel(categorical_col)
            plt.ylabel('Donusum Orani')
            plt.title(f'Donusum Orani - {categorical_col} Bazinda')
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(output_dir / f"chi_square_{categorical_col}_conversion_rate.png")
            plt.close()  # Figure'ı kapat
            
            # Count plot
            plt.figure(figsize=(10, 6))
            cross_tab.plot(kind='bar', stacked=True)
            plt.xlabel(categorical_col)
            plt.ylabel('Kayit Sayisi')
            plt.title(f'Kayit Sayisi - {categorical_col} Bazinda')
            plt.legend(title=target_col)
            plt.tight_layout()
            
            plt.savefig(output_dir / f"chi_square_{categorical_col}_counts.png")
            plt.close()  # Figure'ı kapat
        
        return {
            'feature': categorical_col,
            'chi2': chi2,
            'p_value': p,
            'dof': dof,
            'significant': significant,
            'interpretation': interpretation,
            'contingency_table': cross_tab.to_dict()
        }
    except Exception as e:
        print(f"{categorical_col} için ki-kare testi sırasında hata: {e}")
        plt.close()  # Hata durumunda grafiği kapat
        return {
            'feature': categorical_col,
            'error': str(e)
        }

def t_test_by_group(df: pd.DataFrame, 
                   numeric_col: str, 
                   group_col: str, 
                   equal_var: bool = False, 
                   output_dir: Optional[Union[str, Path]] = None,
                   significance_level: float = 0.05) -> Dict:
    """
    Gruplar arası t-test hesaplar ve görselleştirir
    
    Args:
        df: Veri çerçevesi
        numeric_col: Sayısal değişken 
        group_col: Grup değişkeni (iki değer içermeli)
        equal_var: Eşit varyans varsayımı
        output_dir: Çıktı dizini
        significance_level: Anlamlılık seviyesi
        
    Returns:
        Dict: t-test sonuçları
    """
    try:
        # Veriyi hazırla
        df_clean = df[[numeric_col, group_col]].dropna()
        
        # Grup değişkeninin iki kategorisi olup olmadığını kontrol et
        groups = df_clean[group_col].unique()
        if len(groups) != 2:
            return {
                'feature': numeric_col,
                'error': f"Grup değişkeni ({group_col}) ikiden {'fazla' if len(groups) > 2 else 'az'} kategori içeriyor: {groups}"
            }
        
        # İki gruba ait değerleri al
        group1 = df_clean[df_clean[group_col] == groups[0]][numeric_col]
        group2 = df_clean[df_clean[group_col] == groups[1]][numeric_col]
        
        # Örneklem büyüklüğü kontrolü
        if len(group1) < 2 or len(group2) < 2:
            return {
                'feature': numeric_col,
                'error': f"Yetersiz örneklem: {groups[0]}={len(group1)}, {groups[1]}={len(group2)}"
            }
        
        # t-test
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        # Anlamlılık kontrolü
        is_significant = p_value < significance_level
        
        # Görselleştirme
        if output_dir:
            try:
                # Dizini oluştur
                if not isinstance(output_dir, Path):
                    output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Box plot
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=group_col, y=numeric_col, data=df_clean)
                plt.title(f"{numeric_col} - {group_col} gruplarına göre\nt={t_stat:.4f}, p={p_value:.4f} {'(ANLAMLI)' if is_significant else ''}")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Kaydet
                plt.savefig(output_dir / f"ttest_{numeric_col}_{group_col}.png")
                plt.close()  # Grafiği kapat
                
                # İkinci görselleştirme: Violin plot
                plt.figure(figsize=(10, 6))
                sns.violinplot(x=group_col, y=numeric_col, data=df_clean, inner="quartile")
                plt.title(f"{numeric_col} - {group_col} gruplarına göre dağılım\nt={t_stat:.4f}, p={p_value:.4f}")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Kaydet
                plt.savefig(output_dir / f"ttest_violin_{numeric_col}_{group_col}.png")
                plt.close()  # Grafiği kapat
            except Exception as e:
                print(f"{numeric_col} için t-test grafiği oluşturulurken hata: {e}")
                plt.close()  # Hata durumunda da grafiği kapat
        
        # Sonuçları döndür
        return {
            'feature': numeric_col,
            'group': group_col,
            'group_values': groups.tolist(),
            'group_means': [group1.mean(), group2.mean()],
            'group_counts': [len(group1), len(group2)],
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': is_significant,
            'method': 'Welch' if not equal_var else 'Student'
        }
    except Exception as e:
        print(f"{numeric_col} için t-test sırasında hata: {e}")
        plt.close()  # Hata durumunda grafiği kapat
        return {
            'feature': numeric_col,
            'error': str(e)
        }

def anova_test(df: pd.DataFrame, 
                numeric_col: str, 
                group_col: str, 
                output_dir: Optional[Union[str, Path]] = None,
                significance_level: float = 0.05) -> Dict:
    """
    Gruplar arası ANOVA testi hesaplar ve görselleştirir
    
    Args:
        df: Veri çerçevesi
        numeric_col: Sayısal değişken 
        group_col: Grup değişkeni (ikiden fazla değer içermeli)
        output_dir: Çıktı dizini
        significance_level: Anlamlılık seviyesi
        
    Returns:
        Dict: ANOVA sonuçları
    """
    try:
        # Veriyi hazırla
        df_clean = df[[numeric_col, group_col]].dropna()
        
        # Grup değişkeninin kategorilerini kontrol et
        groups = df_clean[group_col].unique()
        if len(groups) < 2:
            return {
                'feature': numeric_col,
                'error': f"Grup değişkeni ({group_col}) 2'den az kategori içeriyor: {groups}"
            }
        
        # Grup bazında değerleri topla
        group_data = [df_clean[df_clean[group_col] == group][numeric_col].values for group in groups]
        
        # ANOVA testi
        f_stat, p_value = stats.f_oneway(*group_data)
        
        # Anlamlılık kontrolü
        is_significant = p_value < significance_level
        
        # Görselleştirme
        if output_dir:
            try:
                # Dizini oluştur
                if not isinstance(output_dir, Path):
                    output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Box plot
                plt.figure(figsize=(12, 6))
                sns.boxplot(x=group_col, y=numeric_col, data=df_clean)
                plt.title(f"{numeric_col} - {group_col} gruplarına göre\nF={f_stat:.4f}, p={p_value:.6f} {'(ANLAMLI)' if is_significant else ''}")
                plt.xticks(rotation=45)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Kaydet
                plt.savefig(output_dir / f"anova_{numeric_col}_{group_col}.png")
                plt.close()  # Grafiği kapat
                
                # İkinci görselleştirme: Violin plot
                plt.figure(figsize=(12, 6))
                sns.violinplot(x=group_col, y=numeric_col, data=df_clean, inner="quartile")
                plt.title(f"{numeric_col} - {group_col} gruplarına göre dağılım\nF={f_stat:.4f}, p={p_value:.6f}")
                plt.xticks(rotation=45)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Kaydet
                plt.savefig(output_dir / f"anova_violin_{numeric_col}_{group_col}.png")
                plt.close()  # Grafiği kapat
            except Exception as e:
                print(f"{numeric_col} için ANOVA testi sırasında hata: {e}")
                plt.close()  # Hata durumunda da grafiği kapat
        
        # Sonuçları döndür
        return {
            'feature': numeric_col,
            'group': group_col,
            'group_values': groups.tolist(),
            'group_means': [df_clean[df_clean[group_col] == group][numeric_col].mean() for group in groups],
            'group_counts': [len(df_clean[df_clean[group_col] == group]) for group in groups],
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': is_significant
        }
    except Exception as e:
        print(f"{numeric_col} için ANOVA testi sırasında hata: {e}")
        plt.close()  # Hata durumunda grafiği kapat
        return {
            'feature': numeric_col,
            'error': str(e)
        }

def conversion_rate_comparison(df: pd.DataFrame, 
                             categorical_col: str, 
                             target_col: str, 
                             output_dir: Optional[Union[str, Path]] = None,
                             significance_level: float = 0.05) -> Dict:
    """
    Kategorik değişkenin kategorileri arasında dönüşüm oranı farkı analizi
    
    Args:
        df: Veri çerçevesi
        categorical_col: Kategorik değişken
        target_col: Hedef değişken (0/1 binary)
        output_dir: Çıktı dizini
        significance_level: Anlamlılık seviyesi
        
    Returns:
        Dict: Analiz sonuçları
    """
    try:
        # Veriyi hazırla
        df_clean = df[[categorical_col, target_col]].dropna()
        
        # Hedef değişken binary olmalı (0/1)
        if not df_clean[target_col].isin([0, 1]).all() and not df_clean[target_col].isin([False, True]).all():
            return {
                'feature': categorical_col,
                'error': f"Hedef değişken ({target_col}) binary (0/1) olmalı"
            }
        
        # Genel dönüşüm oranı
        overall_rate = df_clean[target_col].mean()
        
        # Kategorilere göre dönüşüm oranları ve anlamlılık testi
        segment_rates = {}
        significant_segments = []
        
        for category in df_clean[categorical_col].unique():
            # Bu kategoriye ait dönüşüm oranı
            segment_data = df_clean[df_clean[categorical_col] == category]
            segment_rate = segment_data[target_col].mean()
            segment_size = len(segment_data)
            
            # Kategorinin segment büyüklüğünü kontrol et (çok küçük segmentler için anlamlılık testi yapma)
            if segment_size < 30:
                segment_rates[category] = {
                    'conversion_rate': segment_rate,
                    'sample_size': segment_size,
                    'significant': False,
                    'p_value': None,
                    'note': 'Segment too small for significance test'
                }
                continue
            
            # Proportions z-test (kategorinin oranı ile genel oran arasında)
            count = segment_data[target_col].sum()
            try:
                # Proporsiyon testi
                z_stat, p_value = proportion_test(
                    count, segment_size,  # Kategori içindeki başarı sayısı ve toplam
                    df_clean[target_col].sum(), len(df_clean)  # Tüm veri setindeki başarı sayısı ve toplam
                )
                
                # Anlamlılık kontrolü
                is_significant = p_value < significance_level
                
                segment_rates[category] = {
                    'conversion_rate': segment_rate,
                    'sample_size': segment_size,
                    'overall_rate': overall_rate,
                    'diff_from_overall': segment_rate - overall_rate,
                    'z_statistic': z_stat,
                    'p_value': p_value,
                    'significant': is_significant
                }
                
                if is_significant:
                    significant_segments.append(category)
            except Exception as e:
                segment_rates[category] = {
                    'conversion_rate': segment_rate,
                    'sample_size': segment_size,
                    'error': str(e)
                }
        
        # Görselleştirme
        if output_dir:
            try:
                # Dizini oluştur
                if not isinstance(output_dir, Path):
                    output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Kategori bazında dönüşüm oranları
                plt.figure(figsize=(12, 6))
                
                # Dönüşüm oranlarını DataFrame'e dönüştür
                df_rates = pd.DataFrame([
                    {'category': cat, **stats}
                    for cat, stats in segment_rates.items()
                ])
                
                # Geçerli değerleri filtrele
                valid_rates = df_rates[~df_rates['conversion_rate'].isna()].copy()
                
                # Dönüşüm oranına göre sırala
                valid_rates = valid_rates.sort_values('conversion_rate', ascending=False)
                
                # Bar renkleri - anlamlı olanlar farklı renkte
                colors = ['red' if cat in significant_segments else 'gray' 
                        for cat in valid_rates['category']]
                
                # Bar plot
                bars = plt.bar(range(len(valid_rates)), 
                            valid_rates['conversion_rate'], 
                            color=colors)
                
                # X ekseni etiketleri
                plt.xticks(range(len(valid_rates)), valid_rates['category'], rotation=90)
                
                # Genel oran çizgisi
                plt.axhline(y=overall_rate, color='green', linestyle='--', 
                        label=f'Genel Oran: {overall_rate:.4f}')
                
                # Grafiği özelleştir
                plt.xlabel(categorical_col)
                plt.ylabel(f'Dönüşüm Oranı ({target_col})')
                plt.title(f'{categorical_col} Kategorilerine Göre Dönüşüm Oranları')
                plt.legend()
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Kaydet
                plt.savefig(output_dir / f"conversion_rate_{categorical_col}.png")
                plt.close()  # Grafiği kapat
                
                # İkinci grafik: Segment büyüklüğüne göre ağırlıklandırılmış scatter plot
                plt.figure(figsize=(10, 6))
                
                # Scatter plot verileri hazırla
                x = valid_rates['sample_size']
                y = valid_rates['conversion_rate']
                labels = valid_rates['category']
                
                # Nokta büyüklüklerini segment boyutuna göre ölçeklendir
                sizes = valid_rates['sample_size'] / valid_rates['sample_size'].max() * 1000
                
                # Renkleri anlamlılığa göre ayarla
                colors = ['red' if cat in significant_segments else 'gray' 
                        for cat in valid_rates['category']]
                
                # Scatter plot çiz
                plt.scatter(x, y, s=sizes, c=colors, alpha=0.6)
                
                # Önemli segmentleri etiketle
                for i, txt in enumerate(labels):
                    if txt in significant_segments:
                        plt.annotate(txt, (x.iloc[i], y.iloc[i]), 
                                fontsize=9, ha='center')
                
                # Genel oran çizgisi
                plt.axhline(y=overall_rate, color='green', linestyle='--', 
                        label=f'Genel Oran: {overall_rate:.4f}')
                
                # Grafiği özelleştir
                plt.xlabel('Segment Büyüklüğü')
                plt.ylabel('Dönüşüm Oranı')
                plt.title(f'{categorical_col} - Segment Büyüklüğü ve Dönüşüm Oranı İlişkisi')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Kaydet
                plt.savefig(output_dir / f"conversion_segment_size_{categorical_col}.png")
                plt.close()  # Grafiği kapat
            except Exception as e:
                print(f"{categorical_col} için dönüşüm oranı analizi sırasında hata: {e}")
                plt.close()  # Hata durumunda da grafiği kapat
        
        # Sonuçları döndür
        return {
            'feature': categorical_col,
            'overall_rate': overall_rate,
            'segment_rates': segment_rates,
            'significant_segments': significant_segments
        }
    except Exception as e:
        print(f"{categorical_col} için dönüşüm oranı analizi sırasında hata: {e}")
        plt.close()  # Hata durumunda grafiği kapat
        return {
            'feature': categorical_col,
            'error': str(e)
        }

def correlation_analysis(df: pd.DataFrame, 
                         numeric_cols: List[str], 
                         target_col: str = None, 
                         method: str = 'pearson',
                         min_abs_corr: float = 0.05,
                         output_dir: Optional[Union[str, Path]] = None) -> Dict:
    """
    Sayısal değişkenler arasında korelasyon analizi yapar
    
    Args:
        df: Veri çerçevesi
        numeric_cols: Sayısal değişkenler listesi
        target_col: Hedef değişken (belirtilirse hedefle korelasyonlar ayrıca hesaplanır)
        method: Korelasyon yöntemi ('pearson', 'spearman', veya 'kendall')
        min_abs_corr: Minimum mutlak korelasyon değeri (daha düşük değerler filtrelenir)
        output_dir: Çıktı dizini
        
    Returns:
        Dict: Korelasyon analizi sonuçları
    """
    try:
        # Değişkenlerin varlığını kontrol et
        all_cols = [col for col in numeric_cols if col in df.columns]
        
        if target_col and target_col in df.columns:
            all_cols.append(target_col)
        
        if not all_cols:
            return {'error': 'Hiçbir sayısal değişken bulunamadı.'}
        
        # Sadece sayısal değişkenleri içeren alt veri seti oluştur
        numeric_df = df[all_cols].copy()
        
        # Korelasyon matrisini hesapla
        corr_matrix = numeric_df.corr(method=method)
        
        # Sonuçları sakla
        correlations = {}
        
        # Hedef değişken ile korelasyonlar
        target_correlations = {}
        if target_col and target_col in corr_matrix.columns:
            for col in corr_matrix.index:
                if col != target_col:
                    corr_value = corr_matrix.loc[col, target_col]
                    if abs(corr_value) >= min_abs_corr:
                        target_correlations[col] = {
                            'correlation': corr_value,
                            'abs_correlation': abs(corr_value)
                        }
        
        # Değişkenler arası yüksek korelasyonlar
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):  # Sadece üst üçgen
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= min_abs_corr:
                    high_correlations.append({
                        'feature1': col1,
                        'feature2': col2,
                        'correlation': corr_value,
                        'abs_correlation': abs(corr_value)
                    })
        
        # Görselleştirme
        if output_dir:
            try:
                # Dizini oluştur
                if not isinstance(output_dir, Path):
                    output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Korelasyon matrisi heat map
                plt.figure(figsize=(max(10, len(all_cols)//2), max(8, len(all_cols)//2)))
                
                # Maskeyi hazırla
                if len(all_cols) > 20:
                    # Çok fazla değişken varsa, sadece yüksek korelasyonlu olanları göster
                    mask = np.abs(corr_matrix) < min_abs_corr
                    
                    # Köşegen elemanlarını göster (1.0 değerleri)
                    np.fill_diagonal(mask, False)
                    
                    # Hedef değişkeni varsa onu da göster
                    if target_col and target_col in corr_matrix.columns:
                        mask[:, corr_matrix.columns.get_loc(target_col)] = False
                        mask[corr_matrix.columns.get_loc(target_col), :] = False
                else:
                    mask = None
                
                # Heatmap
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                         vmin=-1, vmax=1, fmt='.2f', linewidths=0.5,
                         mask=mask, square=True)
                
                plt.title(f'Korelasyon Matrisi ({method.capitalize()})')
                plt.tight_layout()
                
                # Kaydet
                plt.savefig(output_dir / f"correlation_matrix_{method}.png", dpi=150)
                plt.close()  # Grafiği kapat
                
                # Hedef değişken ile korelasyonlar (varsa)
                if target_col and target_col in corr_matrix.columns and target_correlations:
                    plt.figure(figsize=(10, max(6, len(target_correlations)//3)))
                    
                    # Korelasyonları mutlak değere göre sırala
                    sorted_corrs = sorted(target_correlations.items(), 
                                        key=lambda x: abs(x[1]['correlation']), 
                                        reverse=True)
                    
                    features = [x[0] for x in sorted_corrs]
                    corr_values = [x[1]['correlation'] for x in sorted_corrs]
                    
                    # Korelasyon yönüne göre renklendir
                    colors = ['red' if c < 0 else 'blue' for c in corr_values]
                    
                    # Sadece top 20 göster (çok fazla değişken varsa)
                    if len(features) > 20:
                        features = features[:20]
                        corr_values = corr_values[:20]
                        colors = colors[:20]
                    
                    # Değişkenleri ters sırala (en yüksek üstte)
                    features = features[::-1]
                    corr_values = corr_values[::-1]
                    colors = colors[::-1]
                    
                    # Bar plot
                    plt.barh(features, corr_values, color=colors)
                    
                    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    plt.grid(axis='x', linestyle='--', alpha=0.7)
                    plt.xlabel(f'Korelasyon Değeri ({method.capitalize()})')
                    plt.title(f'{target_col} ile Korelasyonlar')
                    plt.tight_layout()
                    
                    # Kaydet
                    plt.savefig(output_dir / f"target_correlations_{method}.png", dpi=150)
                    plt.close()  # Grafiği kapat
            except Exception as e:
                print(f"Korelasyon analizi sırasında hata: {e}")
                plt.close()  # Hata durumunda da grafiği kapat
        
        # Sonuçları döndür
        return {
            'method': method,
            'min_threshold': min_abs_corr,
            'target_correlations': target_correlations,
            'high_correlations': high_correlations,
            'correlation_matrix': corr_matrix.to_dict()
        }
    except Exception as e:
        print(f"Korelasyon analizi sırasında hata: {e}")
        plt.close()  # Hata durumunda grafiği kapat
        return {
            'error': str(e)
        }

def save_test_results(results, test_type, run_name=None):
    """Test sonuçlarını kaydeder.
    
    Args:
        results: Test sonuçları
        test_type: Test tipi (chi_square, t_test, anova, correlation, conversion_rate)
        run_name: Çalıştırma adı (None ise timestamp oluşturulur)
        
    Returns:
        output_path: Sonuçların kaydedildiği dizin
    """
    # MLflow çıktı dizinini yapılandır
    output_dir, run_id = configure_mlflow_output_dir(run_name, "statistical_tests")
    
    # Test tipine göre alt klasör oluştur
    output_path = os.path.join(output_dir, test_type)
    os.makedirs(output_path, exist_ok=True)
    
    # Test tipine göre sonuçları kaydet
    if test_type in ['chi_square', 't_test', 'anova']:
        # DataFrame olarak sonuçları CSV'ye kaydet
        if isinstance(results, pd.DataFrame):
            results.to_csv(os.path.join(output_path, f"{test_type}_results.csv"), index=False)
            
            # Anlamlı sonuçları filtrele
            if 'significant' in results.columns:
                significant = results[results['significant']].copy()
                if not significant.empty:
                    significant.to_csv(os.path.join(output_path, f"{test_type}_significant.csv"), index=False)
            
            # Görselleştirme
            plt.figure(figsize=(10, 6))
            
            if test_type == 'chi_square':
                # En anlamlı 10 özelliği göster
                top_results = results.head(10)
                sns.barplot(x='p_value', y='feature', data=top_results)
                plt.title('Ki-kare Testi: En Anlamlı 10 Kategorik Değişken')
                plt.axvline(x=0.05, color='r', linestyle='--')
            
            elif test_type == 't_test':
                # En anlamlı 10 özelliği göster
                top_results = results.head(10)
                sns.barplot(x='p_value', y='feature', data=top_results)
                plt.title('T-testi: En Anlamlı 10 Değişken')
                plt.axvline(x=0.05, color='r', linestyle='--')
            
            elif test_type == 'anova':
                # En anlamlı 10 özelliği göster
                top_results = results.head(10)
                sns.barplot(x='p_value', y='feature', data=top_results)
                plt.title('ANOVA: En Anlamlı 10 Değişken')
                plt.axvline(x=0.05, color='r', linestyle='--')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"{test_type}_plot.png"))
            plt.close()
    
    elif test_type == 'correlation':
        corr_matrix, target_corr = results
        
        # Korelasyon matrisini kaydet
        corr_matrix.to_csv(os.path.join(output_path, "correlation_matrix.csv"))
        
        # Hedef değişken korelasyonlarını kaydet
        if target_corr is not None:
            target_corr.to_csv(os.path.join(output_path, "target_correlations.csv"))
            
            # Görselleştirme
            plt.figure(figsize=(12, 10))
            
            # Isı haritası
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
            plt.title('Değişkenler Arası Korelasyon Matrisi')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "correlation_heatmap.png"))
            plt.close()
            
            # Hedef korelasyonları
            plt.figure(figsize=(10, 6))
            top_target_corr = target_corr.sort_values(ascending=False)
            top_pos = top_target_corr.head(10)
            top_neg = top_target_corr.tail(10)
            
            # Pozitif
            plt.subplot(1, 2, 1)
            sns.barplot(x=top_pos.values, y=top_pos.index)
            plt.title('Hedef ile En Yüksek 10 Pozitif Korelasyon')
            
            # Negatif
            plt.subplot(1, 2, 2)
            sns.barplot(x=top_neg.values, y=top_neg.index)
            plt.title('Hedef ile En Yüksek 10 Negatif Korelasyon')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "target_correlations.png"))
            plt.close()
    
    elif test_type == 'conversion_rate':
        # Her kategorik değişken için dönüşüm oranlarını kaydet
        for col, conv_rates in results.items():
            conv_rates.to_csv(os.path.join(output_path, f"{col}_conversion_rates.csv"), index=False)
            
            # Görselleştirme
            plt.figure(figsize=(12, 6))
            
            # Dönüşüm oranı çubuğu
            ax = sns.barplot(x=col, y='conversion_rate', data=conv_rates)
            plt.title(f'{col} Değişkenine Göre Dönüşüm Oranları')
            plt.xticks(rotation=45, ha='right')
            
            # Lead sayısını ekle
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                count = conv_rates.iloc[i]['count']
                ax.text(p.get_x() + p.get_width()/2., height + 0.01, f'n={count}', ha="center")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"{col}_conversion_rates.png"))
            plt.close()
    
    print(f"{test_type} test sonuçları {output_path} dizinine kaydedildi.")
    return output_path 