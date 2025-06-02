"""
python -m src.pipelines.stats --type=<analiz_tipi>  (CLI'dan çağrılır)
"""
import os
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.ingestion.loader import read_datamart
from src.preprocessing.cleaning import BasicCleaner
from src.features.engineering import add_temporal_features
from src.utils.logger import get_logger

log = get_logger()
OUTPUT_DIR = Path("outputs/stats")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@click.command()
@click.option('--type', type=int, default=4, 
              help='Analiz tipi: 1=Temel İstatistikler, 2=Korelasyon, 3=Kaynak Bazlı, 4=Tümü')
def main(type):
    """İstatistiksel analizleri gerçekleştirir"""
    log.info("Veriler yükleniyor...")
    df = read_datamart()
    
    log.info("Veri temizleniyor...")
    cleaner = BasicCleaner()
    df = cleaner.fit_transform(df)
    
    log.info("Özellikler ekleniyor...")
    df = add_temporal_features(df)
    
    # Analiz tipine göre işlem yap
    if type == 1 or type == 4:
        basic_stats(df)
    
    if type == 2 or type == 4:
        correlation_analysis(df)
    
    if type == 3 or type == 4:
        source_based_analysis(df)
        cohort_analysis(df)
    
    log.info(f"Analiz tamamlandı. Sonuçlar {OUTPUT_DIR} dizininde.")

def basic_stats(df):
    """Temel istatistikleri hesaplar ve kaydeder"""
    log.info("Temel istatistikler hesaplanıyor...")
    
    # Sayısal sütunların istatistikleri
    num_stats = df.describe().transpose().reset_index()
    num_stats.rename(columns={'index': 'feature'}, inplace=True)
    num_stats.to_csv(OUTPUT_DIR / "numeric_stats.csv", index=False)
    
    # Kategorik sütunların istatistikleri
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if cat_cols:
        cat_stats = pd.DataFrame()
        
        for col in cat_cols:
            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = ['value', 'count']
            value_counts['percentage'] = value_counts['count'] / len(df) * 100
            value_counts['feature'] = col
            cat_stats = pd.concat([cat_stats, value_counts])
        
        cat_stats.to_csv(OUTPUT_DIR / "categorical_stats.csv", index=False)
    
    # Missing value analizi
    missing = pd.DataFrame({
        'feature': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_percentage': df.isnull().mean().values * 100
    }).sort_values('missing_percentage', ascending=False)
    
    missing.to_csv(OUTPUT_DIR / "missing_stats.csv", index=False)
    
    # Görselleştirme: Missing values
    plt.figure(figsize=(12, 8))
    sns.barplot(x='missing_percentage', y='feature', 
               data=missing[missing['missing_percentage'] > 0].sort_values('missing_percentage'))
    plt.title('Missing Values (Eksik Değerler)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missing_values.png")
    plt.close()
    
    # Görselleştirme: Hedef değişken dağılımı
    if 'Target_IsConverted' in df.columns:
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='Target_IsConverted', data=df)
        
        # Yüzdeleri ekle
        total = len(df)
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.annotate(percentage, (x, y), ha='center', va='bottom')
        
        plt.title('Target Dağılımı (Conversion)')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "target_distribution.png")
        plt.close()
    
    log.info("Temel istatistikler kaydedildi.")

def correlation_analysis(df):
    """Korelasyon analizi yapar"""
    log.info("Korelasyon analizi yapılıyor...")
    
    # Sayısal sütunları al
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if 'Target_IsConverted' in num_cols:
        # Target ile korelasyonları hesapla
        target_corrs = df[num_cols].corr()['Target_IsConverted'].sort_values(ascending=False)
        target_corrs = pd.DataFrame({
            'feature': target_corrs.index,
            'correlation': target_corrs.values
        })
        target_corrs.to_csv(OUTPUT_DIR / "target_correlations.csv", index=False)
        
        # Top 20 korelasyon görselleştirme
        plt.figure(figsize=(12, 10))
        sns.barplot(x='correlation', y='feature', 
                   data=target_corrs[target_corrs['feature'] != 'Target_IsConverted'].head(20))
        plt.title('Top 20 Features by Correlation with Target')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "target_correlations.png")
        plt.close()
    
    # Korelasyon matrisi (top 30 feature)
    if len(num_cols) > 5:
        # En çok korelasyona sahip 30 feature'ı seç
        if 'Target_IsConverted' in num_cols:
            top_corr_features = target_corrs['feature'].head(min(30, len(num_cols))).tolist()
            if 'Target_IsConverted' not in top_corr_features:
                top_corr_features.append('Target_IsConverted')
        else:
            # Target yoksa en yüksek varyansa sahip 30 feature'ı seç
            variances = df[num_cols].var().sort_values(ascending=False)
            top_corr_features = variances.index[:min(30, len(num_cols))].tolist()
        
        # Korelasyon matrisi
        corr_matrix = df[top_corr_features].corr()
        
        # Görselleştirme
        plt.figure(figsize=(16, 14))
        mask = np.triu(corr_matrix)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask, 
                   fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "correlation_matrix.png")
        plt.close()
    
    log.info("Korelasyon analizi kaydedildi.")

def source_based_analysis(df):
    """Kaynak bazlı dönüşüm oranlarını analiz eder"""
    log.info("Kaynak bazlı analiz yapılıyor...")
    
    if 'Source_Final__c' in df.columns and 'Target_IsConverted' in df.columns:
        # Kaynak bazlı dönüşüm oranları
        source_conv = df.groupby('Source_Final__c').agg(
            total_count=('Target_IsConverted', 'count'),
            converted=('Target_IsConverted', 'sum')
        ).reset_index()
        
        source_conv['conversion_rate'] = source_conv['converted'] / source_conv['total_count']
        source_conv['percentage_of_total'] = source_conv['total_count'] / source_conv['total_count'].sum() * 100
        
        # Sıralama
        source_conv = source_conv.sort_values('conversion_rate', ascending=False)
        source_conv.to_csv(OUTPUT_DIR / "source_conversion_rates.csv", index=False)
        
        # Görselleştirme: Top 10 kaynak
        plt.figure(figsize=(14, 8))
        top_sources = source_conv.head(10)
        
        ax = sns.barplot(x='Source_Final__c', y='conversion_rate', data=top_sources)
        
        # Yüzdeleri ekle
        for i, row in enumerate(top_sources.itertuples()):
            ax.text(i, row.conversion_rate + 0.01, 
                   f'{row.conversion_rate:.1%}\n({row.total_count})', 
                   ha='center')
        
        plt.title('Top 10 Sources by Conversion Rate')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "top_sources_conversion.png")
        plt.close()
        
        # Görselleştirme: En yaygın 10 kaynak
        plt.figure(figsize=(14, 8))
        top_common = source_conv.sort_values('total_count', ascending=False).head(10)
        
        ax = sns.barplot(x='Source_Final__c', y='conversion_rate', data=top_common)
        
        # Yüzdeleri ve sayıları ekle
        for i, row in enumerate(top_common.itertuples()):
            ax.text(i, row.conversion_rate + 0.01, 
                   f'{row.conversion_rate:.1%}\n({row.total_count})', 
                   ha='center')
        
        plt.title('Most Common 10 Sources and Their Conversion Rates')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "common_sources_conversion.png")
        plt.close()
        
        # Kaynak bazlı performans dosyası
        source_perf = source_conv.copy()
        source_perf.rename(columns={
            'Source_Final__c': 'source',
            'total_count': 'count',
            'converted': 'conversion_count'
        }, inplace=True)
        source_perf.to_csv(OUTPUT_DIR / "source_performance.csv", index=False)
        
        log.info("Kaynak bazlı analiz kaydedildi.")
    else:
        log.warning("Kaynak veya hedef sütunu bulunamadığından kaynak analizi yapılamadı.")

def cohort_analysis(df):
    """Cohort bazlı analiz yapar"""
    log.info("Cohort analizi yapılıyor...")
    
    # Cohort analizi için grup değişkenleri
    cohort_vars = []
    
    if 'Source_Final__c' in df.columns:
        cohort_vars.append('Source_Final__c')
    
    # Tarih bazlı cohort
    date_cols = [col for col in df.columns if 'Date' in col or 'date' in col]
    if date_cols:
        # Tarih sütunlarından yıl ve ay bilgisini çıkar
        for date_col in date_cols:
            if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[f"{date_col}_YearMonth"] = df[date_col].dt.to_period('M').astype(str)
                cohort_vars.append(f"{date_col}_YearMonth")
    
    # Segment veya özel cohort değişkenleri
    segment_cols = [col for col in df.columns if 'Segment' in col or 'segment' in col]
    cohort_vars.extend(segment_cols)
    
    # Diğer potansiyel cohort değişkenleri
    if 'Industry' in df.columns:
        cohort_vars.append('Industry')
    if 'Country' in df.columns:
        cohort_vars.append('Country')
    
    # Cohort analizi yap
    if cohort_vars and 'Target_IsConverted' in df.columns:
        cohort_results = []
        
        for cohort_var in cohort_vars:
            if cohort_var in df.columns:
                # Cohort bazlı dönüşüm oranları
                cohort_conv = df.groupby(cohort_var).agg(
                    count=('Target_IsConverted', 'count'),
                    converted=('Target_IsConverted', 'sum')
                ).reset_index()
                
                cohort_conv['conversion_rate'] = cohort_conv['converted'] / cohort_conv['count']
                cohort_conv['cohort_type'] = cohort_var
                
                # Sütun adını standartlaştır
                cohort_conv.rename(columns={cohort_var: 'cohort_value'}, inplace=True)
                
                cohort_results.append(cohort_conv)
        
        if cohort_results:
            # Tüm cohort sonuçlarını birleştir
            all_cohorts = pd.concat(cohort_results)
            all_cohorts.to_csv(OUTPUT_DIR / "cohort_analysis.csv", index=False)
            
            # Her cohort tipi için görselleştirme
            for cohort_type in all_cohorts['cohort_type'].unique():
                cohort_data = all_cohorts[all_cohorts['cohort_type'] == cohort_type]
                
                # En az 5 kayda sahip olanları al
                cohort_data = cohort_data[cohort_data['count'] >= 5]
                
                # Top 10 (conversion rate'e göre)
                top_cohorts = cohort_data.sort_values('conversion_rate', ascending=False).head(10)
                
                plt.figure(figsize=(14, 8))
                ax = sns.barplot(x='cohort_value', y='conversion_rate', data=top_cohorts)
                
                # Yüzdeleri ve sayıları ekle
                for i, row in enumerate(top_cohorts.itertuples()):
                    ax.text(i, row.conversion_rate + 0.01, 
                           f'{row.conversion_rate:.1%}\n({row.count})', 
                           ha='center')
                
                plt.title(f'Top 10 {cohort_type} by Conversion Rate')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / f"{cohort_type}_conversion.png")
                plt.close()
            
            log.info("Cohort analizi kaydedildi.")
        else:
            log.warning("Cohort değişkenleri bulunamadığından cohort analizi yapılamadı.")
    else:
        log.warning("Cohort değişkenleri veya hedef bulunamadığından cohort analizi yapılamadı.")

if __name__ == "__main__":
    main() 