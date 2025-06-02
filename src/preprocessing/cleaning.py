import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
import logging
from omegaconf import OmegaConf
from pathlib import Path

logger = logging.getLogger(__name__)

class BasicCleaner(BaseEstimator, TransformerMixin):
    """
    • Eksik değer stratejisi
    • Outlier winsorize (z-score, IQR bazlı veya Isolation Forest)
    • Karışık boşluk / özel karakter düzeltme
    """
    def __init__(self, num_fill="median", cat_fill="missing", 
                outlier_method="zscore", outlier_threshold=3.0, 
                winsorize_limits=(0.01, 0.99), apply_outlier_treatment=True,
                iforest_contamination=0.05):
        """
        Parameters
        ----------
        num_fill : str, default="median"
            Sayısal kolonlardaki eksik değerleri doldurma stratejisi ("median", "mean")
        cat_fill : str, default="missing"
            Kategorik kolonlardaki eksik değerleri doldurma stratejisi
        outlier_method : str, default="zscore"
            Outlier belirleme ve baskılama metodu ("zscore", "iqr", "winsorize", "iforest", None)
        outlier_threshold : float, default=3.0
            Z-score metodu için eşik değeri
        winsorize_limits : tuple, default=(0.01, 0.99)
            Winsorize metodu için sınır limitleri (IQR metodu için kullanılmaz)
        apply_outlier_treatment : bool, default=True
            Outlier baskılamanın uygulanıp uygulanmayacağı
        iforest_contamination : float, default=0.05
            Isolation Forest için beklenen anomali oranı (0.0 - 0.5 arası)
        """
        self.num_fill = num_fill
        self.cat_fill = cat_fill
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.winsorize_limits = winsorize_limits
        self.apply_outlier_treatment = apply_outlier_treatment
        self.iforest_contamination = iforest_contamination
        self.iforest_models_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        # sayısalların medyanını veya ortalamasını, kategoriklerin modunu sakla
        if self.num_fill == "median":
            self.num_values_ = X.select_dtypes("number").median()
        elif self.num_fill == "mean":
            self.num_values_ = X.select_dtypes("number").mean()
        else:
            self.num_values_ = X.select_dtypes("number").median()
            
        # Kategorik verileri al (hem 'object' hem 'category' tiplerini içerecek şekilde)
        cat_df = X.select_dtypes(include=['object', 'category'])
        # cat_df boş değilse mode hesapla, boşsa boş Series döndür
        if not cat_df.empty:
            mode_result = cat_df.mode()
            if not mode_result.empty:
                self.cat_mode_ = mode_result.iloc[0]
            else:
                self.cat_mode_ = pd.Series(dtype='object')
        else:
            self.cat_mode_ = pd.Series(dtype='object')
        
        # Outlier limitleri hesapla (eğitim seti üzerinde)
        if self.apply_outlier_treatment and self.outlier_method:
            self.lower_bounds_ = {}
            self.upper_bounds_ = {}
            
            num_cols = X.select_dtypes("number").columns
            for col in num_cols:
                if self.outlier_method == "zscore":
                    # Z-score bazlı sınırlar
                    mean = X[col].mean()
                    std = X[col].std()
                    self.lower_bounds_[col] = mean - self.outlier_threshold * std
                    self.upper_bounds_[col] = mean + self.outlier_threshold * std
                    
                elif self.outlier_method == "iqr":
                    # IQR bazlı sınırlar - standart 0.25 ve 0.75 çeyrekliklerini kullan
                    q1 = X[col].quantile(0.25)
                    q3 = X[col].quantile(0.75)
                    iqr = q3 - q1
                    self.lower_bounds_[col] = q1 - 1.5 * iqr
                    self.upper_bounds_[col] = q3 + 1.5 * iqr
                
                elif self.outlier_method == "winsorize":
                    # Winsorize limitleri (transform'da kullanılacak)
                    # Burada sınırları saklamıyoruz, doğrudan winsorize uygulayacağız
                    pass
                
                elif self.outlier_method == "iforest":
                    # Isolation Forest modeli eğit
                    # Her sayısal kolon için ayrı model
                    values = X[col].values.reshape(-1, 1)
                    iforest = IsolationForest(
                        contamination=self.iforest_contamination,
                        random_state=42
                    )
                    iforest.fit(values)
                    self.iforest_models_[col] = iforest
        
        return self
        
    def transform(self, X):
        X = X.copy()
        num_cols = X.select_dtypes("number").columns
        cat_cols = X.select_dtypes("object").columns
        
        # Eksik değerleri doldur
        X[num_cols] = X[num_cols].fillna(self.num_values_)
        X[cat_cols] = X[cat_cols].fillna(self.cat_mode_)
        
        # Outlier baskılama
        if self.apply_outlier_treatment and self.outlier_method:
            for col in num_cols:
                if self.outlier_method == "zscore" or self.outlier_method == "iqr":
                    # Sınırlar dışındaki değerleri clip et
                    if col in self.lower_bounds_ and col in self.upper_bounds_:
                        # Int64 tipindeki sütunlar için özel işlem
                        if pd.api.types.is_integer_dtype(X[col]) or pd.api.types.is_integer_dtype(X[col].dtype):
                            # Int64 için önce kırpma değerlerini tam sayıya dönüştür
                            lower = int(self.lower_bounds_[col])
                            upper = int(self.upper_bounds_[col])
                            
                            # NA değerleri koruyarak kırpma işlemi yapalım
                            mask = ~X[col].isna()
                            # Sadece NA olmayan değerleri kırpalım
                            if mask.any():
                                X.loc[mask, col] = X.loc[mask, col].clip(lower=lower, upper=upper)
                        else:
                            # Float sütunlar için normal clip işlemi
                            X[col] = X[col].clip(
                                lower=self.lower_bounds_[col], 
                                upper=self.upper_bounds_[col]
                            )
                
                elif self.outlier_method == "winsorize":
                    # Int64 tipindeki sütunlar için özel işlem
                    if pd.api.types.is_integer_dtype(X[col]) or pd.api.types.is_integer_dtype(X[col].dtype):
                        # NA değerleri koruyarak winsorize işlemi
                        mask = ~X[col].isna()
                        if mask.any():
                            # NA olmayan değerleri winsorize et ve integer'a dönüştür
                            winsorized = stats.mstats.winsorize(
                                X.loc[mask, col].values, 
                                limits=self.winsorize_limits
                            )
                            X.loc[mask, col] = np.round(winsorized).astype(X[col].dtype)
                    else:
                        # Float sütunlar için normal winsorize işlemi
                        X[col] = pd.Series(
                            stats.mstats.winsorize(
                                X[col].values, 
                                limits=self.winsorize_limits
                            ), 
                            index=X.index
                        )
                
                elif self.outlier_method == "iforest" and col in self.iforest_models_:
                    # Isolation Forest ile anomalileri tespit et
                    values = X[col].values.reshape(-1, 1)
                    iforest = self.iforest_models_[col]
                    # -1: anomali, 1: normal
                    is_outlier = iforest.predict(values) == -1
                    
                    if is_outlier.any():
                        # Outlier olan değerleri, kolon ortalaması veya medyanı ile değiştir
                        if self.num_fill == "median":
                            replacement = X[col].median()
                        else:
                            replacement = X[col].mean()
                        
                        # Int64 tipindeki sütunlar için yuvarla
                        if pd.api.types.is_integer_dtype(X[col]) or pd.api.types.is_integer_dtype(X[col].dtype):
                            replacement = int(np.round(replacement))
                        
                        X.loc[is_outlier, col] = replacement
        
        return X

class SmartFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Akıllı özellik seçimi için transformer.
    
    Bu sınıf, çeşitli istatistiksel yöntemlerle otomatik özellik seçimi yapar:
    1. Yüksek oranda eksik değer içeren kolonları eler
    2. Duplicate kolonları eler
    3. Düşük varyansa sahip kolonları eler
    4. Outlier oranı yüksek kolonları eler
    5. Hedef değişkenle düşük korelasyona sahip kolonları eler
    6. Yüksek multicollinearity olan kolonları eler
    7. Opsiyonel olarak PCA yapar
    
    Parameters
    ----------
    missing_thresh : float, default=0.3
        Eksik değer oranı bu değerden yüksek olan kolonlar elenecek
    duplicate : bool, default=True
        Birebir aynı olan kolonları ele
    near_zero_var_thresh : float, default=0.01
        Varyans oranı bu değerden düşük olan kolonlar elenecek
    outlier_method : str, default='iqr'
        Outlier tespiti için kullanılacak metod ('iqr' veya 'zscore')
    outlier_thresh : float, default=0.5
        Outlier oranı bu değerden yüksek olan kolonlar elenecek
    correlation_thresh : float, default=0.95
        Kolonlar arası korelasyon bu değerden yüksekse, target ile daha düşük ilişkili olanı ele
    target_correlation_min : float, default=0.02
        Hedef değişkenle korelasyonu bu değerden düşük olan kolonlar elenecek
    pca_components : int, default=None
        PCA uygulanacaksa kullanılacak bileşen sayısı (None: PCA yok)
    pca_variance : float, default=0.95
        PCA uygulanacaksa korunacak varyans oranı (pca_components None ise kullanılır)
    verbose : bool, default=True
        Elenen kolonlar hakkında detaylı bilgi ver
    """
    def __init__(self,
                 missing_thresh: float = 0.3,
                 duplicate: bool = True,
                 near_zero_var_thresh: float = 0.01,
                 outlier_method: str = 'iqr',
                 outlier_thresh: float = 0.5,
                 correlation_thresh: float = 0.95,
                 target_correlation_min: float = 0.02,
                 pca_components: int = None,
                 pca_variance: float = 0.95,
                 verbose: bool = True):
        self.missing_thresh = missing_thresh
        self.duplicate = duplicate
        self.near_zero_var_thresh = near_zero_var_thresh
        self.outlier_method = outlier_method
        self.outlier_thresh = outlier_thresh
        self.correlation_thresh = correlation_thresh
        self.target_correlation_min = target_correlation_min
        self.pca_components = pca_components
        self.pca_variance = pca_variance
        self.verbose = verbose
        
        # İçinde eklenecek:
        self.columns_ = None
        self.to_drop_ = []
        self.pca_ = None
        self.drop_reasons_ = {}
        self.kept_columns_ = None
        self.feature_stats_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Özellikleri analiz eder ve eleme kriterlerini belirler.
        
        Parameters
        ----------
        X : pd.DataFrame
            Özellik seti
        y : array-like, default=None
            Hedef değişken (opsiyonel)
            
        Returns
        -------
        self : object
            Fit edilmiş nesne
        """
        self.columns_ = X.columns.tolist()
        self.to_drop_ = []
        self.drop_reasons_ = {}
        dropped_cols = {}
        
        # Özellik istatistiklerini topla
        self.feature_stats_ = self._collect_feature_stats(X)
        
        # 1) Eksik değer oranı yüksek olan kolonlar
        miss_ratio = X.isna().mean()
        high_missing = miss_ratio[miss_ratio > self.missing_thresh].index.tolist()
        for col in high_missing:
            dropped_cols[col] = f"Eksik deger orani yuksek: {miss_ratio[col]:.2f}"
        self.to_drop_.extend(high_missing)
        
        # 2) Duplicate kolonlar
        if self.duplicate:
            duplicates = []
            # Sütun eşleşmelerini kontrol et
            for i, col1 in enumerate(X.columns):
                if col1 in self.to_drop_:
                    continue
                for col2 in X.columns[i+1:]:
                    if col2 in self.to_drop_ or col2 in duplicates:
                        continue
                    if X[col1].equals(X[col2]):
                        duplicates.append(col2)
                        dropped_cols[col2] = f"Duplikat kolon: {col1} ile ayni"
            self.to_drop_.extend(duplicates)
        
        # 3) Düşük varyansa sahip kolonlar
        var_df = X.select_dtypes(include='number').copy()
        var_df = var_df.drop(columns=self.to_drop_, errors='ignore')
        
        if not var_df.empty:
            # Varyans Threshold için tüm sayısal değerleri normalize et
            for col in var_df.columns:
                if var_df[col].std() > 0:
                    var_df[col] = (var_df[col] - var_df[col].min()) / (var_df[col].max() - var_df[col].min())
            
            var_selector = VarianceThreshold(threshold=self.near_zero_var_thresh)
            var_selector.fit(var_df.fillna(0))
            low_var_cols = var_df.columns[~var_selector.get_support()].tolist()
            
            for col in low_var_cols:
                dropped_cols[col] = f"Dusuk varyans: {var_df[col].var():.4f}"
            self.to_drop_.extend(low_var_cols)
        
        # 4) Outlier oranı yüksek kolonlar
        num_df = X.select_dtypes(include='number').drop(columns=self.to_drop_, errors='ignore')
        if not num_df.empty:
            if self.outlier_method == 'iqr':
                Q1 = num_df.quantile(0.25)
                Q3 = num_df.quantile(0.75)
                IQR = Q3 - Q1
                mask = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR)))
            else:  # 'zscore'
                z_scores = (num_df - num_df.mean()) / num_df.std()
                mask = (abs(z_scores) > 3)
            
            outlier_ratios = mask.mean()
            high_outlier_cols = outlier_ratios[outlier_ratios > self.outlier_thresh].index.tolist()
            
            for col in high_outlier_cols:
                dropped_cols[col] = f"Yuksek outlier orani: {outlier_ratios[col]:.2f}"
            self.to_drop_.extend(high_outlier_cols)
        
        # 5) Hedef değişkenle ilişki (y verilmişse)
        if y is not None:
            y_series = pd.Series(y)
            # Kategorik kolonlar için Mutual Information
            cat_df = X.select_dtypes(exclude='number').drop(columns=self.to_drop_, errors='ignore')
            for col in cat_df.columns:
                try:
                    mi = mutual_info_classif(
                        cat_df[col].fillna('MISSING').values.reshape(-1, 1), 
                        y_series, 
                        discrete_features=True
                    )[0]
                    if mi < self.target_correlation_min:
                        self.to_drop_.append(col)
                        dropped_cols[col] = f"Dusuk mutual information: {mi:.4f}"
                except Exception as e:
                    logger.warning(f"Mutual info hesaplanamadi - {col}: {str(e)}")
            
            # Sayısal kolonlar için Korelasyon
            num_df = X.select_dtypes(include='number').drop(columns=self.to_drop_, errors='ignore')
            if not num_df.empty:
                for col in num_df.columns:
                    try:
                        corr = abs(num_df[col].fillna(num_df[col].median()).corr(y_series))
                        if corr < self.target_correlation_min:
                            self.to_drop_.append(col)
                            dropped_cols[col] = f"Dusuk hedef korelasyonu: {corr:.4f}"
                    except Exception as e:
                        logger.warning(f"Korelasyon hesaplanamadi - {col}: {str(e)}")
        
        # 6) Yüksek korelasyonlu sayısal özelliklerden birini ele
        num_df = X.select_dtypes(include='number').drop(columns=self.to_drop_, errors='ignore')
        if not num_df.empty and len(num_df.columns) > 1:
            corr_matrix = num_df.fillna(0).corr().abs()
            
            # Üçgen üst matris oluştur (kendisiyle korelasyonu hariç tut)
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Yüksek korelasyonlu çiftleri bul
            high_corr_pairs = []
            for col in upper.columns:
                correlated = upper[col][upper[col] > self.correlation_thresh].index.tolist()
                for corr_col in correlated:
                    high_corr_pairs.append((col, corr_col, upper.loc[corr_col, col]))
            
            # Hedef değişkenle daha az ilişkili olanı ele
            if high_corr_pairs and y is not None:
                for col1, col2, corr_val in high_corr_pairs:
                    if col1 in self.to_drop_ or col2 in self.to_drop_:
                        continue
                    
                    try:
                        corr1 = abs(num_df[col1].fillna(num_df[col1].median()).corr(pd.Series(y)))
                        corr2 = abs(num_df[col2].fillna(num_df[col2].median()).corr(pd.Series(y)))
                        
                        # Hedef ile daha az ilişkili olanı ele
                        if corr1 <= corr2:
                            self.to_drop_.append(col1)
                            dropped_cols[col1] = f"Yuksek korelasyon: {col2} ile {corr_val:.2f}, hedef korelasyonu: {corr1:.2f} < {corr2:.2f}"
                        else:
                            self.to_drop_.append(col2)
                            dropped_cols[col2] = f"Yuksek korelasyon: {col1} ile {corr_val:.2f}, hedef korelasyonu: {corr2:.2f} < {corr1:.2f}"
                    except Exception as e:
                        logger.warning(f"Korelasyon karsilastirma hatasi - {col1} vs {col2}: {str(e)}")
        
        # 7) PCA (opsiyonel)
        num_df = X.select_dtypes(include='number').drop(columns=self.to_drop_, errors='ignore')
        if not num_df.empty and self.pca_components is not None:
            if self.pca_components > 0:
                n_components = min(self.pca_components, len(num_df.columns))
                self.pca_ = PCA(n_components=n_components)
            else:
                self.pca_ = PCA(n_components=self.pca_variance, svd_solver='full')
            
            # PCA'yı fit et
            self.pca_.fit(num_df.fillna(0))
            
            if self.verbose:
                explained_var = self.pca_.explained_variance_ratio_.sum()
                logger.info(f"PCA: {self.pca_.n_components_} bilesen ile toplam varyansin %{explained_var*100:.2f}'i aciklandi")
        
        # Elenen kolonları kaydet
        self.to_drop_ = list(set(self.to_drop_))
        self.drop_reasons_ = dropped_cols
        
        # Kalan kolonları kaydet
        self.kept_columns_ = [col for col in self.columns_ if col not in self.to_drop_]
        
        # Sonuçları logla
        if self.verbose:
            logger.info(f"SmartFeatureSelector: Toplam {len(self.columns_)} kolondan {len(self.to_drop_)} tanesi elendi")
            for col, reason in list(dropped_cols.items())[:10]:  # İlk 10 nedeni göster
                logger.info(f"  - {col}: {reason}")
            if len(dropped_cols) > 10:
                logger.info(f"  - ... ve {len(dropped_cols) - 10} diğer kolon")
        
        return self

    def transform(self, X: pd.DataFrame):
        """
        Elenen kolonları kaldırır ve PCA uygular.
        
        Parameters
        ----------
        X : pd.DataFrame
            Dönüştürülecek veri
            
        Returns
        -------
        Xt : pd.DataFrame
            Dönüştürülmüş veri
        """
        if self.columns_ is None:
            raise ValueError("Bu transformer henüz fit edilmedi!")
        
        # Kolonları kontrol et
        missing_cols = [col for col in self.columns_ if col not in X.columns]
        if missing_cols:
            logger.warning(f"Transform edilen veri setinde {len(missing_cols)} kolon eksik: {missing_cols[:5]}...")
        
        # Elenen kolonları kaldır
        to_drop = [col for col in self.to_drop_ if col in X.columns]
        Xt = X.drop(columns=to_drop, errors='ignore').copy()
        
        # PCA sonuçlarını ekle
        if self.pca_ is not None:
            num_df = X.select_dtypes(include='number')
            num_df = num_df.drop(columns=to_drop, errors='ignore')
            
            if not num_df.empty:
                common_cols = [col for col in num_df.columns if col in self.kept_columns_]
                
                # Kolon sıralamasını PCA fit edildiği şekilde kullan
                if len(common_cols) > 0:
                    # Eksik kolonları doldurmak için medyan kullan
                    for col in common_cols:
                        if num_df[col].isna().any():
                            num_df[col] = num_df[col].fillna(num_df[col].median())
                    
                    try:
                        # PCA transform uygula
                        pcs = self.pca_.transform(num_df[common_cols])
                        
                        # Standart ve tutarlı isimlendirme kullan
                        pc_columns = []
                        for i in range(pcs.shape[1]):
                            pc_col_name = f'PC_{i+1:03d}'  # Format: PC_001, PC_002, vb.
                            Xt[pc_col_name] = pcs[:, i]
                            pc_columns.append(pc_col_name)
                            
                        # PCA kolon isimlerini log'la
                        logger.info(f"PCA bileşenleri eklendi: {pc_columns}")
                        
                    except Exception as e:
                        logger.error(f"PCA transform sırasında hata: {str(e)}")
                        logger.error(f"Hata ayrıntıları: Giriş boyutu={num_df[common_cols].shape}, " 
                                    f"PCA bileşen sayısı={self.pca_.n_components_}")
                else:
                    logger.warning("PCA için sayısal özellik bulunamadı!")
        
        return Xt

    def get_feature_names_out(self, input_features=None):
        """
        Dönüştürme sonrası kolon isimlerini döndürür.
        
        Parameters
        ----------
        input_features : array-like, default=None
            Giriş kolon isimleri
            
        Returns
        -------
        feature_names_out : ndarray
            Dönüştürme sonrası kolon isimleri
        """
        if self.columns_ is None:
            raise ValueError("Bu transformer henüz fit edilmedi!")
            
        output_features = [col for col in self.columns_ if col not in self.to_drop_]
        
        # PCA bileşenlerini ekle - transform ile tutarlı isimlendirme
        if self.pca_ is not None:
            for i in range(self.pca_.n_components_):
                output_features.append(f'PC_{i+1:03d}')  # Format: PC_001, PC_002, vb.
        
        return np.array(output_features)

    def _collect_feature_stats(self, X: pd.DataFrame):
        """
        Özellik istatistiklerini toplar.
        
        Parameters
        ----------
        X : pd.DataFrame
            Özellik seti
            
        Returns
        -------
        stats : pd.DataFrame
            Özellik istatistikleri
        """
        stats = pd.DataFrame({
            'dtype': X.dtypes.astype(str),
            'na_ratio': X.isna().mean(),
            'nunique': X.nunique(),
            'nunique_ratio': X.nunique() / len(X)
        })
        
        # Sayısal kolonlar için ek istatistikler
        num_cols = X.select_dtypes(include='number').columns
        if len(num_cols) > 0:
            stats.loc[num_cols, 'mean'] = X[num_cols].mean()
            stats.loc[num_cols, 'std'] = X[num_cols].std()
            stats.loc[num_cols, 'min'] = X[num_cols].min()
            stats.loc[num_cols, 'max'] = X[num_cols].max()
            stats.loc[num_cols, 'skew'] = X[num_cols].skew()
            stats.loc[num_cols, 'kurtosis'] = X[num_cols].kurtosis()
        
        # Kategorik kolonlar için ek istatistikler
        cat_cols = X.select_dtypes(exclude='number').columns
        if len(cat_cols) > 0:
            # En sık görülen değer ve oranı
            for col in cat_cols:
                if X[col].nunique() > 0:
                    most_common = X[col].value_counts().idxmax()
                    most_common_ratio = X[col].value_counts().max() / len(X)
                    stats.loc[col, 'most_common'] = most_common
                    stats.loc[col, 'most_common_ratio'] = most_common_ratio
        
        return stats

def generate_feature_report(selector, output_path=None):
    """
    SmartFeatureSelector sonuçlarını HTML rapor olarak kaydeder.
    
    Parameters
    ----------
    selector : SmartFeatureSelector
        Fit edilmiş selector
    output_path : str, default=None
        Raporun kaydedileceği dosya yolu
        
    Returns
    -------
    report_html : str
        HTML formatında rapor
    """
    if selector.columns_ is None:
        raise ValueError("Bu selector henüz fit edilmedi!")
    
    # Elenen kolonlar hakkında bilgi
    dropped_df = pd.DataFrame({
        'feature': list(selector.drop_reasons_.keys()),
        'reason': list(selector.drop_reasons_.values())
    })
    
    # Kalan kolonlar
    kept_cols = [col for col in selector.columns_ if col not in selector.to_drop_]
    
    # İstatistikler
    stats_df = selector.feature_stats_.copy()
    stats_df['selected'] = stats_df.index.isin(kept_cols)
    
    # HTML rapor oluştur
    html = f"""
    <html>
    <head>
        <title>Feature Selection Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .stats {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            .stats th, .stats td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .stats th {{ background-color: #f2f2f2; }}
            .stats tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .stats tr:hover {{ background-color: #f5f5f5; }}
            .selected {{ background-color: #d4edda !important; }}
            .dropped {{ background-color: #f8d7da !important; }}
            .summary {{ margin-top: 20px; padding: 15px; background-color: #e9f7ef; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Feature Selection Report</h1>
        
        <div class="summary">
            <h2>Özet</h2>
            <p>Toplam özellik sayısı: {len(selector.columns_)}</p>
            <p>Seçilen özellik sayısı: {len(kept_cols)}</p>
            <p>Elenen özellik sayısı: {len(selector.to_drop_)}</p>
            
            <h3>Parametre Değerleri</h3>
            <ul>
                <li>Eksik değer eşiği: {selector.missing_thresh}</li>
                <li>Duplikat özellik kontrolü: {selector.duplicate}</li>
                <li>Düşük varyans eşiği: {selector.near_zero_var_thresh}</li>
                <li>Outlier metodu: {selector.outlier_method}</li>
                <li>Outlier eşiği: {selector.outlier_thresh}</li>
                <li>Korelasyon eşiği: {selector.correlation_thresh}</li>
                <li>Hedef korelasyon minimum: {selector.target_correlation_min}</li>
                <li>PCA bileşen sayısı: {selector.pca_components}</li>
            </ul>
            
            <h3>PCA Bilgisi</h3>
    """
    
    if selector.pca_ is not None:
        explained_var = selector.pca_.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        html += f"""
            <p>PCA bileşen sayısı: {selector.pca_.n_components_}</p>
            <p>Toplam açıklanan varyans: {explained_var.sum():.4f}</p>
            <table class="stats">
                <tr>
                    <th>Bileşen</th>
                    <th>Açıklanan Varyans</th>
                    <th>Kümülatif Varyans</th>
                </tr>
        """
        
        for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var)):
            html += f"""
                <tr>
                    <td>PC_{i+1}</td>
                    <td>{var:.4f}</td>
                    <td>{cum_var:.4f}</td>
                </tr>
            """
        
        html += "</table>"
    else:
        html += "<p>PCA uygulanmadı</p>"
    
    html += """
        </div>
        
        <h2>Elenen Özellikler</h2>
    """
    
    if not dropped_df.empty:
        html += """
        <table class="stats">
            <tr>
                <th>Özellik</th>
                <th>Elenme Nedeni</th>
            </tr>
        """
        
        for _, row in dropped_df.iterrows():
            html += f"""
            <tr class="dropped">
                <td>{row['feature']}</td>
                <td>{row['reason']}</td>
            </tr>
            """
        
        html += "</table>"
    else:
        html += "<p>Hiç özellik elenmedi</p>"
    
    html += """
        <h2>Tüm Özellik İstatistikleri</h2>
        <table class="stats">
            <tr>
                <th>Özellik</th>
                <th>Veri Tipi</th>
                <th>Eksik Oranı</th>
                <th>Benzersiz Değer</th>
                <th>Benzersiz Oran</th>
                <th>Durum</th>
            </tr>
    """
    
    for feature, row in stats_df.iterrows():
        status = "Seçildi" if row['selected'] else "Elendi"
        row_class = "selected" if row['selected'] else "dropped"
        
        html += f"""
        <tr class="{row_class}">
            <td>{feature}</td>
            <td>{row['dtype']}</td>
            <td>{row['na_ratio']:.4f}</td>
            <td>{row['nunique']}</td>
            <td>{row['nunique_ratio']:.4f}</td>
            <td>{status}</td>
        </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    # Raporu dosyaya kaydet
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
    return html

def load_cleaning_config():
    """
    Temizleme yapılandırmasını configs/cleaning.yaml'dan yükler
    
    Returns:
        dict: Temizleme yapılandırması
    """
    config_path = Path("configs/cleaning.yaml")
    if not config_path.exists():
        logger.warning("cleaning.yaml bulunamadı, varsayılan yapılandırma kullanılacak.")
        return {
            "num_fill": "median",
            "cat_fill": "missing",
            "outlier_method": "zscore",
            "outlier_threshold": 3.0,
            "winsorize_limits": [0.01, 0.99],
            "apply_outlier_treatment": True,
            "iforest_contamination": 0.05,
            "feature_selection": {
                "missing_thresh": 0.3,
                "duplicate": True,
                "near_zero_var_thresh": 0.01,
                "outlier_method": "iqr",
                "outlier_thresh": 0.5,
                "correlation_thresh": 0.95,
                "target_correlation_min": 0.02,
                "pca_components": None,
                "pca_variance": 0.95
            }
        }
    
    return OmegaConf.load(config_path)

def stable_feature_subset(importances, top_k=60, stability_thr=0.6):
    """
    Özellik önem derecelerini kullanarak en istikrarlı özellikleri seçer.
    
    Parameters
    ----------
    importances : dict{fold: pd.Series}
        Her fold için özellik önem derecelerini içeren sözlük
    top_k : int, default=60
        Seçilecek maksimum özellik sayısı
    stability_thr : float, default=0.6
        Bir özelliğin seçilmesi için gereken istikrar eşiği (0-1 arası)
        
    Returns
    -------
    list
        Seçilen özelliklerin listesi
    """
    concat = pd.concat(importances, axis=1).fillna(0)
    stability = (concat > 0).mean(1)
    top = concat.mean(1).nlargest(top_k).index
    return [col for col in top if stability[col] >= stability_thr]
