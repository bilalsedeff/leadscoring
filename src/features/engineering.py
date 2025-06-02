# src/features/engineering.py
"""
Advanced, leakage-safe feature engineering for lead-conversion modelling.

Author : you
Date   : 2025-05-28
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, List, Tuple

# -----------------------------------------------------------------------------
# helper ----------------------------------------------------------------------
# -----------------------------------------------------------------------------


def _parse_yearmonth_col(s: pd.Series) -> pd.Series:
    """
    Converts integer-like '52024' → datetime(2024-05-01).

    If the value is already four chars ('1024') treat as 4-digit year?  → skip.
    """
    s = s.astype(str).str.strip()
    year = s.str[-4:].astype(int)
    month = s.str[:-4].replace("", "1").astype(int)  # '12024'→'12', '52024'→'5'
    return pd.to_datetime(
        dict(year=year, month=month, day=1), errors="coerce"
    )


def _is_weekend(dt_series: pd.Series) -> pd.Series:
    return dt_series.dt.weekday >= 5  # Sat=5, Sun=6


# -----------------------------------------------------------------------------
# main class ------------------------------------------------------------------
# -----------------------------------------------------------------------------


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Generates temporal, cumulative, rolling, lag and interaction features.

    Parameters
    ----------
    date_col : str, default='LeadCreatedDate'
    account_col : str, default='account_Id'
    source_col : str, default='Source_Final__c'
    yearmonth_col : str, default='YearMonth'
    windows_days : tuple[int], default=(30, 90, 180)
        Rolling windows (in *days*) for lead counts per source & total.
    keep_original_cols : bool, default=True
        Whether to keep the original columns in the output DataFrame.
    drop_id_cols : bool, default=True
        Whether to drop ID columns from the output DataFrame.
    id_cols : list, default=None
        List of ID columns to drop from the output DataFrame.
    """

    def __init__(
        self,
        date_col: str = "LeadCreatedDate",
        account_col: str = "account_Id",
        source_col: str = "Source_Final__c",
        yearmonth_col: str = "YearMonth",
        windows_days: Union[List[int], Tuple[int, ...]] = (30, 90, 180),
        keep_original_cols: bool = True,
        drop_id_cols: bool = True,
        id_cols: list = None,
    ):
        self.date_col = date_col
        self.account_col = account_col
        self.source_col = source_col
        self.yearmonth_col = yearmonth_col
        
        # Window değerlerini doğrula - geçersiz değerleri düzelt
        self.windows_days = [max(1, int(w)) for w in windows_days]
        self.keep_original_cols = keep_original_cols
        self.drop_id_cols = drop_id_cols
        self.id_cols = id_cols or ["LeadId", "account_Id", "Id"]
        
        # Fit sırasında hesaplanacak istatistikler için boş değişkenler oluştur
        self._source_top_ = []
        self._account_overall_leadcnt_ = pd.Series(dtype='int32')
        
        # Hesaplanmış zaman bilgileri için değişkenler
        self._min_date_ = None
        self._max_date_ = None

    # ---------------------------------------------------------------------
    # Fit : sadece train seti ile hesaplanan "frozen" istatistikleri sakla
    # ---------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None):
        """
        • Hesap (account) bazlı *tarih-bağımsız* istatistikler (örn. toplam lead sayısı)
          veya nadir kategorileri belirleyip kaydeder.
          
        Not: Bu metod sadece train verisi üzerinde çağrılmalıdır!
        """
        df = X.copy()
        
        # Gerekli kolonların varlığını kontrol et
        if self.source_col not in df.columns:
            self._source_top_ = []
            print(f"Uyarı: {self.source_col} kolonu bulunamadı. Kaynak bazlı özellikler oluşturulamayacak.")
        else:
            # Determine rare categories of source/channel to bucket them later
            # Veri sızıntısını önlemek için sadece eğitim verisindeki source kategorilerini kullanıyoruz
            self._source_top_ = (
                df[self.source_col].value_counts(normalize=True)[
                    lambda v: v > 0.01
                ].index.tolist()
            )
        
        # Hesap kolonu kontrolü
        if self.account_col not in df.columns:
            self._account_overall_leadcnt_ = pd.Series(dtype='int32')
            print(f"Uyarı: {self.account_col} kolonu bulunamadı. Hesap bazlı özellikler oluşturulamayacak.")
        else:
            # Account-level global stats - Veri sızıntısını önlemek için sadece eğitim verisindeki hesap istatistiklerini saklıyoruz
            self._account_overall_leadcnt_ = (
                df.groupby(self.account_col).size().astype("int32")
            )
        
        # Tarih kolonu varlığını kontrol et
        if self.date_col in df.columns:
            # Eğitim verisinin tarih aralığını sakla - validation/test setinde bu aralık dışına çıkan değerlerle karşılaştığımızda
            # rolling window hesaplamalarında sorun yaşamamak için
            self._min_date_ = df[self.date_col].min() if pd.api.types.is_datetime64_any_dtype(df[self.date_col]) else None
            self._max_date_ = df[self.date_col].max() if pd.api.types.is_datetime64_any_dtype(df[self.date_col]) else None
        
        return self

    # ---------------------------------------------------------------------
    # Transform : leakage-safe feature construction
    # ---------------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Veri çerçevesine leakage-safe özellikler ekler.
        
        Not: Bu metod fit sırasında hesaplanan istatistikleri kullanır, bu yüzden önce fit edilmelidir.
        Ayrıca, eğitim, validation ve test verilerinde aynı özellikler oluşturulması için tüm verilere
        uygulanabilir.
        """
        df = X.copy()

        # -----------------------------------------------------------------
        # 0) Basic time parsing
        # -----------------------------------------------------------------
        if self.date_col not in df.columns:
            print(f"Hata: {self.date_col} kolonu bulunamadı. Tarih bazlı özellikler oluşturulamayacak.")
            return df
            
        # Tarih kolonunu datetime tipine dönüştür
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
            try:
                df[self.date_col] = pd.to_datetime(df[self.date_col])
            except Exception as e:
                print(f"Tarih dönüşümü sırasında hata: {e}")
                return df
        
        # YearMonth özellikleri oluştur
        if self.yearmonth_col in df.columns:
            df["_ym_date"] = _parse_yearmonth_col(df[self.yearmonth_col])
        else:
            # YearMonth kolonu yoksa tarih kolonundan oluştur
            df["_ym_date"] = pd.to_datetime(dict(
                year=df[self.date_col].dt.year,
                month=df[self.date_col].dt.month,
                day=1
            ))

        # -----------------------------------------------------------------
        # 1) Simple temporal parts
        # -----------------------------------------------------------------
        df["lead_year"] = df[self.date_col].dt.year.astype("int16")
        df["lead_month"] = df[self.date_col].dt.month.astype("int8")
        df["lead_quarter"] = df[self.date_col].dt.quarter.astype("int8")
        df["lead_weekofyear"] = df[self.date_col].dt.isocalendar().week.astype("int8")
        df["lead_weekday"] = df[self.date_col].dt.weekday.astype("int8")
        df["is_weekend"] = _is_weekend(df[self.date_col]).astype("int8")

        # -----------------------------------------------------------------
        # 2) Leakage-safe cumulative & rolling counts
        # -----------------------------------------------------------------
        if self.account_col not in df.columns:
            print(f"Uyarı: {self.account_col} kolonu bulunamadı. Hesap bazlı özellikler oluşturulamayacak.")
        else:
            # Mutlaka önce sıralayın: (DATA LEAK ÖNLEMİ: Her bir veri setini kendi içinde sıralıyoruz)
            df = df.sort_values([self.account_col, self.date_col])

            # per-account cumulative lead count *until current row*
            # Bu akümülatif hesaplama yönü nedeniyle veri sızıntısı yapmıyor - her kayıt sadece kendinden önceki kayıtlara bakıyor
            df["cum_leads_account"] = (
                df.groupby(self.account_col).cumcount().astype("int32")
            )

            # days since previous lead of same account
            # Yine her kayıt sadece kendinden önceki kayıta bakıyor
            df["days_since_prev_lead"] = (
                df.groupby(self.account_col)[self.date_col]
                .diff()
                .dt.days.fillna(-1)
                .astype("int16")
            )

            # Kaynak kolonu kontrolü
            if self.source_col in df.columns and hasattr(self, '_source_top_') and self._source_top_:
                # create one-hot for main sources (rare bucketed as 'Other')
                # FIT aşamasında belirlenen top kaynaklara göre one-hot encoding yapıyoruz
                df["_src_tmp"] = np.where(df[self.source_col].isin(self._source_top_),
                                        df[self.source_col],
                                        "Other")

                for src in self._source_top_ + ["Other"]:
                    df[f"is_src_{src}"] = (df["_src_tmp"] == src).astype("int8")

                # rolling window lead counts per account (total) ve per source
                for window in self.windows_days:
                    # Window değerinin geçerli olduğundan emin ol
                    if window <= 0:
                        continue
                        
                    # etiket: 30'lu tam aylar için L{n}M, değilse {n}D
                    if window % 30 == 0:
                        window_label = f"L{window//30}M"
                    else:
                        window_label = f"{window}D"

                    # --- total lead count over past window excluding current row ---
                    # closed='left' ile current row döneme dahil edilmiyor
                    # LEAKAGE-SAFE: Her kayıt, kendinden önceki kayıtlara bakıyor
                    try:
                        # Kontrol ekleyelim - tarih sütunu datetime mi?
                        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
                            print(f"Hata: {self.date_col} datetime tipinde değil. Rolling hesaplaması yapılamıyor.")
                            continue
                            
                        # Sıralama yapalım, güncel kayıtlar en sonda
                        df_sorted = df.sort_values(by=[self.account_col, self.date_col])
                        
                        # Her hesap için ayrı ayrı rolling window hesaplaması
                        lead_counts = []
                        for acc_id, group in df_sorted.groupby(self.account_col):
                            # Tarih bazlı rolling count - alternatif yaklaşım
                            group = group.set_index(self.date_col)
                            # Size yerine count() kullanıyoruz
                            rolling_counts = group.rolling(f'{window}D', closed='left').count().reset_index()
                            if rolling_counts.shape[1] > 1:  # En az bir sayısal kolon varsa
                                # İlk sayısal kolonu kullan
                                first_col = next((col for col in rolling_counts.columns if col != self.date_col), None)
                                if first_col:
                                    rolling_counts = rolling_counts[[self.date_col, first_col]]
                                    rolling_counts[self.account_col] = acc_id
                                    rolling_counts = rolling_counts.rename(columns={first_col: f"lead_cnt_{window_label}"})
                                    lead_counts.append(rolling_counts)
                            
                        if lead_counts:
                            # Tüm hesapların sonuçlarını birleştir
                            all_counts = pd.concat(lead_counts, ignore_index=True)
                            # Tarih ve hesap ID'ye göre birleştir
                            df = df.merge(
                                all_counts[[self.date_col, self.account_col, f"lead_cnt_{window_label}"]], 
                                on=[self.date_col, self.account_col], 
                                how='left'
                            )
                            # Eksik değerleri 0 ile doldur
                            df[f"lead_cnt_{window_label}"] = df[f"lead_cnt_{window_label}"].fillna(0).astype("int16")
                        else:
                            # Eğer hesaplama yapılamadıysa, sıfır doldur
                            df[f"lead_cnt_{window_label}"] = 0

                        # --- per-source rolling counts ---
                        for src in self._source_top_:
                            col_name = f"lead_cnt_{src}_{window_label}"
                            
                            # Her kaynak için ayrı hesaplama
                            src_lead_counts = []
                            for acc_id, group in df_sorted[df_sorted[self.source_col] == src].groupby(self.account_col):
                                if len(group) > 0:  # Grup boş değilse
                                    group = group.set_index(self.date_col)
                                    # Size yerine count() kullanıyoruz
                                    rolling_counts = group.rolling(f'{window}D', closed='left').count().reset_index()
                                    if rolling_counts.shape[1] > 1:  # En az bir sayısal kolon varsa
                                        # İlk sayısal kolonu kullan
                                        first_col = next((col for col in rolling_counts.columns if col != self.date_col), None)
                                        if first_col:
                                            rolling_counts = rolling_counts[[self.date_col, first_col]]
                                            rolling_counts[self.account_col] = acc_id
                                            rolling_counts = rolling_counts.rename(columns={first_col: col_name})
                                            src_lead_counts.append(rolling_counts)
                            
                            # Önce tüm değerleri 0 ata
                            df[col_name] = 0
                            
                            if src_lead_counts:
                                # Tüm hesapların sonuçlarını birleştir
                                all_src_counts = pd.concat(src_lead_counts, ignore_index=True)
                                # Tarih ve hesap ID'ye göre birleştir
                                df = df.merge(
                                    all_src_counts[[self.date_col, self.account_col, col_name]], 
                                    on=[self.date_col, self.account_col], 
                                    how='left'
                                )
                                # Eksik değerleri 0 ile doldur
                                df[col_name] = df[col_name].fillna(0).astype("int16")
                    except Exception as e:
                        print(f"{window} günlük window için rolling hesaplama hatası: {e}")
                        print(f"Muhtemelen geçersiz window değeri: {window}")
                        continue

                # now compute source ratios within each window
                for window in self.windows_days:
                    if window <= 0:  # Geçersiz window değerlerini atla
                        continue
                        
                    window_label = f"L{window//30}M" if window % 30 == 0 else f"{window}D"
                    total_col = f"lead_cnt_{window_label}"
                    
                    # Total kolon yoksa atla
                    if total_col not in df.columns:
                        continue
                        
                    # sıfırları NaN yap ki bölme sıfır-dan kaçınsın
                    tot = df[total_col].replace(0, np.nan)

                    for src in self._source_top_:
                        src_col = f"lead_cnt_{src}_{window_label}"
                        
                        # Kaynak kolonu yoksa atla
                        if src_col not in df.columns:
                            continue
                            
                        ratio_col = f"ratio_{src}_{window_label}"
                        df[ratio_col] = (
                            df[src_col] / tot
                        ).fillna(0).astype("float32")

        # -----------------------------------------------------------------
        # 3) Interaction examples
        # -----------------------------------------------------------------
        if "Channel_Final__c" in df.columns and self.source_col in df.columns:
            try:
                df["src_chan_cross"] = (
                    df[self.source_col].astype(str) + "_" + df["Channel_Final__c"].astype(str)
                )

                # frequency encode high-card cross - Sadece mevcut veri içindeki frekansları kullan
                # DATA LEAK ÖNLEMİ: Eğitim setinde kaynak bazlı olasılıklar hesaplandıysa, onları kullan
                if hasattr(self, '_source_channel_freq_') and self._source_channel_freq_ is not None:
                    # Fit sırasında hesaplanan frekansları kullan
                    df["src_chan_freq_enc"] = df["src_chan_cross"].map(self._source_channel_freq_).fillna(0).astype("float32")
                else:
                    # İlk kez hesaplanıyorsa (fit sırasında) frekansları sakla
                    freq = df["src_chan_cross"].value_counts(normalize=True)
                    df["src_chan_freq_enc"] = df["src_chan_cross"].map(freq).astype("float32")
                    # Sadece fit sırasında çağrılmışsa frekansları sakla
                    if not hasattr(self, '_source_channel_freq_'):
                        self._source_channel_freq_ = freq
            except Exception as e:
                print(f"Kaynak/Kanal çapraz özelliği oluşturma hatası: {e}")

        # -----------------------------------------------------------------
        # 4) Bucket rare categories of RecordType
        # -----------------------------------------------------------------
        if "Recordtypedevelopername" in df.columns:
            try:
                # DATA LEAK ÖNLEMİ: Eğitim setinde nadir kategoriler hesaplandıysa, onları kullan
                if hasattr(self, '_rare_record_types_') and self._rare_record_types_ is not None:
                    # Fit sırasında belirlenen nadir record type'ları kullan
                    df["RecordType_bucketed"] = df["Recordtypedevelopername"].replace(
                        self._rare_record_types_, "RARE"
                    )
                else:
                    # İlk kez hesaplanıyorsa (fit sırasında)
                    rec_counts = df["Recordtypedevelopername"].value_counts()
                    rare_recs = rec_counts[rec_counts < 50].index
                    df["RecordType_bucketed"] = df["Recordtypedevelopername"].replace(
                        rare_recs, "RARE"
                    )
                    # Sadece fit sırasında çağrılmışsa nadir kategorileri sakla
                    if not hasattr(self, '_rare_record_types_'):
                        self._rare_record_types_ = rare_recs
            except Exception as e:
                print(f"RecordType gruplama hatası: {e}")

        # -----------------------------------------------------------------
        # 5) Global account statistics (train-frozen)
        # -----------------------------------------------------------------
        if self.account_col in df.columns and hasattr(self, '_account_overall_leadcnt_') and not self._account_overall_leadcnt_.empty:
            try:
                # DATA LEAK ÖNLEMİ: Eğitim verisinde hesaplanan hesap istatistiklerini kullan
                df["account_total_leads"] = df[self.account_col].map(
                    self._account_overall_leadcnt_
                ).fillna(0).astype("int32")
            except Exception as e:
                print(f"Hesap toplam lead sayısı hesaplama hatası: {e}")

        # -----------------------------------------------------------------
        # 6) Drop helpers and ID columns
        # -----------------------------------------------------------------
        if "_src_tmp" in df.columns:
            df.drop(columns=["_src_tmp"], inplace=True)
        
        # Kimlik sütunlarını düşür (kullanıcı isterse)
        if self.drop_id_cols:
            id_cols_to_drop = [col for col in self.id_cols if col in df.columns]
            if id_cols_to_drop:
                df.drop(columns=id_cols_to_drop, inplace=True)
                
        if not self.keep_original_cols:
            cols_to_drop = []
            for col in [self.date_col, self.source_col, self.yearmonth_col, 
                       "Channel_Final__c", "Recordtypedevelopername"]:
                if col in df.columns:
                    cols_to_drop.append(col)
            
            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True)
                
        return df.reset_index(drop=True)

def add_temporal_features(df, date_col=None, window_sizes=[30, 90, 180], verbose=True):
    """Veri setine zaman bazlı özellikler ekler
    
    Args:
        df: Pandas DataFrame
        date_col: Tarih kolonu. None ise algılamaya çalışır.
        window_sizes: Rolling window boyutları (gün cinsinden)
        verbose: Detaylı loglama
        
    Returns:
        DataFrame: Zaman bazlı özellikler eklenmiş veri seti
    """
    # Kopya oluştur
    df_copy = df.copy()
    
    # Tarih kolonu yoksa algıla
    if date_col is None:
        date_cols = [col for col in df.columns 
                    if ('date' in col.lower() or 'time' in col.lower() or 'tarih' in col.lower())
                    and not col.endswith('_YearMonth') 
                    and 'relative' not in col.lower()]
        
        if not date_cols:
            if verbose:
                print("Tarih kolonu bulunamadı, zaman bazlı özellikler eklenemedi.")
            return df_copy
        
        # İlk tarih kolonunu kullan
        date_col = date_cols[0]
        if verbose:
            print(f"Tarih kolonu otomatik algılandı: {date_col}")
    
    # Tarih kolonu datetime tipinde değilse dönüştür
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        try:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            if verbose:
                print(f"{date_col} kolonu datetime tipine dönüştürüldü")
        except Exception as e:
            if verbose:
                print(f"Tarih dönüştürme hatası: {e}")
            return df_copy
    
    # Veriyi tarihe göre sırala
    df_copy = df_copy.sort_values(date_col)
    
    # Zaman bileşenleri
    df_copy[f'{date_col}_Year'] = df_copy[date_col].dt.year
    df_copy[f'{date_col}_Month'] = df_copy[date_col].dt.month
    df_copy[f'{date_col}_Day'] = df_copy[date_col].dt.day
    df_copy[f'{date_col}_DayOfWeek'] = df_copy[date_col].dt.dayofweek
    df_copy[f'{date_col}_Quarter'] = df_copy[date_col].dt.quarter
    df_copy[f'{date_col}_DayOfYear'] = df_copy[date_col].dt.dayofyear
    df_copy[f'{date_col}_WeekOfYear'] = df_copy[date_col].dt.isocalendar().week
    
    # Yıl ve ay kombinasyonu (örn: 202101)
    df_copy[f'{date_col}_YearMonth'] = df_copy[date_col].dt.strftime('%Y%m')
    
    # Hafta sonu mu?
    df_copy[f'{date_col}_IsWeekend'] = df_copy[date_col].dt.dayofweek.isin([5, 6]).astype(int)
    
    # İş günü mü? (Pazartesi-Cuma)
    df_copy[f'{date_col}_IsBusinessDay'] = ~df_copy[date_col].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Mevsim
    month = df_copy[date_col].dt.month
    df_copy[f'{date_col}_Season'] = np.select(
        [month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11]), month.isin([12, 1, 2])],
        ['Spring', 'Summer', 'Fall', 'Winter']
    )
    
    # Ay içindeki günler: Başlangıç, orta, son
    day = df_copy[date_col].dt.day
    days_in_month = df_copy[date_col].dt.daysinmonth
    df_copy[f'{date_col}_MonthPart'] = np.select(
        [day <= days_in_month/3, day <= 2*days_in_month/3, day <= days_in_month],
        ['Early', 'Mid', 'Late']
    )
    
    # Günün kısmı (sabah, öğleden sonra, akşam, gece)
    if hasattr(df_copy[date_col].dt, 'hour'):  # datetime saat bilgisi içeriyorsa
        hour = df_copy[date_col].dt.hour
        df_copy[f'{date_col}_DayPart'] = np.select(
            [hour < 6, hour < 12, hour < 18, hour < 24],
            ['Night', 'Morning', 'Afternoon', 'Evening']
        )
    
    # Source_Final__c kolonu varsa kaynak bazlı zaman analizleri yap
    if 'Source_Final__c' in df_copy.columns:
        try:
            # Kaynak bazlı istatistikler 
            for source in df_copy['Source_Final__c'].unique():
                # Kaynak adını güvenli hale getir (özel karakterleri temizle)
                safe_source = source.replace(' ', '_').replace('-', '_')
                
                # Her source için rolling window hesaplamaları
                for window in window_sizes:
                    try:
                        # Window değerini kontrol et
                        if window <= 0:
                            if verbose:
                                print(f"Geçersiz window değeri: {window}, pozitif bir değer olmalıdır.")
                            continue
                        
                        # Rolling window zamansal çerçeve
                        feature_name = f'lead_cnt_{safe_source}_L{window//30}M'  # L1M, L3M, L6M formatı
                        
                        # Tarih aralığı ve sayıları
                        source_df = df_copy[df_copy['Source_Final__c'] == source].copy()
                        
                        if len(source_df) <= 1:
                            # Bu source için yetersiz veri
                            df_copy[feature_name] = 0
                            continue
                        
                        # Tarih bazlı gruplama ve rolling hesaplama
                        date_counts = source_df.groupby(pd.Grouper(key=date_col, freq='D')).size()
                        
                        # Tarih indeksli seri oluştur
                        date_range = pd.date_range(start=date_counts.index.min(), end=date_counts.index.max())
                        full_date_series = pd.Series(0, index=date_range)
                        
                        # Var olan tarihleri doldur
                        for date, count in date_counts.items():
                            if date in full_date_series.index:
                                full_date_series[date] = count
                        
                        # Rolling window hesapla (min_periods=1 ile)
                        rolling_counts = full_date_series.rolling(window=f'{window}D', min_periods=1).sum()
                        
                        # Her gözlem için uygun rolling değerini bul
                        df_copy[feature_name] = df_copy[date_col].map(
                            lambda d: rolling_counts.get(pd.Timestamp(d).normalize(), 0)
                        )
                    except Exception as e:
                        if verbose:
                            print(f"{window} günlük window için rolling hesaplama hatası: '{feature_name}'")
                            print(f"Hata: {e}")
        except Exception as e:
            if verbose:
                print(f"Kaynak bazlı hesaplamada hata: {e}")
    
    return df_copy
