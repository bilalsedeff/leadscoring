from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from .binwise import evaluate_segments, assign_segments, lift_table
import logging

logger = logging.getLogger(__name__)

class BaseCalibrator:
    """Temel kalibratör sınıfı"""
    
    def __init__(self):
        self.calibrator = None
    
    def fit(self, y_prob, y_true):
        """
        Kalibratörü fit eder
        
        Args:
            y_prob: Kalibre edilmemiş olasılıklar
            y_true: Gerçek değerler
            
        Returns:
            self: Fit edilmiş kalibratör
        """
        raise NotImplementedError("Implementasyon alt sınıflarda yapılmalı")
    
    def calibrate(self, y_prob):
        """
        Olasılıkları kalibre eder
        
        Args:
            y_prob: Kalibre edilmemiş olasılıklar
            
        Returns:
            array: Kalibre edilmiş olasılıklar
        """
        if self.calibrator is None:
            raise ValueError("Kalibratör önce fit edilmeli")
        
        return self.calibrator.predict(y_prob.reshape(-1, 1))
    
    def plot_calibration(self, y_true, y_prob, output_dir=None):
        """
        Kalibrasyon eğrisini çizer
        
        Args:
            y_true: Gerçek değerler
            y_prob: Kalibre edilmemiş olasılıklar
            output_dir: Çıktı dizini
            
        Returns:
            dict: Kalibrasyon metrikleri
        """
        if self.calibrator is None:
            raise ValueError("Kalibratör önce fit edilmeli")
        
        # Kalibre edilmiş olasılıklar
        cal_prob = self.calibrate(y_prob)
        
        # Brier skoru
        uncal_brier = _brier(y_true, y_prob)
        cal_brier = _brier(y_true, cal_prob)
        
        # Plot
        plt.figure(figsize=(10, 6))
        _plot_calibration_curve(y_true, y_prob, "Uncalibrated", n_bins=10)
        _plot_calibration_curve(y_true, cal_prob, f"Calibrated ({self.__class__.__name__})", n_bins=10)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.title('Calibration Curve')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/calibration_curve.png", dpi=300)
        
        plt.close()
        
        return {
            "uncalibrated_brier": uncal_brier,
            "calibrated_brier": cal_brier,
            "improvement": (uncal_brier - cal_brier) / uncal_brier * 100  # % iyileşme
        }


class IsotonicCalibrator(BaseCalibrator):
    """Isotonic Regression ile kalibrasyon yapar"""
    
    def __init__(self):
        super().__init__()
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
    
    def fit(self, y_prob, y_true):
        """
        Isotonic Regression kalibratörünü fit eder
        
        Args:
            y_prob: Kalibre edilmemiş olasılıklar
            y_true: Gerçek değerler
            
        Returns:
            self: Fit edilmiş kalibratör
        """
        # Reshape ederek fit et
        self.calibrator.fit(y_prob.reshape(-1, 1), y_true)
        return self


class PlattCalibrator(BaseCalibrator):
    """Platt Scaling (Logistic Regression) ile kalibrasyon yapar"""
    
    def __init__(self, C=1.0):
        super().__init__()
        # Logistic Regression'ı kalibratör olarak kullan
        # Düşük C değeri daha fazla regularizasyon ve daha düzgün bir eğri sağlar
        self.calibrator = LogisticRegression(C=C, solver='lbfgs')
    
    def fit(self, y_prob, y_true):
        """
        Platt Scaling kalibratörünü fit eder
        
        Args:
            y_prob: Kalibre edilmemiş olasılıklar
            y_true: Gerçek değerler
            
        Returns:
            self: Fit edilmiş kalibratör
        """
        # Reshape ederek fit et
        self.calibrator.fit(y_prob.reshape(-1, 1), y_true)
        return self

def choose_calibrator(model, X_val, y_val, X_test=None, y_test=None, output_dir=None):
    """
    En iyi kalibrasyon yöntemini seçer ve uygular (validation seti üzerinde)
    
    Args:
        model: Eğitilmiş model
        X_val: Validation özellikleri
        y_val: Validation hedef değişkeni
        X_test: Test özellikleri (SADECE final değerlendirme için, seçim için değil)
        y_test: Test hedef değişkeni (SADECE final değerlendirme için, seçim için değil)
        output_dir: Çıktı dizini (None ise görselleştirme yapılmaz)
        
    Returns:
        Kalibre edilmiş model
    """
    # Kalibre edilmemiş olasılıklar
    uncal_probs = None
    if hasattr(model, 'predict_proba'):
        uncal_probs = model.predict_proba(X_val)[:, 1]
    else:
        uncal_probs = model.predict(X_val)
    
    logger.info(f"Kalibrasyon öncesi Brier skoru: {_brier(y_val, uncal_probs):.4f}")
    
    # Kalibrasyon modellerini eğit
    # 1) sigmoid (Platt scaling)
    sig = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
    sig.fit(X_val, y_val)
    sig_probs = sig.predict_proba(X_val)[:, 1]
    brier_sig = _brier(y_val, sig_probs)
    logger.info(f"Sigmoid kalibrasyon Brier skoru: {brier_sig:.4f}")

    # 2) isotonic
    iso = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    iso.fit(X_val, y_val)
    iso_probs = iso.predict_proba(X_val)[:, 1]
    brier_iso = _brier(y_val, iso_probs)
    logger.info(f"Isotonic kalibrasyon Brier skoru: {brier_iso:.4f}")
    
    # 3) Segmentlerin conversion rate'lerini ve dağılımlarını değerlendir
    sig_segments = assign_segments(sig_probs)
    iso_segments = assign_segments(iso_probs)
    
    # Segment bazlı değerlendirme
    sig_lift_df = lift_table(y_val, sig_segments)
    iso_lift_df = lift_table(y_val, iso_segments)
    
    # Sigmoid kalibrasyon için segment conversion rate kontrolü
    sig_segment_order_correct = _check_segment_order(sig_lift_df)
    logger.info(f"Sigmoid kalibrasyon segment sırası doğru mu: {sig_segment_order_correct}")
    
    # Isotonic kalibrasyon için segment conversion rate kontrolü
    iso_segment_order_correct = _check_segment_order(iso_lift_df)
    logger.info(f"Isotonic kalibrasyon segment sırası doğru mu: {iso_segment_order_correct}")
    
    # Segment boşluk kontrolü (her segmentte örnek var mı?)
    sig_all_segments_have_samples = _all_segments_have_samples(sig_lift_df)
    iso_all_segments_have_samples = _all_segments_have_samples(iso_lift_df)
    logger.info(f"Sigmoid kalibrasyon tüm segmentlerde örnek var mı: {sig_all_segments_have_samples}")
    logger.info(f"Isotonic kalibrasyon tüm segmentlerde örnek var mı: {iso_all_segments_have_samples}")
    
    # Kalibrasyon seçimini yaparken birden fazla kriteri dikkate al
    # 1. Brier skoru
    # 2. Segment conversion rate'lerinin artan olması
    # 3. Tüm segmentlerde yeterli örnek olması
    
    # Brier skoru farkı çok küçükse (<%1) segment sırasını ve dağılımını öncelikle dikkate al
    brier_diff = abs(brier_sig - brier_iso) / max(brier_sig, brier_iso)
    
    if brier_diff < 0.01:  # Brier skorları çok yakınsa
        if sig_segment_order_correct and not iso_segment_order_correct:
            best_calibrator = sig
            best_name = "Sigmoid"
            best_probs = sig_probs
            best_brier = brier_sig
            logger.info("Sigmoid seçildi: Segment sırası daha doğru")
        elif iso_segment_order_correct and not sig_segment_order_correct:
            best_calibrator = iso
            best_name = "Isotonic"
            best_probs = iso_probs
            best_brier = brier_iso
            logger.info("Isotonic seçildi: Segment sırası daha doğru")
        elif sig_all_segments_have_samples and not iso_all_segments_have_samples:
            best_calibrator = sig
            best_name = "Sigmoid"
            best_probs = sig_probs
            best_brier = brier_sig
            logger.info("Sigmoid seçildi: Tüm segmentlerde örnek var")
        elif iso_all_segments_have_samples and not sig_all_segments_have_samples:
            best_calibrator = iso
            best_name = "Isotonic"
            best_probs = iso_probs
            best_brier = brier_iso
            logger.info("Isotonic seçildi: Tüm segmentlerde örnek var")
        else:
            # Yukarıdaki kriterlerle karar verilemezse Brier'a dön
            best_calibrator = sig if brier_sig <= brier_iso else iso
            best_name = "Sigmoid" if brier_sig <= brier_iso else "Isotonic"
            best_probs = sig_probs if brier_sig <= brier_iso else iso_probs
            best_brier = min(brier_sig, brier_iso)
            logger.info(f"Brier skoru daha düşük olduğu için {best_name} seçildi")
    else:
        # Brier skorları arasında belirgin fark varsa, öncelikle bunu dikkate al
        best_calibrator = sig if brier_sig <= brier_iso else iso
        best_name = "Sigmoid" if brier_sig <= brier_iso else "Isotonic"
        best_probs = sig_probs if brier_sig <= brier_iso else iso_probs
        best_brier = min(brier_sig, brier_iso)
        logger.info(f"Brier skoru belirgin şekilde daha düşük olduğu için {best_name} seçildi")
    
    logger.info(f"Seçilen kalibratör: {best_name} (Brier: {best_brier:.4f})")
    
    # Kalibrasyon sonuçlarını görselleştir
    if output_dir:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Validation seti üzerinde kalibrasyon değerlendirmesi
        val_metrics, val_lift_df, val_segments = evaluate_segments(
            y_val, best_probs, output_dir / "validation"
        )
        
        # Kalibrasyon karşılaştırma grafiği
        plt.figure(figsize=(12, 8))
        
        # Bin sayısı
        bin_count = min(10, len(y_val) // 50)  # En az bin başına 50 örnek
        
        # Her kalibrasyon yöntemi için kalibrasyon eğrisi
        _plot_calibration_curve(y_val, uncal_probs, "Uncalibrated", bin_count)
        _plot_calibration_curve(y_val, sig_probs, "Sigmoid (Platt)", bin_count)
        _plot_calibration_curve(y_val, iso_probs, "Isotonic", bin_count)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.title('Calibration Curves (Validation Set)')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "calibration_comparison.png", dpi=300)
        plt.close()
        
        # Segment dağılımı
        uncal_segments = assign_segments(uncal_probs)
        cal_segments = val_segments
        
        # Kalibrasyon öncesi/sonrası segment dağılımı karşılaştırması
        segment_comparison = pd.DataFrame({
            'Segment': pd.Series(uncal_segments).value_counts().index,
            'Uncalibrated': pd.Series(uncal_segments).value_counts().values,
            'Calibrated': pd.Series(cal_segments).value_counts().reindex(
                pd.Series(uncal_segments).value_counts().index).values
        })
        
        # Dağılım grafiği
        plt.figure(figsize=(10, 6))
        segment_comparison.set_index('Segment').plot(kind='bar')
        plt.title('Segment Distribution Before/After Calibration (Validation Set)')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "segment_distribution_comparison.png", dpi=300)
        plt.close()
        
        # Binwise calibration değerlendirmesi
        _evaluate_binwise_calibration(y_val, best_probs, output_dir / "binwise_calibration_val")
        
        # Metrikleri kaydet
        cal_metrics = pd.DataFrame({
            'Calibrator': ['Uncalibrated', 'Sigmoid', 'Isotonic', f'Best ({best_name})'],
            'Brier_Score': [_brier(y_val, uncal_probs), brier_sig, brier_iso, best_brier],
            'Segment_Order_Correct': [
                _check_segment_order(lift_table(y_val, uncal_segments)),
                sig_segment_order_correct,
                iso_segment_order_correct,
                _check_segment_order(val_lift_df)
            ],
            'All_Segments_Have_Samples': [
                _all_segments_have_samples(lift_table(y_val, uncal_segments)),
                sig_all_segments_have_samples,
                iso_all_segments_have_samples,
                _all_segments_have_samples(val_lift_df)
            ]
        })
        cal_metrics.to_csv(output_dir / "calibration_metrics.csv", index=False)
        
        # Test seti üzerinde SADECE DEĞERLENDİRME (seçim için değil)
        if X_test is not None and y_test is not None:
            logger.info("Test seti üzerinde SADECE değerlendirme yapılıyor (kalibratör seçimi için değil)")
            # Test seti üzerinde tahminler
            test_uncal_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
            test_cal_probs = best_calibrator.predict_proba(X_test)[:, 1]
            
            # Test seti üzerinde değerlendirme
            test_metrics, test_lift_df, test_segments = evaluate_segments(
                y_test, test_cal_probs, output_dir / "test"
            )
            
            # Test kalibrasyon eğrisi
            plt.figure(figsize=(12, 8))
            _plot_calibration_curve(y_test, test_uncal_probs, "Uncalibrated", bin_count)
            _plot_calibration_curve(y_test, test_cal_probs, f"Calibrated ({best_name})", bin_count)
            plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
            plt.title('Calibration Curves (Test Set)')
            plt.xlabel('Mean predicted probability')
            plt.ylabel('Fraction of positives')
            plt.legend(loc='best')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "test_calibration_curve.png", dpi=300)
            plt.close()
            
            # Test segment dağılımı
            test_uncal_segments = assign_segments(test_uncal_probs)
            
            # Kalibrasyon öncesi/sonrası segment dağılımı karşılaştırması (Test)
            test_segment_comparison = pd.DataFrame({
                'Segment': pd.Series(test_uncal_segments).value_counts().index,
                'Uncalibrated': pd.Series(test_uncal_segments).value_counts().values,
                'Calibrated': pd.Series(test_segments).value_counts().reindex(
                    pd.Series(test_uncal_segments).value_counts().index).values
            })
            
            # Dağılım grafiği (Test)
            plt.figure(figsize=(10, 6))
            test_segment_comparison.set_index('Segment').plot(kind='bar')
            plt.title('Segment Distribution Before/After Calibration (Test Set)')
            plt.ylabel('Count')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "test_segment_distribution.png", dpi=300)
            plt.close()
            
            # Binwise calibration değerlendirmesi (Test)
            _evaluate_binwise_calibration(y_test, test_cal_probs, output_dir / "binwise_calibration_test")
            
            # Test metrikleri kaydet
            test_cal_metrics = pd.DataFrame({
                'Metric': ['Brier_Score', 'ROC_AUC', 'PR_AUC', 'Segment_Order_Correct', 'All_Segments_Have_Samples'],
                'Uncalibrated': [
                    _brier(y_test, test_uncal_probs),
                    val_metrics['roc_auc'],
                    val_metrics['pr_auc'],
                    _check_segment_order(lift_table(y_test, test_uncal_segments)),
                    _all_segments_have_samples(lift_table(y_test, test_uncal_segments))
                ],
                'Calibrated': [
                    _brier(y_test, test_cal_probs),
                    test_metrics['roc_auc'],
                    test_metrics['pr_auc'],
                    _check_segment_order(test_lift_df),
                    _all_segments_have_samples(test_lift_df)
                ]
            })
            test_cal_metrics.to_csv(output_dir / "test_calibration_metrics.csv", index=False)
            
            # Validation vs Test karşılaştırması
            comparison_metrics = pd.DataFrame({
                'Metric': ['Brier_Score', 'ROC_AUC', 'PR_AUC', 'Segment_Order_Correct', 'All_Segments_Have_Samples'],
                'Validation': [
                    best_brier,
                    val_metrics['roc_auc'],
                    val_metrics['pr_auc'],
                    _check_segment_order(val_lift_df),
                    _all_segments_have_samples(val_lift_df)
                ],
                'Test': [
                    _brier(y_test, test_cal_probs),
                    test_metrics['roc_auc'],
                    test_metrics['pr_auc'],
                    _check_segment_order(test_lift_df),
                    _all_segments_have_samples(test_lift_df)
                ]
            })
            comparison_metrics.to_csv(output_dir / "validation_vs_test_metrics.csv", index=False)
            
            logger.info(f"Test seti üzerinde Brier skoru: {_brier(y_test, test_cal_probs):.4f}")
            logger.info(f"Test seti segment sırası doğru mu: {_check_segment_order(test_lift_df)}")
    
    return best_calibrator

def _check_segment_order(lift_df):
    """
    Segment conversion rate'lerinin High > Medium > Low şeklinde artan olup olmadığını kontrol eder
    
    Args:
        lift_df: lift_table() fonksiyonundan dönen DataFrame
        
    Returns:
        bool: Segment sırası doğru mu
    """
    # Standart segment isimleri ["Low", "Medium", "High"]
    expected_order = {"Low": 0, "Medium": 1, "High": 2}
    
    # Tüm beklenen segmentlerin olup olmadığını kontrol et
    if not all(seg in lift_df['seg'].values for seg in expected_order.keys()):
        return False
    
    # Conversion rate'lere göre sırala
    sorted_df = lift_df.sort_values('conv_rate')
    
    # Sıranın doğru olup olmadığını kontrol et
    for i, row in enumerate(sorted_df.iterrows()):
        seg = row[1]['seg']
        if expected_order[seg] != i:
            return False
    
    return True

def _all_segments_have_samples(lift_df, min_samples=10):
    """
    Her segmentte (Low/Medium/High) en az min_samples kadar örnek olup olmadığını kontrol eder.
    
    Args:
        lift_df: Lift tablosu
        min_samples: Bir segmentteki minimum örnek sayısı
        
    Returns:
        bool: Tüm segmentlerde en az min_samples kadar örnek varsa True
    """
    if lift_df.empty:
        return False
    
    # Toplam örnek sayısı 
    total_samples = lift_df['count'].sum()
    
    # Dinamik min_samples hesaplama - validation seti küçükse eşiği düşür
    # Çok küçük veri setlerinde uyum sağlaması için
    if total_samples < 100:
        # Çok küçük veri setleri için minimum 2 örnek olmasını sağla
        dynamic_min_samples = max(2, int(total_samples * 0.05))
    elif total_samples < 1000:
        # Küçük-orta veri setleri için
        dynamic_min_samples = max(5, int(total_samples * 0.02))
    else:
        # Büyük veri setleri için verilen değeri kullan
        dynamic_min_samples = min_samples
    
    # En az beklenen örnek sayısını loglama
    logging.debug(f"Toplam {total_samples} örnek için dinamik min_samples: {dynamic_min_samples}")
    
    # Her segmentte yeterli örnek var mı kontrol et
    for segment in ['Low', 'Medium', 'High']:
        if segment not in lift_df['seg'].values:
            logging.warning(f"{segment} segmenti lift_df'te yok!")
            return False
        
        segment_count = lift_df[lift_df['seg'] == segment]['count'].values[0]
        if segment_count < dynamic_min_samples:
            logging.warning(f"{segment} segmentinde yetersiz örnek: {segment_count} < {dynamic_min_samples}")
            return False
    
    return True

def _evaluate_binwise_calibration(y_true, y_prob, output_dir):
    """
    Binwise kalibrasyon değerlendirmesi yapar
    
    Args:
        y_true: Gerçek değerler
        y_prob: Tahmin olasılıkları
        output_dir: Çıktı dizini
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Bin sayısı ve aralıkları
    n_bins = 10
    bins = np.linspace(0, 1, n_bins+1)
    
    # Olasılık değerlerini bin'lere ayır
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    # Her bin için gerçek oranları ve ortalama tahminleri hesapla
    binwise_metrics = []
    
    for i in range(len(bins)-1):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_y_true = y_true[mask]
            bin_y_prob = y_prob[mask]
            
            true_rate = np.mean(bin_y_true)
            avg_pred = np.mean(bin_y_prob)
            bin_min = bins[i]
            bin_max = bins[i+1]
            bin_count = np.sum(mask)
            bin_error = abs(true_rate - avg_pred)
            
            binwise_metrics.append({
                'bin_min': bin_min,
                'bin_max': bin_max,
                'count': bin_count,
                'true_rate': true_rate,
                'avg_pred': avg_pred,
                'error': bin_error
            })
    
    # DataFrame oluştur
    binwise_df = pd.DataFrame(binwise_metrics)
    
    # Kaydet
    binwise_df.to_csv(output_dir / "binwise_metrics.csv", index=False)
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    
    # Bin bazlı kalibrasyon grafiği
    plt.scatter(binwise_df['avg_pred'], binwise_df['true_rate'], 
               s=binwise_df['count']/binwise_df['count'].sum()*1000, alpha=0.6)
    
    # Mükemmel kalibrasyon çizgisi
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    
    # Etiketler
    for i, row in binwise_df.iterrows():
        plt.text(row['avg_pred'], row['true_rate'], 
                f"{row['bin_min']:.1f}-{row['bin_max']:.1f}\n(n={row['count']})", 
                ha='center', va='center', fontsize=8)
    
    plt.title('Binwise Calibration Evaluation')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability (fraction of positives)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "binwise_calibration.png", dpi=300)
    plt.close()
    
    return binwise_df

def _plot_calibration_curve(y_true, y_prob, label, n_bins=10):
    """
    Kalibrasyon eğrisini çizer
    
    Args:
        y_true: Gerçek değerler
        y_prob: Tahmin olasılıkları
        label: Eğri etiketi
        n_bins: Bin sayısı
    """
    # Bin'lere ayır
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    # Her bin için gerçek oranları hesapla
    bin_sums = np.bincount(bin_indices, minlength=len(bins) - 1)
    bin_true = np.bincount(bin_indices, weights=y_true, minlength=len(bins) - 1)
    bin_pred = np.bincount(bin_indices, weights=y_prob, minlength=len(bins) - 1)
    
    nonzero = bin_sums > 0
    bin_true[nonzero] /= bin_sums[nonzero]
    bin_pred[nonzero] /= bin_sums[nonzero]
    
    # Orta noktaları hesapla
    bin_middle = (bins[:-1] + bins[1:]) / 2
    
    # Eğriyi çiz
    plt.plot(bin_middle[nonzero], bin_true[nonzero], '-o', label=label)

def _brier(y, p): 
    """
    Brier skoru hesaplar (düşük değer daha iyi)
    
    Args:
        y: Gerçek değerler (0/1)
        p: Tahmin olasılıkları
        
    Returns:
        float: Brier skoru
    """
    return np.mean((y - p) ** 2)
