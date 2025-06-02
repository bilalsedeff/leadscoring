"""
Lead Scoring kalibrasyon ve segmentasyon modüllerini test etmek için birim testleri.
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Proje kök dizinini sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Test edilen modülleri import et
from src.calibration.calibrators import IsotonicCalibrator, PlattCalibrator
from src.calibration.binner import ProbabilityBinner
from src.calibration.binwise import calculate_binwise_metrics

class TestCalibration(unittest.TestCase):
    """Kalibrasyon modüllerini test eder."""
    
    def setUp(self):
        """Test verisi oluştur."""
        # Test için sentetik sınıflandırma verisi oluştur
        X, y = make_classification(
            n_samples=1000, 
            n_features=10, 
            n_informative=5, 
            n_redundant=2, 
            random_state=42
        )
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Model eğit - kasıtlı olarak calibrasyon gerektiren bir model
        self.model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Olasılık tahminleri
        self.y_proba_train = self.model.predict_proba(self.X_train)[:, 1]
        self.y_proba_test = self.model.predict_proba(self.X_test)[:, 1]
    
    def test_isotonic_calibration(self):
        """Isotonic kalibrasyon test edilir."""
        # IsotonicCalibrator sınıfını test et
        calibrator = IsotonicCalibrator()
        
        # Eğit
        calibrator.fit(self.y_proba_train, self.y_train)
        
        # Kalibre et
        calibrated_probs = calibrator.calibrate(self.y_proba_test)
        
        # Kalibre edilmiş olasılıklar [0, 1] aralığında olmalı
        self.assertTrue(np.all(calibrated_probs >= 0))
        self.assertTrue(np.all(calibrated_probs <= 1))
        
        # Ortalama kalibre edilmiş olasılık, gerçek pozitif oranına yakın olmalı
        # (Mükemmel kalibrasyon için)
        mean_proba = np.mean(calibrated_probs)
        actual_rate = np.mean(self.y_test)
        
        # Toleransla kontrol et
        self.assertAlmostEqual(mean_proba, actual_rate, delta=0.1, 
                               msg="Kalibre edilmiş olasılıkların ortalaması gerçek pozitif oranına yakın olmalı")
    
    def test_platt_calibration(self):
        """Platt kalibrasyon test edilir."""
        # PlattCalibrator sınıfını test et
        calibrator = PlattCalibrator()
        
        # Eğit
        calibrator.fit(self.y_proba_train, self.y_train)
        
        # Kalibre et
        calibrated_probs = calibrator.calibrate(self.y_proba_test)
        
        # Kalibre edilmiş olasılıklar [0, 1] aralığında olmalı
        self.assertTrue(np.all(calibrated_probs >= 0))
        self.assertTrue(np.all(calibrated_probs <= 1))
    
    def test_probability_binner(self):
        """ProbabilityBinner sınıfı test edilir."""
        # Özel edges ve labels ile binner oluştur
        edges = [0.0, 0.25, 0.75, 1.0]
        labels = ['Low', 'Medium', 'High']
        binner = ProbabilityBinner(edges=edges, labels=labels)
        
        # Farklı olasılık değerleri için test et
        test_probs = np.array([0.1, 0.3, 0.6, 0.8, 0.95])
        expected_bins = np.array(['Low', 'Medium', 'Medium', 'High', 'High'])
        
        # Bin ata
        assigned_bins = binner.assign_bins(test_probs)
        
        # Beklenen sonuçlarla karşılaştır
        np.testing.assert_array_equal(assigned_bins, expected_bins)
        
        # edge durumlarını test et
        edge_probs = np.array([0.0, 0.25, 0.75, 1.0])
        expected_edge_bins = np.array(['Low', 'Medium', 'High', 'High'])
        
        edge_bins = binner.assign_bins(edge_probs)
        np.testing.assert_array_equal(edge_bins, expected_edge_bins)
    
    def test_binwise_metrics(self):
        """Bin bazlı metrikler test edilir."""
        # Olasılık tahminleri - kasıtlı olarak düşük/orta/yüksek potansiyel gruplarını oluştur
        n = 300
        low_probs = np.random.uniform(0, 0.25, n)
        med_probs = np.random.uniform(0.25, 0.75, n)
        high_probs = np.random.uniform(0.75, 1.0, n)
        
        # Tahmin olasılıklarına göre gerçek etiketler oluştur
        # Düşük potansiyel: %10, Orta: %40, Yüksek: %80 dönüşüm oranı
        np.random.seed(42)
        low_actual = np.random.binomial(1, 0.1, n)  # %10 dönüşüm
        med_actual = np.random.binomial(1, 0.4, n)  # %40 dönüşüm
        high_actual = np.random.binomial(1, 0.8, n) # %80 dönüşüm
        
        # Birleştir
        y_proba = np.concatenate([low_probs, med_probs, high_probs])
        y_true = np.concatenate([low_actual, med_actual, high_actual])
        
        # Binner oluştur
        binner = ProbabilityBinner()
        
        # Bin bazlı metrikler hesapla
        metrics = calculate_binwise_metrics(y_true, y_proba, binner)
        
        # Sonuçları kontrol et
        self.assertIn('bins', metrics)
        self.assertIn('bin_counts', metrics)
        self.assertIn('conversion_rates', metrics)
        self.assertIn('avg_proba', metrics)
        
        # Beklenen segment sırası - Low, Medium, High olmalı
        expected_bins = np.array(['Low', 'Medium', 'High'])
        np.testing.assert_array_equal(metrics['bins'], expected_bins)
        
        # conversion_rates listesinde segmentlerin sırası ['Low', 'Medium', 'High'] şeklinde olmalı
        conversion_rates = metrics['conversion_rates']
        
        # Dönüşüm oranları artan bir hiyerarşide olmalı
        self.assertLess(conversion_rates[0], conversion_rates[1], 
                       "Düşük potansiyel dönüşüm oranı orta potansiyelden düşük olmalı")
        self.assertLess(conversion_rates[1], conversion_rates[2], 
                       "Orta potansiyel dönüşüm oranı yüksek potansiyelden düşük olmalı")
        
        # Dönüşüm oranları beklenen değerlere yakın olmalı
        self.assertAlmostEqual(conversion_rates[0], 0.1, delta=0.1, 
                              msg="Düşük potansiyel dönüşüm oranı ~%10 olmalı")
        self.assertAlmostEqual(conversion_rates[1], 0.4, delta=0.1, 
                              msg="Orta potansiyel dönüşüm oranı ~%40 olmalı")
        self.assertAlmostEqual(conversion_rates[2], 0.8, delta=0.1, 
                              msg="Yüksek potansiyel dönüşüm oranı ~%80 olmalı")
    
    def test_yaml_config_edges(self):
        """Binner'ın configs/split.yaml'dan eşik değerlerini doğru okuduğunu test eder."""
        # Geçici dizin oluştur
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Configs dizini oluştur
            os.makedirs(os.path.join(temp_dir, "configs"), exist_ok=True)
            
            # split.yaml dosyası oluştur
            config_content = """
            # Test split konfigurasyonu
            bin_edges: [0.0, 0.3, 0.6, 1.0]
            bin_labels: ['Very Low', 'Low', 'High']
            """
            
            with open(os.path.join(temp_dir, "configs", "split.yaml"), "w") as f:
                f.write(config_content)
            
            # Orijinal çalışma dizinini sakla
            original_dir = os.getcwd()
            
            try:
                # Çalışma dizinini geçici dizine değiştir
                os.chdir(temp_dir)
                
                # Binner oluştur - otomatik olarak split.yaml'dan değerleri okumalı
                binner = ProbabilityBinner()
                
                # Değerleri kontrol et
                np.testing.assert_array_equal(binner.edges, [0.0, 0.3, 0.6, 1.0], 
                                            "Binner split.yaml'dan eşik değerlerini doğru okumamış")
                np.testing.assert_array_equal(binner.labels, ['Very Low', 'Low', 'High'], 
                                            "Binner split.yaml'dan etiketleri doğru okumamış")
            finally:
                # Orijinal çalışma dizinine geri dön
                os.chdir(original_dir)
        finally:
            # Geçici dizini temizle
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    unittest.main() 