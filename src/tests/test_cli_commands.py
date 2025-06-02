"""
Lead Scoring CLI komutlarını test etmek için birim testleri.
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from click.testing import CliRunner

# Proje kök dizinini sys.path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Test edilen CLI komutlarını import et
from lead_scoring import cli, run, stats, feature_imp, train, predict, dashboard, mlflow

class TestCLICommands(unittest.TestCase):
    """Lead Scoring CLI komutlarını test eder."""
    
    @classmethod
    def setUpClass(cls):
        """Test ortamını hazırla."""
        # Test için geçici dizin
        cls.temp_dir = tempfile.mkdtemp()
        
        # Dizin yapısını oluştur
        os.makedirs(os.path.join(cls.temp_dir, "configs"), exist_ok=True)
        os.makedirs(os.path.join(cls.temp_dir, "outputs"), exist_ok=True)
        os.makedirs(os.path.join(cls.temp_dir, "data", "raw"), exist_ok=True)
        
        # Test verisi oluştur
        np.random.seed(42)
        n = 200
        
        # Lead ve hesap verileri
        data = []
        for i in range(n):
            yearmonth = 12023 if i < 50 else (62023 if i < 100 else (102023 if i < 150 else 12024))
            source = np.random.choice(['Web', 'App', 'Ecommerce', 'Social Media'])
            account_id = f"A{i % 50:03d}"  # 50 unique account
            target = np.random.binomial(1, 0.3 if source == 'Web' else 0.6)
            
            data.append({
                'LeadId': f"L{i:05d}",
                'account_Id': account_id,
                'YearMonth': yearmonth,
                'Source_Final__c': source,
                'Target_IsConverted': target,
                'NumericFeature1': np.random.normal(5, 2),
                'NumericFeature2': np.random.normal(10, 3),
                'CategoricalFeature': np.random.choice(['A', 'B', 'C']),
                'LeadCreatedDate': pd.Timestamp(f"20{str(yearmonth)[-2:]}-{str(yearmonth)[:-4].zfill(2)}-15")
            })
        
        cls.df = pd.DataFrame(data)
        
        # Test verisini kaydet
        cls.df.to_csv(os.path.join(cls.temp_dir, "data", "raw", "Conversion_Datamart.csv"), index=False)
        
        # Temel yapılandırma dosyalarını oluştur
        # split.yaml
        split_config = """
        # Veri bolme yapilandirmasi
        split_method: time
        time_col: YearMonth
        train_cutoff: 72023
        val_cutoff: 122023
        group_col: account_Id
        target_col: Target_IsConverted
        cv_folds: 3
        bin_edges: [0.0, 0.25, 0.75, 1.0]
        bin_labels: ['Low', 'Medium', 'High']
        """
        
        with open(os.path.join(cls.temp_dir, "configs", "split.yaml"), "w", encoding="utf-8") as f:
            f.write(split_config)
        
        # experiment.yaml
        experiment_config = """
        # Experiment yapilandirmasi
        experiment_name: test_experiment
        model_family: "ensemble"
        use_optuna: false
        n_trials: 10
        timeout: 600
        metric: "roc_auc"
        direction: "maximize"
        cv_folds: 3
        random_state: 42
        
        # Numeric and categorical features
        num_cols: []
        cat_cols: []
        """
        
        with open(os.path.join(cls.temp_dir, "configs", "experiment.yaml"), "w", encoding="utf-8") as f:
            f.write(experiment_config)
        
        # model.yaml (basitleştirilmiş)
        model_config = """
        # Model yapilandirmasi
        lightgbm:
          name: "LightGBM"
          num_leaves: [31]
          max_depth: [5]
          learning_rate: [0.1]
          n_estimators: [100]
          
        voting:
          name: "Voting Ensemble"
          base_models: ["lightgbm", "random_forest"]
          voting_type: "soft"
          optimize_weights: true
        """
        
        with open(os.path.join(cls.temp_dir, "configs", "model.yaml"), "w", encoding="utf-8") as f:
            f.write(model_config)
        
        # cleaning.yaml
        cleaning_config = """
        # Veri temizleme yapilandirmasi
        num_fill: "median"
        cat_fill: "missing"
        outlier_method: "zscore"
        outlier_threshold: 3.0
        
        # Özellik seçimi
        feature_selection:
          missing_thresh: 0.3
          duplicate: true
          near_zero_var_thresh: 0.01
        """
        
        with open(os.path.join(cls.temp_dir, "configs", "cleaning.yaml"), "w", encoding="utf-8") as f:
            f.write(cleaning_config)
        
        # data.yaml (basitleştirilmiş)
        data_config = """
        # Veri yapilandirmasi
        paths:
          raw_csv: data/raw/Conversion_Datamart.csv
          
        dtypes:
          LeadId: str
          account_Id: str
          Source_Final__c: category
          Target_IsConverted: int8
          
        datetime_cols:
          - LeadCreatedDate
        """
        
        with open(os.path.join(cls.temp_dir, "configs", "data.yaml"), "w", encoding="utf-8") as f:
            f.write(data_config)
        
        # Orijinal çalışma dizinini sakla
        cls.original_dir = os.getcwd()
        
        # Çalışma dizinini geçici dizine değiştir
        os.chdir(cls.temp_dir)
    
    @classmethod
    def tearDownClass(cls):
        """Test ortamını temizle."""
        # Orijinal çalışma dizinine geri dön
        os.chdir(cls.original_dir)
        
        # Geçici dizini sil
        shutil.rmtree(cls.temp_dir)
    
    def test_run_command(self):
        """run komutu test edilir."""
        runner = CliRunner()
        result = runner.invoke(run, [
            '--split_method=time',
            '--train_cutoff=72023',
            '--val_cutoff=122023',
            '--create_new_run'
        ])
        
        # Komut başarılı olmalı
        self.assertEqual(result.exit_code, 0, f"Run komutu başarısız: {result.output}")
        
        # Split dizini oluşturulmuş olmalı
        split_dirs = []
        for root, dirs, files in os.walk("outputs"):
            for d in dirs:
                if d == "split":
                    split_dirs.append(os.path.join(root, d))
        
        self.assertGreater(len(split_dirs), 0, "Split dizini oluşturulmamış")
        
        # Train, validation ve test dosyaları oluşturulmuş olmalı
        for split_dir in split_dirs:
            self.assertTrue(os.path.exists(os.path.join(split_dir, "train.csv")),
                           f"Train dosyası oluşturulmamış: {split_dir}")
            self.assertTrue(os.path.exists(os.path.join(split_dir, "validation.csv")),
                           f"Validation dosyası oluşturulmamış: {split_dir}")
    
    def test_stats_command(self):
        """stats komutu test edilir."""
        # Önce run komutunu çalıştırarak split dosyalarını oluştur
        runner = CliRunner()
        runner.invoke(run, [
            '--split_method=time',
            '--train_cutoff=72023',
            '--val_cutoff=122023',
            '--create_new_run'
        ])
        
        # Sonra stats komutunu çalıştır
        result = runner.invoke(stats, ['--test_type=all'])
        
        # Komut başarılı olmalı
        self.assertEqual(result.exit_code, 0, f"Stats komutu başarısız: {result.output}")
        
        # İstatistiksel test dizini oluşturulmuş olmalı
        stat_dirs = []
        for root, dirs, files in os.walk("outputs"):
            for d in dirs:
                if "statistical_tests" in d:
                    stat_dirs.append(os.path.join(root, d))
        
        self.assertGreater(len(stat_dirs), 0, "İstatistiksel test dizini oluşturulmamış")
        
        # En azından chi-square test sonuçları bulunmalı
        chi_square_files = []
        for stat_dir in stat_dirs:
            for root, dirs, files in os.walk(stat_dir):
                chi_square_files.extend([f for f in files if "chi_square" in f])
        
        self.assertGreater(len(chi_square_files), 0, "Chi-square test sonuçları oluşturulmamış")
    
    def test_feature_imp_command(self):
        """feature-imp komutu test edilir."""
        # Önce run komutunu çalıştırarak split dosyalarını oluştur
        runner = CliRunner()
        runner.invoke(run, [
            '--split_method=time',
            '--train_cutoff=72023',
            '--val_cutoff=122023',
            '--create_new_run'
        ])
        
        # Sonra feature-imp komutunu çalıştır
        result = runner.invoke(feature_imp, ['--method=shap'])
        
        # Komut başarılı olmalı (shap modülü kuruluysa)
        try:
            import shap
            self.assertEqual(result.exit_code, 0, f"Feature-imp komutu başarısız: {result.output}")
            
            # Feature importance dizini oluşturulmuş olmalı
            found = False
            for root, dirs, files in os.walk("outputs"):
                for d in dirs:
                    if "feature_importance" in d:
                        found = True
                        break
                if found:
                    break
            
            self.assertTrue(found, "Feature importance dizini oluşturulmamış")
        except ImportError:
            self.skipTest("shap modülü kurulu değil, test atlanıyor")
    
    def test_train_command_with_yaml_config(self):
        """Train komutunun YAML yapılandırma dosyalarından değerleri okuduğunu test eder."""
        # Önce run komutunu çalıştırarak split dosyalarını oluştur
        runner = CliRunner()
        runner.invoke(run, [
            '--split_method=time',
            '--train_cutoff=72023',
            '--val_cutoff=122023',
            '--create_new_run'
        ])
        
        # configs/experiment.yaml dosyasına bakalım ve değeri kontrol edelim
        with open("configs/experiment.yaml", "r") as f:
            experiment_config = f.read()
        
        # model_family değerini manuel olarak değiştirelim
        experiment_config = experiment_config.replace('model_family: "ensemble"', 'model_family: "tree"')
        
        with open("configs/experiment.yaml", "w") as f:
            f.write(experiment_config)
        
        # Ardından train komutunu çalıştıralım
        # --model_type=2 (LightGBM) kullanıyoruz, bu tree-based model kategorisinde
        result = runner.invoke(train, ['--model_type=2'])
        
        # model_family "tree" olarak değiştirildiği için komut başarılı olmalı
        self.assertEqual(result.exit_code, 0, f"Train komutu başarısız: {result.output}")
        
        # Çıktıda "tree" kelimesi geçmeli
        self.assertIn("tree", result.output.lower(), "Güncellenmiş model_family değeri kullanılmamış")
        
        # Eğitilen model dosyaları oluşturulmuş olmalı
        models_dirs = []
        for root, dirs, files in os.walk("outputs"):
            for d in dirs:
                if d == "models":
                    models_dirs.append(os.path.join(root, d))
        
        self.assertGreater(len(models_dirs), 0, "Model dosyaları oluşturulmamış")

if __name__ == '__main__':
    unittest.main() 