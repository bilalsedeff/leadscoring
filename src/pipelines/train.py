"""
Model eğitimi ve değerlendirme pipeline'ı.

Usage:
    python -m src.pipelines.train [options]

Options:
    --split-dir=<dir>    Split dizini [default: outputs/latest/split]
    --output-dir=<dir>   Çıktı dizini [default: None]
    --source-specific   Kaynak bazlı model eğitimi yapılsın mı? [default: False]
    --source-col=<col>  Kaynak kolonu [default: Source_Final__c]
"""

import hydra, mlflow, numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from typing import Dict, Any
from src.models.baseline import ElasticNetLR
from src.models.tree_based import LightGBMModel, XGBModel, RandomForestModel, CatBoostModel
from src.models.ensemble import build_voting_ensemble
from src.features.engineering import AdvancedFeatureEngineer
from src.features.interactions import add_pairwise_ratios
from src.preprocessing.cleaning import BasicCleaner
from src.preprocessing.encoders import build_preprocessor
from src.evaluation.metrics import calculate_classification_metrics, plot_roc_curve, plot_precision_recall_curve
from src.imbalance.samplers import balanced_resample
from src.utils.logger import get_logger
from src.calibration.binner import ProbabilityBinner
from src.calibration.calibrators import choose_calibrator
import matplotlib.pyplot as plt
import pandas as pd
import shap
import joblib
import os
import glob
import datetime

log = get_logger()

def train_model(
    cfg: OmegaConf,
    split_dir: str = None,
    output_dir: str = None,
    source_specific: bool = False,
    source_col: str = "Source_Final__c",
    model_type: int = 2,  # 1: Baseline, 2: LightGBM, 3: XGBoost, 4: Ensemble
    include_calibration: bool = True,
    experiment_name: str = None,
    random_state: int = 42,
    target_col: str = "Target_IsConverted",
):
    """
    End-to-end model eğitim ve değerlendirme pipeline'ı.
    
    Args:
        cfg: Hydra konfigürasyonu
        split_dir: Veri setlerinin bulunduğu dizin
        output_dir: Çıktı dizini
        source_specific: Kaynak bazlı model eğitimi yapılsın mı?
        source_col: Kaynak kolonu
        model_type: Model tipi (1: Baseline, 2: LightGBM, 3: XGBoost, 4: Ensemble)
        include_calibration: Kalibrasyon yapılsın mı?
        experiment_name: MLflow experiment adı
        random_state: Random state
        target_col: Hedef değişken kolonu
    """
    if not experiment_name:
        experiment_name = f"model_{model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    log.info(f"Model eğitimi başlatılıyor... Model Tipi: {model_type}, Experiment: {experiment_name}")
    
    # Output directory setup
    if output_dir is None:
        from src.utils.paths import get_experiment_dir
        output_dir = get_experiment_dir()
    
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Locate data
    if split_dir is None:
        # try to find latest outputs/run_*/split directory
        from glob import glob
        split_dirs = sorted(glob("outputs/run_*/split"), reverse=True)
        if split_dirs:
            split_dir = split_dirs[0]
            log.info(f"En son split dizini bulundu: {split_dir}")
        else:
            # check for outputs/split
            if os.path.exists("outputs/split"):
                split_dir = "outputs/split"
                log.info(f"Split dizini kullanılıyor: {split_dir}")
            else:
                raise ValueError("Split directory not found. Run split first.")
    
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    
    # Check for training data
    train_path = split_dir / "train.csv"
    val_path = split_dir / "validation.csv"
    test_path = split_dir / "test.csv"
    
    if not train_path.exists():
        raise ValueError(f"Training data not found: {train_path}")
    
    log.info(f"Eğitim verisi: {train_path}")
    log.info(f"Doğrulama verisi: {val_path if val_path.exists() else 'Bulunamadı'}")
    log.info(f"Test verisi: {test_path if test_path.exists() else 'Bulunamadı'}")
    
    # Load data
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path) if val_path.exists() else None
    df_test = pd.read_csv(test_path) if test_path.exists() else None
    
    has_val_set = df_val is not None and len(df_val) > 0
    has_test_set = df_test is not None and len(df_test) > 0
    
    # Drop LeadId or other ID columns (keep account_Id for feature engineering)
    id_cols_to_drop = ["LeadId"]
    for df in [df_train, df_val, df_test]:
        if df is not None:
            for col in id_cols_to_drop:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
    
    log.info(f"Veri seti boyutları - Train: {df_train.shape}, "
             f"Validation: {df_val.shape if has_val_set else 'Yok'}, "
             f"Test: {df_test.shape if has_test_set else 'Yok'}")
    
    # 1. Veri temizleme (train setinde fit, tüm setlere transform)
    log.info("Veri temizleme başlatılıyor...")
    cleaner = BasicCleaner()
    cleaner.fit(df_train)  # Sadece train verisi ile fit et
    
    df_train_clean = cleaner.transform(df_train)
    df_val_clean = cleaner.transform(df_val) if has_val_set else None
    df_test_clean = cleaner.transform(df_test) if has_test_set else None
    
    # 2. Zamansal özellikler ekle
    log.info("Zamansal özellikler ekleniyor...")
    # Configs'ten pencere günlerini al (varsayılan değerler kullan, yaml'da yoksa)
    windows_days = cfg.get('feature_engineering', {}).get('windows_days', (30, 90, 180))
    
    feature_engineer = AdvancedFeatureEngineer(
        date_col="LeadCreatedDate",
        account_col="account_Id",
        source_col=source_col,
        yearmonth_col="YearMonth",
        windows_days=windows_days
    )
    feature_engineer.fit(df_train_clean)  # Sadece train verisi ile fit et
    
    df_train_temp = feature_engineer.transform(df_train_clean)
    df_val_temp = feature_engineer.transform(df_val_clean) if has_val_set else None
    df_test_temp = feature_engineer.transform(df_test_clean) if has_test_set else None
    
    # İşlem sonrası veri seti kontrolü
    log.info(f"Zamansal özellikler sonrası - Train: {df_train_temp.shape}, "
             f"Validation: {df_val_temp.shape if has_val_set else 'Yok'}, "
             f"Test: {df_test_temp.shape if has_test_set else 'Yok'}")
    
    # 3. Sayısal kolonları belirle (target ve ID kolonlarını çıkar)
    # Sayısal kolonları belirle (veri tipine göre)
    numerical_columns = df_train_temp.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Target ve ID kolonlarını çıkar
    numerical_columns = [col for col in numerical_columns if col not in ['Target_IsConverted', 'LeadId', 'account_Id']]
    
    # 4. Özellik etkileşimleri ekle
    log.info("Özellik etkileşimleri ekleniyor...")
    
    df_train_inter = add_pairwise_ratios(df_train_temp, numerical_columns)
    df_val_inter = add_pairwise_ratios(df_val_temp, numerical_columns) if has_val_set else None
    df_test_inter = add_pairwise_ratios(df_test_temp, numerical_columns) if has_test_set else None
    
    # 5. Özellik seçimi ve ön işleme
    log.info("Özellik seçimi ve ön işleme yapılıyor...")
    
    # Target kolonunu ayır
    y_train = df_train_inter[target_col]
    X_train = df_train_inter.drop(columns=[target_col])
    
    if has_val_set:
        y_val = df_val_inter[target_col]
        X_val = df_val_inter.drop(columns=[target_col])
    
    if has_test_set:
        y_test = df_test_inter[target_col]
        X_test = df_test_inter.drop(columns=[target_col])
    
    # account_Id kolonunu düşür (eğer hala varsa)
    id_cols = ['account_Id']
    for df in [X_train, X_val if has_val_set else None, X_test if has_test_set else None]:
        if df is not None:
            for col in id_cols:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
    
    # Kategorik kolonları belirle
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Preprocessing pipeline oluştur (kategorik kolonlar için OneHotEncoder, sayısal kolonlar için StandardScaler)
    preprocessor = build_preprocessor(X_train, categorical_columns, numerical_columns)
    
    # Preprocessor'ı sadece train verisi üzerinde fit et
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Validation ve test verilerine transform uygula
    if has_val_set:
        X_val_processed = preprocessor.transform(X_val)
    
    if has_test_set:
        X_test_processed = preprocessor.transform(X_test)
    
    # Özellik isimlerini al
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # Eski sklearn sürümleri için
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                names = transformer.get_feature_names_out(cols)
            elif hasattr(transformer, 'get_feature_names'):
                names = transformer.get_feature_names(cols)
            else:
                names = cols
            feature_names.extend(names)
    
    log.info(f"İşlenmiş özellik sayısı: {len(feature_names)}")
    
    # 6. Model seçimi ve eğitimi - Konfigürasyondan parametreleri al
    if model_type == 1:
        # Baseline: Elastic Net Logistic Regression
        log.info("Elastic Net Logistic Regression modeli eğitiliyor...")
        # Model parametrelerini configs/model.yaml'dan al
        baseline_params = cfg.get('model', {}).get('baseline_lr', {})
        model = ElasticNetLR(
            C=baseline_params.get('C', 1.0),
            l1_ratio=baseline_params.get('l1_ratio', 0.5),
            max_iter=baseline_params.get('max_iter', 1000),
            random_state=random_state
        )
    elif model_type == 2:
        # LightGBM
        log.info("LightGBM modeli eğitiliyor...")
        # Model parametrelerini configs/model.yaml'dan al
        lgbm_params = cfg.get('model', {}).get('lightgbm', {})
        model = LightGBMModel(
            num_leaves=lgbm_params.get('num_leaves', 31),
            max_depth=lgbm_params.get('max_depth', 5),
            learning_rate=lgbm_params.get('learning_rate', 0.05),
            n_estimators=lgbm_params.get('n_estimators', 300),
            min_child_samples=lgbm_params.get('min_child_samples', 20),
            subsample=lgbm_params.get('subsample', 0.8),
            colsample_bytree=lgbm_params.get('colsample_bytree', 0.8),
            reg_alpha=lgbm_params.get('reg_alpha', 0.1),
            reg_lambda=lgbm_params.get('reg_lambda', 0.1),
            random_state=random_state
        )
    elif model_type == 3:
        # XGBoost
        log.info("XGBoost modeli eğitiliyor...")
        # Model parametrelerini configs/model.yaml'dan al
        xgb_params = cfg.get('model', {}).get('xgboost', {})
        model = XGBModel(
            max_depth=xgb_params.get('max_depth', 5),
            learning_rate=xgb_params.get('learning_rate', 0.05),
            n_estimators=xgb_params.get('n_estimators', 300),
            subsample=xgb_params.get('subsample', 0.8),
            colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
            reg_alpha=xgb_params.get('reg_alpha', 0.1),
            reg_lambda=xgb_params.get('reg_lambda', 0.1),
            random_state=random_state
        )
    elif model_type == 4:
        # Ensemble (Voting)
        log.info("Ensemble (Voting) modeli eğitiliyor...")
        # Ensemble parametrelerini configs/model.yaml'dan al
        lgbm_params = cfg.get('model', {}).get('lightgbm', {})
        xgb_params = cfg.get('model', {}).get('xgboost', {})
        rf_params = cfg.get('model', {}).get('random_forest', {})
        
        # YAML'dan çekilen parametrelerle modelleri oluştur
        base_models = [
            ('lgbm', LightGBMModel(
                num_leaves=lgbm_params.get('num_leaves', 31),
                max_depth=lgbm_params.get('max_depth', 5),
                learning_rate=lgbm_params.get('learning_rate', 0.05),
                n_estimators=lgbm_params.get('n_estimators', 300),
                min_child_samples=lgbm_params.get('min_child_samples', 20),
                subsample=lgbm_params.get('subsample', 0.8),
                colsample_bytree=lgbm_params.get('colsample_bytree', 0.8),
                reg_alpha=lgbm_params.get('reg_alpha', 0.1),
                reg_lambda=lgbm_params.get('reg_lambda', 0.1),
                random_state=random_state
            )),
            ('xgb', XGBModel(
                max_depth=xgb_params.get('max_depth', 5),
                learning_rate=xgb_params.get('learning_rate', 0.05),
                n_estimators=xgb_params.get('n_estimators', 300),
                subsample=xgb_params.get('subsample', 0.8),
                colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
                reg_alpha=xgb_params.get('reg_alpha', 0.1),
                reg_lambda=xgb_params.get('reg_lambda', 0.1),
                random_state=random_state
            )),
            ('rf', RandomForestModel(
                n_estimators=rf_params.get('n_estimators', 300),
                max_depth=rf_params.get('max_depth', 10),
                min_samples_split=rf_params.get('min_samples_split', 5),
                min_samples_leaf=rf_params.get('min_samples_leaf', 5),
                max_features=rf_params.get('max_features', 'sqrt'),
                random_state=random_state
            ))
        ]
        model = build_voting_ensemble(base_models)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    
    # 7. Model eğitimi ve değerlendirme
    log.info("Model eğitiliyor...")
    
    # Sınıf dengesizliği kontrolü
    class_balance = y_train.mean()
    log.info(f"Sınıf dağılımı - Positive: {class_balance:.2%}, Negative: {1-class_balance:.2%}")
    
    if class_balance < 0.2:
        log.warning(f"Dengesiz sınıf dağılımı: {class_balance:.2%} positive")
        
        # Kategorik özellik indekslerini belirle (SMOTENC için)
        cat_idx = []
        for i, (name, transformer, cols) in enumerate(preprocessor.transformers_):
            if name == 'cat' and cols:
                start_idx = 0
                for prev_name, prev_transformer, prev_cols in preprocessor.transformers_[:i]:
                    if hasattr(prev_transformer, 'transform'):
                        prev_sample = prev_transformer.transform(X_train[prev_cols].iloc[:1])
                        start_idx += prev_sample.shape[1]
                
                cat_sample = transformer.transform(X_train[cols].iloc[:1])
                cat_idx.extend(range(start_idx, start_idx + cat_sample.shape[1]))
        
        log.info(f"SMOTENC için kategorik özellik indeksleri: {len(cat_idx)} indeks")
        
        # Dengesiz veri işleme
        try:
            X_train_resampled, y_train_resampled = balanced_resample(X_train_processed, y_train, cat_idx)
            log.info(f"Resampled veri boyutu: {X_train_resampled.shape}, Positive ratio: {y_train_resampled.mean():.2%}")
            
            # Resampled veri ile modeli eğit
            model.fit(X_train_resampled, y_train_resampled)
        except Exception as e:
            log.error(f"Resampling sırasında hata: {e}")
            log.warning("Orijinal veri ile devam ediliyor...")
            model.fit(X_train_processed, y_train)
        else:
            # Normal eğitim (sınıf dengeli)
            model.fit(X_train_processed, y_train)
    
    # 8. Kalibrasyon
    if include_calibration and has_val_set:
        log.info("Model kalibrasyonu yapılıyor...")
        
        # Kalibrasyon dizini oluştur
        calibration_dir = output_dir / "calibration"
        os.makedirs(calibration_dir, exist_ok=True)
        
        # En iyi kalibratörü seç (validation seti üzerinde)
        calibrated_model = choose_calibrator(
            model, 
            X_val_processed, 
            y_val,
            X_test_processed if has_test_set else None,
            y_test if has_test_set else None,
            output_dir=calibration_dir
        )
        
        # Modeli calibrated_model ile değiştir
        model = calibrated_model
    
    # 9. Değerlendirme
    log.info("Model değerlendiriliyor...")
    
    # Train seti değerlendirme
    y_prob_train = model.predict_proba(X_train_processed)[:, 1]
    metrics_train = calculate_classification_metrics(y_train, y_prob_train)
    log.info(f"Train metrics: {metrics_train}")
    
    # Validation seti değerlendirme
    if has_val_set:
        y_prob_val = model.predict_proba(X_val_processed)[:, 1]
        metrics_val = calculate_classification_metrics(y_val, y_prob_val)
        log.info(f"Validation metrics: {metrics_val}")
        
        # ROC ve PR eğrileri
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plot_roc_curve(y_val, y_prob_val, title="ROC Curve (Validation)")
        
        plt.subplot(1, 2, 2)
        plot_precision_recall_curve(y_val, y_prob_val, title="Precision-Recall Curve (Validation)")
        
        plt.tight_layout()
        plt.savefig(output_dir / "validation_curves.png")
        plt.close()
    
    # Test seti değerlendirme
    if has_test_set:
        y_prob_test = model.predict_proba(X_test_processed)[:, 1]
        metrics_test = calculate_classification_metrics(y_test, y_prob_test)
        log.info(f"Test metrics: {metrics_test}")
        
        # ROC ve PR eğrileri
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plot_roc_curve(y_test, y_prob_test, title="ROC Curve (Test)")
        
        plt.subplot(1, 2, 2)
        plot_precision_recall_curve(y_test, y_prob_test, title="Precision-Recall Curve (Test)")
        
        plt.tight_layout()
        plt.savefig(output_dir / "test_curves.png")
        plt.close()
    
    # 10. Bin bazlı kalibrasyon
    if has_val_set or has_test_set:
        log.info("Bin bazlı kalibrasyon değerlendirmesi yapılıyor...")
        
        # ProbabilityBinner ile segmentlere ayır
        binner = ProbabilityBinner()
        
        # Validation seti bin bazlı metrikler
        if has_val_set:
            val_segments = binner.transform(y_prob_val)
            val_bins = pd.DataFrame({
                'segment': val_segments,
                'actual': y_val,
                'predicted': y_prob_val
            })
            
            # Segment bazlı metrikler
            val_bin_metrics = val_bins.groupby('segment').agg({
                'actual': ['count', 'mean'],
                'predicted': 'mean'
            })
            val_bin_metrics.columns = ['count', 'conversion_rate', 'avg_prediction']
            val_bin_metrics['pct_total'] = val_bin_metrics['count'] / len(val_bins) * 100
            
            log.info(f"Validation bin metrics:\n{val_bin_metrics}")
            
            # CSV'ye kaydet
            val_bin_metrics.to_csv(output_dir / "validation_bin_metrics.csv")
        
        # Test seti bin bazlı metrikler
        if has_test_set:
            test_segments = binner.transform(y_prob_test)
            test_bins = pd.DataFrame({
                'segment': test_segments,
                'actual': y_test,
                'predicted': y_prob_test
            })
            
            # Segment bazlı metrikler
            test_bin_metrics = test_bins.groupby('segment').agg({
                'actual': ['count', 'mean'],
                'predicted': 'mean'
            })
            test_bin_metrics.columns = ['count', 'conversion_rate', 'avg_prediction']
            test_bin_metrics['pct_total'] = test_bin_metrics['count'] / len(test_bins) * 100
            
            log.info(f"Test bin metrics:\n{test_bin_metrics}")
            
            # CSV'ye kaydet
            test_bin_metrics.to_csv(output_dir / "test_bin_metrics.csv")
            
            # Low-Medium-High segment doğrulaması
            bin_order = ['Low', 'Medium', 'High']
            if all(segment in test_bin_metrics.index for segment in bin_order):
                conversion_rates = [test_bin_metrics.loc[segment, 'conversion_rate'] for segment in bin_order]
                if conversion_rates[0] < conversion_rates[1] < conversion_rates[2]:
                    log.info("✓ Segment sıralaması doğru: Low < Medium < High")
                else:
                    log.warning("✕ Segment sıralaması hatalı!")
                    log.warning(f"Conversion rates: Low={conversion_rates[0]:.2%}, Medium={conversion_rates[1]:.2%}, High={conversion_rates[2]:.2%}")
    
    # 11. Model ve bileşenleri kaydet
    log.info("Model ve bileşenleri kaydediliyor...")
    
    # Models dizini oluştur
    models_dir = output_dir / "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Model bileşenlerini kaydet
    joblib.dump(cleaner, models_dir / "cleaner.pkl")
    joblib.dump(feature_engineer, models_dir / "feature_engineer.pkl")
    joblib.dump(preprocessor, models_dir / "preprocessor.pkl")
    joblib.dump(model, models_dir / "model.pkl")
    joblib.dump(binner, models_dir / "binner.pkl")
    
    # Özellik isimleri ve önemlerini kaydet
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_[0]
    elif hasattr(model, "estimators_") and len(model.estimators_) > 0:
        # Voting ensemble için ilk modelin önem değerlerini kullan
        for name, estimator in model.named_estimators_.items():
            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
                break
            elif hasattr(estimator, "coef_"):
                importances = estimator.coef_[0]
                break
        else:
            importances = np.ones(len(feature_names))
    else:
        importances = np.ones(len(feature_names))
    
    # Özellik önemleri
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # CSV'ye kaydet
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    
    # Top 20 özellik grafiği
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 20 Features by Importance')
    plt.tight_layout()
    plt.savefig(output_dir / "top_features.png")
    plt.close()
    
    # SHAP değerleri (TreeExplainer için)
    if isinstance(model, (LightGBMModel, XGBModel, RandomForestModel)) or model_type in [2, 3]:
        try:
            log.info("SHAP değerleri hesaplanıyor...")
            
            # Explainer oluştur
            if hasattr(model, "estimators_") and len(model.estimators_) > 0:
                # Voting ensemble için ilk modeli kullan
                for name, estimator in model.named_estimators_.items():
                    if hasattr(estimator, "feature_importances_"):
                        explainer = shap.TreeExplainer(estimator)
                        break
                else:
                    explainer = None
            else:
                explainer = shap.TreeExplainer(model)
            
            if explainer is not None:
                # SHAP değerlerini hesapla (train setinin ilk 1000 örneği)
                X_sample = X_train_processed[:min(1000, len(X_train_processed))]
                shap_values = explainer.shap_values(X_sample)
                
                # SHAP değerlerini doğru formata getir
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    # Binary sınıflandırma için, pozitif sınıfın SHAP değerlerini al
                    shap_values = shap_values[1]
                
                # SHAP summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(output_dir / "shap_summary.png")
                plt.close()
                
                # SHAP dependency plot (top 2 özellik için)
                try:
                    top_2_features = feature_importance['feature'].iloc[:2].tolist()
                    for feature in top_2_features:
                        if feature in feature_names:
                            feature_idx = list(feature_names).index(feature)
                            plt.figure(figsize=(10, 6))
                            shap.dependence_plot(feature_idx, shap_values, X_sample, feature_names=feature_names, show=False)
                            plt.tight_layout()
                            plt.savefig(output_dir / f"shap_dependence_{feature.replace(' ', '_')}.png")
                            plt.close()
                except Exception as e:
                    log.warning(f"SHAP dependency plot oluşturma hatası: {e}")
        except Exception as e:
            log.warning(f"SHAP değerleri hesaplanırken hata: {e}")
    
    # 12. MLflow tracking
    try:
        # MLflow yapılandırması
        from src.utils.paths import get_mlflow_tracking_uri
        mlflow.set_tracking_uri(get_mlflow_tracking_uri())
        
        # Yeni experiment oluştur veya mevcut olanı kullan
        mlflow.set_experiment(experiment_name)
        
        # Run başlat
        with mlflow.start_run(run_name=f"model_{model_type}"):
            # Parametreleri logla
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("include_calibration", include_calibration)
            mlflow.log_param("source_specific", source_specific)
            mlflow.log_param("random_state", random_state)
            
            # Eğitim ve değerlendirme metriklerini logla
            for key, value in metrics_train.items():
                mlflow.log_metric(f"train_{key}", value)
            
            if has_val_set:
                for key, value in metrics_val.items():
                    mlflow.log_metric(f"val_{key}", value)
            
            if has_test_set:
                for key, value in metrics_test.items():
                    mlflow.log_metric(f"test_{key}", value)
            
            # Artifact'ları logla
            mlflow.log_artifact(str(output_dir / "top_features.png"))
            
            if (output_dir / "validation_curves.png").exists():
                mlflow.log_artifact(str(output_dir / "validation_curves.png"))
            
            if (output_dir / "test_curves.png").exists():
                mlflow.log_artifact(str(output_dir / "test_curves.png"))
            
            if (output_dir / "shap_summary.png").exists():
                mlflow.log_artifact(str(output_dir / "shap_summary.png"))
            
            # Model kaydet (scikit-learn formatında)
            mlflow.sklearn.log_model(model, "model")
    except Exception as e:
        log.warning(f"MLflow tracking hatası: {e}")
    
    log.info(f"Model eğitimi tamamlandı! Çıktılar: {output_dir}")
    
    # Sonuçları döndür
    return {
        'model': model,
        'cleaner': cleaner,
        'feature_engineer': feature_engineer,
        'preprocessor': preprocessor,
        'binner': binner,
        'feature_importance': feature_importance,
        'output_dir': output_dir
    }

@hydra.main(version_base=None, config_path="../../configs", config_name="experiment")
def main(cfg):
    """
    Model eğitimi için CLI.
    """
    from src.utils.paths import get_experiment_dir
    output_dir = get_experiment_dir()
    
    # Model tipini belirle - Hydra konfigürasyonundan al
    model_type = cfg.get('model', 2)
    experiment_name = cfg.get('experiment_name', f"model_{model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Eğitim - konfigürasyonu da geçir
    train_model(
        cfg=cfg,
        output_dir=output_dir,
        model_type=model_type,
        experiment_name=experiment_name
    )

if __name__ == "__main__":
    main()
