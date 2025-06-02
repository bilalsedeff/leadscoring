from typing import Any, Dict, Optional, List, Union
import mlflow, joblib, tempfile, os
import pandas as pd
import numpy as np
import time
from pathlib import Path
import yaml
from omegaconf import OmegaConf
from ..utils.paths import get_experiment_dir, get_mlflow_tracking_uri
from ..preprocessing.cleaning import BasicCleaner
from ..preprocessing.type_cast import apply_type_map
from ..features.engineering import add_temporal_features
from ..features.interactions import add_pairwise_ratios, add_products
import json
import logging
import uuid
import datetime
import sys
import sklearn

# Set logging
logger = logging.getLogger(__name__)

class _PyFuncWrapper(mlflow.pyfunc.PythonModel):
    """MLflow uyumlu model wrapper. Tüm işlem zincirini (cleaner → preproc → model → cal → binner) paketler."""
    
    def load_context(self, context):
        # Eğer cleaner varsa yükle, yoksa varsayılan cleaner oluştur
        self.cleaner = joblib.load(context.artifacts["cleaner"]) if "cleaner" in context.artifacts else BasicCleaner()
        self.preproc = joblib.load(context.artifacts["preproc"])
        self.model = joblib.load(context.artifacts["model"])
        self.calibrator = joblib.load(context.artifacts["calibrator"])
        self.binner = joblib.load(context.artifacts["binner"])
        
    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """LeadId → probability → segment
        
        Args:
            context: MLflow model context
            model_input: Giriş verisi (DataFrame)
            
        Returns:
            results: Tahmin sonuçları (DataFrame)
        """
        # Model input pandas DataFrame olmalı
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        
        # Tam preprocessing pipeline uygula
        # 1. Temizleme
        cleaned_data = self.cleaner.transform(model_input)
        
        # 2. Veri tipi dönüşümü
        typed_data = apply_type_map(cleaned_data)
        
        # 3. Temporal özellikler
        data_with_temporal = add_temporal_features(typed_data)
        
        # 4. Özellik etkileşimleri
        num_features = data_with_temporal.select_dtypes(include=['int64', 'float64']).columns.tolist()
        data_with_ratios = add_pairwise_ratios(data_with_temporal, num_features, max_pairs=20)
        processed_data = add_products(data_with_ratios, num_features, top_k=10)
            
        # 5. Transformer uygula
        X = self.preproc.transform(processed_data)
        
        # Tahmin
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X)
            if isinstance(y_prob, np.ndarray) and y_prob.ndim > 1:
                y_prob = y_prob[:, 1]  # Pozitif sınıf olasılığı
        else:
            y_prob = self.model.predict(X)
            
        # Kalibrasyon
        if hasattr(self.calibrator, 'predict_proba'):
            y_cal = self.calibrator.predict_proba(y_prob.reshape(-1, 1))
            if isinstance(y_cal, np.ndarray) and y_cal.ndim > 1:
                y_cal = y_cal[:, 1]
        else:
            y_cal = y_prob
            
        # Segment dönüşümü
        segments = self.binner.transform(y_cal)
        
        # Sonuçları DataFrame olarak döndür
        results = pd.DataFrame({
            "lead_id": model_input.get("LeadId", range(len(model_input))),
            "conversion_probability": y_cal,
            "segment": segments
        })
        
        return results

def load_config(config_name):
    """Hydra konfigürasyon dosyasını yükler.
    
    Args:
        config_name: Konfigürasyon dosyası adı (data, split, model, experiment)
        
    Returns:
        config: Yüklenen konfigürasyon
    """
    config_path = Path(f"configs/{config_name}.yaml")
    
    if not config_path.exists():
        print(f"UYARI: {config_path} bulunamadı. Varsayılan değerler kullanılacak.")
        return {}
    
    try:
        # OmegaConf ile yükle (Hydra uyumlu)
        config = OmegaConf.load(config_path)
        return config
    except Exception as e:
        # Düz yaml ile yükle
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e2:
            print(f"Konfigürasyon yüklenirken hata: {e2}")
            return {}

def configure_mlflow_output_dir(run_name=None, component="models"):
    """MLflow için çıktı dizinini yapılandırır.
    
    Args:
        run_name: Çalıştırma adı (None ise timestamp oluşturulur)
        component: Komponent adı (models, metrics, vb.)
        
    Returns:
        output_dir: MLflow çıktı dizini
        run_id: Çalıştırma kimliği (timestamp bazlı)
    """
    # Timestamp oluştur
    run_id = run_name or f"run_{int(time.time())}"
    
    # Outputs dizini altında run_id/component klasörü oluştur
    if run_id.startswith("run_"):
        output_dir = Path(f"outputs/{run_id}/{component}")
    else:
        output_dir = Path(f"outputs/run_{int(time.time())}/{run_id}/{component}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # MLflow için tracking URI ve artifact dizini ayarla
    # Outputs klasörü altındaki mlruns kullan
    experiment_dir = output_dir.parent
    tracking_uri = get_mlflow_tracking_uri(experiment_dir)
    mlflow.set_tracking_uri(tracking_uri)
    
    # Experiment adını ayarla (yoksa oluştur)
    experiment_name = f"LeadScoring_{run_id}"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name, artifact_location=f"file:{output_dir}")
    except Exception as e:
        print(f"MLflow experiment oluşturulurken hata: {e}")
    
    mlflow.set_experiment(experiment_name)
    
    return output_dir, run_id

def log_model_to_registry(
    preprocessor, 
    model, 
    calibrator=None, 
    binner=None, 
    experiment_name="lead_scoring",
    auto_selector=None,
    cleaner=None,
    metadata=None
):
    """
    Modeli MLflow model registry'ye kaydet
    
    Args:
        preprocessor: Preprocessing pipeline
        model: Eğitilmiş model
        calibrator: Olasılık kalibratörü (isteğe bağlı)
        binner: Segment etiketleyici (isteğe bağlı)
        experiment_name: MLflow experiment adı
        auto_selector: SmartFeatureSelector (isteğe bağlı)
        cleaner: Veri temizleyici (isteğe bağlı)
        metadata: Ek meta veriler (isteğe bağlı)
    """
    # MLflow tracking URI'yi ayarla
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    
    # Experiment oluştur
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    # Mevcut run varsa kullan, yoksa yeni run oluştur
    try:
        active_run = mlflow.active_run()
        if active_run is None:
            active_run = mlflow.start_run(experiment_id=experiment_id)
    except Exception as e:
        logger.warning(f"MLflow run başlatılırken hata: {e}")
        active_run = mlflow.start_run(experiment_id=experiment_id)
    
    run_id = active_run.info.run_id
    logger.info(f"MLflow run_id: {run_id}")
    
    # Geçici dizin oluştur
    tmp_dir = Path(f"./tmp/mlflow_artifacts_{run_id}")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Model ve bileşenleri kaydet
    model_path = tmp_dir / "model.pkl"
    joblib.dump(model, model_path)
    
    preprocessor_path = tmp_dir / "preproc.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    
    if calibrator is not None:
        calibrator_path = tmp_dir / "calibrator.pkl"
        joblib.dump(calibrator, calibrator_path)
    
    if binner is not None:
        binner_path = tmp_dir / "binner.pkl"
        joblib.dump(binner, binner_path)
    
    if cleaner is not None:
        cleaner_path = tmp_dir / "cleaner.pkl"
        joblib.dump(cleaner, cleaner_path)
    
    # SmartFeatureSelector'ı kaydet
    if auto_selector is not None:
        auto_selector_path = tmp_dir / "auto_selector.pkl"
        joblib.dump(auto_selector, auto_selector_path)
    
    # Metadata dosyası
    meta = metadata or {}
    if hasattr(model, "get_params"):
        meta["model_params"] = model.get_params()
    if hasattr(model, "__class__"):
        meta["model_type"] = model.__class__.__name__
    
    # SmartFeatureSelector için özet metaverileri ekle
    if auto_selector is not None:
        try:
            meta["auto_selector_info"] = {
                "n_features_in": len(auto_selector.columns_) if hasattr(auto_selector, "columns_") else None,
                "n_features_selected": len(auto_selector.kept_columns_) if hasattr(auto_selector, "kept_columns_") else None,
                "n_features_dropped": len(auto_selector.to_drop_) if hasattr(auto_selector, "to_drop_") else None,
                "pca_components": auto_selector.pca_.n_components_ if hasattr(auto_selector, "pca_") and auto_selector.pca_ is not None else None
            }
        except Exception as e:
            logger.warning(f"SmartFeatureSelector metaverisi oluşturulurken hata: {e}")
    
    # Metadata kaydet
    with open(tmp_dir / "metadata.json", "w") as f:
        json.dump(meta, f, default=str)
    
    # Python environment'i kaydet
    conda_env = {
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            f"python={'.'.join(map(str, list(map(int, pd.__version__.split('.')))))}",
            "pip",
            {"pip": [
                f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}",
                f"scikit-learn=={sklearn.__version__}",
                f"mlflow=={mlflow.__version__}",
                f"joblib=={joblib.__version__}"
            ]}
        ],
        "name": "lead_scoring_env"
    }
    
    with open(tmp_dir / "conda.yaml", "w") as f:
        yaml.dump(conda_env, f)
    
    # MLModel dosyası
    mlmodel = {
        "flavors": {
            "python_function": {
                "loader_module": "mlflow.sklearn",
                "python_version": ".".join(map(str, sys.version_info[:3])),
                "data": "model.pkl"
            },
            "sklearn": {
                "pickled_model": "model.pkl",
                "sklearn_version": sklearn.__version__,
                "serialization_format": "joblib"
            }
        },
        "run_id": run_id,
        "model_uuid": str(uuid.uuid4()),
        "utc_time_created": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "mlflow_version": mlflow.__version__
    }
    
    with open(tmp_dir / "MLmodel", "w") as f:
        yaml.dump(mlmodel, f)
    
    # MLflow'a artifact olarak kaydet
    mlflow.log_artifacts(str(tmp_dir), "model")
    
    # Parametreleri log'la
    mlflow.log_params({
        "model_type": meta.get("model_type", "Unknown"),
        "has_calibrator": calibrator is not None,
        "has_binner": binner is not None,
        "has_cleaner": cleaner is not None,
        "has_auto_selector": auto_selector is not None
    })
    
    # Modeli MLflow model registry'ye kaydet
    mlflow.sklearn.log_model(
        model,
        "registered_model",
        registered_model_name=f"lead_scoring_{experiment_name}"
    )
    
    logger.info(f"Model MLflow'a kaydedildi: experiment={experiment_name}, run_id={run_id}")
    
    return run_id
