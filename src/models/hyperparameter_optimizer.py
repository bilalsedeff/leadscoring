"""
Optuna ile hiperparametre optimizasyonu yapan modül.
model.yaml içindeki hiperparametre ayarlarını kullanarak optimizasyon yapar.
"""
import optuna
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
import warnings
import logging

from src.utils.logger import get_logger

log = get_logger()

# Uyarıları bastır
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_hyperparameter_config(model_type: str = None) -> Dict:
    """
    model.yaml'dan hiperparametre konfigürasyonunu yükler
    
    Args:
        model_type: Belirli bir model türü için konfigürasyon yüklemek için
                   (örn. "lightgbm", "xgboost", "baseline_lr")
    
    Returns:
        Dict: Hiperparametre konfigürasyonu
    """
    config_path = Path("configs/model.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")
    
    try:
        # OmegaConf ile konfigürasyonu yükle
        cfg = OmegaConf.load(config_path)
        
        # Dict'e dönüştür
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Belirli bir model türü istendiyse, sadece onu döndür
        if model_type is not None:
            if model_type not in config_dict:
                raise ValueError(f"Model türü bulunamadı: {model_type}")
            return config_dict[model_type]
        
        return config_dict
    except Exception as e:
        log.error(f"Hiperparametre konfigürasyonu yüklenirken hata: {e}")
        raise


def get_metrics_func(metric_name: str) -> Callable:
    """
    Metrik adına göre ilgili metrik fonksiyonunu döndürür
    
    Args:
        metric_name: Metrik adı (örn. "roc_auc", "f1", "precision", "recall", "average_precision")
    
    Returns:
        Callable: Metrik fonksiyonu
    """
    metrics = {
        "roc_auc": roc_auc_score,
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score,
        "average_precision": average_precision_score,
    }
    
    if metric_name not in metrics:
        raise ValueError(f"Desteklenmeyen metrik: {metric_name}. Desteklenen metrikler: {list(metrics.keys())}")
    
    return metrics[metric_name]


def get_pruner(pruner_name: str, **kwargs) -> optuna.pruners.BasePruner:
    """
    Pruner adına göre Optuna pruner oluşturur
    
    Args:
        pruner_name: Pruner adı (örn. "hyperband", "median", "percentile", "none")
        **kwargs: Pruner için ek parametreler
    
    Returns:
        optuna.pruners.BasePruner: Optuna pruner
    """
    if pruner_name == "hyperband":
        return optuna.pruners.HyperbandPruner(**kwargs)
    elif pruner_name == "median":
        return optuna.pruners.MedianPruner(**kwargs)
    elif pruner_name == "percentile":
        return optuna.pruners.PercentilePruner(**kwargs)
    elif pruner_name == "none":
        return optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Desteklenmeyen pruner: {pruner_name}")


def create_model_from_params(model_type: str, params: Dict[str, Any]) -> Any:
    """
    Model türüne ve parametrelere göre model nesnesi oluşturur
    
    Args:
        model_type: Model türü (örn. "lightgbm", "xgboost", "baseline_lr")
        params: Model parametreleri
    
    Returns:
        Any: Model nesnesi
    """
    if model_type == "lightgbm":
        return lgb.LGBMClassifier(**params, random_state=42)
    elif model_type == "xgboost":
        return xgb.XGBClassifier(**params, random_state=42)
    elif model_type == "catboost":
        return CatBoostClassifier(**params, random_state=42, verbose=False)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params, random_state=42)
    elif model_type == "baseline_lr":
        return LogisticRegression(penalty="elasticnet", solver="saga", **params, random_state=42)
    elif model_type == "ridge_lr":
        return LogisticRegression(penalty="l2", solver="lbfgs", **params, random_state=42)
    elif model_type == "lasso_lr":
        return LogisticRegression(penalty="l1", solver="liblinear", **params, random_state=42)
    elif model_type == "svm":
        return SVC(**params, random_state=42)
    else:
        raise ValueError(f"Desteklenmeyen model türü: {model_type}")


def create_objective(model_type: str, X: pd.DataFrame, y: pd.Series, 
                     metric: str = "roc_auc", cv_folds: int = 5,
                     random_state: int = 42) -> Callable:
    """
    Optuna için amaç fonksiyonu oluşturur
    
    Args:
        model_type: Model türü (örn. "lightgbm", "xgboost", "baseline_lr")
        X: Eğitim verileri
        y: Hedef değişken
        metric: Optimizasyon metriği (örn. "roc_auc", "f1")
        cv_folds: Cross-validation fold sayısı
        random_state: Randomizasyon için seed değeri
        
    Returns:
        Callable: Optuna amaç fonksiyonu
    """
    # Hiperparametre konfigürasyonunu yükle
    config = load_hyperparameter_config(model_type)
    
    def objective(trial):
        # Trial'dan hiperparametreleri al
        params = {}
        
        for param_name, param_values in config.items():
            if param_name == "name":  # İsim alanını atla
                continue
                
            if isinstance(param_values, list):
                if all(isinstance(x, (int, float)) for x in param_values) and not any(isinstance(x, bool) for x in param_values):
                    # Sayısal değerler
                    if all(isinstance(x, int) for x in param_values):
                        params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                    else:
                        params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                else:
                    # Kategorik değerler
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
        
        # Model oluştur
        model = create_model_from_params(model_type, params)
        
        # Cross-validation ile değerlendir
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
        
        return scores.mean()
    
    return objective


def optimize_hyperparameters(X: pd.DataFrame, y: pd.Series, 
                            model_type: str = "lightgbm",
                            n_trials: int = 50, 
                            timeout: Optional[int] = 3600,
                            metric: str = "roc_auc",
                            direction: str = "maximize",
                            pruner_name: str = "hyperband",
                            cv_folds: int = 5,
                            random_state: int = 42,
                            early_stopping_rounds: Optional[int] = 30) -> Tuple[Dict[str, Any], float]:
    """
    Belirtilen model için hiperparametre optimizasyonu yapar
    
    Args:
        X: Eğitim verileri
        y: Hedef değişken
        model_type: Model türü (örn. "lightgbm", "xgboost", "baseline_lr")
        n_trials: Deneme sayısı
        timeout: Maksimum çalışma süresi (saniye)
        metric: Optimizasyon metriği (örn. "roc_auc", "f1")
        direction: Optimizasyon yönü ("maximize" veya "minimize")
        pruner_name: Pruner adı (örn. "hyperband", "median", "percentile", "none")
        cv_folds: Cross-validation fold sayısı
        random_state: Randomizasyon için seed değeri
        early_stopping_rounds: Early stopping için gerekli round sayısı
        
    Returns:
        Tuple[Dict[str, Any], float]: En iyi parametreler ve en iyi skor
    """
    log.info(f"{model_type} için hiperparametre optimizasyonu başlatılıyor...")
    
    # Pruner oluştur
    pruner = get_pruner(pruner_name)
    
    # Çalışma oluştur
    study = optuna.create_study(
        direction=direction,
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    # Amaç fonksiyonu oluştur
    objective = create_objective(
        model_type=model_type,
        X=X,
        y=y,
        metric=metric,
        cv_folds=cv_folds,
        random_state=random_state
    )
    
    # Optimizasyonu çalıştır
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[optuna.callbacks.MaxTrialsCallback(n_trials)],
        show_progress_bar=True
    )
    
    log.info(f"Optimizasyon tamamlandı. En iyi skor: {study.best_value:.4f}")
    
    # En iyi parametreleri döndür
    return study.best_params, study.best_value


def create_ensemble_model(X: pd.DataFrame, y: pd.Series, 
                         ensemble_type: str = "voting",
                         base_estimators: List[str] = None,
                         weights: List[float] = None,
                         final_estimator: str = "baseline_lr",
                         cv: int = 5,
                         passthrough: bool = True,
                         optimize_base: bool = False) -> Union[VotingClassifier, StackingClassifier]:
    """
    Ensemble model oluşturur (Voting veya Stacking)
    
    Args:
        X: Eğitim verileri
        y: Hedef değişken
        ensemble_type: Ensemble türü ("voting" veya "stacking")
        base_estimators: Temel model türleri listesi
        weights: Voting için ağırlıklar (None = eşit ağırlık)
        final_estimator: Stacking için final estimator türü
        cv: Stacking için cross-validation fold sayısı
        passthrough: Stacking için orijinal özellikleri geçirme
        optimize_base: Temel modelleri optimize etme
        
    Returns:
        Union[VotingClassifier, StackingClassifier]: Ensemble model
    """
    if base_estimators is None:
        base_estimators = ["lightgbm", "xgboost", "random_forest", "baseline_lr"]
    
    # Konfigürasyonu yükle
    config = load_hyperparameter_config()
    
    # Temel modelleri oluştur
    estimators = []
    for est_type in base_estimators:
        if optimize_base:
            # Temel modelleri optimize et
            best_params, _ = optimize_hyperparameters(
                X=X, 
                y=y, 
                model_type=est_type,
                n_trials=20,  # Daha az deneme yaparak hızlandır
                timeout=1800   # Daha kısa timeout
            )
            model = create_model_from_params(est_type, best_params)
        else:
            # Varsayılan parametrelerle oluştur
            model_config = config[est_type]
            params = {k: v[0] if isinstance(v, list) and k != "name" else v 
                     for k, v in model_config.items() if k != "name"}
            model = create_model_from_params(est_type, params)
        
        estimators.append((est_type, model))
    
    # Ensemble model oluştur
    if ensemble_type == "voting":
        return VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=weights
        )
    elif ensemble_type == "stacking":
        # Final estimator oluştur
        if optimize_base:
            final_params, _ = optimize_hyperparameters(
                X=X, 
                y=y, 
                model_type=final_estimator,
                n_trials=20
            )
            final_model = create_model_from_params(final_estimator, final_params)
        else:
            final_config = config[final_estimator]
            final_params = {k: v[0] if isinstance(v, list) and k != "name" else v 
                           for k, v in final_config.items() if k != "name"}
            final_model = create_model_from_params(final_estimator, final_params)
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_model,
            cv=cv,
            passthrough=passthrough
        )
    else:
        raise ValueError(f"Desteklenmeyen ensemble türü: {ensemble_type}")


def create_source_specific_models(X: pd.DataFrame, y: pd.Series,
                                 source_col: str = "Source_Final__c",
                                 base_model: str = "lightgbm",
                                 optimize_models: bool = False) -> Dict[str, Any]:
    """
    Kaynak bazlı modeller oluşturur
    
    Args:
        X: Eğitim verileri
        y: Hedef değişken
        source_col: Kaynak kolonu
        base_model: Temel model türü
        optimize_models: Modelleri optimize etme
        
    Returns:
        Dict[str, Any]: Kaynak bazlı modeller sözlüğü
    """
    source_models = {}
    
    # Global model
    if optimize_models:
        global_params, _ = optimize_hyperparameters(
            X=X, 
            y=y, 
            model_type=base_model,
            n_trials=30
        )
        global_model = create_model_from_params(base_model, global_params)
    else:
        config = load_hyperparameter_config(base_model)
        params = {k: v[0] if isinstance(v, list) and k != "name" else v 
                 for k, v in config.items() if k != "name"}
        global_model = create_model_from_params(base_model, params)
    
    # Global modeli fit et
    global_model.fit(X, y)
    source_models["global"] = global_model
    
    # Kaynak bazlı modeller
    for source in X[source_col].unique():
        mask = X[source_col] == source
        if sum(mask) >= 50:  # Yeterli veri varsa
            X_source = X[mask]
            y_source = y[mask]
            
            if optimize_models:
                source_params, _ = optimize_hyperparameters(
                    X=X_source, 
                    y=y_source, 
                    model_type=base_model,
                    n_trials=20
                )
                source_model = create_model_from_params(base_model, source_params)
            else:
                config = load_hyperparameter_config(base_model)
                params = {k: v[0] if isinstance(v, list) and k != "name" else v 
                         for k, v in config.items() if k != "name"}
                source_model = create_model_from_params(base_model, params)
            
            source_model.fit(X_source, y_source)
            source_models[source] = source_model
    
    return source_models 