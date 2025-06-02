from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
import numpy as np
import itertools
from typing import List, Tuple, Dict, Any, Optional
from .baseline import ElasticNetLR
from .tree_based import LightGBMModel, XGBModel, RandomForestModel, CatBoostModel
from src.utils.logger import get_logger

log = get_logger()

def build_ensemble(base_models: List[Tuple[str, Any]] = None, weights: List[float] = None):
    """
    Ensemble model oluşturur.
    
    Args:
        base_models: (model_adı, model_nesnesi) çiftlerinden oluşan liste
                    Eğer None ise, varsayılan modeller kullanılır
        weights: Model ağırlıkları. None ise eşit ağırlık kullanılır
        
    Returns:
        Ensemble model
    """
    if base_models is None:
        base_models = [
            ("lgb", LightGBMModel().sk_model()),
            ("xgb", XGBModel().sk_model())
        ]
    
    meta = ElasticNetLR(C=5.0, l1_ratio=0.0).model
    
    # Stacking Classifier
    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=meta, 
        n_jobs=-1, 
        passthrough=True)
    
    return stack

def build_voting_ensemble(base_models: List[Tuple[str, Any]], weights: Optional[List[float]] = None):
    """
    Voting ensemble modeli oluşturur.
    
    Args:
        base_models: (model_adı, model_nesnesi) çiftlerinden oluşan liste
        weights: Model ağırlıkları. None ise eşit ağırlık kullanılır
        
    Returns:
        VotingClassifier: Voting ensemble model
    """
    # Base modellerin formatını kontrol et ve gerekirse dönüştür
    # scikit-learn 1.6+'da VotingClassifier estimators parametresi liste olmalı
    if not isinstance(base_models, list):
        base_models = list(base_models)
    
    return VotingClassifier(
        estimators=base_models,
        voting='soft',
        weights=weights,
        n_jobs=-1
    )

def optimize_ensemble_weights(base_models: List[Tuple[str, Any]], X_val: np.ndarray, y_val: np.ndarray, 
                             metric: str = "roc_auc", n_steps: int = 10) -> Tuple[List[float], float]:
    """
    Validation seti üzerinde ensemble ağırlıklarını optimize eder
    
    Args:
        base_models: (model_adı, model_nesnesi) çiftlerinden oluşan liste
        X_val: Validation verisi
        y_val: Validation hedef değişkeni
        metric: Optimizasyon metriği ("roc_auc", "f1", "precision", "recall", "average_precision")
        n_steps: Ağırlık aralığını böleceğimiz adım sayısı
        
    Returns:
        Tuple[List[float], float]: (En iyi ağırlıklar, En iyi skor)
    """
    log.info(f"Ensemble ağırlıkları optimize ediliyor (metrik: {metric})...")
    
    metric_func = {
        "roc_auc": roc_auc_score,
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score,
        "average_precision": average_precision_score
    }.get(metric, roc_auc_score)
    
    n_models = len(base_models)
    best_weights = None
    best_score = 0.0
    
    # Tüm modellerin tahminlerini hesapla
    predictions = []
    for _, model in base_models:
        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(X_val)[:, 1]
        else:
            pred = model.predict(X_val)
        predictions.append(pred)
    
    # Grid search için ağırlık kombinasyonlarını oluştur
    # Daha efektif bir yaklaşım için basitleştirilmiş grid search kullanıyoruz
    # n_models > 2 olduğunda kombinatorik patlama olmaması için adım sayısını azaltıyoruz
    weight_step = 1.0 / min(n_steps, 5 if n_models > 2 else 10)
    
    # Tüm ağırlıkların toplamı 1 olacak şekilde ağırlık kombinasyonları oluştur
    weight_combinations = []
    
    if n_models == 2:
        # İki model durumunda daha basit
        for w1 in np.arange(0.0, 1.0 + weight_step, weight_step):
            w2 = 1.0 - w1
            weight_combinations.append([w1, w2])
    else:
        # Çok modelli durumda Optuna kullanmak daha iyi olabilir,
        # ancak basit bir yaklaşım olarak random sampling kullanıyoruz
        np.random.seed(42)
        # 100 farklı random ağırlık kombinasyonu oluştur
        for _ in range(100):
            weights = np.random.random(n_models)
            weights = weights / weights.sum()  # Normalize et
            weight_combinations.append(weights.tolist())
        
        # Modellerin eşit ağırlıklı kombinasyonunu da ekle
        weight_combinations.append([1.0/n_models] * n_models)
    
    # Her ağırlık kombinasyonu için skoru hesapla
    for weights in weight_combinations:
        # Ağırlıklı tahmin hesapla
        weighted_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += weights[i] * pred
        
        # Skoru hesapla
        try:
            if metric in ["f1", "precision", "recall"]:
                # Bu metrikler için olasılık değil, sınıf etiketi gerekir
                score = metric_func(y_val, (weighted_pred >= 0.5).astype(int))
            else:
                score = metric_func(y_val, weighted_pred)
            
            # Daha iyi bir skor bulundu mu?
            if score > best_score:
                best_score = score
                best_weights = weights
        except Exception as e:
            log.warning(f"Ağırlık optimizasyonu sırasında hata: {e}")
    
    if best_weights is None:
        log.warning("Optimum ağırlık bulunamadı, eşit ağırlıklar kullanılacak.")
        best_weights = [1.0/n_models] * n_models
    
    log.info(f"Optimum ağırlıklar: {[round(w, 3) for w in best_weights]}, Skor: {best_score:.4f}")
    return best_weights, best_score

def train_multiple_models(X_train, y_train, model_types: List[str]) -> List[Tuple[str, Any]]:
    """
    Birden fazla model türünü eğitir
    
    Args:
        X_train: Eğitim verisi
        y_train: Eğitim hedef değişkeni
        model_types: Eğitilecek model türlerinin listesi
        
    Returns:
        List[Tuple[str, Any]]: (model_adı, model_nesnesi) çiftlerinden oluşan liste
    """
    trained_models = []
    
    for model_type in model_types:
        log.info(f"{model_type} modeli eğitiliyor...")
        
        if model_type == "lightgbm":
            model = LightGBMModel().sk_model()
        elif model_type == "xgboost":
            model = XGBModel().sk_model()
        elif model_type == "catboost":
            model = CatBoostModel().sk_model()
        elif model_type == "random_forest":
            model = RandomForestModel().sk_model()
        elif model_type == "baseline_lr":
            model = ElasticNetLR().model
        else:
            log.warning(f"Desteklenmeyen model türü: {model_type}, atlanıyor.")
            continue
        
        model.fit(X_train, y_train)
        trained_models.append((model_type, model))
        log.info(f"{model_type} modeli eğitimi tamamlandı.")
    
    return trained_models
