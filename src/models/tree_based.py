from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

# CatBoost opsiyonel olarak yüklenir
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

class LightGBMModel(BaseEstimator, ClassifierMixin):
    """LightGBM için wrapper sınıf"""
    
    def __init__(self, num_leaves=31, max_depth=-1, learning_rate=0.1, 
                 n_estimators=300, scale_pos_weight=1, n_jobs=-1, random_state=42):
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.scale_pos_weight = scale_pos_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model = None
        
    def fit(self, X, y, sample_weight=None, **kwargs):
        self.model = LGBMClassifier(
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            scale_pos_weight=self.scale_pos_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **kwargs
        )
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
    
    def predict_proba(self, X):
        """Pozitif sınıf olasılıklarını döndürür"""
        if self.model is None:
            raise RuntimeError("Model eğitilmemiş! Önce fit() çağırın.")
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        """Sınıf tahminlerini döndürür"""
        if self.model is None:
            raise RuntimeError("Model eğitilmemiş! Önce fit() çağırın.")
        return self.model.predict(X)
    
    def sk_model(self):
        """Sklearn uyumlu modeli döndürür"""
        return LGBMClassifier(
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            scale_pos_weight=self.scale_pos_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
    def feature_importances(self):
        """Feature importance skorlarını döndürür"""
        if self.model is None:
            raise RuntimeError("Model eğitilmemiş! Önce fit() çağırın.")
        return self.model.feature_importances_


class XGBModel(BaseEstimator, ClassifierMixin):
    """XGBoost için wrapper sınıf"""
    
    def __init__(self, max_depth=7, eta=0.1, subsample=0.7, 
                 colsample_bytree=0.7, n_estimators=500, 
                 scale_pos_weight=1, n_jobs=-1, random_state=42):
        self.max_depth = max_depth
        self.eta = eta
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_estimators = n_estimators
        self.scale_pos_weight = scale_pos_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model = None
        
    def fit(self, X, y, sample_weight=None, **kwargs):
        self.model = XGBClassifier(
            max_depth=self.max_depth,
            eta=self.eta,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_estimators=self.n_estimators,
            scale_pos_weight=self.scale_pos_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **kwargs
        )
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
    
    def predict_proba(self, X):
        """Pozitif sınıf olasılıklarını döndürür"""
        if self.model is None:
            raise RuntimeError("Model eğitilmemiş! Önce fit() çağırın.")
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        """Sınıf tahminlerini döndürür"""
        if self.model is None:
            raise RuntimeError("Model eğitilmemiş! Önce fit() çağırın.")
        return self.model.predict(X)
    
    def sk_model(self):
        """Sklearn uyumlu modeli döndürür"""
        return XGBClassifier(
            max_depth=self.max_depth,
            eta=self.eta,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_estimators=self.n_estimators,
            scale_pos_weight=self.scale_pos_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
    def feature_importances(self):
        """Feature importance skorlarını döndürür"""
        if self.model is None:
            raise RuntimeError("Model eğitilmemiş! Önce fit() çağırın.")
        return self.model.feature_importances_


class RandomForestModel(BaseEstimator, ClassifierMixin):
    """RandomForest için wrapper sınıf"""
    
    def __init__(self, n_estimators=500, max_depth=10, min_samples_split=2, 
                 min_samples_leaf=1, max_features="sqrt", bootstrap=True, 
                 n_jobs=-1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model = None
        
    def fit(self, X, y, sample_weight=None, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **kwargs
        )
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
    
    def predict_proba(self, X):
        """Pozitif sınıf olasılıklarını döndürür"""
        if self.model is None:
            raise RuntimeError("Model eğitilmemiş! Önce fit() çağırın.")
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        """Sınıf tahminlerini döndürür"""
        if self.model is None:
            raise RuntimeError("Model eğitilmemiş! Önce fit() çağırın.")
        return self.model.predict(X)
    
    def sk_model(self):
        """Sklearn uyumlu modeli döndürür"""
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
    def feature_importances(self):
        """Feature importance skorlarını döndürür"""
        if self.model is None:
            raise RuntimeError("Model eğitilmemiş! Önce fit() çağırın.")
        return self.model.feature_importances_


class CatBoostModel(BaseEstimator, ClassifierMixin):
    """CatBoost için wrapper sınıf"""
    
    def __init__(self, iterations=1000, depth=6, learning_rate=0.1, 
                 l2_leaf_reg=3, random_strength=1, bagging_temperature=1,
                 thread_count=-1, random_seed=42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.thread_count = thread_count
        self.random_seed = random_seed
        self.model = None
        
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost kütüphanesi yüklü değil. 'pip install catboost' ile yükleyebilirsiniz.")
        
    def fit(self, X, y, sample_weight=None, **kwargs):
        self.model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            thread_count=self.thread_count,
            random_seed=self.random_seed,
            verbose=False,
            **kwargs
        )
        
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Pozitif sınıf olasılıklarını döndürür"""
        if self.model is None:
            raise RuntimeError("Model eğitilmemiş! Önce fit() çağırın.")
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        """Sınıf tahminlerini döndürür"""
        if self.model is None:
            raise RuntimeError("Model eğitilmemiş! Önce fit() çağırın.")
        return self.model.predict(X)
    
    def sk_model(self):
        """Sklearn uyumlu modeli döndürür"""
        return CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            thread_count=self.thread_count,
            random_seed=self.random_seed,
            verbose=False
        )
        
    def feature_importances(self):
        """Feature importance skorlarını döndürür"""
        if self.model is None:
            raise RuntimeError("Model eğitilmemiş! Önce fit() çağırın.")
        return self.model.feature_importances_
