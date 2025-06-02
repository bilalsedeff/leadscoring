from collections import defaultdict
from .tree_based import LightGBMModel

class SourceWiseMoE:
    """Her Source_Final__c için ayrı model; fallback = global."""
    def fit(self, X, y, source_col="Source_Final__c", **kw):
        self.models = defaultdict(LightGBMModel)
        self.global_model = LightGBMModel()
        self.global_model.fit(X, y, **kw)
        for src in X[source_col].unique():
            idx = X[source_col] == src
            self.models[src].fit(X[idx], y[idx], **kw)
        return self
    def predict_proba(self, X):
        p = self.global_model.predict_proba(X)
        for src, mdl in self.models.items():
            mask = X["Source_Final__c"] == src
            p[mask] = mdl.predict_proba(X[mask])
        return p
