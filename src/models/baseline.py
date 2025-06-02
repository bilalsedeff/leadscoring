from sklearn.linear_model import LogisticRegression
class ElasticNetLR:
    def __init__(self, C=1.0, l1_ratio=0.2):
        self.model = LogisticRegression(
            penalty="elasticnet", solver="saga",
            l1_ratio=l1_ratio, C=C, max_iter=4000, n_jobs=-1)
    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
