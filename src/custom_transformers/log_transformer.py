from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    LogTransformer applies log1p and expm1 for reversible transformation.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)

    def inverse_transform(self, X):
        return np.expm1(X)
