import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies log1p transformation and supports inverse transform.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)

    def inverse_transform(self, X):
        return np.expm1(X)
