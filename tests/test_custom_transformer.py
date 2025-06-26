import numpy as np
from src.features.custom_transformers import LogTransformer

def test_log_transformer():
    transformer = LogTransformer()
    data = np.array([1, 10, 100])
    transformed = transformer.transform(data)
    inversed = transformer.inverse_transform(transformed)
    assert np.allclose(data, inversed)
