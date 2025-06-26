import numpy as np
from src.custom_transformers.log_transformer import LogTransformer

def test_log_transformer():
    data = np.array([[1], [10], [100]])
    transformer = LogTransformer()
    transformed = transformer.transform(data)
    restored = transformer.inverse_transform(transformed)
    np.testing.assert_array_almost_equal(data, restored)
