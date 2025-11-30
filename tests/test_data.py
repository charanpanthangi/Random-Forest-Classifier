import pandas as pd

from app.data import load_iris_dataset


def test_load_iris_dataset_shape():
    X, y = load_iris_dataset()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == 150
    assert X.shape[1] == 4
    assert y.shape[0] == 150
