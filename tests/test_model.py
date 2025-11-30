from app.data import load_iris_dataset
from app.model import build_model
from app.preprocess import split_data


def test_model_fit_and_predict():
    X, y = load_iris_dataset()
    X_train, X_test, y_train, _ = split_data(X, y, test_size=0.3, random_state=0)
    model = build_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
