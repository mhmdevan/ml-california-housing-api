from __future__ import annotations

import numpy as np

from src import train_model
from src import predict as predict_module


def test_predict_one_sample():
    # Ensure model is trained and saved
    train_model.main()

    model, feature_names, target_name, model_name = predict_module.load_model_and_metadata()

    X_sample = np.zeros((1, len(feature_names)))
    y_pred = model.predict(X_sample)[0]

    # Just check it's a numeric value and no exception is raised
    assert isinstance(float(y_pred), float)