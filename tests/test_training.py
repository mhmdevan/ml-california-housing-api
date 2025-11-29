from __future__ import annotations

from sklearn.model_selection import train_test_split

from src import train_model


def test_training_pipeline_runs():
    df = train_model.load_data()
    X = df[train_model.FEATURE_COLS].values
    y = df[train_model.TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = train_model.train_and_evaluate_models(X_train, X_test, y_train, y_test)

    assert isinstance(results, dict)
    assert len(results) >= 2

    best_name, best_entry = train_model.choose_best_model(results)

    assert "rmse" in best_entry
    assert best_entry["rmse"] > 0
