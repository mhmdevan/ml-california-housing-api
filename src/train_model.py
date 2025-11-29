#!/usr/bin/env python

"""
train_model.py

Project 1 – California housing price prediction.

Steps:
    - Load the California housing dataset into a Pandas DataFrame.
    - Run basic EDA and save plots (histogram, correlation heatmap, scatter).
    - Split data into train and test sets.
    - Train multiple models:
        * LinearRegression (raw features)
        * LinearRegression (StandardScaler + LinearRegression)
        * RandomForestRegressor (baseline)
        * RandomForestRegressor (tuned with GridSearchCV)
    - Compare models using MSE / RMSE / MAE on the test set.
    - Save the best model and metadata (feature names, target name, CV params).

Run:
    python src/train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import joblib


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET_COL = "MedHouseVal"


def print_section(title: str) -> None:
    """Pretty section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------
# EDA helpers
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame) -> None:
    """
    Basic EDA:
        - print shape/info
        - target histogram
        - correlation heatmap
        - example scatter
    """
    print_section("EDA – basic info")
    print("Shape:", df.shape)
    print("\nDataFrame head:")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isna().sum())

    # Histogram of target
    print_section("EDA – target histogram")
    plt.figure(figsize=(8, 5))
    plt.hist(df[TARGET_COL], bins=50, edgecolor="black")
    plt.xlabel("Median house value (x 100,000 USD)")
    plt.ylabel("Frequency")
    plt.title("Target distribution – MedHouseVal")
    plt.tight_layout()
    target_hist_path = PLOTS_DIR / "target_hist.png"
    plt.savefig(target_hist_path, dpi=140)
    plt.close()
    print(f"[PLOT] Saved target histogram to {target_hist_path}")

    # Correlation heatmap
    print_section("EDA – correlation heatmap")
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    heatmap_path = PLOTS_DIR / "corr_heatmap.png"
    plt.savefig(heatmap_path, dpi=140)
    plt.close()
    print(f"[PLOT] Saved correlation heatmap to {heatmap_path}")

    # Example scatter plot for an important feature
    print_section("EDA – scatter: MedInc vs MedHouseVal")
    plt.figure(figsize=(8, 5))
    plt.scatter(df["MedInc"], df[TARGET_COL], alpha=0.3)
    plt.xlabel("MedInc (median income)")
    plt.ylabel("MedHouseVal")
    plt.title("Income vs. house value")
    plt.tight_layout()
    scatter_path = PLOTS_DIR / "feature_scatter_medinc.png"
    plt.savefig(scatter_path, dpi=140)
    plt.close()
    print(f"[PLOT] Saved scatter plot to {scatter_path}")


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Fetch California housing dataset as a Pandas DataFrame."""
    print_section("Loading California housing dataset")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    expected_cols = set(FEATURE_COLS + [TARGET_COL])
    if not expected_cols.issubset(df.columns):
        raise RuntimeError(
            f"Dataset columns do not match expected ones.\n"
            f"Expected at least: {sorted(expected_cols)}\n"
            f"Got: {sorted(df.columns.tolist())}"
        )
    return df


def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict]:
    """
    Train multiple models and evaluate them on the test set.

    Returns
    -------
    results : dict
        Mapping from model name to:
            {
                "model": fitted estimator,
                "mse": float,
                "rmse": float,
                "mae": float,
                (optional) "cv_best_params": dict
            }
    """
    results: dict[str, dict] = {}

    # --------------------------------------------------
    # Linear Regression (raw features)
    # --------------------------------------------------
    print_section("Training LinearRegression (raw features)")
    lin_raw = LinearRegression()
    lin_raw.fit(X_train, y_train)
    y_pred_lin_raw = lin_raw.predict(X_test)

    mse_lin_raw = mean_squared_error(y_test, y_pred_lin_raw)
    mae_lin_raw = mean_absolute_error(y_test, y_pred_lin_raw)
    rmse_lin_raw = np.sqrt(mse_lin_raw)

    print(f"LinearRegression (raw) – MSE:  {mse_lin_raw:.4f}")
    print(f"LinearRegression (raw) – RMSE: {rmse_lin_raw:.4f}")
    print(f"LinearRegression (raw) – MAE:  {mae_lin_raw:.4f}")

    results["LinearRegression_raw"] = {
        "model": lin_raw,
        "mse": mse_lin_raw,
        "rmse": rmse_lin_raw,
        "mae": mae_lin_raw,
    }

    # --------------------------------------------------
    # Linear Regression with StandardScaler
    # --------------------------------------------------
    print_section("Training LinearRegression (StandardScaler + LinearRegression)")
    lin_scaled = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    lin_scaled.fit(X_train, y_train)
    y_pred_lin_scaled = lin_scaled.predict(X_test)

    mse_lin_scaled = mean_squared_error(y_test, y_pred_lin_scaled)
    mae_lin_scaled = mean_absolute_error(y_test, y_pred_lin_scaled)
    rmse_lin_scaled = np.sqrt(mse_lin_scaled)

    print(f"LinearRegression (scaled) – MSE:  {mse_lin_scaled:.4f}")
    print(f"LinearRegression (scaled) – RMSE: {rmse_lin_scaled:.4f}")
    print(f"LinearRegression (scaled) – MAE:  {mae_lin_scaled:.4f}")

    results["LinearRegression_scaled"] = {
        "model": lin_scaled,
        "mse": mse_lin_scaled,
        "rmse": rmse_lin_scaled,
        "mae": mae_lin_scaled,
    }

    # --------------------------------------------------
    # Baseline RandomForestRegressor
    # --------------------------------------------------
    print_section("Training RandomForestRegressor (baseline)")
    rf_base = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf_base.fit(X_train, y_train)
    y_pred_rf_base = rf_base.predict(X_test)

    mse_rf_base = mean_squared_error(y_test, y_pred_rf_base)
    mae_rf_base = mean_absolute_error(y_test, y_pred_rf_base)
    rmse_rf_base = np.sqrt(mse_rf_base)

    print(f"RandomForest (baseline) – MSE:  {mse_rf_base:.4f}")
    print(f"RandomForest (baseline) – RMSE: {rmse_rf_base:.4f}")
    print(f"RandomForest (baseline) – MAE:  {mae_rf_base:.4f}")

    results["RandomForest_baseline"] = {
        "model": rf_base,
        "mse": mse_rf_base,
        "rmse": rmse_rf_base,
        "mae": mae_rf_base,
    }

    # --------------------------------------------------
    # Tuned RandomForest with GridSearchCV
    # --------------------------------------------------
    print_section("Training RandomForestRegressor (GridSearchCV tuned)")

    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_

    print("Best RandomForest params from CV:", grid.best_params_)

    y_pred_rf_tuned = best_rf.predict(X_test)
    mse_rf_tuned = mean_squared_error(y_test, y_pred_rf_tuned)
    mae_rf_tuned = mean_absolute_error(y_test, y_pred_rf_tuned)
    rmse_rf_tuned = np.sqrt(mse_rf_tuned)

    print(f"RandomForest (tuned) – MSE:  {mse_rf_tuned:.4f}")
    print(f"RandomForest (tuned) – RMSE: {rmse_rf_tuned:.4f}")
    print(f"RandomForest (tuned) – MAE:  {mae_rf_tuned:.4f}")

    results["RandomForest_tuned"] = {
        "model": best_rf,
        "mse": mse_rf_tuned,
        "rmse": rmse_rf_tuned,
        "mae": mae_rf_tuned,
        "cv_best_params": grid.best_params_,
    }

    return results


def choose_best_model(results: dict[str, dict]) -> tuple[str, dict]:
    """Pick the model with the lowest RMSE."""
    best_name: str | None = None
    best_entry: dict | None = None
    best_rmse = float("inf")

    for name, entry in results.items():
        rmse = entry["rmse"]
        print(f"Model {name}: RMSE = {rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_entry = entry

    if best_name is None or best_entry is None:
        raise RuntimeError("No best model found. Something went wrong.")

    print_section("Best model")
    print(f"Best model: {best_name}")
    print(f"RMSE: {best_entry['rmse']:.4f}, MAE: {best_entry['mae']:.4f}")
    return best_name, best_entry


def save_model(
    model,
    feature_names: list[str],
    target_name: str,
    model_name: str,
    extra: dict | None = None,
) -> None:
    """
    Save the trained model and metadata (feature names, target name, model type).
    """
    model_path = MODELS_DIR / "best_model.joblib"
    metadata_path = MODELS_DIR / "model_metadata.json"

    joblib.dump(model, model_path)
    print(f"[MODEL] Saved best model to {model_path}")

    metadata: dict = {
        "model_name": model_name,
        "feature_names": feature_names,
        "target_name": target_name,
    }
    if extra:
        metadata.update(extra)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[MODEL] Saved metadata to {metadata_path}")


def main() -> None:
    # 1) Load data
    df = load_data()

    # 2) EDA and plots
    run_eda(df)

    # 3) Prepare features/target
    print_section("Preparing features and target")
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    print("X shape (n_samples, n_features):", X.shape)
    print("y shape (n_samples,):", y.shape)

    # 4) Train-test split
    print_section("Train-test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    # 5) Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # 6) Choose best model and save
    best_name, best_entry = choose_best_model(results)

    extra: dict | None = {}
    if "cv_best_params" in best_entry:
        extra["cv_best_params"] = best_entry["cv_best_params"]

    save_model(
        model=best_entry["model"],
        feature_names=FEATURE_COLS,
        target_name=TARGET_COL,
        model_name=best_name,
        extra=extra,
    )


if __name__ == "__main__":
    main()
