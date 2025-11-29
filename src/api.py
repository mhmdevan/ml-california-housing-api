#!/usr/bin/env python

"""
api.py

FastAPI application that exposes a /predict endpoint
for California housing price prediction.

Run:

    uvicorn src.api:app --reload

Swagger / OpenAPI docs:

    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


# ---------------------------------------------------------------------------
# FastAPI app with OpenAPI / Swagger metadata
# ---------------------------------------------------------------------------

app = FastAPI(
    title="California Housing Price Prediction API",
    description=(
        "Predict median house value (MedHouseVal) for California districts "
        "using a trained ML model (RandomForest or LinearRegression).\n\n"
        "The model is trained offline via src/train_model.py and loaded here "
        "for real-time HTTP prediction.\n\n"
        "Request body is a JSON with 8 numeric features; "
        "response includes the predicted MedHouseVal (units: 100,000 USD)."
    ),
    version="1.0.0",
    docs_url="/docs",    # Swagger UI
    redoc_url="/redoc",  # ReDoc UI
)


# ---------------------------------------------------------------------------
# Globals for model & metadata
# ---------------------------------------------------------------------------

model: object | None = None
feature_names: list[str] = []
target_name: str = "MedHouseVal"
model_name: str = "unknown"


def load_model_and_metadata() -> None:
    """
    Load the trained model and metadata into global variables.

    This is called once on startup (and lazily if needed).
    """
    global model, feature_names, target_name, model_name

    if model is not None:
        # already loaded
        return

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            "Model or metadata not found. "
            "Make sure you have run src/train_model.py first."
        )

    loaded_model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Basic validation
    fn = metadata.get("feature_names")
    tn = metadata.get("target_name")
    mn = metadata.get("model_name")

    if not isinstance(fn, list):
        raise ValueError("feature_names in metadata must be a list of strings.")

    model = loaded_model
    feature_names.clear()
    feature_names.extend(fn)
    target_name = str(tn)
    model_name = str(mn)


# ---------------------------------------------------------------------------
# Pydantic models for Swagger / OpenAPI
# ---------------------------------------------------------------------------

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


class PredictionResponse(BaseModel):
    model: str
    target_name: str
    prediction: float
    input: HouseFeatures


# ---------------------------------------------------------------------------
# Startup event – load model once
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup_event() -> None:
    """Load model once when the API starts."""
    load_model_and_metadata()


# ---------------------------------------------------------------------------
# Root endpoint (healthcheck)
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return {"status": "ok", "message": "California Housing Price Prediction API"}


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Predict California house price",
    description=(
        "Predict the median house value (`MedHouseVal`) for a single district.\n\n"
        "**Input:** 8 numeric features matching the training schema.\n\n"
        "**Output:** Predicted `MedHouseVal` in units of 100,000 USD, "
        "plus basic model metadata."
    ),
)
def predict_price(features: HouseFeatures) -> PredictionResponse:
    """
    Predict house price for a single sample.

    Input: JSON with all 8 features.
    Output: JSON with predicted price and model info.
    """
    # In case startup event didn't fire for some reason (tests, etc.)
    if model is None:
        load_model_and_metadata()

    data_dict = features.dict()

    # Keep feature order consistent with training
    row = [data_dict[name] for name in feature_names]
    sample_df = pd.DataFrame([row], columns=feature_names)

    pred = float(model.predict(sample_df.values)[0])  # type: ignore[call-arg]

    return PredictionResponse(
        model=model_name,
        target_name=target_name,
        prediction=pred,
        input=features,
    )
