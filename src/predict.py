#!/usr/bin/env python

"""
predict.py

Use the trained model (saved by train_model.py) to predict a house price
for a single sample.

Usage example:

    python src/predict.py --values 8.3 20 6.5 1.0 500 2.5 37.86 -122.22

where the values correspond to:

    ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
     "Population", "AveOccup", "Latitude", "Longitude"]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def load_model_and_metadata():
    """Load the trained model and metadata from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            f"Run src/train_model.py first."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}. "
            f"Run src/train_model.py first."
        )

    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_names = metadata["feature_names"]
    target_name = metadata["target_name"]
    model_name = metadata["model_name"]

    return model, feature_names, target_name, model_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict house price using the trained model. "
            "Provide feature values in the correct order."
        )
    )
    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        required=True,
        help=(
            "Feature values in the correct order: "
            "[MedInc, HouseAge, AveRooms, AveBedrms, "
            "Population, AveOccup, Latitude, Longitude]"
        ),
    )
    return parser


def main() -> None:
    model, feature_names, target_name, model_name = load_model_and_metadata()

    parser = build_parser()
    args = parser.parse_args()

    values = args.values
    if len(values) != len(feature_names):
        raise ValueError(
            f"Expected {len(feature_names)} feature values, "
            f"but got {len(values)}."
        )

    data_dict = {name: [val] for name, val in zip(feature_names, values)}
    sample_df = pd.DataFrame(data_dict)

    prediction = model.predict(sample_df.values)[0]

    print("\nModel:", model_name)
    print("Features order:", feature_names)
    print("Input values:", values)
    print(f"\nPredicted {target_name}: {prediction:.4f} (units: 100,000 USD)")


if __name__ == "__main__":
    main()
