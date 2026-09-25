from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(
    "models/diabetes_model.joblib"
)


def load_model():
    """
    Load the trained model artifact.
    """
    return joblib.load(
        MODEL_PATH
    )


def predict_diabetes(input_data):
    """
    Generate a diabetes prediction from patient features.
    input_data should contain the same feature names
    used during model training.
    """

    artifact = load_model()

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]
    threshold = artifact["threshold"]

    data = pd.DataFrame(
        [input_data]
    )

    data = data.reindex(
        columns=feature_names
    )

    data = data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if data.isnull().any().any():
        raise ValueError(
            "Input contains missing or invalid values."
        )

    scaled_data = scaler.transform(
        data
    )

    probability = float(
        model.predict_proba(
            scaled_data
        )[0][1]
    )

    # Apply selected threshold
    prediction = int(
        probability >= threshold
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold
    }