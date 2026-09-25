from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.data_preprocessing import (
    load_dataset,
    clean_dataset,
    prepare_data
)


DATA_PATH = Path(
    "data/diabetes_binary_health.csv"
)

MODEL_PATH = Path(
    "models/diabetes_model.joblib"
)

def main():

    print("=" * 70)
    print("DIABETES PREDICTION THRESHOLD ANALYSIS")
    print("=" * 70)

    df = load_dataset(DATA_PATH)

    df = clean_dataset(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = prepare_data(df)

    artifact = joblib.load(
        MODEL_PATH
    )

    model = artifact["model"]

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    thresholds = [
        0.50,
        0.45,
        0.40,
        0.35,
        0.30,
        0.25,
        0.20
    ]

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append({
            "threshold": threshold,

            "accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "f1_score": f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        })

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    output_path = Path(
        "outputs/threshold_analysis.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\n✓ Results saved to: {output_path}"
    )


if __name__ == "__main__":
    main()