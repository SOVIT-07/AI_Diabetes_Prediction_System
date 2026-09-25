import pandas as pd

from src.predict import predict_diabetes


DATA_PATH = "data/diabetes_binary_health.csv"


def main():

    print("=" * 60)
    print("DIABETES PREDICTION MODULE TEST")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    df = df.drop_duplicates().reset_index(drop=True)

    sample = df.iloc[0].copy()

    input_data = sample.drop(
        labels=["Diabetes_binary"]
    ).to_dict()

    print("\nInput features:")

    for feature, value in input_data.items():
        print(
            f"{feature}: {value}"
        )

    result = predict_diabetes(
        input_data
    )

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(
        f"Prediction: {result['prediction']}"
    )

    print(
        f"Probability: "
        f"{result['probability'] * 100:.2f}%"
    )

    print(
        f"Threshold: "
        f"{result['threshold']}"
    )

    if result["prediction"] == 1:
        print(
            "\nResult: Diabetic"
        )
    else:
        print(
            "\nResult: Non-Diabetic"
        )

    print(
        "\nPrediction module test completed successfully."
    )


if __name__ == "__main__":
    main()