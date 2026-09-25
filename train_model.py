from pathlib import Path

from src.data_preprocessing import (
    load_dataset,
    clean_dataset,
    prepare_data
)

from src.train import (
    train_random_forest,
    save_model,
    get_feature_importance
)

from src.evaluate import (
    evaluate_model,
    save_confusion_matrix,
    save_roc_curve,
    save_classification_report
)


DATA_PATH = Path(
    "data/diabetes_binary_health.csv"
)


def main():

    print("=" * 60)
    print("AI-BASED DIABETES PREDICTION SYSTEM")
    print("RANDOM FOREST TRAINING")
    print("=" * 60)

    print("\nLoading dataset...")

    df = load_dataset(
        DATA_PATH
    )

    print(
        f"Original dataset: {df.shape}"
    )

    print("\nCleaning dataset...")

    df = clean_dataset(
        df
    )

    print(
        f"Cleaned dataset: {df.shape}"
    )

    print("\nPreparing training data...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_names
    ) = prepare_data(
        df
    )

    print(
        f"Training data: {X_train.shape}"
    )

    print(
        f"Testing data:  {X_test.shape}"
    )

    print("\nTraining Random Forest...")

    model = train_random_forest(
        X_train,
        y_train
    )

    print(
        "Random Forest training completed."
    )

    print("\nEvaluating model...")

    (
        metrics,
        predictions,
        probabilities
    ) = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1-Score : {metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC  : {metrics['roc_auc']:.4f}"
    )

    print("\nGenerating evaluation plots...")

    save_confusion_matrix(
        y_test,
        predictions
    )

    print(
        "✓ Confusion matrix saved"
    )

    save_roc_curve(
        model,
        X_test,
        y_test
    )

    print(
        "✓ ROC curve saved"
    )

    report = save_classification_report(
        y_test,
        predictions
    )

    print(
        "✓ Classification report saved"
    )

    save_model(
        model,
        scaler,
        feature_names,
        threshold=0.25
    )

    print("\nCalculating feature importance...")

    importance = get_feature_importance(
        model,
        feature_names
    )

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    print(
        importance.to_string(
            index=False
        )
    )

    importance.to_csv(
        "outputs/feature_importance.csv",
        index=False
    )

    print(
        "\n✓ Feature importance saved to "
        "outputs/feature_importance.csv"
)

    print("\n" + "=" * 60)
    print(
        "MODEL TRAINING PIPELINE COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()