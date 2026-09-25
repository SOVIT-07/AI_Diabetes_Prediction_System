from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


OUTPUT_DIR = Path("outputs")


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained classification model.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
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
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return (
        metrics,
        predictions,
        probabilities
    )


def save_confusion_matrix(
    y_test,
    predictions
):
    """
    Save confusion matrix visualization.
    """

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Non-Diabetic",
            "Diabetic"
        ]
    )

    display.plot()

    plt.title(
        "Diabetes Prediction Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()


def save_roc_curve(
    model,
    X_test,
    y_test
):
    """
    Save ROC curve.
    """

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.title(
        "Diabetes Prediction ROC Curve"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "roc_curve.png",
        dpi=300
    )

    plt.close()


def save_classification_report(
    y_test,
    predictions
):
    """
    Save classification report as text.
    """

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Non-Diabetic",
            "Diabetic"
        ],
        zero_division=0
    )

    report_path = (
        OUTPUT_DIR /
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    return report