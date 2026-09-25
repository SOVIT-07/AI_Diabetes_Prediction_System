import joblib
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train, y_train):
    """
    Train the Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def save_model(
    model,
    scaler,
    feature_names,
    threshold=0.25,
    model_path="models/diabetes_model.joblib"
):
    """
    Save model, scaler and feature names together.
    """

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "threshold": threshold
    }

    joblib.dump(
        artifact,
        model_path
    )

    print(f"\nModel saved to: {model_path}")

    print(f"Prediction threshold: {threshold}")

def get_feature_importance(model, feature_names):
    """
    Return Random Forest feature importance values.
    """

    import pandas as pd

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="importance",
        ascending=False
    )

    return importance 