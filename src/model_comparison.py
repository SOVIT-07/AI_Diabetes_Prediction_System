from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/diabetes_binary_health.csv")
OUTPUT_PATH = Path("outputs/model_comparison.csv")

TARGET = "Diabetes_binary"
RANDOM_STATE = 42

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Original records: {len(df):,}")

df = df.drop_duplicates().reset_index(drop=True)

print(f"Records after duplicate removal: {len(df):,}")

X = df.drop(columns=[TARGET])
y = df[TARGET]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    ),
}

results = []

print("\nStarting model comparison...\n")

for name, model in models.items():

    print(f"Training: {name}")

    # Logistic Regression benefits from scaled features.
    # Tree-based models do not require scaling, but using the
    # same transformed feature representation keeps the experiment
    # consistent for this comparison.
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_probability = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
        }
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("-" * 50)

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by="ROC-AUC",
    ascending=False,
).reset_index(drop=True)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

comparison_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\nModel Comparison Results")
print("=" * 80)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nComparison saved to:")
print(OUTPUT_PATH)