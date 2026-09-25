from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

DATA_PATH = Path("data/diabetes_binary_health.csv")
MODEL_PATH = Path("models/diabetes_model.joblib")
OUTPUT_DIR = Path("outputs")

TARGET = "Diabetes_binary"
RANDOM_STATE = 42
SAMPLE_SIZE = 500

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Remove exact duplicates
df = df.drop_duplicates().reset_index(drop=True)

print(f"Clean records: {len(df):,}")

print("Loading trained model...")

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
scaler = artifact["scaler"]
feature_names = artifact["feature_names"]

print("Model loaded successfully.")
print(f"Number of features: {len(feature_names)}")

X = df[feature_names].copy()

X = X.apply(pd.to_numeric)

# Apply the SAME scaler used during model training
X_scaled = scaler.transform(X)

sample_size = min(
    SAMPLE_SIZE,
    len(X_scaled),
)

X_sample_scaled = X_scaled[:sample_size]

# Keep readable feature names
X_sample = pd.DataFrame(
    X.iloc[:sample_size].values,
    columns=feature_names,
)

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_sample_scaled
)

print("SHAP values calculated successfully.")

if isinstance(shap_values, list):

    if len(shap_values) > 1:
        shap_class_1 = shap_values[1]
    else:
        shap_class_1 = shap_values[0]

else:

    if getattr(shap_values, "ndim", 2) == 3:

        # Binary classification:
        # samples × features × classes
        shap_class_1 = shap_values[:, :, 1]

    else:

        shap_class_1 = shap_values

mean_abs_shap = abs(shap_class_1).mean(axis=0)

shap_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_abs_shap,
    }
)

shap_importance = shap_importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False,
).reset_index(drop=True)

importance_csv = (
    OUTPUT_DIR / "shap_feature_importance.csv"
)

shap_importance.to_csv(
    importance_csv,
    index=False,
)

print("\nSHAP feature importance saved to:")
print(importance_csv)

plt.figure(figsize=(10, 8))

top_features = shap_importance.head(15)

plt.barh(
    top_features["Feature"][::-1],
    top_features["Mean_Absolute_SHAP"][::-1],
)

plt.xlabel("Mean Absolute SHAP Value")

plt.ylabel("Feature")

plt.title(
    "Top 15 Features by SHAP Importance"
)

plt.tight_layout()

importance_plot = (
    OUTPUT_DIR / "shap_feature_importance.png"
)

plt.savefig(
    importance_plot,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("SHAP feature importance plot saved to:")
print(importance_plot)

print("\nGenerating SHAP summary plot...")

summary_plot = (
    OUTPUT_DIR / "shap_summary.png"
)

plt.figure()

shap.summary_plot(
    shap_class_1,
    X_sample,
    show=False,
    max_display=15,
)

plt.tight_layout()

plt.savefig(
    summary_plot,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("SHAP summary plot saved to:")
print(summary_plot)


print("\nTop SHAP Features")
print("=" * 60)

print(
    shap_importance.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\n" + "=" * 60)
print("Explainable AI analysis completed successfully.")
print("=" * 60)