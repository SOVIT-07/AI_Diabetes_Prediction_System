import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = "data/diabetes_binary_health.csv"
OUTPUT_DIR = "outputs"


def load_data():
    """Load and remove duplicate records."""
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def create_output_directory():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_target_distribution(df):
    """Plot diabetic vs non-diabetic distribution."""

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="Diabetes_binary"
    )

    plt.title("Diabetes Class Distribution")
    plt.xlabel("Diabetes Status")
    plt.ylabel("Number of Records")

    plt.xticks(
        [0, 1],
        ["Non-Diabetic", "Diabetic"]
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/diabetes_distribution.png",
        dpi=300
    )

    plt.close()


def plot_bmi_distribution(df):
    """Plot BMI distribution."""

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="BMI",
        bins=30,
        kde=True
    )

    plt.title("BMI Distribution")
    plt.xlabel("BMI")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/bmi_distribution.png",
        dpi=300
    )

    plt.close()


def plot_age_distribution(df):
    """Plot age category distribution."""

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Age"
    )

    plt.title("Age Category Distribution")
    plt.xlabel("Age Category")
    plt.ylabel("Number of Records")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/age_distribution.png",
        dpi=300
    )

    plt.close()


def plot_correlation_heatmap(df):
    """Generate correlation heatmap."""

    correlation = df.corr(numeric_only=True)

    plt.figure(
        figsize=(16, 13)
    )

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Feature Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/correlation_heatmap.png",
        dpi=300
    )

    plt.close()


def main():

    print("=" * 60)
    print("AI-BASED DIABETES PREDICTION SYSTEM")
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    create_output_directory()

    df = load_data()

    print("\nDataset shape after removing duplicates:")
    print(df.shape)

    print("\nGenerating visualizations...")

    plot_target_distribution(df)

    print("✓ Diabetes distribution")

    plot_bmi_distribution(df)

    print("✓ BMI distribution")

    plot_age_distribution(df)

    print("✓ Age distribution")

    plot_correlation_heatmap(df)

    print("✓ Correlation heatmap")

    print("\nEDA completed successfully.")

    print(f"\nCharts saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()