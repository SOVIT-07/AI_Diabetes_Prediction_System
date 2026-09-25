import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "Diabetes_binary"


def load_dataset(file_path):
    """
    Load the diabetes dataset from a CSV file.
    """
    return pd.read_csv(file_path)


def clean_dataset(df):
    """
    Clean the dataset by removing exact duplicate rows.

    The dataset contains no missing values, so no imputation
    is required at this stage.
    """
    df = df.copy()

    initial_rows = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    removed_duplicates = initial_rows - len(df)

    print(f"Initial records: {initial_rows}")
    print(f"Duplicates removed: {removed_duplicates}")
    print(f"Records after cleaning: {len(df)}")

    return df


def prepare_data(df):
    """
    Prepare features and target for machine learning.

    Returns:
        X_train_scaled
        X_test_scaled
        y_train
        y_test
        scaler
        feature_names
    """

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_names
    )