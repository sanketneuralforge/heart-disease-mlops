"""
src/preprocess.py

MLOps Assignment 01 (AIMLCZG523) — non-interactive counterpart to the
Task 1 EDA notebook's cleaning/splitting/preprocessing steps.

The notebook (notebooks/01_eda_preprocessing.ipynb) is for human-readable
EDA and visualization. This script contains the same target conversion,
train/test split, and ColumnTransformer fitting logic, but as a plain
script — so it can run non-interactively in CI/CD, Docker, or from the
command line, without a Jupyter kernel.

Usage:
    python src/preprocess.py
"""

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
RAW_PATH = "data/raw/heart_disease_raw.csv"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

NUMERICAL_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def load_and_clean(raw_path=RAW_PATH):
    """Load the raw dataset and convert the 0-4 severity target to binary."""
    df = pd.read_csv(raw_path)

    if "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)
        df = df.drop(columns=["num"])

    return df


def build_preprocessor():
    """
    Builds the ColumnTransformer: median-impute + scale for numerical columns,
    most-frequent-impute + one-hot encode for categorical columns.
    Identical logic to the EDA notebook's Section 11.
    """
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numerical_transformer, NUMERICAL_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ])


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Loading raw data from {RAW_PATH} ...")
    df = load_and_clean()
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)  # fit ONLY on train data — no leakage

    X_train.assign(target=y_train.values).to_csv(
        os.path.join(PROCESSED_DIR, "train.csv"), index=False
    )
    X_test.assign(target=y_test.values).to_csv(
        os.path.join(PROCESSED_DIR, "test.csv"), index=False
    )
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))

    print("Saved:")
    print(f" - {PROCESSED_DIR}/train.csv")
    print(f" - {PROCESSED_DIR}/test.csv")
    print(f" - {MODELS_DIR}/preprocessor.joblib")


if __name__ == "__main__":
    main()