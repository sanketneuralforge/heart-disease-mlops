"""
src/data_download.py

MLOps Assignment 01 (AIMLCZG523) — Task 1: Data Acquisition.

Downloads and saves the Heart Disease UCI dataset via the official
ucimlrepo package (UCI Machine Learning Repository, id=45, Cleveland subset).

Usage:
    python src/data_download.py
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_heart_disease_data(output_dir="data"):
    os.makedirs(f"{output_dir}/raw", exist_ok=True)

    print("Fetching Heart Disease UCI dataset...")
    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Combine features and target into a single dataframe
    df = pd.concat([X, y], axis=1)

    raw_path = f"{output_dir}/raw/heart_disease_raw.csv"
    df.to_csv(raw_path, index=False)

    print(f"Saved raw dataset to {raw_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    download_heart_disease_data()
