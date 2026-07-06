"""
Download and save the Heart Disease UCI dataset.
Source: UCI Machine Learning Repository (dataset id=45, Cleveland subset)
"""

import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

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