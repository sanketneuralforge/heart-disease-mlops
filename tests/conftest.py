"""
tests/conftest.py

Shared pytest fixtures for the heart disease MLOps test suite.
Makes project modules importable and loads artifacts using absolute paths,
so tests behave identically regardless of the directory pytest is invoked from.
"""

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Point the API at the model artifact using an absolute path BEFORE app.main
# is imported anywhere, so it doesn't depend on the current working directory.
os.environ["MODEL_PATH"] = str(PROJECT_ROOT / "models" / "final_model.joblib")


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def preprocessor():
    path = PROJECT_ROOT / "models" / "preprocessor.joblib"
    if not path.exists():
        pytest.skip(f"Preprocessor not found at {path}. Run the EDA notebook first.")
    return joblib.load(path)


@pytest.fixture(scope="session")
def final_model():
    path = PROJECT_ROOT / "models" / "final_model.joblib"
    if not path.exists():
        pytest.skip(f"Final model not found at {path}. Run src/train.py first.")
    return joblib.load(path)


@pytest.fixture(scope="session")
def numerical_cols():
    return ["age", "trestbps", "chol", "thalach", "oldpeak"]


@pytest.fixture(scope="session")
def categorical_cols():
    return ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


@pytest.fixture
def sample_raw_row():
    """A single valid, realistic patient record in raw (untransformed) form."""
    return pd.DataFrame([{
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }])


@pytest.fixture
def sample_raw_row_with_missing():
    """A record with missing ca/thal, mirroring real-world incomplete data."""
    return pd.DataFrame([{
        "age": 58, "sex": 0, "cp": 2, "trestbps": 130, "chol": 197,
        "fbs": 0, "restecg": 1, "thalach": 131, "exang": 0,
        "oldpeak": 0.6, "slope": 1, "ca": None, "thal": None,
    }])


@pytest.fixture
def api_client():
    """FastAPI TestClient wrapping the app, with the model already loaded."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def valid_api_payload():
    return {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }