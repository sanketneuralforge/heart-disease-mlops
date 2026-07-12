"""
app/main.py

MLOps Assignment 01 (AIMLCZG523) — Task 6: Model Containerization (API layer).

FastAPI service exposing a /predict endpoint that loads the packaged
preprocessor+model pipeline (models/final_model.joblib, produced by
src/train.py) and returns a prediction plus confidence score for a
single patient record.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then visit http://localhost:8000/docs for interactive Swagger UI testing.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("heart-disease-api")

MODEL_PATH = os.getenv("MODEL_PATH", "models/final_model.joblib")

# Holds the loaded model pipeline; populated at startup, not per-request.
ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model once when the API starts, instead of on every request."""
    logger.info(f"Loading model from {MODEL_PATH} ...")
    try:
        ml_model["pipeline"] = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    ml_model.clear()
    logger.info("Model unloaded. Shutting down.")


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="MLOps Assignment 01 — predicts heart disease risk from patient data.",
    version="1.0.0",
    lifespan=lifespan,
)

# Exposes /metrics with request counts, latency histograms, status codes, etc.
# Prometheus scrapes this endpoint; see docker-compose.yml + prometheus.yml.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

class PatientData(BaseModel):
    """
    Raw (untransformed) patient features, matching the Heart Disease UCI schema.
    These are passed straight through the saved preprocessing pipeline —
    the client does not need to scale, encode, or impute anything.
    """
    age: float = Field(..., description="Age in years")
    sex: int = Field(..., description="1 = male, 0 = female")
    cp: int = Field(..., description="Chest pain type (0-3)")
    trestbps: float = Field(..., description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)")
    restecg: int = Field(..., description="Resting ECG results (0-2)")
    thalach: float = Field(..., description="Max heart rate achieved")
    exang: int = Field(..., description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., description="Slope of peak exercise ST segment (0-2)")
    ca: Optional[float] = Field(None, description="Number of major vessels colored by fluoroscopy (0-3)")
    thal: Optional[float] = Field(None, description="Thalassemia type (3=normal, 6=fixed defect, 7=reversible defect)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = no disease, 1 = disease present")
    prediction_label: str
    confidence: float = Field(..., description="Model's probability for the predicted class")
    probability_disease: float = Field(..., description="Raw probability of class 1 (disease present)")


@app.get("/", tags=["Health"])
def root():
    return {"message": "Heart Disease Risk Prediction API is running. See /docs for usage."}


@app.get("/health", tags=["Health"])
def health_check():
    """Used for Docker/Kubernetes liveness and readiness probes."""
    model_loaded = "pipeline" in ml_model
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientData):
    """
    Accepts a single patient's raw feature values as JSON, runs them through
    the saved preprocessing + model pipeline, and returns the predicted class
    plus a confidence score.
    """
    if "pipeline" not in ml_model:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    logger.info(f"Received prediction request: {patient.model_dump()}")

    try:
        input_df = pd.DataFrame([patient.model_dump()])
        pipeline = ml_model["pipeline"]

        pred = int(pipeline.predict(input_df)[0])
        proba = pipeline.predict_proba(input_df)[0]

        confidence = float(proba[pred])
        probability_disease = float(proba[1])

        response = PredictionResponse(
            prediction=pred,
            prediction_label="Disease Present" if pred == 1 else "No Disease",
            confidence=round(confidence, 4),
            probability_disease=round(probability_disease, 4),
        )

        logger.info(f"Prediction result: {response.model_dump()}")
        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")