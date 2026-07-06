"""
src/train.py

MLOps Assignment 01 (AIMLCZG523) — Task 2 & 3:
Feature Engineering & Model Development + Experiment Tracking (MLflow).

Trains at least two classifiers (Logistic Regression, Random Forest, XGBoost),
tunes each with cross-validated hyperparameter search, logs params/metrics/artifacts
to MLflow for every run, and saves the best full pipeline (preprocessor + model)
as a single reusable artifact for Task 4 (packaging) and the serving API.

Usage:
    python src/train.py
"""

import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    # Catches ImportError (package not installed) AND XGBoostError (native
    # libxgboost.dylib/.so fails to load, e.g. missing libomp on macOS).
    # On macOS this is usually fixed with: brew install libomp
    print(f"WARNING: XGBoost unavailable ({type(e).__name__}: {e}). "
          f"Continuing with Logistic Regression and Random Forest only.")
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42
DATA_DIR = "data/processed"
MODELS_DIR = "models"
REPORT_DIR = "report"
MLFLOW_EXPERIMENT_NAME = "heart-disease-classification"


def load_data():
    """Load processed train/test splits and the fitted preprocessor from Task 1."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))

    return X_train, X_test, y_train, y_test, preprocessor


def get_model_configs():
    """
    Defines each candidate model plus its hyperparameter grid for tuning.
    At least two classifiers are required by the assignment; three are included here
    (Logistic Regression, Random Forest, XGBoost) so the best can be chosen on evidence.
    """
    configs = {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "param_grid": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
            },
        },
    }

    if XGBOOST_AVAILABLE:
        configs["xgboost"] = {
            "estimator": XGBClassifier(
                random_state=RANDOM_STATE, eval_metric="logloss"
            ),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.01, 0.1],
            },
        }

    return configs


def evaluate_predictions(y_true, y_pred, y_proba):
    """Computes the assignment-required metrics: accuracy, precision, recall, F1, ROC-AUC."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def log_confusion_matrix(y_true, y_pred, model_name, run_dir):
    """Saves and logs a confusion matrix plot as an MLflow artifact."""
    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    path = os.path.join(run_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(path)


def log_roc_curve(y_true, y_proba, model_name, run_dir):
    """Saves and logs an ROC curve plot as an MLflow artifact."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})", color="#C44E52")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(run_dir, f"roc_curve_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(path)


def train_and_log_model(
    name, estimator, param_grid, preprocessor,
    X_train, y_train, X_test, y_test, run_dir,
):
    """
    Runs a GridSearchCV hyperparameter search inside a single MLflow run,
    logging params, cross-validated + test metrics, and diagnostic plots.
    Returns the fitted best pipeline and its test-set metrics.
    """
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name=name):
        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        best_params = search.best_params_
        cv_best_score = search.best_score_

        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_predictions(y_test, y_pred, y_proba)

        # --- MLflow logging ---
        mlflow.log_param("model_type", name)
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        mlflow.log_metric("cv_best_roc_auc", cv_best_score)
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        log_confusion_matrix(y_test, y_pred, name, run_dir)
        log_roc_curve(y_test, y_proba, name, run_dir)

        # MLflow 3.x defaults sklearn model logging to 'skops' serialization, which
        # rejects ColumnTransformer/OneHotEncoder internals as "untrusted types".
        # Pickle is explicitly listed as an acceptable format in the assignment FAQ,
        # so we pin to it here for a serialization that just works.
        mlflow.sklearn.log_model(
            best_pipeline, artifact_path="model", serialization_format="pickle"
        )

        print(f"\n[{name}] best CV ROC-AUC: {cv_best_score:.4f}")
        print(f"[{name}] best params: {best_params}")
        print(f"[{name}] test metrics: {json.dumps(test_metrics, indent=2)}")

    return best_pipeline, test_metrics


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # MLflow 3.x deprecated the raw filesystem store; use a local SQLite backend
    # instead (also lets you query runs with mlflow.search_runs()).
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test, preprocessor = load_data()

    configs = get_model_configs()
    results = {}
    fitted_pipelines = {}

    for name, cfg in configs.items():
        pipeline, metrics = train_and_log_model(
            name, cfg["estimator"], cfg["param_grid"], preprocessor,
            X_train, y_train, X_test, y_test, REPORT_DIR,
        )
        results[name] = metrics
        fitted_pipelines[name] = pipeline

    # --- Model comparison table ---
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values("roc_auc", ascending=False)
    comparison_path = os.path.join(REPORT_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path)

    print("\n=== Model Comparison (sorted by test ROC-AUC) ===")
    print(comparison_df)

    # --- Select and save the best model ---
    best_model_name = comparison_df.index[0]
    best_pipeline = fitted_pipelines[best_model_name]

    final_model_path = os.path.join(MODELS_DIR, "final_model.joblib")
    joblib.dump(best_pipeline, final_model_path)

    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump({
            "best_model": best_model_name,
            "test_metrics": results[best_model_name],
        }, f, indent=2)

    print(f"\nBest model: {best_model_name}")
    print(f"Saved full pipeline (preprocessor + model) to: {final_model_path}")
    print("This single artifact is what the FastAPI /predict endpoint will load.")


if __name__ == "__main__":
    main()