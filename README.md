# Heart Disease Risk Prediction — MLOps Pipeline

End-to-end MLOps pipeline that predicts heart disease presence from patient health data — built for AIMLCZG523 (MLOps), Assignment 01, BITS Pilani.

Data → preprocessing → model training with experiment tracking → CI/CD → containerized API → Kubernetes deployment → live monitoring.

## Results

Three classifiers were trained and tuned (5-fold CV + GridSearchCV). Logistic Regression won on test ROC-AUC despite being the simplest model — with 303 records and a largely linear feature-target relationship, the extra flexibility of tree-based models didn't pay off.

| Model | Test Accuracy | Test Recall | Test ROC-AUC |
|---|---|---|---|
| **Logistic Regression** | 0.885 | 0.929 | **0.966** |
| Random Forest | 0.869 | 0.893 | 0.943 |
| XGBoost | 0.869 | 0.857 | 0.920 |

## Stack

Python 3.12 · scikit-learn · XGBoost · MLflow · FastAPI · Pytest · Docker · Kubernetes · Prometheus · Grafana · GitHub Actions

## Project Structure

```
heart-disease-mlops/
├── notebooks/01_eda_preprocessing.ipynb   # EDA + preprocessing (exploration)
├── src/
│   ├── data_download.py                   # pulls UCI Heart Disease dataset
│   ├── preprocess.py                      # cleaning, split, ColumnTransformer (CI-runnable)
│   └── train.py                           # trains + tunes 3 models, logs to MLflow
├── app/main.py                            # FastAPI serving layer
├── tests/                                 # 20 pytest tests (data, model, API)
├── models/                                # final_model.joblib, preprocessor.joblib
├── k8s/                                   # deployment.yaml, service.yaml
├── .github/workflows/ci.yml               # lint -> download -> preprocess -> train -> test
├── Dockerfile, docker-compose.yml, prometheus.yml
├── report/                                # EDA plots, model comparison CSV
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/sanketneuralforge/heart-disease-mlops.git
cd heart-disease-mlops
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python src/data_download.py
python src/preprocess.py
python src/train.py
pytest tests/ -v
```

## Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

## Experiment Tracking

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

## Docker

```bash
docker build -t heart-disease-api:latest .
docker run -d -p 8000:8000 heart-disease-api:latest
```

## Monitoring Stack (API + Prometheus + Grafana)

```bash
docker-compose up -d
# API: localhost:8000  Prometheus: localhost:9090  Grafana: localhost:3000
```

## Kubernetes Deployment

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl get svc heart-disease-api-service
```

## Notes on Reproducibility

The preprocessing pipeline (impute → scale → one-hot encode) and the winning model are bundled into a single artifact, `models/final_model.joblib` — one load reproduces the entire inference path. Dependency versions are pinned in `requirements.txt`; install from that file rather than ad hoc `pip install` commands, since an unpinned upgrade mid-project once broke a serialized artifact during development (details in the full report).

---

**Sanket Panchalwar** · BITS Pilani, MTech (AI/ML) · [Full report in this repo](./MLOps_Assignment01_Report.docx)