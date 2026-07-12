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

## Running the Full Project (Step by Step)

This mirrors exactly how the project is run end-to-end, including the monitoring stack and Kubernetes deployment. Each numbered step assumes its own terminal tab, since several stay running simultaneously.

### 1. Clone & set up the environment

```bash
git clone https://github.com/sanketneuralforge/heart-disease-mlops.git
cd heart-disease-mlops
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the pipeline fresh

```bash
python src/data_download.py     # downloads UCI Heart Disease dataset
python src/preprocess.py        # cleans, splits, fits the preprocessing pipeline
python src/train.py             # trains + tunes 3 models, logs everything to MLflow
pytest tests/ -v                # should end in "20 passed"
```

### 3. View experiment tracking (new terminal, leave running)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
Open **http://localhost:5001** → `heart-disease-classification` experiment → compare the 3 runs → click into `logistic_regression` (the winner) to see its metrics and logged confusion matrix / ROC curve.

### 4. Run the API + monitoring stack (new terminal, leave running)

```bash
docker ps                       # check ports 8000/9090/3000 are free
docker-compose up -d
docker ps                       # confirm all 3 containers are Up, api is healthy
```
Open:
- **http://localhost:8000/docs** — Swagger UI, try `/predict` directly
- **http://localhost:9090/targets** — Prometheus, confirm `heart-disease-api` is UP
- **http://localhost:3000** — Grafana (login `admin`/`admin`), open the saved dashboard

### 5. Deploy to Kubernetes (new terminal)

```bash
kubectl get nodes                              # confirm cluster is up
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods                               # wait for both Running, 1/1
kubectl get svc heart-disease-api-service
```

Test it through the service (not the direct container port):
```bash
curl -X POST http://localhost/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

### 6. Generate traffic to see monitoring live (new terminal)

```bash
for i in {1..30}; do
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}' > /dev/null
sleep 0.5
done
```
Watch the Grafana dashboard (step 4) update in real time.

### 7. Final sanity check

```bash
curl -s http://localhost:8000/health           # API healthy
curl -s http://localhost:9090/-/healthy         # Prometheus healthy
kubectl get pods --no-headers | grep -c Running # should print 2
```

### Shutting everything down

```bash
docker-compose down
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml
deactivate
```

---

## Individual Commands (Reference)

<details>
<summary>Run just the API locally (no Docker)</summary>

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
</details>

<details>
<summary>Build the Docker image manually</summary>

```bash
docker build -t heart-disease-api:latest .
docker run -d -p 8000:8000 heart-disease-api:latest
```
</details>

## Notes on Reproducibility

The preprocessing pipeline (impute → scale → one-hot encode) and the winning model are bundled into a single artifact, `models/final_model.joblib` — one load reproduces the entire inference path. Dependency versions are pinned in `requirements.txt`; install from that file rather than ad hoc `pip install` commands, since an unpinned upgrade mid-project once broke a serialized artifact during development (details in the full report).

---

**Sanket Panchalwar** · BITS Pilani, MTech (AI/ML) · [Full report in this repo](https://docs.google.com/document/d/1unvI48AUX2bxj8PPEuII5EzmeeY87URB/edit)