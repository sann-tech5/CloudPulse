# CloudPulse – Intelligent Cloud Resource Monitoring & Optimization Platform

CloudPulse is an MLOps-based cloud resource monitoring and prediction project. It collects CPU and memory usage metrics, stores them for analysis, trains machine learning models, and provides a dashboard and API to monitor and predict resource usage.

## Features

- **Resource Monitoring:** Collects CPU and memory metrics using Prometheus and cAdvisor.
- **Automated Data Collection:** Saves resource metrics to a CSV file at regular intervals.
- **Data Preprocessing:** Cleans and prepares collected data for machine learning.
- **Machine Learning:** Uses Random Forest regression models to predict CPU and memory usage.
- **Model Tracking:** Uses MLflow to track experiments, parameters, metrics, and model artifacts.
- **Model Validation:** Compares new model performance with previously saved models before replacing them.
- **Prediction API:** Provides predictions through a FastAPI endpoint.
- **Interactive Dashboard:** Displays resource analytics, predictions, system health, and model performance using Streamlit.
- **Health Alerts:** Identifies high resource usage and provides recommendations.
- **Pipeline Automation:** Runs preprocessing and model training through a single pipeline command.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Containerization | Docker |
| Container Metrics | cAdvisor |
| Metrics Collection | Prometheus |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Experiment Tracking | MLflow |
| Backend API | FastAPI |
| Dashboard | Streamlit, Plotly |
| Model Storage | Joblib |
| Data Storage | CSV |
| API Server | Uvicorn |

## Project Architecture

1. Docker runs Prometheus and cAdvisor.
2. cAdvisor exposes container resource metrics.
3. Prometheus collects and stores those metrics.
4. The Python collector retrieves CPU and memory metrics from Prometheus.
5. Metrics are stored in `data/raw/metrics.csv`.
6. The preprocessing module cleans the collected data.
7. Random Forest models are trained using the prepared dataset.
8. MLflow tracks model experiments and performance.
9. The FastAPI service provides resource predictions.
10. The Streamlit dashboard displays monitoring data, predictions, and system health.

## Project Structure

```text
CloudPulse/
│
├── .github/
│   └── workflows/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── grafana/
│   └── provisioning/
│
├── logs/
│
├── models/
│
├── prometheus/
│   └── prometheus.yml
│
├── src/
│   ├── api/
│   ├── data_collection/
│   ├── models/
│   ├── monitoring/
│   ├── preprocessing/
│   ├── recommendations/
│   └── run_pipeline.py
│
├── .gitignore
├── docker-compose.yml
├── requirements.txt
├── start_collector.bat
├── run_cloudpulse_pipeline.bat
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd CloudPulse
```

### 2. Create a virtual environment

```cmd
python -m venv .venv
```

### 3. Activate the environment (Windows)

```cmd
.venv\Scripts\activate
```

### 4. Install dependencies

```cmd
pip install -r requirements.txt
```

### 5. Start Docker services

Make sure Docker Desktop is running, then execute:

```cmd
docker compose up -d
```

This starts Prometheus and cAdvisor.

### 6. Start resource monitoring

```cmd
python src\data_collection\collect_continuously.py
```

The collector retrieves metrics every 10 seconds and saves them to the raw CSV dataset.

### 7. Run the complete ML pipeline

Open another terminal and execute:

```cmd
python src\run_pipeline.py
```

This runs data preprocessing followed by model training and validation.

Alternatively, double-click:

```text
run_cloudpulse_pipeline.bat
```

### 8. Start the prediction API

```cmd
uvicorn src.api.main:app --reload
```

API documentation:

http://localhost:8000/docs

### 9. Start the Streamlit dashboard

Open another terminal:

```cmd
streamlit run dashboard/app.py
```

The dashboard will normally be available at:

http://localhost:8501

## Service URLs

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| FastAPI Documentation | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| cAdvisor | http://localhost:8080 |
| MLflow Tracking UI | http://localhost:5000 |

To view MLflow experiments in a browser, start its UI separately:

```cmd
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Then open http://localhost:5000.

## Machine Learning

CloudPulse uses Random Forest Regressors to predict future CPU and memory usage based on previous resource measurements.

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- R-squared (R²) score

MLflow records experiment parameters, evaluation metrics, and model artifacts.

Model performance depends on the amount, quality, and variation of collected data. Predictions should be treated as estimates, not guaranteed future resource usage.

## Health Monitoring

CloudPulse checks resource usage against configured thresholds and reports:

- Healthy status
- Warning status
- Critical status

It also provides recommendations when CPU or memory usage becomes high.

## Important Notes and Limitations

- This project is configured for a Windows development environment using Docker Desktop.
- cAdvisor metrics collected through Docker Desktop represent the monitored Linux container environment, not necessarily the physical Windows host's CPU and memory usage.
- The machine learning models require sufficient historical observations for meaningful evaluation.
- AWS deployment and production cloud infrastructure are possible future extensions; they are not part of the current local setup.

## Future Improvements

- Deploy CloudPulse on AWS EC2.
- Add cloud-host monitoring and resource optimization.
- Improve prediction accuracy with larger datasets and additional features.
- Add more advanced anomaly detection.
- Introduce automated model retraining and deployment.
- Add Grafana dashboards and CI/CD workflows.

## Project Objective

The objective of CloudPulse is to demonstrate how cloud monitoring, data engineering, machine learning, model tracking, APIs, and interactive dashboards can work together in an end-to-end MLOps pipeline.

---

**CloudPulse – Monitor. Predict. Optimize.**
