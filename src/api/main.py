
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

app = FastAPI(
    title="CloudPulse API",
    description="Intelligent Cloud Resource Monitoring and Prediction",
    version="3.1.0"
)


class ResourceMetrics(BaseModel):
    cpu_history: list[float] = Field(min_length=5, max_length=5)
    memory_history: list[float] = Field(min_length=5, max_length=5)


def load_models():
    cpu_path = MODEL_DIR / "cpu_model.pkl"
    memory_path = MODEL_DIR / "memory_model.pkl"

    if not cpu_path.exists() or not memory_path.exists():
        raise RuntimeError("Forecast models not found. Train the models first.")

    return (
        joblib.load(cpu_path),
        joblib.load(memory_path)
    )


cpu_model_data, memory_model_data = load_models()


FEATURES = [
    "cpu_lag_1",
    "cpu_lag_2",
    "cpu_lag_3",
    "cpu_lag_5",
    "memory_lag_1",
    "memory_lag_2",
    "memory_lag_3",
    "memory_lag_5",
    "cpu_mean_3",
    "memory_mean_3",
    "cpu_mean_5",
    "memory_mean_5"
]


def build_features(cpu_history, memory_history):
    data = {}

    for lag in [1, 2, 3, 5]:
        data[f"cpu_lag_{lag}"] = cpu_history[-lag]
        data[f"memory_lag_{lag}"] = memory_history[-lag]

    data["cpu_mean_3"] = sum(cpu_history[-3:]) / 3
    data["memory_mean_3"] = sum(memory_history[-3:]) / 3

    data["cpu_mean_5"] = sum(cpu_history) / 5
    data["memory_mean_5"] = sum(memory_history) / 5

    return pd.DataFrame([data], columns=FEATURES)


def make_prediction(model_data, input_data, history):

    strategy = model_data["strategy"]

    if strategy == "random_forest":
        prediction = model_data["model"].predict(input_data)[0]

    elif strategy == "previous_value":
        prediction = history[-1]

    elif strategy == "moving_average":
        prediction = sum(history) / len(history)

    else:
        raise ValueError(f"Unknown prediction strategy: {strategy}")

    return float(prediction)


@app.get("/")
def home():
    return {
        "message": "Welcome to CloudPulse API",
        "status": "Running"
    }


@app.get("/model-info")
def model_info():
    return {
        "cpu_prediction_strategy": cpu_model_data["strategy"],
        "memory_prediction_strategy": memory_model_data["strategy"],
        "forecast_horizon_observations": 5,
        "forecast_horizon_seconds": 50
    }


@app.post("/predict")
def predict(metrics: ResourceMetrics):

    if any(x < 0 or x > 100 for x in metrics.cpu_history):
        raise HTTPException(
            status_code=422,
            detail="CPU values must be between 0 and 100."
        )

    if any(x < 0 for x in metrics.memory_history):
        raise HTTPException(
            status_code=422,
            detail="Memory values cannot be negative."
        )

    input_data = build_features(
        metrics.cpu_history,
        metrics.memory_history
    )

    current_cpu = metrics.cpu_history[-1]
    current_memory = metrics.memory_history[-1]

    predicted_cpu = make_prediction(
        cpu_model_data,
        input_data,
        metrics.cpu_history
    )

    predicted_memory = make_prediction(
        memory_model_data,
        input_data,
        metrics.memory_history
    )

    # Keep predictions within physically valid ranges.
    predicted_cpu = max(0.0, min(100.0, predicted_cpu))
    predicted_memory = max(0.0, predicted_memory)

    return {
        "current_cpu_percent": round(current_cpu, 2),
        "predicted_cpu_percent": round(predicted_cpu, 2),
        "cpu_difference": round(predicted_cpu - current_cpu, 2),

        "current_memory_mb": round(current_memory, 2),
        "predicted_memory_mb": round(predicted_memory, 2),
        "memory_difference_mb": round(predicted_memory - current_memory, 2),

        "cpu_strategy": cpu_model_data["strategy"],
        "memory_strategy": memory_model_data["strategy"],

        "forecast_horizon_observations": 5,
        "forecast_horizon_seconds": 50
    }
