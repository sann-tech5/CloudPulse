
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import joblib


# PROJECT CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

cpu_model_data = joblib.load(MODEL_DIR / "cpu_model.pkl")
memory_model_data = joblib.load(MODEL_DIR / "memory_model.pkl")


app = FastAPI(
    title="CloudPulse API",
    description="Intelligent Cloud Resource Monitoring and Prediction",
    version="2.0.0"
)


# INPUT VALIDATION

class ResourceMetrics(BaseModel):
    cpu_percent: float = Field(ge=0, le=100)
    memory_mb: float = Field(ge=0)


# PREDICTION FUNCTION

def make_prediction(model_data, input_data, previous_value):

    strategy = model_data["strategy"]

    if strategy == "random_forest":
        model = model_data["model"]
        prediction = model.predict(input_data)[0]

    elif strategy == "previous_value":
        prediction = previous_value

    else:
        raise ValueError(f"Unknown prediction strategy: {strategy}")

    return float(prediction)


# API ENDPOINTS

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
        "memory_prediction_strategy": memory_model_data["strategy"]
    }


@app.post("/predict")
def predict(metrics: ResourceMetrics):

    input_data = [[
        metrics.cpu_percent,
        metrics.memory_mb
    ]]

    predicted_cpu = make_prediction(
        cpu_model_data,
        input_data,
        metrics.cpu_percent
    )

    predicted_memory = make_prediction(
        memory_model_data,
        input_data,
        metrics.memory_mb
    )

    return {
        "current_cpu_percent": metrics.cpu_percent,
        "predicted_cpu_percent": round(predicted_cpu, 2),
        "current_memory_mb": metrics.memory_mb,
        "predicted_memory_mb": round(predicted_memory, 2),
        "cpu_strategy": cpu_model_data["strategy"],
        "memory_strategy": memory_model_data["strategy"]
    }
