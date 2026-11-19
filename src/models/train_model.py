
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# PROJECT CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "prepared_metrics.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
mlflow.set_experiment("CloudPulse-Resource-Prediction")


# FORECAST CONFIGURATION

HORIZON = 5
MOVING_AVERAGE_WINDOW = 5

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


# FEATURE ENGINEERING
# All calculations remain within individual monitoring segments.

def create_features(df):

    df = df.copy()

    if "segment_id" not in df.columns:
        raise ValueError(
            "segment_id missing. Run preprocessing first."
        )

    groups = df.groupby("segment_id", sort=False)

    # Lag features
    for lag in [1, 2, 3, 5]:

        df[f"cpu_lag_{lag}"] = (
            groups["cpu_percent"].shift(lag)
        )

        df[f"memory_lag_{lag}"] = (
            groups["memory_mb"].shift(lag)
        )

    # Rolling averages
    for window in [3, 5]:

        df[f"cpu_mean_{window}"] = (
            df.groupby("segment_id")["cpu_percent"]
            .transform(
                lambda s: s.rolling(window).mean().shift(1)
            )
        )

        df[f"memory_mean_{window}"] = (
            df.groupby("segment_id")["memory_mb"]
            .transform(
                lambda s: s.rolling(window).mean().shift(1)
            )
        )

    # Future targets within the same segment
    df["target_cpu"] = (
        groups["cpu_percent"].shift(-HORIZON)
    )

    df["target_memory"] = (
        groups["memory_mb"].shift(-HORIZON)
    )

    # Remove incomplete forecasting observations
    df = df.dropna(
        subset=FEATURES + [
            "target_cpu",
            "target_memory"
        ]
    )

    return df.reset_index(drop=True)


# MODEL TRAINING AND STRATEGY SELECTION

def train_target(
    X_train,
    X_test,
    y_train,
    y_test,
    previous_baseline,
    moving_average_baseline,
    path,
    name
):

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Evaluate Random Forest
    rf_mae = mean_absolute_error(
        y_test, predictions
    )

    # Evaluate previous-value baseline
    previous_mae = mean_absolute_error(
        y_test, previous_baseline
    )

    # Evaluate moving-average baseline
    moving_average_mae = mean_absolute_error(
        y_test, moving_average_baseline
    )

    # Compare all strategies
    strategies = {
        "random_forest": rf_mae,
        "previous_value": previous_mae,
        "moving_average": moving_average_mae
    }

    selected_strategy = min(
        strategies,
        key=strategies.get
    )

    selected_model = (
        model
        if selected_strategy == "random_forest"
        else None
    )

    model_r2 = r2_score(y_test, predictions)

    # Save selected strategy
    joblib.dump(
        {
            "strategy": selected_strategy,
            "model": selected_model,
            "features": FEATURES,
            "horizon": HORIZON,
            "model_mae": float(rf_mae),
            "baseline_mae": float(previous_mae),
            "moving_average_mae": float(moving_average_mae)
        },
        path
    )

    print(f"\n{name} RESULTS")

    print("Random Forest MAE:", round(rf_mae, 4))
    print("Previous-value MAE:", round(previous_mae, 4))
    print("Moving-average MAE:", round(moving_average_mae, 4))
    print("Random Forest R2:", round(model_r2, 4))

    print("Selected strategy:", selected_strategy)

    return {
        "model": model,
        "model_mae": rf_mae,
        "baseline_mae": previous_mae,
        "moving_average_mae": moving_average_mae,
        "r2": model_r2,
        "strategy": selected_strategy
    }


# MAIN TRAINING PIPELINE

def train_model():

    print("\n========== CLOUDPULSE FORECAST TRAINING ==========")

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(
            "Not enough observations for forecasting."
        )

    print("Raw observations:", len(df))
    print("Monitoring segments:", df["segment_id"].nunique())

    # Generate segment-aware features
    df = create_features(df)

    if len(df) < 50:
        raise ValueError(
            "Not enough valid forecasting observations."
        )

    X = df[FEATURES]

    y_cpu = df["target_cpu"]
    y_memory = df["target_memory"]

    # Chronological train-test split
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_cpu_train = y_cpu.iloc[:split_index]
    y_cpu_test = y_cpu.iloc[split_index:]

    y_memory_train = y_memory.iloc[:split_index]
    y_memory_test = y_memory.iloc[split_index:]

    # Previous-value baselines
    cpu_previous = df["cpu_lag_1"].iloc[split_index:]
    memory_previous = df["memory_lag_1"].iloc[split_index:]

    # Five-observation moving-average baselines
    cpu_moving_average = df["cpu_mean_5"].iloc[split_index:]
    memory_moving_average = df["memory_mean_5"].iloc[split_index:]

    print("Valid forecasting observations:", len(df))
    print("Training observations:", len(X_train))
    print("Testing observations:", len(X_test))
    print("Forecast horizon:", HORIZON, "observations")

    with mlflow.start_run(
        run_name="CloudPulse_Three_Strategy_Forecasting"
    ):

        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": 200,
            "max_depth": 8,
            "forecast_horizon": HORIZON,
            "train_test_split": 0.8,
            "selection_method": "Lowest MAE",
            "feature_engineering": "segment_aware",
            "monitoring_gap_seconds": 30,
            "moving_average_window": MOVING_AVERAGE_WINDOW
        })

        # CPU MODEL
        cpu_results = train_target(
            X_train,
            X_test,
            y_cpu_train,
            y_cpu_test,
            cpu_previous,
            cpu_moving_average,
            MODEL_DIR / "cpu_model.pkl",
            "CPU"
        )

        # MEMORY MODEL
        memory_results = train_target(
            X_train,
            X_test,
            y_memory_train,
            y_memory_test,
            memory_previous,
            memory_moving_average,
            MODEL_DIR / "memory_model.pkl",
            "Memory"
        )

        # LOG EVALUATION METRICS
        mlflow.log_metrics({

            "cpu_random_forest_mae": cpu_results["model_mae"],
            "cpu_previous_value_mae": cpu_results["baseline_mae"],
            "cpu_moving_average_mae": cpu_results["moving_average_mae"],
            "cpu_random_forest_r2": cpu_results["r2"],

            "memory_random_forest_mae": memory_results["model_mae"],
            "memory_previous_value_mae": memory_results["baseline_mae"],
            "memory_moving_average_mae": memory_results["moving_average_mae"],
            "memory_random_forest_r2": memory_results["r2"]

        })

        mlflow.log_params({
            "cpu_selected_strategy": cpu_results["strategy"],
            "memory_selected_strategy": memory_results["strategy"]
        })

        # Log Random Forest candidate models
        mlflow.sklearn.log_model(
            cpu_results["model"],
            name="cpu_random_forest_candidate",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        mlflow.sklearn.log_model(
            memory_results["model"],
            name="memory_random_forest_candidate",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print("\nMLflow run:", mlflow.active_run().info.run_id)

    print("\n========== TRAINING COMPLETED ==========")

    print("CPU strategy:", cpu_results["strategy"])
    print("Memory strategy:", memory_results["strategy"])

    print("Saved models in:", MODEL_DIR)


if __name__ == "__main__":
    train_model()
