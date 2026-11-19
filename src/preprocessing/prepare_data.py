
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "metrics.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "prepared_metrics.csv"


def prepare_data():
    print("Starting CloudPulse data preprocessing...")

    df = pd.read_csv(INPUT_FILE)
    print(f"Original observations: {len(df)}")

    # Clean timestamps
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Convert measurements to numeric values
    numeric_columns = [
        "cpu_percent",
        "memory_bytes",
        "memory_mb"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(subset=numeric_columns)

    # Remove impossible measurements
    df = df[
        (df["cpu_percent"] >= 0) &
        (df["memory_bytes"] >= 0) &
        (df["memory_mb"] >= 0)
    ].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Start a new segment after every gap above 30 seconds
    gaps = df["timestamp"].diff().dt.total_seconds()

    df["segment_id"] = (gaps > 30).cumsum()

    # Save prepared data
    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nPreprocessing completed!")
    print(f"Final observations: {len(df)}")
    print(f"Monitoring segments: {df['segment_id'].nunique()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Saved dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    prepare_data()
