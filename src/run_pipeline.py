
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

STEPS = [
    ("Data Preprocessing", "src/preprocessing/prepare_data.py"),
    ("Model Training", "src/models/train_model.py"),
]

def run_pipeline():
    print("\n========== CLOUDPULSE MLOPS PIPELINE ==========\n")

    for step_name, script_path in STEPS:
        print(f"\nStarting: {step_name}")

        result = subprocess.run(
            [PYTHON, str(PROJECT_ROOT / script_path)],
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            print(f"\nPipeline failed during: {step_name}")
            sys.exit(result.returncode)

        print(f"\nCompleted: {step_name}")

    print("\n========== PIPELINE COMPLETED SUCCESSFULLY ==========")

if __name__ == "__main__":
    run_pipeline()
