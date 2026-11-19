
@echo off
cd /d "C:\Users\s3777\CloudPulse"

echo ========================================
echo       CLOUDPULSE MLOPS PIPELINE
echo ========================================

".venv\Scripts\python.exe" "src\run_pipeline.py"

echo.
echo ========================================
echo Pipeline execution finished.
echo ========================================
pause
