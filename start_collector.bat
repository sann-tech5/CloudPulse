@echo off
cd /d "C:\Users\s3777\CloudPulse"

if not exist "logs" mkdir "logs"

echo Waiting for Prometheus to become available...

:WAIT_PROMETHEUS
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://localhost:9090/-/ready' -TimeoutSec 3 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }"

if errorlevel 1 (
    timeout /t 10 /nobreak >nul
    goto WAIT_PROMETHEUS
)

echo Prometheus is ready!
echo Starting CloudPulse monitoring...

".venv\Scripts\python.exe" -u "src\data_collection\collect_continuously.py" >> "logs\collector.log" 2>&1
