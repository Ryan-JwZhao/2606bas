@echo off
setlocal
cd /d "%~dp0"

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        py -3 -m venv .venv
    )
)

if not exist "%VENV_PY%" (
    echo Failed to create .venv. Please install Python 3.10+ and retry.
    pause
    exit /b 1
)

echo Installing runtime dependencies...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Runtime dependency installation failed.
    pause
    exit /b 1
)

echo Installing YOLO inference dependencies...
"%VENV_PY%" -m pip install -r requirements-yolo.txt
if errorlevel 1 (
    echo YOLO dependency installation failed.
    pause
    exit /b 1
)

"%VENV_PY%" -m bas doctor
pause
