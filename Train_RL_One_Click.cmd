@echo off
setlocal
cd /d "%~dp0"

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "SAMPLES_DIR=%CD%\rl\data\samples"
set "MODEL_PATH=%CD%\rl\models\ranker.json"

if not exist "%VENV_PY%" (
    echo Missing virtual environment: %VENV_PY%
    echo Run Setup_Environment.cmd once to repair the local .venv.
    pause
    exit /b 1
)

"%VENV_PY%" -c "import torch, numpy, yaml; import rl.train" >nul 2>nul
if errorlevel 1 (
    echo RL training dependency check failed. Run Setup_Environment.cmd once to repair the local .venv.
    pause
    exit /b 1
)

echo Inspecting learning samples...
"%VENV_PY%" -m rl.inspect_samples --samples "%SAMPLES_DIR%"
if errorlevel 1 (
    echo Failed to inspect samples.
    pause
    exit /b 1
)

echo.
echo Training learning ranker...
"%VENV_PY%" -m rl.train --samples "%SAMPLES_DIR%" --out "%MODEL_PATH%"
if errorlevel 1 (
    echo Training failed. Collect learning samples in BAS first, then run this script again.
    pause
    exit /b 1
)

echo.
echo Exported model:
echo %MODEL_PATH%
echo.
echo Set BAS Settings - Learning ranker model to this path, or use configs/default.yaml learning.ranker_model_path.
pause
