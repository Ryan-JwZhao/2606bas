@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "VENV_PYW=.venv\Scripts\pythonw.exe"
set "YOLO_CONFIG_DIR=%CD%\local_settings\ultralytics"
set "MPLCONFIGDIR=%CD%\local_settings\matplotlib"

if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%" >nul 2>nul
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%" >nul 2>nul

if not exist "%VENV_PY%" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        py -3 -m venv .venv
    )
)

if not exist "%VENV_PY%" (
    echo Failed to create .venv. Please install Python 3.10+ and run again.
    pause
    exit /b 1
)

"%VENV_PY%" -c "import numpy, cv2, PyQt5, yaml" >nul 2>nul
if errorlevel 1 (
    echo Installing runtime dependencies...
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed.
        pause
        exit /b 1
    )
)

"%VENV_PY%" -m bas.dependency_check --needs-yolo >nul 2>nul
if not errorlevel 1 (
    "%VENV_PY%" -m bas.dependency_check --yolo-available >nul 2>nul
    if errorlevel 1 (
        echo Active detector backend is Ultralytics. Installing YOLO runtime dependencies...
        "%VENV_PY%" -m pip install -r requirements-yolo.txt
        if errorlevel 1 (
            echo YOLO dependency installation failed.
            echo You can retry manually:
            echo   "%VENV_PY%" -m pip install -r requirements-yolo.txt
            pause
            exit /b 1
        )
    )
)

if exist "%VENV_PYW%" (
    start "" "%VENV_PYW%" -m bas ui
) else (
    start "" "%VENV_PY%" -m bas ui
)

exit /b 0
