@echo off
setlocal
cd /d "%~dp0"

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "YOLO_CONFIG_DIR=%CD%\local_settings\ultralytics"
set "MPLCONFIGDIR=%CD%\local_settings\matplotlib"
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "VENV_PYW=%CD%\.venv\Scripts\pythonw.exe"

if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%" >nul 2>nul
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%" >nul 2>nul

if not exist "%VENV_PY%" (
    echo Missing virtual environment: %VENV_PY%
    echo The dependencies have been installed in this workspace's .venv. If this folder was moved, run Setup_Environment.cmd once.
    pause
    exit /b 1
)

"%VENV_PY%" -c "import bas, cv2, numpy, PyQt5, yaml" >nul 2>nul
if errorlevel 1 (
    echo Runtime dependency check failed. Run Setup_Environment.cmd once to repair the local .venv.
    pause
    exit /b 1
)

if exist "%VENV_PYW%" (
    start "BAS" "%VENV_PYW%" -m bas ui
) else (
    start "BAS" "%VENV_PY%" -m bas ui
)

exit /b 0
