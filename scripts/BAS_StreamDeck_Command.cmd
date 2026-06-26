@echo off
setlocal

set "ROOT=%~dp0.."
pushd "%ROOT%"

if "%~1"=="" (
    echo Usage: BAS_StreamDeck_Command.cmd action-name
    echo Example: BAS_StreamDeck_Command.cmd free-shot-once
    popd
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m bas.cli remote-control %*
) else (
    python -m bas.cli remote-control %*
)

set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%
