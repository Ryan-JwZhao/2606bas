@echo off
setlocal
call "%~dp0BAS_StreamDeck_Command.cmd" toggle-shot-mode
exit /b %ERRORLEVEL%
