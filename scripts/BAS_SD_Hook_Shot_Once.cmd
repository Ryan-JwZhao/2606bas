@echo off
setlocal
call "%~dp0BAS_StreamDeck_Command.cmd" hook-shot-once
exit /b %ERRORLEVEL%
