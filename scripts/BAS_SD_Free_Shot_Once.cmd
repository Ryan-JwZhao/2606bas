@echo off
setlocal
call "%~dp0BAS_StreamDeck_Command.cmd" free-shot-once
exit /b %ERRORLEVEL%
