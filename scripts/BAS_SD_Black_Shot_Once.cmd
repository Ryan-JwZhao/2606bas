@echo off
setlocal
call "%~dp0BAS_StreamDeck_Command.cmd" black-shot-once
exit /b %ERRORLEVEL%
