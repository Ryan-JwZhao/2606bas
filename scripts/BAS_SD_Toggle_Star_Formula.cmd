@echo off
setlocal
call "%~dp0BAS_StreamDeck_Command.cmd" toggle-star-formula
exit /b %ERRORLEVEL%
