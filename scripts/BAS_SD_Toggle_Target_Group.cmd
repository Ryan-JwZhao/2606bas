@echo off
setlocal
call "%~dp0BAS_StreamDeck_Command.cmd" toggle-target-group
exit /b %ERRORLEVEL%
