@echo off
setlocal
call "%~dp0BAS_StreamDeck_Command.cmd" start-projection
exit /b %ERRORLEVEL%
