@echo off
setlocal
call "%~dp0BAS_StreamDeck_Command.cmd" start-capture
exit /b %ERRORLEVEL%
