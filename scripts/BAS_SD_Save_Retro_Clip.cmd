@echo off
setlocal
call "%~dp0BAS_StreamDeck_Command.cmd" save-retro-clip
exit /b %ERRORLEVEL%
