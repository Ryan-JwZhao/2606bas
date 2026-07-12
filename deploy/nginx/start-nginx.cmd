@echo off
setlocal

set "NGINX_EXE=%NGINX_EXE%"
if "%NGINX_EXE%"=="" set "NGINX_EXE=nginx.exe"

if not exist "%~dp0logs" mkdir "%~dp0logs"
if not exist "%~dp0temp\proxy_temp" mkdir "%~dp0temp\proxy_temp"

"%NGINX_EXE%" -p "%~dp0" -c nginx.conf -t
if errorlevel 1 exit /b %errorlevel%

"%NGINX_EXE%" -p "%~dp0" -c nginx.conf
exit /b %errorlevel%
