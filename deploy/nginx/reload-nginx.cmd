@echo off
setlocal

set "NGINX_EXE=%NGINX_EXE%"
if "%NGINX_EXE%"=="" set "NGINX_EXE=nginx.exe"

"%NGINX_EXE%" -p "%~dp0" -c nginx.conf -t
if errorlevel 1 exit /b %errorlevel%

"%NGINX_EXE%" -p "%~dp0" -c nginx.conf -s reload
exit /b %errorlevel%
