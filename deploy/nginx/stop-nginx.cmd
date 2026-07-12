@echo off
setlocal

set "NGINX_EXE=%NGINX_EXE%"
if "%NGINX_EXE%"=="" set "NGINX_EXE=nginx.exe"

"%NGINX_EXE%" -p "%~dp0" -c nginx.conf -s quit
exit /b %errorlevel%
