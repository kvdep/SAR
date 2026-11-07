@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Открытие автономного интерактивного отчета Sentinel-1 SAR...
start "" "html\index.html"
