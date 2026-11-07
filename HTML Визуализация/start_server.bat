@echo off
chcp 65001 >nul
title Аппаратный сервер Sentinel-1 SAR (порт 8080)
cd /d "%~dp0"

echo =========================================================================
echo  Sentinel-1 SAR: Запуск вычислительного сервера конструктора (GPU/CUDA)
echo =========================================================================

REM Поиск исполняемого файла Python
set "PY_CMD="
where python >nul 2>&1
if %errorlevel% equ 0 set "PY_CMD=python"
if not defined PY_CMD if exist "D:\Programs\Python\python.exe" set "PY_CMD=D:\Programs\Python\python.exe"
if not defined PY_CMD if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY_CMD if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY_CMD if exist "%ProgramFiles%\Python312\python.exe" set "PY_CMD=%ProgramFiles%\Python312\python.exe"

if not defined PY_CMD (
    echo [ОШИБКА] Интерпретатор Python не найден в системе.
    echo Установите Python 3.10-3.12 и зависимости из requirements.txt:
    echo     pip install -r requirements.txt
    pause
    exit /b 1
)

echo [ИНФОРМАЦИЯ] Используемый Python: %PY_CMD%
echo [ИНФОРМАЦИЯ] Рабочий каталог: %CD%

REM Автоматическое открытие страницы визуализации в браузере через 2 секунды
start /b cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8080/html/index.html"

REM Запуск HTTP сервера инференса нейросетей
"%PY_CMD%" "код\sar_server.py" 8080

if %errorlevel% neq 0 (
    echo [ОШИБКА] Сервер завершил работу с кодом ошибки %errorlevel%.
    pause
)
