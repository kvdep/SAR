@echo off
chcp 65001 >nul
title Генерация автономного HTML-бенчмарка Sentinel-1 SAR
cd /d "%~dp0"

echo =========================================================================
echo  Sentinel-1 SAR: Полная перегенерация интерактивного отчета (index.html)
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
echo [ИНФОРМАЦИЯ] Запуск сборки: код\build_full_benchmark.py
echo.

"%PY_CMD%" "код\build_full_benchmark.py"

if %errorlevel% equ 0 (
    echo.
    echo =========================================================================
    echo  [УСПЕХ] Генерация завершена. Файл сохранен: html\index.html
    echo =========================================================================
) else (
    echo.
    echo [ОШИБКА] Сборка завершилась с кодом ошибки %errorlevel%.
)

pause
