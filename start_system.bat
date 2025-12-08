@echo off
title Face Recognition System Launcher (Debug Mode)
color 0A

echo ========================================================
echo    FACE RECOGNITION ATTENDANCE SYSTEM
echo ========================================================
echo.

:: 1. Define paths clearly
set VENV_PATH=%~dp0venv_name
set PYTHON_EXE=%VENV_PATH%\Scripts\python.exe

:: Check if python exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Could not find Python at %PYTHON_EXE%
    echo Please make sure the virtual environment 'venv_name' exists.
    pause
    exit /b
)

echo [1/3] Using Python from: %PYTHON_EXE%

echo.
echo [2/3] Starting Web Dashboard...
:: Start app.py in a new window using the specific python executable
start "Attendance Dashboard" cmd /k "%PYTHON_EXE% app.py"

:: Wait a moment for server to start
timeout /t 5 >nul

:: Automatically open the browser
echo.
echo [2.5/3] Opening Browser...
start http://localhost:8000


echo.
echo [3/3] Starting Webcam Recognition...
echo.
echo    -----------------------------------------------------
echo    Controls:
echo    [q] Quit
echo    [j] Save JSON  |  [c] Save CSV
echo    -----------------------------------------------------
echo.

:: Run the webcam script using the specific python executable
"%PYTHON_EXE%" webcam_recognition.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: The webcam script crashed or failed to start.
    echo Please check the error message above.
    pause
)

echo.
echo System Stopped.
pause
