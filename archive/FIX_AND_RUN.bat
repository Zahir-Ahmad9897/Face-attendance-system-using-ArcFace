@echo off
color 0A
echo ========================================================
echo    FIXING AND STARTING SYSTEM
echo ========================================================
echo.

:: 1. Activate the environment
call venv_name\Scripts\activate.bat

:: 2. Install missing requirements (just to be safe)
echo Installing missing web libraries...
pip install fastapi uvicorn[standard] websockets python-multipart aiofiles

:: 3. Start the Web Server
echo.
echo Starting Web Dashboard (In a new window)...
start "Web Dashboard" cmd /k "python app.py"

:: 4. Start the Webcam
echo.
echo Starting Webcam...
python webcam_recognition.py

pause
