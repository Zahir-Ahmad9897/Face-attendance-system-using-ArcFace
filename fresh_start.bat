@echo off
REM ============================================================================
REM Fresh Start - Clear Attendance & Run System
REM Deletes old attendance and starts clean
REM ============================================================================

echo ======================================================================
echo Fresh Start - Reset Attendance System
echo ======================================================================
echo.

REM Backup current database before deleting
if exist "attendance.db" (
    echo Creating backup before reset...
    set TIMESTAMP=%date:~-4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%
    set TIMESTAMP=%TIMESTAMP: =0%
    if not exist "backups" mkdir backups
    copy "attendance.db" "backups\attendance_before_reset_%TIMESTAMP%.db" >nul
    echo ✅ Backup created: backups\attendance_before_reset_%TIMESTAMP%.db
    echo.
)

REM Delete attendance database
echo Deleting old attendance data...
if exist "attendance.db" (
    del "attendance.db"
    echo ✅ Old database deleted
) else (
    echo ℹ️  No database found (will create fresh)
)

REM Delete old JSON backup if exists
if exist "attendance.json" (
    del "attendance.json"
    echo ✅ Old JSON deleted
)

echo.
echo ======================================================================
echo Starting Fresh System
echo ======================================================================
echo.

REM Start dashboard in new window
echo [1/2] Starting Web Dashboard...
start "Attendance Dashboard" cmd /k "python app.py"
timeout /t 3 >nul

REM Start webcam recognition in new window
echo [2/2] Starting Face Recognition...
start "Face Recognition Camera" cmd /k "python webcam_recognition.py"

echo.
echo ======================================================================
echo ✅ System Started with Fresh Database!
echo ======================================================================
echo.
echo 🌐 Dashboard: http://localhost:8000
echo 📹 Camera: Running in separate window
echo.
echo What to do:
echo 1. Wait 5 seconds for services to start
echo 2. Open browser: http://localhost:8000
echo 3. Show face to camera
echo 4. Dashboard will update in real-time!
echo.
echo Press Ctrl+C in each window to stop
echo ======================================================================
echo.
echo Opening dashboard in browser...
timeout /t 5 >nul
start http://localhost:8000

echo.
echo ✅ Done! Check the browser and camera windows.
pause
