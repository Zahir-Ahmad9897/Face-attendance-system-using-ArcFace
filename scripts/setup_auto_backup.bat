@echo off
REM ============================================================================
REM Setup Windows Task Scheduler for Automated Daily Backup
REM Run this once to schedule automatic backups at 11:59 PM daily
REM ============================================================================

echo Setting up automated daily backup...
echo.

REM Create backup task
schtasks /Create /SC DAILY /TN "FaceDetection_Backup" /TR "%CD%\backup_database.bat" /ST 23:59 /F

if %errorlevel%==0 (
    echo ✅ Automated backup scheduled successfully!
    echo    Task Name: FaceDetection_Backup
    echo    Schedule: Daily at 11:59 PM
    echo    Action: Run backup_database.bat
    echo.
    echo To view/modify: Control Panel → Task Scheduler → FaceDetection_Backup
) else (
    echo ❌ Failed to create scheduled task
    echo Run this script as Administrator
)

echo.
pause
