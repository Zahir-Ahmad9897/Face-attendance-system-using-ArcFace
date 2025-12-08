@echo off
REM ============================================================================
REM Database Backup Script for Face Recognition Attendance System
REM ============================================================================

echo Starting database backup...

REM Create backups directory if it doesn't exist
if not exist "backups" mkdir backups

REM Get current date in YYYYMMDD format
set YEAR=%date:~-4%
set MONTH=%date:~-7,2%
set DAY=%date:~-10,2%
set BACKUP_DATE=%YEAR%%MONTH%%DAY%

REM Get current time for filename
set HOUR=%time:~0,2%
set MINUTE=%time:~3,2%
if "%HOUR:~0,1%" == " " set HOUR=0%HOUR:~1,1%
set BACKUP_TIME=%HOUR%%MINUTE%

REM Backup filename
set BACKUP_FILE=backups\attendance_%BACKUP_DATE%_%BACKUP_TIME%.db

REM Copy database
if exist "attendance.db" (
    copy "attendance.db" "%BACKUP_FILE%"
    echo ✅ Backup created: %BACKUP_FILE%
) else (
    echo ❌ Database file not found: attendance.db
    exit /b 1
)

REM Also backup email config
if exist "email_config.json" (
    copy "email_config.json" "backups\email_config_%BACKUP_DATE%_%BACKUP_TIME%.json"
    echo ✅ Email config backed up
)

REM Delete backups older than 30 days
echo.
echo Cleaning up old backups (keeping last 30 days)...
forfiles /P "backups" /M attendance_*.db /D -30 /C "cmd /c del @path" 2>nul
forfiles /P "backups" /M email_config_*.json /D -30 /C "cmd /c del @path" 2>nul

echo.
echo ✅ Backup completed successfully!
echo Total backups: 
dir /B backups\attendance_*.db | find /C /V ""

pause
