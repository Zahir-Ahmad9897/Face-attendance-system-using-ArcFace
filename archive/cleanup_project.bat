@echo off
REM ============================================================================
REM Project Cleanup Script - Archive Documentation and Trial Files
REM ============================================================================

echo ======================================================================
echo Project Cleanup - Archiving Documentation and Trial Files
echo ======================================================================
echo.

REM Create archive directories
if not exist "archive" mkdir archive
if not exist "archive\docs" mkdir archive\docs
if not exist "archive\guides" mkdir archive\guides
if not exist "archive\planning" mkdir archive\planning

echo Creating archive folders...
echo.

REM Move planning and implementation files
echo [1/4] Archiving planning documents...
if exist "implementation_plan.md" move "implementation_plan.md" "archive\planning\" >nul
if exist "task.md" move "task.md" "archive\planning\" >nul
if exist "walkthrough.md" move "walkthrough.md" "archive\planning\" >nul

REM Move documentation files  
echo [2/4] Archiving documentation...
if exist "IMPROVEMENTS.md" move "IMPROVEMENTS.md" "archive\docs\" >nul
if exist "EDGE_AI_DEPLOYMENT.md" move "EDGE_AI_DEPLOYMENT.md" "archive\docs\" >nul
if exist "WEB_APP_GUIDE.md" move "WEB_APP_GUIDE.md" "archive\docs\" >nul
if exist "JSON_ATTENDANCE_GUIDE.md" move "JSON_ATTENDANCE_GUIDE.md" "archive\docs\" >nul
if exist "SETUP_VIRTUAL_ENV.md" move "SETUP_VIRTUAL_ENV.md" "archive\docs\" >nul

REM Move guide files
echo [3/4] Archiving setup guides...
if exist "TESTING_GUIDE.md" move "TESTING_GUIDE.md" "archive\guides\" >nul
if exist "COMPLETE_PIPELINE.md" move "COMPLETE_PIPELINE.md" "archive\guides\" >nul

REM Move test/trial files
echo [4/4] Archiving test files...
if exist "test_email.py" move "test_email.py" "archive\" >nul
if exist "csv_to_json.py" move "csv_to_json.py" "archive\" >nul
if exist "test.jpg" move "test.jpg" "archive\" >nul
if exist "attendance.csv" move "attendance.csv" "archive\" >nul

echo.
echo ======================================================================
echo ✅ Cleanup Complete!
echo ======================================================================
echo.
echo 📁 Archived Files:
dir /B archive\docs 2>nul | find /C /V ""
echo    documentation files moved to: archive\docs\
echo.
dir /B archive\guides 2>nul | find /C /V ""
echo    guide files moved to: archive\guides\
echo.
dir /B archive\planning 2>nul | find /C /V ""
echo    planning files moved to: archive\planning\
echo.
echo 📂 Main Directory Now Contains Only:
echo    - Production code files (*.py)
echo    - Configuration files (*.json, *.txt)
echo    - Batch scripts (*.bat)
echo    - README.md (main documentation)
echo    - Database and backups
echo    - Static web files
echo.
echo ======================================================================
pause
