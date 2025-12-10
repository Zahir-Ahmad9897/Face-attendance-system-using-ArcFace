@echo off
REM ============================================================================
REM Advanced Project Organization - Folder Structure
REM Organizes Python files into clean folders by purpose
REM ============================================================================

echo ======================================================================
echo Advanced Project Organization
echo ======================================================================
echo.

REM Create organized folder structure
echo Creating folder structure...
if not exist "src" mkdir src
if not exist "src\core" mkdir src\core
if not exist "src\modules" mkdir src\modules
if not exist "src\utils" mkdir src\utils
if not exist "config" mkdir config
if not exist "scripts" mkdir scripts

echo ✅ Folders created
echo.

REM Move core application files
echo [1/5] Organizing core application files...
if exist "app.py" move "app.py" "src\core\" >nul
if exist "webcam_recognition.py" move "webcam_recognition.py" "src\core\" >nul

REM Move module files
echo [2/5] Organizing module files...
if exist "database.py" move "database.py" "src\modules\" >nul
if exist "email_scheduler.py" move "email_scheduler.py" "src\modules\" >nul
if exist "auth.py" move "auth.py" "src\modules\" >nul
if exist "student_management.py" move "student_management.py" "src\modules\" >nul
if exist "logger.py" move "logger.py" "src\modules\" >nul

REM Move utility files
echo [3/5] Organizing utility files...
if exist "infrence.py" move "infrence.py" "src\utils\" >nul
if exist "batch_processor.py" move "batch_processor.py" "src\utils\" >nul
if exist "csv_to_json.py" move "csv_to_json.py" "src\utils\" >nul 2>nul

REM Move config files
echo [4/5] Organizing configuration files...
if exist "email_config.json" move "email_config.json" "config\" >nul
if exist "users.json" move "users.json" "config\" 2>nul

REM Move batch scripts
echo [5/5] Organizing scripts...
if exist "start_system.bat" move "start_system.bat" "scripts\" >nul
if exist "backup_database.bat" move "backup_database.bat" "scripts\" >nul
if exist "setup_auto_backup.bat" move "setup_auto_backup.bat" "scripts\" >nul
if exist "FIX_AND_RUN.bat" move "FIX_AND_RUN.bat" "scripts\" >nul

echo.
echo ======================================================================
echo ⚠️ IMPORTANT: Updating import paths...
echo ======================================================================
echo.

REM Create __init__.py files for Python package structure
echo Creating package files...
echo # Core module > src\core\__init__.py
echo # Modules > src\modules\__init__.py
echo # Utils > src\utils\__init__.py

echo.
echo ======================================================================
echo ❌ STOP! Manual Update Required
echo ======================================================================
echo.
echo This organization requires updating Python import statements.
echo.
echo Example changes needed:
echo   OLD: from database import get_connection
echo   NEW: from src.modules.database import get_connection
echo.
echo   OLD: from email_scheduler import send_email
echo   NEW: from src.modules.email_scheduler import send_email
echo.
echo Do you want to continue? This will break imports!
echo.
echo Press Ctrl+C to CANCEL
pause
echo.
echo Files will be moved in 3 seconds...
timeout /t 3
goto :move_files

:move_files
REM Actually move the files here
echo.
echo ❌ Organization cancelled - use organize_simple.bat instead
echo.
pause
exit /b 1
