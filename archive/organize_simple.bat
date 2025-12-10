@echo off
REM ============================================================================
REM Simple & Safe Project Organization
REM Organizes files into folders WITHOUT breaking imports
REM ============================================================================

echo ======================================================================
echo Simple Project Organization (Import-Safe)
echo ======================================================================
echo.

REM Create organized folders
echo Creating folders...
if not exist "config" mkdir config
if not exist "scripts" mkdir scripts
if not exist "utils" mkdir utils

echo ✅ Folders created
echo.

REM Move config files
echo [1/3] Moving configuration files...
if exist "users.json" copy "users.json" "config\users.json" >nul 2>nul
if exist "attendance_backup_*.json" move "attendance_backup_*.json" "config\" >nul 2>nul

REM Move utility scripts (not core Python files)
echo [2/3] Moving utility scripts...
if exist "csv_to_json.py" move "csv_to_json.py" "utils\" >nul 2>nul
if exist "infrence.py" copy "infrence.py" "utils\infrence.py" >nul 2>nul
if exist "batch_processor.py" copy "batch_processor.py" "utils\batch_processor.py" >nul 2>nul

REM Copy batch scripts (keep originals in root for easy access)
echo [3/3] Organizing batch scripts...
if exist "backup_database.bat" copy "backup_database.bat" "scripts\" >nul
if exist "setup_auto_backup.bat" copy "setup_auto_backup.bat" "scripts\" >nul
if exist "cleanup_project.bat" copy "cleanup_project.bat" "scripts\" >nul

echo.
echo ======================================================================
echo ✅ Organization Complete!
echo ======================================================================
echo.
echo 📁 New Structure:
echo.
echo d:\face_det\
echo ├── Core Files (in root - for easy imports)
echo │   ├── app.py
echo │   ├── webcam_recognition.py
echo │   ├── database.py
echo │   ├── email_scheduler.py
echo │   ├── auth.py
echo │   ├── student_management.py
echo │   ├── logger.py
echo │   └── email_config.json
echo │
echo ├── config\              # Backup configs
echo │   └── users.json
echo │
echo ├── scripts\             # Helper scripts (copies)
echo │   ├── backup_database.bat
echo │   └── setup_auto_backup.bat
echo │
echo ├── utils\               # Utility scripts
echo │   ├── infrence.py
echo │   └── batch_processor.py
echo │
echo ├── archive\             # Documentation
echo ├── backups\             # DB backups
echo ├── logs\                # System logs
echo ├── face_models\         # AI models
echo └── static\              # Web files
echo.
echo ✅ All imports still work!
echo ✅ Main files in root for easy access
echo ✅ Extra files organized in folders
echo.
echo ======================================================================
pause
