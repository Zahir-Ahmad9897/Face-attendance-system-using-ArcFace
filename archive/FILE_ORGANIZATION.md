# Project File Organization

## ✅ Production Files (Keep in Main Directory)

### Core Application
- app.py
- webcam_recognition.py  
- infrence.py
- batch_processor.py

### Modules
- database.py
- email_scheduler.py
- auth.py
- student_management.py
- logger.py

### Configuration
- email_config.json
- requirements.txt
- web_requirements.txt
- users.json (auto-generated)

### Batch Scripts
- start_system.bat
- backup_database.bat
- setup_auto_backup.bat
- cleanup_project.bat
- FIX_AND_RUN.bat

### Documentation
- README.md (main documentation)

### Directories
- face_models/ (AI models)
- static/ (web dashboard)
- logs/ (system logs)
- backups/ (database backups)
- venv_name/ (Python environment)

---

## 📦 Archived Files (Moved to archive/)

### Documentation (archive/docs/)
- IMPROVEMENTS.md
- EDGE_AI_DEPLOYMENT.md
- WEB_APP_GUIDE.md
- JSON_ATTENDANCE_GUIDE.md
- SETUP_VIRTUAL_ENV.md

### Guides (archive/guides/)
- TESTING_GUIDE.md
- COMPLETE_PIPELINE.md

### Planning (archive/planning/)
- implementation_plan.md
- task.md
- walkthrough.md

### Test Files (archive/)
- test_email.py
- csv_to_json.py
- test.jpg
- attendance.csv

---

## 🎯 Why This Organization?

### Main Directory Benefits
✅ Clean workspace
✅ Only production files visible
✅ Easy to navigate
✅ Clear purpose

### Archive Benefits
✅ Documentation preserved
✅ Historical reference available
✅ No clutter in main directory
✅ Easy to find when needed

---

## 📁 Final Structure

```
d:\face_det\
│
├── Core Files (Python)
├── Config Files (JSON, TXT)
├── Scripts (BAT)
├── README.md
│
├── face_models\
├── static\
├── logs\
├── backups\
│
└── archive\
    ├── docs\
    ├── guides\
    └── planning\
```

---

**Run `cleanup_project.bat` to organize automatically!**
