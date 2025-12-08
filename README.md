# 🎯 Face Recognition Attendance System

## Production-Ready Edge AI Attendance System

Real-time face recognition with automated attendance tracking, database storage, and daily email reports.

---

## 🚀 Quick Start

### Option 1: Auto-Start Everything
```bash
start_system.bat
```

### Option 2: Manual Start
```bash
# Terminal 1: Web Dashboard
python app.py

# Terminal 2: Camera Recognition  
python webcam_recognition.py

# Browser: http://localhost:8000
```

---

## ✨ Key Features

- 🎥 **Real-time Face Detection** - MTCNN + ArcFace recognition
- 💾 **SQLite Database** - Persistent attendance storage
- 📧 **Daily Email Reports** - Automated HTML reports to teachers
- 📊 **Web Dashboard** - Responsive monitoring interface
- 🤖 **AI Chat Assistant** - Ask about attendance
- 📱 **Fully Responsive** - Desktop, tablet, mobile support
- 🔐 **Authentication** - Secure dashboard access
- 💾 **Auto Backup** - Database backup system
- 📝 **Logging** - Complete activity tracking

---

## 📋 System Requirements

- **Python**: 3.9+
- **Camera**: Webcam or IP camera
- **OS**: Windows, Linux, macOS
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space

---

## 🎓 Registered Students

- Mehran
- Yousaf  
- Zahir

*To add/remove students, see Student Management section*

---

## 📧 Email Configuration

1. Open dashboard: http://localhost:8000
2. Click "Email Setup"
3. Configure Gmail settings:
   - Sender email
   - App password ([Generate here](https://myaccount.google.com/apppasswords))
   - Teacher email
   - Send time
4. Enable automated emails
5. Test with "Send Test Email"

**Current Config**: Daily reports at 21:00

---

## 🗂️ Project Structure

```
d:\face_det\
├── app.py                    # Web server & API
├── webcam_recognition.py     # Face recognition
├── database.py               # SQLite database
├── email_scheduler.py        # Email automation
├── auth.py                   # Authentication
├── student_management.py     # Student CRUD
├── logger.py                 # Logging system
├── *.bat                     # Helper scripts
├── README.md                 # This file
│
├── face_models/              # Trained AI models
├── static/                   # Web dashboard files
├── logs/                     # System logs
├── backups/                  # Database backups
└── archive/                  # Documentation archive
    ├── docs/                 # Technical docs
    ├── guides/               # Setup guides  
    └── planning/             # Development files
```

---

## 🛠️ Batch Scripts

| Script | Purpose |
|--------|---------|
| `start_system.bat` | Start dashboard + camera |
| `backup_database.bat` | Backup database manually |
| `setup_auto_backup.bat` | Schedule daily backups |
| `cleanup_project.bat` | Organize files & archive docs |

---

## 📊 Dashboard Features

- **Live Attendance Feed** - Real-time arrivals
- **Present/Absent Tracking** - Who's here, who's missing
- **Date Filtering** - View any date's attendance
- **CSV Export** - Download reports
- **Email Configuration** - Manage email settings
- **AI Chat** - Ask questions about attendance

---

## 🔐 Default Login

**Admin**:
- Username: `admin`
- Password: `admin123`

**Teacher**:
- Username: `teacher`
- Password: `teacher123`

⚠️ Change these passwords immediately!

---

## 💾 Database Backup

**Manual**:
```bash
backup_database.bat
```

**Automatic** (daily at 11:59 PM):
```bash
setup_auto_backup.bat
```

**Location**: `backups/attendance_YYYYMMDD_HHMM.db`

---

## 👥 Student Management

**View Students**:
```
GET http://localhost:8000/api/students/all
```

**Add Student**:
```
POST http://localhost:8000/api/students/add
Body: {"name": "Student Name"}
```

**Remove Student**:
```
POST http://localhost:8000/api/students/remove  
Body: {"name": "Student Name"}
```

*Web UI for student management coming soon*

---

## 📝 Logs

**Location**: `logs/attendance_YYYYMMDD.log`

**View Today's Log**:
```bash
type logs\attendance_20251207.log
```

**Auto-cleanup**: Keeps 30 days of logs

---

## 🌐 Edge AI Deployment

This system is ready for edge deployment:
- Run on Raspberry Pi, Jetson Nano, or any PC
- Camera at entrance (runs `webcam_recognition.py`)
- Supervisor accesses dashboard remotely
- Works offline, no cloud required

See `archive/docs/EDGE_AI_DEPLOYMENT.md` for details

---

## 📚 Documentation

All documentation archived in `archive/`:

- **Technical Docs**: `archive/docs/`
  - IMPROVEMENTS.md
  - EDGE_AI_DEPLOYMENT.md
  - WEB_APP_GUIDE.md
  
- **Setup Guides**: `archive/guides/`
  - TESTING_GUIDE.md
  - COMPLETE_PIPELINE.md
  
- **Development**: `archive/planning/`
  - implementation_plan.md
  - walkthrough.md

---

## 🧪 Testing

**Test Camera**:
```bash
python webcam_recognition.py
```

**Test Email**:
```bash
python archive/test_email.py
```

**Test Dashboard**:
```
http://localhost:8000
```

---

## 🆘 Troubleshooting

**Camera not detected**:
- Check USB connection
- Try different camera index (0, 1, 2)
- Check permissions

**Email not sending**:
- Generate new app password
- Remove spaces from password
- Enable 2FA on Gmail
- Run `test_email.py`

**Database locked**:
- Close all connections
- Restart server

**Dashboard not loading**:
- Check if server is running
- Try http://127.0.0.1:8000
- Clear browser cache

---

## 📦 Dependencies

See `requirements.txt` and `web_requirements.txt`

**Install**:
```bash
pip install -r requirements.txt
pip install -r web_requirements.txt
```

---

## 🎯 Production Checklist

- [ ] Camera tested and working
- [ ] Database created and migrated
- [ ] Email configured and tested
- [ ] Default passwords changed
- [ ] Auto-backup scheduled
- [ ] Dashboard accessible
- [ ] Students list updated
- [ ] Logs directory created

---

## 📞 Support

For issues:
1. Check `logs/` for errors
2. Review documentation in `archive/`
3. Test email with `test_email.py`
4. Verify database with `attendance.db`

---

## 🌟 Future Enhancements

- Mobile app
- Multi-camera support
- SMS notifications
- Advanced analytics
- Cloud sync
- Anti-spoofing detection

---

**Version**: 2.0 (Production)  
**Updated**: December 2025  
**Status**: ✅ Production Ready

---

Made with ❤️ for automated attendance tracking
