# Edge AI Deployment Guide

## System Architecture

This face recognition attendance system is **already designed for edge AI deployment** with the following architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    EDGE AI DEVICE                        │
│  (Camera Location - Can be Raspberry Pi, Jetson, etc.)  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  📹 Camera → webcam_recognition.py                       │
│              (Face Detection & Recognition)              │
│                      ↓                                    │
│              SQLite Database                             │
│              (attendance.db)                             │
│                      ↓                                    │
│              app.py (FastAPI Server)                     │
│              Email Scheduler                             │
│                                                           │
└─────────────────────────────────────────────────────────┘
                          ↓
                    Network (WiFi/LAN)
                          ↓
        ┌─────────────────────────────────────┐
        │   SUPERVISOR DASHBOARD               │
        │   (Any Device with Browser)          │
        ├─────────────────────────────────────┤
        │  💻 Desktop                          │
        │  📱 Tablet                           │
        │  📲 Mobile Phone                     │
        │                                      │
        │  Access: http://DEVICE_IP:8000      │
        └─────────────────────────────────────┘

```

## ✅ Current System = Edge AI Ready

Your system **already works** as an edge AI solution:

### Camera Side (Edge Device)
- **webcam_recognition.py** runs independently
- Detects faces in real-time
- Recognizes students using AI model
- Saves to local database automatically
- No internet required for face recognition

### Supervisor Side (Dashboard)
- **app.py** serves web dashboard
- Read-only monitoring interface
- Real-time updates via WebSocket
- Access from any device on network
- Responsive mobile/tablet/desktop

### Key Features Already Implemented
✅ **Offline Operation**: Works without internet
✅ **Real-time Updates**: Dashboard updates live
✅ **Edge Processing**: All AI runs on local device
✅ **Database**: Efficient SQLite storage
✅ **Email Reports**: Automatic daily summaries
✅ **Multi-device Access**: Responsive web interface

---

## Deployment Options

### Option 1: Single Edge Device (Recommended)

**Hardware**: Raspberry Pi 4 (4GB+), Jetson Nano, or any PC

**Setup**:
```bash
# Install dependencies
pip install -r requirements.txt

# Auto-start on boot (Linux/Raspberry Pi)
sudo nano /etc/rc.local
# Add before 'exit 0':
cd /home/pi/face_det
python3 app.py &
python3 webcam_recognition.py &

# Or use systemd service (better)
sudo nano /etc/systemd/system/attendance.service
```

**Systemd Service File**:
```ini
[Unit]
Description=Face Recognition Attendance System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/face_det
ExecStart=/usr/bin/python3 /home/pi/face_det/start_attendance.sh
Restart=always

[Install]
WantedBy=multi-user.target
```

**start_attendance.sh**:
```bash
#!/bin/bash
cd /home/pi/face_det
python3 app.py &
sleep 5
python3 webcam_recognition.py &
wait
```

---

### Option 2: Multiple Cameras (Multiple Edge Devices)

**Architecture**:
- Each camera = separate edge device (Raspberry Pi)
- Each runs its own `webcam_recognition.py`
- Central server runs `app.py` with consolidated database
- All edge devices sync to central database

**Setup**:

**On each camera device**:
```bash
# Run only recognition, save to network database
python webcam_recognition.py
```

**On central server**:
```bash
# Run dashboard and email service
python app.py
```

**Database sync**: Use network file share (NFS/SMB) or database replication

---

## Network Access Configuration

### 1. Find Edge Device IP
```bash
# Linux/Raspberry Pi
hostname -I

# Windows
ipconfig
```

### 2. Access Dashboard from Supervisor Device
```
Open browser:
http://DEVICE_IP:8000

Example: http://192.168.1.100:8000
```

### 3. Open Firewall (if needed)
```bash
# Linux
sudo ufw allow 8000

# Windows
# Windows Defender Firewall → Allow port 8000
```

### 4. Static IP (Recommended)
Set static IP on edge device so supervisor always knows the address.

**On Raspberry Pi**:
```bash
sudo nano /etc/dhcpcd.conf

# Add:
interface wlan0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1
```

---

## Edge AI Hardware Recommendations

### Budget Option: Raspberry Pi 4
- **Model**: 4GB RAM minimum, 8GB recommended
- **Camera**: Raspberry Pi Camera Module or USB Webcam
- **Storage**: 32GB+ microSD card (Class 10 or better)
- **Power**: Official 5V 3A power supply
- **Cost**: ~$75-100

**Performance**:
- Face detection: 5-10 FPS
- Good for single entrance
- Can handle 1-3 faces simultaneously

### Performance Option: NVIDIA Jetson Nano
- **Model**: 4GB variant
- **Camera**: USB or MIPI-CSI camera
- **Storage**: 64GB+ microSD
- **Power**: 5V 4A barrel jack
- **Cost**: ~$100-150

**Performance**:
- Face detection: 15-30 FPS
- Excellent for multiple entrances
- Can handle 5-10 faces simultaneously
- GPU acceleration for AI

### High Performance: Jetson Xavier NX
- **Cost**: ~$400-500
- **Performance**: 50+ FPS, 20+ simultaneous faces
- Best for large deployments

---

## Installation on Edge Device

### Quick Setup Script
```bash
#!/bin/bash
# Edge AI Attendance System Setup

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Python dependencies
sudo apt-get install -y python3-pip python3-opencv
sudo apt-get install -y libatlas-base-dev libjasper-dev
sudo apt-get install -y libqt4-test libqtgui4

# Install project dependencies
cd /home/pi/face_det
pip3 install -r requirements.txt

# Test camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read(); print('Camera OK' if ret else 'Camera FAILED'); cap.release()"

# Configure auto-start
sudo systemctl enable attendance.service
sudo systemctl start attendance.service

echo "✅ Edge AI setup complete!"
echo "Access dashboard at: http://$(hostname -I | awk '{print $1}'):8000"
```

---

## Supervisor Dashboard Features

### Read-Only Monitoring
- View current attendance status
- See who's present/absent
- Export reports as CSV
- Filter by date
- Real-time updates

### Actions Available
- Configure email settings
- Send test emails
- Send daily reports manually
- Export data
- View statistics

### No Direct Control of Camera
- Camera runs independently
- Supervisor can only monitor
- This ensures reliability
- Camera works even if dashboard is offline

---

## Troubleshooting Edge Deployment

### Camera Not Detected
```bash
# List cameras
ls /dev/video*

# Test camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"

# Try different camera index
# In webcam_recognition.py line 168, change:
cap = cv2.VideoCapture(1)  # Try 0, 1, 2, etc.
```

### Dashboard Not Accessible from Network
```bash
# Check if server is running
netstat -tulpn | grep 8000

# Test local access
curl http://localhost:8000

# Check firewall
sudo ufw status

# Bind to all interfaces (in app.py, line 393)
uvicorn.run(app, host="0.0.0.0", port=8000)  # Already set correctly
```

### Database Permission Issues
```bash
# Fix permissions
cd /home/pi/face_det
chmod 666 attendance.db
chmod 777 .
```

### Email Not Sending from Edge Device
```bash
# Test internet connection
ping google.com

# Check email config
python3 -c "from email_scheduler import send_test_email; send_test_email()"
```

---

## Performance Optimization

### For Raspberry Pi
```python
# In webcam_recognition.py, adjust:
process_every_n_frames = 10  # Increase from 5 to process fewer frames
```

### For Jetson Nano (Use GPU)
```bash
# Install CUDA-enabled OpenCV
# Use TensorRT for faster inference
# Enable GPU acceleration in model loading
```

### Reduce Resolution
```python
# In webcam_recognition.py, after line 188:
ret, frame = cap.read()
frame = cv2.resize(frame, (640, 480))  # Reduce from higher resolution
```

---

## Security Best Practices

### 1. Change Default Port
```python
# In app.py, line 393:
uvicorn.run(app, host="0.0.0.0", port=8080)  # Use non-standard port
```

### 2. Enable HTTPS (Optional)
```bash
# Generate SSL certificate
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# Update app.py:
uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
```

### 3. Network Isolation
- Use separate VLAN for cameras
- Restrict access to supervisor network only

### 4. Database Backup
```bash
# Auto-backup script
#!/bin/bash
cp /home/pi/face_det/attendance.db /backup/attendance_$(date +%Y%m%d).db
find /backup -name "attendance_*.db" -mtime +30 -delete
```

---

## Production Checklist

- [ ] Edge device has static IP address
- [ ] Auto-start service configured
- [ ] Camera tested and working
- [ ] Dashboard accessible from supervisor device
- [ ] Email configured and tested
- [ ] Database backup scheduled
- [ ] Firewall rules configured
- [ ] Performance acceptable (5+ FPS)
- [ ] Power supply reliable (UPS recommended)
- [ ] Monitoring/alerts configured

---

## Summary

✅ **Your system is already edge AI ready!**

**What you have**:
- Independent camera recognition (edge processing)
- Local database storage (no cloud needed)
- Web dashboard for supervisor (any device)
- Responsive mobile interface
- Automated email reports
- Offline operation

**How to deploy**:
1. Install on edge device (Raspberry Pi/Jetson)
2. Connect camera
3. Configure auto-start
4. Access dashboard from supervisor device
5. Configure email (optional)

**The architecture is perfect for edge AI deployment with camera at entrance and supervisor monitoring remotely via web dashboard!**
