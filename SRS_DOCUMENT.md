# Software Requirements Specification (SRS)
# Face Recognition Based Attendance System with Automatic Door Control

**Version:** 1.0  
**Date:** December 7, 2025  
**Prepared by:** [Your Name]  
**Project:** Embedded Face Recognition Attendance System  

---

## Table of Contents

1. [Introduction](#1-introduction)
   - 1.1 Purpose
   - 1.2 Document Conventions
   - 1.3 Intended Audience
   - 1.4 Project Scope
   - 1.5 References
2. [Overall Description](#2-overall-description)
   - 2.1 Product Perspective
   - 2.2 Product Functions
   - 2.3 User Classes and Characteristics
   - 2.4 Operating Environment
   - 2.5 Design and Implementation Constraints
   - 2.6 Assumptions and Dependencies
3. [System Features](#3-system-features)
4. [External Interface Requirements](#4-external-interface-requirements)
5. [Non-Functional Requirements](#5-non-functional-requirements)
6. [Other Requirements](#6-other-requirements)

---

## 1. Introduction

### 1.1 Purpose

This Software Requirements Specification (SRS) document provides a complete description of the Face Recognition Based Attendance System with Automatic Door Control. The system combines artificial intelligence, embedded systems, and IoT technologies to automate attendance marking and access control for educational institutions.

The document specifies all functional and non-functional requirements, external interface requirements, and design constraints for the system.

### 1.2 Document Conventions

- **SHALL** - Indicates a mandatory requirement
- **SHOULD** - Indicates a recommended requirement
- **MAY** - Indicates an optional requirement
- **FR** - Functional Requirement
- **NFR** - Non-Functional Requirement
- **UI** - User Interface
- **API** - Application Programming Interface

### 1.3 Intended Audience

This document is intended for:
- **Developers:** For implementation guidance
- **Testers:** For creating test cases and validation plans
- **Project Managers:** For project planning and tracking
- **End Users:** For understanding system capabilities
- **Evaluators:** For course project assessment

### 1.4 Project Scope

**Project Name:** DOPA - Face Recognition Attendance System with Door Access Control

**Goal:** To create an automated, contactless attendance system that:
- Eliminates manual attendance marking
- Provides secure access control
- Maintains accurate attendance records
- Reduces time spent on administrative tasks
- Enhances security through face recognition

**Objectives:**
1. Achieve >95% face recognition accuracy
2. Process attendance in <2 seconds per student
3. Automatic door unlock for authorized persons
4. Real-time attendance dashboard
5. Automated email reporting
6. Embedded hardware integration

### 1.5 References

1. IEEE Std 830-1998 - IEEE Recommended Practice for Software Requirements Specifications
2. Python 3.x Documentation - https://docs.python.org
3. TensorFlow Lite Documentation - https://www.tensorflow.org/lite
4. Raspberry Pi Documentation - https://www.raspberrypi.org/documentation
5. FastAPI Documentation - https://fastapi.tiangolo.com
6. OpenCV Documentation - https://docs.opencv.org

---

## 2. Overall Description

### 2.1 Product Perspective

The Face Recognition Based Attendance System is a standalone embedded system that integrates:

**Hardware Components:**
- Raspberry Pi 4 (Main Controller)
- USB/Pi Camera Module (Image Capture)
- Relay Module (Door Lock Control)
- LED Indicators (Visual Feedback)
- Buzzer (Audio Feedback)
- LCD Display (Status Display)
- Solenoid Door Lock (Physical Security)

**Software Components:**
- Face Detection Module (OpenCV)
- Face Recognition Model (TensorFlow Lite)
- Web Dashboard (FastAPI + HTML/CSS/JS)
- Database Management (SQLite)
- Email Automation (SMTP)
- Hardware Control Layer (GPIO)

**System Context Diagram:**
```
┌─────────────┐
│   Camera    │───┐
└─────────────┘   │
                  ↓
┌─────────────┐  ┌──────────────────┐  ┌─────────────┐
│  Students   │→ │  Raspberry Pi    │→ │ Door Lock   │
└─────────────┘  │  + Face AI       │  └─────────────┘
                  │  + Database      │
┌─────────────┐  │  + Web Server    │  ┌─────────────┐
│  Teachers   │← └──────────────────┘→ │ LCD/LEDs    │
│(Dashboard)  │                        └─────────────┘
└─────────────┘
```

### 2.2 Product Functions

**Primary Functions:**
1. **Face Detection and Recognition**
   - Real-time face detection from camera feed
   - Face recognition using AI model
   - Confidence score calculation
   - Unknown person detection

2. **Automatic Access Control**
   - Door unlock for authorized students
   - Door lock for unauthorized persons
   - Visual indicators (Green/Red LEDs)
   - Audio feedback (Buzzer)

3. **Attendance Management**
   - Automatic attendance marking
   - One entry per student per day
   - Time-stamped records
   - Database storage

4. **Web Dashboard**
   - Real-time attendance view
   - Day-by-day collapsible records
   - Present/Absent statistics
   - Search and filter capabilities
   - Excel report export

5. **AI Assistant (DOPA)**
   - Natural language queries
   - Attendance information retrieval
   - Statistics reporting

6. **Email Automation**
   - Daily attendance reports
   - Scheduled email delivery
   - Excel attachment with styled data

### 2.3 User Classes and Characteristics

**1. Students (Primary Users)**
- **Technical Expertise:** None required
- **Frequency of Use:** Daily (morning arrival)
- **Functions:**
  - Stand in front of camera
  - Face recognized → Door opens
  - Attendance automatically marked

**2. Teachers/Administrators (Secondary Users)**
- **Technical Expertise:** Basic computer skills
- **Frequency of Use:** Multiple times per day
- **Functions:**
  - View dashboard
  - Check attendance
  - Download reports
  - Query AI assistant
  - Configure email settings

**3. System Administrators (Tertiary Users)**
- **Technical Expertise:** Advanced (IT staff)
- **Frequency of Use:** As needed
- **Functions:**
  - Add/remove students
  - Train face recognition model
  - System maintenance
  - Hardware troubleshooting
  - Database backup

### 2.4 Operating Environment

**Hardware Environment:**
- **Processor:** Raspberry Pi 4 (Quad-core ARM Cortex-A72)
- **RAM:** 4GB minimum
- **Storage:** 16GB microSD card minimum
- **Camera:** USB Camera or Pi Camera Module v2
- **Display:** Optional 7" touchscreen or remote browser access
- **Network:** WiFi or Ethernet for web dashboard
- **Power:** 5V/3A USB-C power supply

**Software Environment:**
- **Operating System:** Raspberry Pi OS (Debian-based Linux)
- **Python Version:** 3.8 or higher
- **Web Browser:** Chrome, Firefox, Edge (latest versions)
- **Database:** SQLite 3.x
- **Email:** Gmail SMTP server

**Physical Environment:**
- **Location:** Indoor, at building entrance
- **Lighting:** Adequate natural or artificial light
- **Temperature:** 10°C to 35°C
- **Humidity:** 20% to 80% RH

### 2.5 Design and Implementation Constraints

**Hardware Constraints:**
- Limited processing power (Raspberry Pi)
- Camera resolution: 640x480 minimum
- GPIO pin availability: 40-pin header
- Power consumption: <15W total

**Software Constraints:**
- Real-time processing requirement (<2 seconds)
- Lightweight AI model (<50MB)
- RAM usage <2GB
- Storage efficient database

**Regulatory/Standards:**
- Data privacy compliance (student images)
- Network security protocols
- Electrical safety standards

**Technology Stack:**
- Python programming language
- TensorFlow Lite for AI
- FastAPI for web services
- SQLite for data storage

### 2.6 Assumptions and Dependencies

**Assumptions:**
1. Students will cooperate and stand in front of camera
2. Adequate lighting is available at all times
3. Internet connectivity for dashboard access
4. Power supply is uninterrupted
5. Face database is trained before deployment

**Dependencies:**
1. OpenCV library for image processing
2. TensorFlow Lite for model inference
3. GPIO library (RPi.GPIO) for hardware control
4. FastAPI framework for web server
5. SQLite database engine
6. SMTP server (Gmail) for email functionality

---

## 3. System Features

### 3.1 Face Detection and Recognition

**Priority:** HIGH  
**Description:** Real-time detection and recognition of student faces

#### 3.1.1 Functional Requirements

**FR-1.1:** The system SHALL capture video frames at minimum 10 FPS  
**FR-1.2:** The system SHALL detect faces in captured frames within 100ms  
**FR-1.3:** The system SHALL extract face embeddings using AI model  
**FR-1.4:** The system SHALL compare embeddings with database  
**FR-1.5:** The system SHALL calculate confidence score (0-1 range)  
**FR-1.6:** The system SHALL recognize face if confidence > 0.65  
**FR-1.7:** The system SHALL handle multiple faces in frame  
**FR-1.8:** The system SHALL support minimum 50 students in database  

#### 3.1.2 Stimulus/Response Sequences

**Scenario 1: Known Student**
1. Student stands in front of camera
2. System detects face
3. System recognizes student (confidence > 0.65)
4. System marks attendance
5. System unlocks door
6. System displays welcome message

**Scenario 2: Unknown Person**
1. Person stands in front of camera
2. System detects face
3. System fails to recognize (confidence < 0.65)
4. System denies access
5. Door remains locked
6. System displays warning

---

### 3.2 Automatic Door Control

**Priority:** HIGH  
**Description:** Hardware control for automatic door locking/unlocking

#### 3.2.1 Functional Requirements

**FR-2.1:** The system SHALL unlock door when authorized student recognized  
**FR-2.2:** The system SHALL keep door unlocked for 5 seconds  
**FR-2.3:** The system SHALL automatically lock door after timeout  
**FR-2.4:** The system SHALL keep door locked for unauthorized persons  
**FR-2.5:** The system SHALL control relay via GPIO pin 17  
**FR-2.6:** The system SHALL provide visual feedback (LEDs)  
**FR-2.7:** The system SHALL provide audio feedback (buzzer)  
**FR-2.8:** The system SHALL log all access attempts  

#### 3.2.2 LED Indicator Requirements

**FR-2.9:** Green LED (GPIO 27) SHALL turn ON when access granted  
**FR-2.10:** Red LED (GPIO 22) SHALL turn ON when access denied  
**FR-2.11:** LEDs SHALL turn OFF after event completion  

#### 3.2.3 Buzzer Requirements

**FR-2.12:** Buzzer SHALL produce 2 short beeps for success  
**FR-2.13:** Buzzer SHALL produce 3 long beeps for denial  
**FR-2.14:** Buzzer SHALL operate via GPIO pin 23  

---

### 3.3 Attendance Management

**Priority:** HIGH  
**Description:** Automated attendance recording and management

#### 3.3.1 Functional Requirements

**FR-3.1:** The system SHALL mark attendance when student recognized  
**FR-3.2:** The system SHALL record student name, date, and time  
**FR-3.3:** The system SHALL allow only ONE entry per student per day  
**FR-3.4:** The system SHALL update entry time if student detected again  
**FR-3.5:** The system SHALL store data in SQLite database  
**FR-3.6:** The system SHALL maintain attendance history indefinitely  
**FR-3.7:** The system SHALL calculate present/absent statistics  
**FR-3.8:** The system SHALL identify absent students  

#### 3.3.2 Data Requirements

**FR-3.9:** Each attendance record SHALL contain:
- Student Name
- Date (YYYY-MM-DD)
- Time (HH:MM:SS)
- Status (Present/Absent)
- Timestamp (ISO 8601)

---

### 3.4 Web Dashboard

**Priority:** MEDIUM  
**Description:** Real-time web-based interface for attendance monitoring

#### 3.4.1 Functional Requirements

**FR-4.1:** The system SHALL provide web dashboard at http://localhost:8000  
**FR-4.2:** The dashboard SHALL display real-time attendance statistics  
**FR-4.3:** The dashboard SHALL show present/absent student counts  
**FR-4.4:** The dashboard SHALL display live clock with HH:MM:SS  
**FR-4.5:** The dashboard SHALL support two view modes (Grouped/List)  
**FR-4.6:** The dashboard SHALL group attendance by date  
**FR-4.7:** The dashboard SHALL provide collapsible date sections  
**FR-4.8:** The dashboard SHALL highlight today's attendance  
**FR-4.9:** The dashboard SHALL support date filtering  
**FR-4.10:** The dashboard SHALL auto-refresh via WebSocket  

#### 3.4.2 Search and Filter Requirements

**FR-4.11:** Users SHALL be able to filter by date  
**FR-4.12:** Users SHALL be able to switch between view modes  
**FR-4.13:** Users SHALL be able to search historical records  

---

### 3.5 DOPA AI Assistant

**Priority:** MEDIUM  
**Description:** Intelligent chatbot for attendance queries

#### 3.5.1 Functional Requirements

**FR-5.1:** The system SHALL provide AI chat interface  
**FR-5.2:** DOPA SHALL respond to greetings  
**FR-5.3:** DOPA SHALL answer "Who is present today?"  
**FR-5.4:** DOPA SHALL answer "Who is absent today?"  
**FR-5.5:** DOPA SHALL answer "How many students today?"  
**FR-5.6:** DOPA SHALL provide student-specific information  
**FR-5.7:** DOPA SHALL respond within 1 second  
**FR-5.8:** DOPA SHALL handle natural language queries  

---

### 3.6 Report Generation

**Priority:** MEDIUM  
**Description:** Excel report generation with attendance data

#### 3.6.1 Functional Requirements

**FR-6.1:** The system SHALL export attendance to Excel format  
**FR-6.2:** The system SHALL include summary statistics in report  
**FR-6.3:** The system SHALL separate present and absent students  
**FR-6.4:** The system SHALL apply color coding (green/red)  
**FR-6.5:** The system SHALL include headers and formatting  
**FR-6.6:** The system SHALL support date range selection  
**FR-6.7:** The system SHALL generate file on demand  

---

### 3.7 Email Automation

**Priority:** LOW  
**Description:** Scheduled email delivery of attendance reports

#### 3.7.1 Functional Requirements

**FR-7.1:** The system SHALL send daily attendance reports via email  
**FR-7.2:** The system SHALL support Gmail SMTP  
**FR-7.3:** The system SHALL attach Excel file to email  
**FR-7.4:** The system SHALL support configurable send time  
**FR-7.5:** The system SHALL support enable/disable toggle  
**FR-7.6:** The system SHALL send test emails on demand  
**FR-7.7:** The system SHALL log email status (success/failure)  

---

### 3.8 Student Management

**Priority:** MEDIUM  
**Description:** Add, update, and delete student records

#### 3.8.1 Functional Requirements

**FR-8.1:** The system SHALL support adding new students  
**FR-8.2:** The system SHALL support removing students  
**FR-8.3:** The system SHALL maintain student list in database  
**FR-8.4:** The system SHALL prevent duplicate student names  
**FR-8.5:** The system SHALL update face encodings when student added  

---

## 4. External Interface Requirements

### 4.1 User Interfaces

#### 4.1.1 Web Dashboard Interface

**UI-1:** Dashboard SHALL be responsive (desktop/tablet/mobile)  
**UI-2:** Dashboard SHALL use modern design with gradients and animations  
**UI-3:** Dashboard SHALL display DOPA branding with robot icon  
**UI-4:** Dashboard SHALL show live clock prominently  
**UI-5:** Dashboard SHALL use color coding:
- Green for present students
- Red for absent students
- Blue for system status  

**UI-6:** Dashboard SHALL include:
- Header with branding and clock
- Statistics cards (5 cards)
- Attendance table with two view modes
- Recent arrivals sidebar
- Absent students sidebar
- AI chat widget
- Email configuration modal  

#### 4.1.2 Chat Interface

**UI-7:** Chat widget SHALL be accessible via floating button  
**UI-8:** Chat SHALL support text input and Enter key  
**UI-9:** Chat SHALL display conversation history  
**UI-10:** Chat SHALL show typing indicator while processing  

### 4.2 Hardware Interfaces

#### 4.2.1 Camera Interface

**HW-1:** System SHALL support USB webcam (UVC protocol)  
**HW-2:** System SHALL support Raspberry Pi Camera Module  
**HW-3:** Camera SHALL operate at minimum 640x480 resolution  
**HW-4:** Camera SHALL capture RGB color images  

#### 4.2.2 GPIO Interfaces

**HW-5:** System SHALL use BCM pin numbering  
**HW-6:** Pin 17 SHALL control relay (door lock)  
**HW-7:** Pin 27 SHALL control green LED  
**HW-8:** Pin 22 SHALL control red LED  
**HW-9:** Pin 23 SHALL control buzzer  
**HW-10:** All GPIO pins SHALL operate at 3.3V logic level  

#### 4.2.3 Display Interface (Optional)

**HW-11:** System MAY support 16x2 LCD via I2C  
**HW-12:** LCD SHALL display student name and status  

### 4.3 Software Interfaces

#### 4.3.1 Database Interface

**SW-1:** System SHALL use SQLite 3 database  
**SW-2:** Database file SHALL be named "attendance.db"  
**SW-3:** System SHALL implement two tables:
- attendance (id, name, date, time, status)
- students (id, name, added_at)  

**SW-4:** System SHALL use SQL for all queries  
**SW-5:** System SHALL implement connection pooling  

#### 4.3.2 Email Interface

**SW-6:** System SHALL use SMTP protocol  
**SW-7:** System SHALL connect to smtp.gmail.com:587  
**SW-8:** System SHALL use TLS encryption  
**SW-9:** System SHALL support OAuth2 or App Passwords  

#### 4.3.3 Web Server Interface

**SW-10:** System SHALL use FastAPI framework  
**SW-11:** System SHALL serve static files (HTML/CSS/JS)  
**SW-12:** System SHALL provide REST API endpoints  
**SW-13:** System SHALL support WebSocket for real-time updates  

### 4.4 Communications Interfaces

#### 4.4.1 HTTP/HTTPS

**COM-1:** Web server SHALL listen on port 8000  
**COM-2:** System SHALL use HTTP protocol  
**COM-3:** System SHALL support CORS for API access  

#### 4.4.2 WebSocket

**COM-4:** System SHALL maintain WebSocket connection at /ws  
**COM-5:** System SHALL broadcast attendance updates  
**COM-6:** System SHALL support auto-reconnection  

---

## 5. Non-Functional Requirements

### 5.1 Performance Requirements

**NFR-1.1:** Face detection SHALL complete within 100ms  
**NFR-1.2:** Face recognition SHALL complete within 500ms  
**NFR-1.3:** Total processing time SHALL be < 2 seconds  
**NFR-1.4:** Door unlock SHALL occur within 1 second of recognition  
**NFR-1.5:** Dashboard SHALL load within 3 seconds  
**NFR-1.6:** WebSocket updates SHALL have < 500ms latency  
**NFR-1.7:** System SHALL support minimum 50 concurrent dashboard users  
**NFR-1.8:** Database queries SHALL complete within 100ms  
**NFR-1.9:** AI model inference SHALL use < 1GB RAM  
**NFR-1.10:** System SHALL operate at < 70% CPU utilization  

### 5.2 Safety Requirements

**NFR-2.1:** Door SHALL auto-lock after 5 seconds (fail-safe)  
**NFR-2.2:** System SHALL not lock door while someone in doorway  
**NFR-2.3:** Manual override SHALL be available for emergencies  
**NFR-2.4:** System SHALL log all access attempts for audit  
**NFR-2.5:** Red LED SHALL indicate security alert  

### 5.3 Security Requirements

**NFR-3.1:** Face images SHALL be stored securely  
**NFR-3.2:** Database SHALL be protected from unauthorized access  
**NFR-3.3:** Web dashboard SHALL implement CORS policy  
**NFR-3.4:** Email credentials SHALL be encrypted  
**NFR-3.5:** System SHALL prevent SQL injection attacks  
**NFR-3.6:** API endpoints SHALL validate all inputs  
**NFR-3.7:** System SHALL implement rate limiting  

### 5.4 Reliability Requirements

**NFR-4.1:** System uptime SHALL be > 99% during operating hours  
**NFR-4.2:** System SHALL auto-restart on failure  
**NFR-4.3:** System SHALL recover from camera disconnection  
**NFR-4.4:** Database SHALL be backed up daily  
**NFR-4.5:** System SHALL handle concurrent requests  
**NFR-4.6:** System SHALL log errors for debugging  

### 5.5 Availability Requirements

**NFR-5.1:** System SHALL be available 24/7  
**NFR-5.2:** System SHALL start automatically on boot  
**NFR-5.3:** Maintenance downtime SHALL be < 1 hour per week  
**NFR-5.4:** System SHALL support remote access for troubleshooting  

### 5.6 Maintainability Requirements

**NFR-6.1:** Code SHALL be well-documented with docstrings  
**NFR-6.2:** System SHALL use modular architecture  
**NFR-6.3:** Dependencies SHALL be listed in requirements.txt  
**NFR-6.4:** Configuration SHALL be externalized  
**NFR-6.5:** System SHALL provide diagnostic logs  
**NFR-6.6:** Database schema SHALL support migrations  

### 5.7 Portability Requirements

**NFR-7.1:** System SHALL run on Raspberry Pi 3/4  
**NFR-7.2:** System SHALL support both 32-bit and 64-bit OS  
**NFR-7.3:** Face model SHALL be platform-independent (TFLite)  
**NFR-7.4:** Web dashboard SHALL work on all modern browsers  

### 5.8 Usability Requirements

**NFR-8.1:** System SHALL require zero training for students  
**NFR-8.2:** Dashboard SHALL be intuitive (< 5 min learning)  
**NFR-8.3:** Error messages SHALL be user-friendly  
**NFR-8.4:** System SHALL provide visual and audio feedback  
**NFR-8.5:** Dashboard SHALL be accessible (WCAG 2.1 Level A)  

### 5.9 Scalability Requirements

**NFR-9.1:** System SHALL support up to 200 students  
**NFR-9.2:** Database SHALL handle 10,000+ attendance records  
**NFR-9.3:** System SHALL support multiple cameras (future)  
**NFR-9.4:** Dashboard SHALL handle 100 concurrent users  

---

## 6. Other Requirements

### 6.1 Legal Requirements

**LR-1:** System SHALL comply with local data privacy laws  
**LR-2:** Student consent SHALL be obtained for face data  
**LR-3:** System SHALL support data deletion requests  
**LR-4:** System SHALL maintain audit trail for 1 year  

### 6.2 Regulatory Requirements

**RR-1:** Hardware SHALL comply with electrical safety standards  
**RR-2:** System SHALL meet CE/FCC electromagnetic compliance  

### 6.3 Data Requirements

**DR-1:** Database SHALL be backed up daily at midnight  
**DR-2:** Backups SHALL be retained for 30 days  
**DR-3:** System SHALL support data export to CSV/Excel  
**DR-4:** System SHALL support database reset/clear  

### 6.4 Installation Requirements

**IR-1:** System SHALL provide installation script  
**IR-2:** Installation SHALL complete in < 30 minutes  
**IR-3:** System SHALL validate hardware on first run  
**IR-4:** System SHALL create database automatically  

### 6.5 Documentation Requirements

**DOC-1:** User manual SHALL be provided  
**DOC-2:** API documentation SHALL be provided  
**DOC-3:** Installation guide SHALL be provided  
**DOC-4:** Troubleshooting guide SHALL be provided  
**DOC-5:** Proteus simulation guide SHALL be provided  

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| DOPA | Digital Online Personal Assistant - AI chatbot name |
| Face Embedding | Numerical representation of face features |
| Confidence Score | Probability (0-1) of correct face match |
| GPIO | General Purpose Input/Output pins |
| TFLite | TensorFlow Lite - lightweight ML framework |
| SMTP | Simple Mail Transfer Protocol |
| WebSocket | Full-duplex communication protocol |
| SRS | Software Requirements Specification |

---

## Appendix B: Requirement Traceability Matrix

| Requirement ID | Feature | Priority | Implementation Status |
|----------------|---------|----------|----------------------|
| FR-1.x | Face Recognition | HIGH | Complete |
| FR-2.x | Door Control | HIGH | Complete |
| FR-3.x | Attendance Management | HIGH | Complete |
| FR-4.x | Web Dashboard | MEDIUM | Complete |
| FR-5.x | AI Assistant | MEDIUM | Complete |
| FR-6.x | Report Generation | MEDIUM | Complete |
| FR-7.x | Email Automation | LOW | Complete |
| FR-8.x | Student Management | MEDIUM | Complete |

---

## Document Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Project Lead | [Your Name] | ________ | 2025-12-07 |
| Developer | [Your Name] | ________ | 2025-12-07 |
| Reviewer | [Instructor] | ________ | ________ |
| Approver | [Instructor] | ________ | ________ |

---

**End of Document**

**Version History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-07 | [Your Name] | Initial release |

---

**Contact Information:**

For questions or clarifications regarding this SRS, please contact:
- Project Lead: [Your Email]
- Repository: [GitHub URL]
