import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

DATABASE_FILE = "attendance.db"
ATTENDANCE_JSON = "attendance.json"

# Default student list - can be updated
ALL_STUDENTS = ["Mehran", "Yousaf", "Zahir"]

# ============================================================================
# Database Setup
# ============================================================================

def get_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn

def init_database():
    """Initialize database tables"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create attendance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'Present',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_attendance_date 
        ON attendance(date)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_attendance_name 
        ON attendance(name)
    """)
    
    conn.commit()
    conn.close()
    
    # Initialize students if table is empty
    initialize_students()

def initialize_students():
    """Add default students to database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    for student in ALL_STUDENTS:
        try:
            cursor.execute("INSERT INTO students (name) VALUES (?)", (student,))
        except sqlite3.IntegrityError:
            # Student already exists, skip
            pass
    
    conn.commit()
    conn.close()

# ============================================================================
# Migration
# ============================================================================

def migrate_json_to_db():
    """Migrate existing JSON attendance data to database"""
    if not os.path.exists(ATTENDANCE_JSON):
        print("No JSON file to migrate.")
        return 0
    
    try:
        with open(ATTENDANCE_JSON, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        if not records:
            print("JSON file is empty.")
            return 0
        
        conn = get_connection()
        cursor = conn.cursor()
        
        migrated_count = 0
        for record in records:
            try:
                cursor.execute("""
                    INSERT INTO attendance (name, date, time, status)
                    VALUES (?, ?, ?, ?)
                """, (
                    record.get('Name', ''),
                    record.get('Date', ''),
                    record.get('Time', ''),
                    record.get('Status', 'Present')
                ))
                migrated_count += 1
            except Exception as e:
                print(f"Error migrating record: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"✅ Migrated {migrated_count} records from JSON to database")
        
        # Backup JSON file
        backup_name = f"attendance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.rename(ATTENDANCE_JSON, backup_name)
        print(f"📦 JSON file backed up as: {backup_name}")
        
        return migrated_count
    
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        return 0

# ============================================================================
# Insert Operations
# ============================================================================

def add_attendance(name: str, date: str = None, time: str = None, status: str = 'Present'):
    """Add attendance record to database - Only one entry per student per day"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    if time is None:
        time = datetime.now().strftime('%H:%M:%S')
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if student already has attendance for this date
    cursor.execute("""
        SELECT id FROM attendance 
        WHERE name = ? AND date = ?
    """, (name, date))
    
    existing = cursor.fetchone()
    
    if existing:
        # Student already marked for today - update time to latest
        cursor.execute("""
            UPDATE attendance 
            SET time = ?, status = ?
            WHERE name = ? AND date = ?
        """, (time, status, name, date))
        record_id = existing['id']
        print(f"Updated {name}'s attendance for {date} to {time}")
    else:
        # New entry for today
        cursor.execute("""
            INSERT INTO attendance (name, date, time, status)
            VALUES (?, ?, ?, ?)
        """, (name, date, time, status))
        record_id = cursor.lastrowid
        print(f"Added {name}'s attendance for {date} at {time}")
    
    conn.commit()
    conn.close()
    
    return record_id

def add_student(name: str):
    """Add a new student to the database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("INSERT INTO students (name) VALUES (?)", (name,))
        conn.commit()
        student_id = cursor.lastrowid
        conn.close()
        return student_id
    except sqlite3.IntegrityError:
        conn.close()
        return None  # Student already exists

# ============================================================================
# Query Operations
# ============================================================================

def get_all_students() -> List[str]:
    """Get list of all registered students"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM students ORDER BY name")
    students = [row['name'] for row in cursor.fetchall()]
    
    conn.close()
    return students if students else ALL_STUDENTS

def get_today_attendance() -> List[Dict]:
    """Get today's attendance records"""
    today = datetime.now().strftime('%Y-%m-%d')
    return get_attendance_by_date(today)

def get_attendance_by_date(date: str) -> List[Dict]:
    """Get attendance records for a specific date"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT name, date, time, status
        FROM attendance
        WHERE date = ?
        ORDER BY time
    """, (date,))
    
    records = []
    for row in cursor.fetchall():
        records.append({
            'Name': row['name'],
            'Date': row['date'],
            'Time': row['time'],
            'Status': row['status']
        })
    
    conn.close()
    return records

def get_all_attendance() -> List[Dict]:
    """Get all attendance records"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT name, date, time, status
        FROM attendance
        ORDER BY date DESC, time DESC
    """)
    
    records = []
    for row in cursor.fetchall():
        records.append({
            'Name': row['name'],
            'Date': row['date'],
            'Time': row['time'],
            'Status': row['status']
        })
    
    conn.close()
    return records

def get_present_students(date: str = None) -> List[str]:
    """Get list of students present on a specific date"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT name
        FROM attendance
        WHERE date = ? AND status = 'Present'
    """, (date,))
    
    present = [row['name'] for row in cursor.fetchall()]
    conn.close()
    
    return present

def get_absent_students(date: str = None) -> List[str]:
    """Get list of students absent on a specific date"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    all_students = get_all_students()
    present = get_present_students(date)
    
    absent = [student for student in all_students if student not in present]
    return absent

def get_statistics(date: str = None) -> Dict:
    """Get attendance statistics for a specific date"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    present = get_present_students(date)
    absent = get_absent_students(date)
    total_students = len(get_all_students())
    
    # Get total records for today
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM attendance WHERE date = ?", (date,))
    total_records = cursor.fetchone()['count']
    
    # Get total days with attendance
    cursor.execute("SELECT COUNT(DISTINCT date) as count FROM attendance")
    total_days = cursor.fetchone()['count']
    conn.close()
    
    return {
        'date': date,
        'total_students': total_students,
        'total_people_today': len(present),  # For frontend compatibility
        'total_absent_today': len(absent),   # For frontend compatibility
        'total_records_today': total_records,
        'total_days': total_days,
        'people_present_today': present,
        'people_absent_today': absent,
        'present_count': len(present),
        'absent_count': len(absent),
        'present_students': present,
        'absent_students': absent,
        'attendance_percentage': round((len(present) / total_students * 100), 2) if total_students > 0 else 0,
        'last_updated': datetime.now().isoformat()
    }

def get_date_range_statistics(start_date: str, end_date: str) -> Dict:
    """Get statistics for a date range"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT date, COUNT(DISTINCT name) as present_count
        FROM attendance
        WHERE date BETWEEN ? AND ?
        GROUP BY date
        ORDER BY date
    """, (start_date, end_date))
    
    daily_stats = []
    for row in cursor.fetchall():
        daily_stats.append({
            'date': row['date'],
            'present_count': row['present_count']
        })
    
    conn.close()
    return daily_stats

# ============================================================================
# Utility Functions
# ============================================================================

def clear_all_attendance():
    """Clear all attendance records (use with caution)"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance")
    conn.commit()
    conn.close()

def is_student_present_today(name: str) -> bool:
    """Check if a student is present today"""
    today = datetime.now().strftime('%Y-%m-%d')
    present = get_present_students(today)
    return name in present

# ============================================================================
# Initialize on import
# ============================================================================

# Auto-initialize database on first import
if not os.path.exists(DATABASE_FILE):
    print("🔧 Creating database for the first time...")
    init_database()
    
    # Auto-migrate JSON data if exists
    if os.path.exists(ATTENDANCE_JSON):
        print("📋 Found existing JSON data, migrating to database...")
        migrate_json_to_db()
    
    print("✅ Database ready!")
else:
    # Ensure tables exist
    init_database()
