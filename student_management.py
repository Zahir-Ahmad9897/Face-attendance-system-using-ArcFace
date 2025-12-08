"""
Student Management Module
Add, remove, and manage students in the system
"""

from database import get_all_students, add_student as db_add_student
from logger import db_logger
import sqlite3

def get_student_list():
    """Get current list of all students"""
    try:
        students = get_all_students()
        db_logger.info(f"Retrieved {len(students)} students")
        return {"success": True, "students": students, "count": len(students)}
    except Exception as e:
        db_logger.error(f"Error getting student list: {e}")
        return {"success": False, "error": str(e)}

def add_new_student(name: str):
    """Add a new student to the system"""
    try:
        # Validate name
        name = name.strip().title()
        
        if not name:
            return {"success": False, "error": "Student name cannot be empty"}
        
        if len(name) < 2:
            return {"success": False, "error": "Student name too short"}
        
        # Check if already exists
        existing_students = get_all_students()
        if name in existing_students:
            return {"success": False, "error": f"Student '{name}' already exists"}
        
        # Add to database
        student_id = db_add_student(name)
        
        if student_id:
            db_logger.info(f"Added new student: {name}")
            return {"success": True, "message": f"Student '{name}' added successfully", "student_id": student_id}
        else:
            return {"success": False, "error": "Failed to add student to database"}
    
    except Exception as e:
        db_logger.error(f"Error adding student {name}: {e}")
        return {"success": False, "error": str(e)}

def remove_student(name: str):
    """Remove a student from the system (admin only)"""
    try:
        from database import get_connection
        
        name = name.strip().title()
        
        # Check if student exists
        existing_students = get_all_students()
        if name not in existing_students:
            return {"success": False, "error": f"Student '{name}' not found"}
        
        # Remove from database
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM students WHERE name = ?", (name,))
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()
        
        if rows_affected > 0:
            db_logger.warning(f"Removed student: {name}")
            return {"success": True, "message": f"Student '{name}' removed successfully"}
        else:
            return {"success": False, "error": "Failed to remove student"}
    
    except Exception as e:
        db_logger.error(f"Error removing student {name}: {e}")
        return {"success": False, "error": str(e)}

def get_student_statistics(name: str):
    """Get attendance statistics for a specific student"""
    try:
        from database import get_all_attendance, get_connection
        
        name = name.strip().title()
        
        # Get all attendance records for this student
        all_records = get_all_attendance()
        student_records = [r for r in all_records if r['Name'] == name]
        
        if not student_records:
            return {
                "success": True,
                "student": name,
                "total_days": 0,
                "attendance_rate": 0,
                "last_attendance": None
            }
        
        # Calculate statistics
        total_days = len(student_records)
        last_record = student_records[0]  # Already sorted by date desc
        
        # Get total working days (unique dates in system)
        all_dates = set(r['Date'] for r in all_records)
        total_working_days = len(all_dates)
        
        attendance_rate = (total_days / total_working_days * 100) if total_working_days > 0 else 0
        
        return {
            "success": True,
            "student": name,
            "total_days_present": total_days,
            "total_working_days": total_working_days,
            "attendance_rate": round(attendance_rate, 2),
            "last_attendance": {
                "date": last_record['Date'],
                "time": last_record['Time']
            }
        }
    
    except Exception as e:
        db_logger.error(f"Error getting statistics for {name}: {e}")
        return {"success": False, "error": str(e)}

def bulk_add_students(names: list):
    """Add multiple students at once"""
    results = []
    
    for name in names:
        result = add_new_student(name)
        results.append({
            "name": name,
            "success": result["success"],
            "message": result.get("message") or result.get("error")
        })
    
    successful = sum(1 for r in results if r["success"])
    
    return {
        "success": True,
        "total": len(names),
        "successful": successful,
        "failed": len(names) - successful,
        "details": results
    }
