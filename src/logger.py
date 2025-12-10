import logging
import os
from datetime import datetime
from pathlib import Path

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
def setup_logging():
    """Setup comprehensive logging for the attendance system"""
    
    # Log filename with date
    log_filename = LOGS_DIR / f"attendance_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler - keeps 30 days of logs
            logging.FileHandler(log_filename, encoding='utf-8'),
            # Console handler - only errors and warnings
            logging.StreamHandler()
        ]
    )
    
    # Set console handler to only show warnings and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Create specific loggers
    app_logger = logging.getLogger('attendance_app')
    email_logger = logging.getLogger('email_scheduler')
    db_logger = logging.getLogger('database')
    camera_logger = logging.getLogger('camera')
    
    return {
        'app': app_logger,
        'email': email_logger,
        'db': db_logger,
        'camera': camera_logger
    }

# Cleanup old logs (keep last 30 days)
def cleanup_old_logs(days_to_keep=30):
    """Remove log files older than specified days"""
    import time
    
    if not LOGS_DIR.exists():
        return
    
    current_time = time.time()
    
    for log_file in LOGS_DIR.glob("attendance_*.log"):
        file_age_days = (current_time - log_file.stat().st_mtime) / 86400
        
        if file_age_days > days_to_keep:
            try:
                log_file.unlink()
                print(f"Deleted old log: {log_file.name}")
            except Exception as e:
                print(f"Error deleting log {log_file.name}: {e}")

# Initialize logging
loggers = setup_logging()
cleanup_old_logs()

# Export loggers
app_logger = loggers['app']
email_logger = loggers['email']
db_logger = loggers['db']
camera_logger = loggers['camera']

# Example usage:
# from logger import app_logger
# app_logger.info("Application started")
# app_logger.error("Something went wrong", exc_info=True)
