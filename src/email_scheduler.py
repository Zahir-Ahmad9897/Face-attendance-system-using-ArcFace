import asyncio
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, time
from pathlib import Path

try:
    from database import get_statistics, get_all_students
except ImportError:
    print("Warning: database module not found")

CONFIG_FILE = "email_config.json"

# ============================================================================
# Configuration Management
# ============================================================================

def load_config():
    """Load email configuration from JSON file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return get_default_config()
    return get_default_config()

def get_default_config():
    """Get default configuration"""
    return {
        "sender_email": "your-email@gmail.com",
        "app_password": "your-app-password",
        "teacher_email": "teacher@example.com",
        "send_time": "17:00",
        "enabled": False,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587
    }

def save_config(config: dict):
    """Save email configuration to JSON file"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def is_configured():
    """Check if email is properly configured"""
    config = load_config()
    return (
        config.get('sender_email', '') != 'your-email@gmail.com' and
        config.get('app_password', '') != 'your-app-password' and
        config.get('teacher_email', '') != 'teacher@example.com' and
        config.get('enabled', False)
    )

# ============================================================================
# Email Sending
# ============================================================================

def send_email(subject: str, body: str, recipient_email: str = None):
    """Send email using configured SMTP settings"""
    config = load_config()
    
    if recipient_email is None:
        recipient_email = config.get('teacher_email', '')
    
    sender_email = config.get('sender_email', '')
    app_password = config.get('app_password', '')
    smtp_server = config.get('smtp_server', 'smtp.gmail.com')
    smtp_port = config.get('smtp_port', 587)
    
    # Validate configuration
    if sender_email == 'your-email@gmail.com' or not sender_email:
        return False, "Sender email not configured"
    
    if app_password == 'your-app-password' or not app_password:
        return False, "App password not configured"
    
    if not recipient_email or recipient_email == 'teacher@example.com':
        return False, "Teacher email not configured"
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to server and send
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, app_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        print(f"✅ Email sent successfully to {recipient_email}")
        return True, "Email sent successfully"
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error sending email: {error_msg}")
        return False, error_msg

def generate_daily_report_html(date: str = None):
    """Generate HTML email report for daily attendance"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    stats = get_statistics(date)
    
    # Create HTML email
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px 10px 0 0;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 24px;
            }}
            .content {{
                background: #f9f9f9;
                padding: 30px;
                border-radius: 0 0 10px 10px;
            }}
            .stats {{
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
            }}
            .stat-box {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                flex: 1;
                margin: 0 10px;
            }}
            .stat-number {{
                font-size: 36px;
                font-weight: bold;
                color: #667eea;
            }}
            .stat-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }}
            .student-list {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .student-list h3 {{
                margin-top: 0;
                color: #333;
            }}
            .student {{
                padding: 10px;
                margin: 5px 0;
                border-left: 4px solid;
                background: #f0f0f0;
                border-radius: 4px;
            }}
            .present {{
                border-color: #10b981;
            }}
            .absent {{
                border-color: #ef4444;
            }}
            .footer {{
                text-align: center;
                color: #666;
                font-size: 12px;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Daily Attendance Report</h1>
            <p style="margin: 5px 0 0 0;">{date}</p>
        </div>
        
        <div class="content">
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-number">{stats['total_students']}</div>
                    <div class="stat-label">Total Students</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" style="color: #10b981;">{stats['present_count']}</div>
                    <div class="stat-label">Present</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" style="color: #ef4444;">{stats['absent_count']}</div>
                    <div class="stat-label">Absent</div>
                </div>
            </div>
            
            <div class="student-list">
                <h3>✅ Present Students ({stats['present_count']})</h3>
    """
    
    if stats['present_students']:
        for student in stats['present_students']:
            html += f'<div class="student present">✓ {student}</div>\n'
    else:
        html += '<p style="color: #666; font-style: italic;">No students present</p>'
    
    html += """
            </div>
            
            <div class="student-list">
                <h3>❌ Absent Students ({absent_count})</h3>
    """.format(absent_count=stats['absent_count'])
    
    if stats['absent_students']:
        for student in stats['absent_students']:
            html += f'<div class="student absent">✗ {student}</div>\n'
    else:
        html += '<p style="color: #666; font-style: italic;">All students present!</p>'
    
    html += f"""
            </div>
            
            <p style="margin-top: 30px; padding: 15px; background: #e0e7ff; border-radius: 8px; text-align: center;">
                <strong>Attendance Rate: {stats['attendance_percentage']}%</strong>
            </p>
        </div>
        
        <div class="footer">
            <p>Generated by Face Recognition Attendance System</p>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """
    
    return html

def send_daily_report(date: str = None, recipient_email: str = None):
    """Send daily attendance report"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    subject = f"Daily Attendance Report - {date}"
    body = generate_daily_report_html(date)
    
    return send_email(subject, body, recipient_email)

# ============================================================================
# Scheduler
# ============================================================================

async def email_scheduler_task():
    """Background task to send daily emails at scheduled time"""
    print("📧 Email scheduler started")
    
    last_sent_date = None
    
    while True:
        try:
            if not is_configured():
                # Wait 60 seconds before checking again
                await asyncio.sleep(60)
                continue
            
            config = load_config()
            send_time_str = config.get('send_time', '17:00')
            
            # Parse send time
            hour, minute = map(int, send_time_str.split(':'))
            send_time_obj = time(hour, minute)
            
            now = datetime.now()
            current_time = now.time()
            current_date = now.strftime('%Y-%m-%d')
            
            # Check if it's time to send AND we haven't sent today yet
            if (current_time.hour == send_time_obj.hour and 
                current_time.minute == send_time_obj.minute and
                last_sent_date != current_date):
                
                print(f"⏰ Sending scheduled daily report for {current_date}")
                success, message = send_daily_report()
                
                if success:
                    last_sent_date = current_date
                    print(f"✅ Daily report sent successfully")
                else:
                    print(f"❌ Failed to send daily report: {message}")
            
            # Wait 60 seconds before next check
            await asyncio.sleep(60)
        
        except Exception as e:
            print(f"Error in email scheduler: {e}")
            await asyncio.sleep(60)

# ============================================================================
# Test Functions
# ============================================================================

def send_test_email():
    """Send a test email to verify configuration"""
    subject = "Test Email - Face Recognition Attendance System"
    
    body = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
            }
            .content {
                background: #f9f9f9;
                padding: 30px;
                margin-top: 20px;
                border-radius: 10px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>✅ Test Email</h1>
        </div>
        <div class="content">
            <h2>Configuration Successful!</h2>
            <p>Your email configuration is working correctly.</p>
            <p>Daily attendance reports will be sent at the scheduled time.</p>
            <p><strong>Time:</strong> {}</p>
        </div>
    </body>
    </html>
    """.format(load_config().get('send_time', '17:00'))
    
    return send_email(subject, body)

if __name__ == "__main__":
    # Test email sending
    print("Testing email configuration...")
    success, message = send_test_email()
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
