"""
Quick Email Configuration Setup Script
This script helps you set up email configuration interactively.
"""

import json
import os
from pathlib import Path

def setup_email_config():
    """Interactive setup for email configuration"""
    print("="*70)
    print("📧 EMAIL CONFIGURATION SETUP")
    print("="*70)
    print("\nThis script will help you configure email settings.")
    print("You'll need:")
    print("  1. A Gmail account")
    print("  2. Gmail App Password (not your regular password)")
    print("  3. Recipient email address")
    print("\n" + "="*70)
    
    # Get user inputs
    print("\n📝 Enter your details:\n")
    
    sender_email = input("Your Gmail address: ").strip()
    app_password = input("Gmail App Password (16 characters): ").strip()
    teacher_email = input("Recipient email address: ").strip()
    
    print("\n⏰ Email send time (for scheduled mode, not used for immediate send):")
    send_time = input("Send time [HH:MM, default 17:00]: ").strip() or "17:00"
    
    # Validate inputs
    if not sender_email or '@' not in sender_email:
        print("❌ Invalid sender email!")
        return False
    
    if not app_password or len(app_password) < 10:
        print("❌ Invalid app password!")
        return False
    
    if not teacher_email or '@' not in teacher_email:
        print("❌ Invalid recipient email!")
        return False
    
    # Create configuration
    config = {
        "sender_email": sender_email,
        "app_password": app_password,
        "teacher_email": teacher_email,
        "send_time": send_time,
        "enabled": True,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587
    }
    
    # Determine config file path
    config_dir = Path(__file__).parent.parent / "config"
    config_file = config_dir / "email_config.json"
    
    # Create directory if needed
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("✅ Configuration saved successfully!")
    print(f"📁 Location: {config_file}")
    print("="*70)
    
    # Ask if user wants to send test email
    test = input("\n🧪 Send test email? (y/n): ").strip().lower()
    
    if test == 'y':
        print("\nSending test email...")
        try:
            from email_scheduler import send_test_email
            success, message = send_test_email()
            if success:
                print(f"\n✅ {message}")
                print("📧 Check your inbox!")
            else:
                print(f"\n❌ {message}")
                print("\nCommon issues:")
                print("  - Make sure 2FA is enabled on Gmail")
                print("  - Use App Password, not regular password")
                print("  - Check internet connection")
        except ImportError:
            print("❌ email_scheduler module not found")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*70)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("  1. Run: python webcam_recognition.py")
    print("  2. Press 'q' to quit")
    print("  3. Email will be sent automatically!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    try:
        setup_email_config()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
