"""
Authentication module for Face Recognition Attendance System
Provides basic HTTP authentication for dashboard access
"""

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import hashlib
import json
import os

security = HTTPBasic()

# Default credentials file
CREDENTIALS_FILE = "users.json"

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    if os.path.exists(CREDENTIALS_FILE):
        try:
            with open(CREDENTIALS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users: dict):
    """Save users to JSON file"""
    with open(CREDENTIALS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def create_default_user():
    """Create default admin user if no users exist"""
    users = load_users()
    
    if not users:
        # Default credentials
        default_user = {
            "admin": {
                "password": hash_password("admin123"),
                "role": "admin"
            },
            "teacher": {
                "password": hash_password("teacher123"),
                "role": "teacher"
            }
        }
        save_users(default_user)
        print("=" * 70)
        print("🔐 Created default users:")
        print("   Username: admin    | Password: admin123    | Role: admin")
        print("   Username: teacher  | Password: teacher123  | Role: teacher")
        print("=" * 70)
        print("⚠️  CHANGE THESE PASSWORDS IMMEDIATELY!")
        print("=" * 70)
        return default_user
    
    return users

def verify_credentials(credentials: HTTPBasicCredentials = Security(security)):
    """Verify username and password"""
    users = load_users()
    
    username = credentials.username
    password = credentials.password
    
    # Check if user exists
    if username not in users:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    # Verify password
    stored_password_hash = users[username]["password"]
    password_hash = hash_password(password)
    
    if not secrets.compare_digest(password_hash, stored_password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return {
        "username": username,
        "role": users[username].get("role", "user")
    }

def change_password(username: str, old_password: str, new_password: str) -> bool:
    """Change user password"""
    users = load_users()
    
    if username not in users:
        return False
    
    # Verify old password
    old_hash = hash_password(old_password)
    if not secrets.compare_digest(old_hash, users[username]["password"]):
        return False
    
    # Update password
    users[username]["password"] = hash_password(new_password)
    save_users(users)
    return True

def add_user(admin_username: str, new_username: str, new_password: str, role: str = "user") -> bool:
    """Add new user (admin only)"""
    users = load_users()
    
    # Verify admin
    if users.get(admin_username, {}).get("role") != "admin":
        return False
    
    # Add user
    if new_username in users:
        return False  # User already exists
    
    users[new_username] = {
        "password": hash_password(new_password),
        "role": role
    }
    save_users(users)
    return True

# Initialize default users on import
create_default_user()
