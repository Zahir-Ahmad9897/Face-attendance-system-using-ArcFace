# 🤝 Contributing to Face Recognition Attendance System

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

---

## 📜 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

---

## 🚀 Getting Started

### 1. Fork the Repository
```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Face-attendance-system-using-ArcFace.git
cd Face-attendance-system-using-ArcFace
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r web_requirements.txt
```

### 3. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

---

## 💻 Development Workflow

### Project Structure
```
src/          # Source code
models/       # AI models
static/       # Web frontend
scripts/      # Automation scripts
docs/         # Documentation
data/         # Data files (gitignored)
```

### Running Tests
```bash
# Test the web server
python src/app.py

# Test face recognition
python src/webcam_recognition.py

# Test specific modules
python -m pytest tests/  # (if tests exist)
```

---

## 📝 Coding Standards

### Python Style Guide

Follow **PEP 8** guidelines:

```python
# Good
def process_attendance(student_id, timestamp):
    """
    Process student attendance record.
    
    Args:
        student_id (int): Student identification number
        timestamp (datetime): Attendance timestamp
    
    Returns:
        bool: True if successful, False otherwise
    """
    pass

# Bad
def processAttendance(id,ts):
    pass
```

### Key Principles

- **Use meaningful variable names**
- **Add docstrings to functions**
- **Keep functions small and focused**
- **Comment complex logic**
- **Handle errors gracefully**

### File Organization

- Place Python source in `src/`
- Place documentation in `docs/`
- Place scripts in `scripts/`
- Don't commit sensitive data (databases, config files)

---

## 🔄 Submitting Changes

### 1. Commit Your Changes
```bash
git add .
git commit -m "feat: add student bulk upload feature"
```

### Commit Message Format
```
<type>: <description>

[optional body]
[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat: add CSV export for attendance reports"
git commit -m "fix: resolve camera disconnection issue"
git commit -m "docs: update installation instructions"
```

### 2. Push to Your Fork
```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template:
   - **Title**: Clear, concise description
   - **Description**: What changed and why
   - **Testing**: How you tested the changes
   - **Screenshots**: If UI changes

---

## 🐛 Reporting Bugs

### Before Reporting

1. Check existing issues
2. Try latest version
3. Reproduce the bug

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**Steps to Reproduce**
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 10]
- Python Version: [e.g., 3.9]
- Browser: [e.g., Chrome 96]

**Logs/Screenshots**
Attach relevant logs or screenshots
```

---

## 💡 Feature Requests

### Feature Request Template

```markdown
**Problem**
Describe the problem you're trying to solve

**Solution**
Describe your proposed solution

**Alternatives**
Alternative solutions you've considered

**Additional Context**
Any other relevant information
```

---

## 🧪 Testing Guidelines

Before submitting:

- [ ] Code runs without errors
- [ ] Tested with real webcam
- [ ] Web dashboard loads correctly
- [ ] No console errors
- [ ] Database operations work
- [ ] Email functionality works
- [ ] Documentation updated

---

## 📚 Documentation

When adding features:

1. Update relevant `.md` files in `docs/`
2. Add docstrings to new functions
3. Update `README.md` if needed
4. Add examples if applicable

---

## 🎯 Priority Areas

We welcome contributions in these areas:

- 🐛 Bug fixes
- ✨ New features from roadmap
- 📝 Documentation improvements
- 🧪 Test coverage
- 🎨 UI/UX enhancements
- ⚡ Performance optimizations
- 🌐 Internationalization

---

## ❓ Questions?

- **GitHub Issues**: For bugs and features
- **Discussions**: For questions and ideas
- **Email**: For private inquiries

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing! 🙏**

Every contribution, no matter how small, makes a difference!
