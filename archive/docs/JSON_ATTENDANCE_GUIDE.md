# JSON Attendance Format - Quick Guide

## What Changed

Your webcam recognition system now saves attendance in **JSON format by default** instead of CSV!

## JSON Format Example

```json
[
  {
    "Name": "Mehran",
    "Date": "2025-12-07",
    "Time": "16:58:52",
    "Status": "Present"
  },
  {
    "Name": "Zahir",
    "Date": "2025-12-07",
    "Time": "17:00:46",
    "Status": "Present"
  }
]
```

## How to Use

### Run Webcam Recognition
```bash
python webcam_recognition.py
```

### New Keyboard Controls
- **Press 'q'** - Quit and auto-save as JSON
- **Press 'j'** - Save as JSON manually
- **Press 'c'** - Save as CSV manually
- **Press 'r'** - Reset today's attendance

### Auto-Save on Exit
When you quit (press 'q'), the system automatically saves to `attendance.json`

## Convert Existing CSV to JSON

If you have old CSV files, convert them easily:

```bash
# Convert attendance.csv to attendance.json
python csv_to_json.py attendance.csv

# Or specify custom output
python csv_to_json.py old_attendance.csv new_attendance.json
```

## Why JSON?

### Advantages:
- ✅ Better for APIs and web applications
- ✅ Easier to parse in JavaScript/Python
- ✅ More structured and readable
- ✅ Supports nested data (for future features)
- ✅ Industry standard for data exchange

### When to Use CSV:
- 📊 Opening in Excel/Google Sheets
- 📈 Data analysis with pandas
- 📋 Simple tabular reports

## File Locations

After running the webcam system:
- **JSON**: `d:\face_det\attendance.json` (default)
- **CSV**: `d:\face_det\attendance.csv` (if you press 'c')

## Reading JSON in Python

```python
import json

# Read attendance
with open('attendance.json', 'r') as f:
    attendance = json.load(f)

# Print all records
for record in attendance:
    print(f"{record['Name']} - {record['Date']} {record['Time']}")

# Filter by name
zahir_records = [r for r in attendance if r['Name'] == 'Zahir']
```

## Reading JSON in JavaScript

```javascript
// Fetch and parse
fetch('attendance.json')
  .then(response => response.json())
  .then(data => {
    data.forEach(record => {
      console.log(`${record.Name} - ${record.Date} ${record.Time}`);
    });
  });
```

## Summary

✅ **Default format**: JSON
✅ **Auto-save**: Enabled on exit
✅ **Manual save**: Press 'j' for JSON, 'c' for CSV
✅ **Converter**: Use `csv_to_json.py` for old files

Your attendance system is now more modern and flexible! 🎉
