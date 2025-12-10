import pandas as pd
import json
import sys
import os

def csv_to_json(csv_file, json_file=None):
    """Convert CSV attendance file to JSON format"""
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        return False
    
    try:
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Convert to list of dictionaries
        data = df.to_dict('records')
        
        # Determine output filename
        if json_file is None:
            json_file = csv_file.replace('.csv', '.json')
        
        # Save as JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully converted!")
        print(f"   Input:  {csv_file}")
        print(f"   Output: {json_file}")
        print(f"   Records: {len(data)}")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        json_file = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        csv_file = "attendance.csv"
        json_file = "attendance.json"
    
    csv_to_json(csv_file, json_file)
