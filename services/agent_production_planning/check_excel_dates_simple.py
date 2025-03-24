import pandas as pd
import numpy as np
from datetime import datetime
import re

# Path to the Excel file
excel_file = "/Users/carrickcheah/llms_project/services/agent_production_planning/mydata.xlsx"

# Load the Excel file with pandas, keeping dates as strings initially
print(f"Loading Excel file: {excel_file}")
df_raw = pd.read_excel(excel_file, sheet_name='Jobs PlanningDetails-Draft', header=3, dtype=str)

# Find the START_DATE column
start_date_cols = [col for col in df_raw.columns if 'START_DATE' in str(col).strip()]
if start_date_cols:
    start_date_col = start_date_cols[0]
    print(f"Found START_DATE column: '{start_date_col}'")
    
    # Print the first few raw values
    print("\nRaw START_DATE values:")
    sample_values = df_raw[start_date_col].dropna().head(10)
    for idx, val in sample_values.items():
        print(f"Row {idx}: '{val}' (type: {type(val)})")
    
    # Now load with pandas default date parsing
    print("\nLoading with pandas default date parsing:")
    df = pd.read_excel(excel_file, sheet_name='Jobs PlanningDetails-Draft', header=3)
    
    # Check how pandas parsed the dates
    if pd.api.types.is_datetime64_any_dtype(df[start_date_col]):
        print(f"Pandas parsed START_DATE as datetime64 type: {df[start_date_col].dtype}")
        sample_dates = df[start_date_col].dropna().head(10)
        for idx, val in sample_dates.items():
            print(f"Row {idx}: {val} - Hour: {val.hour}, Minute: {val.minute}")
    else:
        print(f"Pandas did not parse START_DATE as datetime. Type: {df[start_date_col].dtype}")
        
    # Try to manually parse with different formats
    print("\nTrying different date formats:")
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y %H:%M',
        '%d/%m/%Y',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M',
        '%m/%d/%Y'
    ]
    
    for fmt in formats:
        try:
            parsed_dates = pd.to_datetime(df_raw[start_date_col], format=fmt, errors='coerce')
            valid_count = parsed_dates.notna().sum()
            if valid_count > 0:
                print(f"Format {fmt}: {valid_count} valid dates")
                sample = parsed_dates.dropna().head(3)
                for idx, val in sample.items():
                    print(f"  Row {idx}: {val} - Hour: {val.hour}, Minute: {val.minute}")
        except Exception as e:
            print(f"Error with format {fmt}: {e}")
    
    # Check if there are any time components in the raw strings
    print("\nChecking for time components in raw strings:")
    time_pattern = r'(\d{1,2}):(\d{2})'
    for idx, val in df_raw[start_date_col].dropna().head(10).items():
        if isinstance(val, str):
            time_match = re.search(time_pattern, val)
            if time_match:
                print(f"Row {idx}: Found time component in '{val}': {time_match.group(0)}")
            else:
                print(f"Row {idx}: No time component found in '{val}'")
else:
    print("START_DATE column not found in the Excel file")
