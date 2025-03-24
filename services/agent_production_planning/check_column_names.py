#!/usr/bin/env python3
"""
Script to inspect Excel column names to help solve the date column disparency issue.
"""

import pandas as pd

# Path to Excel file
file_path = "/Users/carrickcheah/llms_project/services/agent_production_planning/mydata.xlsx"
sheet_name = "Jobs PlanningDetails-Draft"

print(f"Reading Excel file: {file_path}, Sheet: {sheet_name}")
print("-" * 70)

# Read Excel file
df = pd.read_excel(file_path, sheet_name=sheet_name, header=3)

# Display column information
print("Excel columns found:")
for i, col in enumerate(df.columns):
    print(f"{i+1}. '{col}' (type: {type(col).__name__})")

print("-" * 70)

# Look for date columns specifically
date_keywords = ['DATE', 'START', 'END', 'COMPLETION']
print("Possible date columns:")
for col in df.columns:
    if any(keyword in str(col).upper() for keyword in date_keywords):
        # Check if pandas detected it as datetime
        is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
        print(f"- '{col}' (datetime: {is_datetime})")
        
        # Show sample values
        if not df[col].empty:
            sample = df[col].dropna().head(3)
            print(f"  Sample values: {sample.tolist()}")
