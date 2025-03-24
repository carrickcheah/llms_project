import pandas as pd

# Path to the Excel file
excel_file = "/Users/carrickcheah/llms_project/services/agent_production_planning/mydata.xlsx"

# Load the Excel file with pandas
print(f"Loading Excel file: {excel_file}")

# First, check available sheet names
xl = pd.ExcelFile(excel_file)
print(f"Available sheets: {xl.sheet_names}")

# Try to load with different header positions
for header_row in range(10):
    try:
        df = pd.read_excel(excel_file, sheet_name=xl.sheet_names[0], header=header_row)
        print(f"\nWith header at row {header_row}:")
        print(f"Columns: {list(df.columns)}")
        
        # Check for columns containing 'START'
        start_cols = [col for col in df.columns if 'START' in str(col)]
        if start_cols:
            print(f"Found columns containing 'START': {start_cols}")
            
            # Print sample values for these columns
            for col in start_cols:
                print(f"\nSample values for '{col}':")
                sample = df[col].dropna().head(5)
                for idx, val in sample.items():
                    print(f"  Row {idx}: {val} (type: {type(val)})")
    except Exception as e:
        print(f"Error with header at row {header_row}: {e}")
