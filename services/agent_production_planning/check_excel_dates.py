import pandas as pd
import openpyxl
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the Excel file
excel_file = "/Users/carrickcheah/llms_project/services/agent_production_planning/mydata.xlsx"

# Load the Excel file directly with openpyxl to check cell formats
logger.info(f"Loading Excel file with openpyxl: {excel_file}")
wb = openpyxl.load_workbook(excel_file, data_only=True)
sheet = wb['Jobs PlanningDetails-Draft']

# Find the header row (row 4)
header_row = 4
header_cells = sheet[header_row]
column_indices = {}

# Find the START_DATE column index
for idx, cell in enumerate(header_cells, 1):
    if cell.value and 'START_DATE' in str(cell.value).strip():
        column_indices['START_DATE'] = idx
        logger.info(f"Found START_DATE column at index {idx}, value: '{cell.value}'")

# Check the first few START_DATE values and their formats
if 'START_DATE' in column_indices:
    start_date_col = column_indices['START_DATE']
    logger.info(f"Checking START_DATE values in column {start_date_col}")
    
    for row_idx in range(header_row + 1, header_row + 11):  # Check first 10 data rows
        cell = sheet.cell(row=row_idx, column=start_date_col)
        
        # Get the cell value
        value = cell.value
        
        # Get the cell's number format
        number_format = cell.number_format
        
        # Get the cell's style
        style = cell._style
        
        logger.info(f"Row {row_idx}: Value={value}, Type={type(value)}, Format={number_format}")
        
        # If it's a datetime, print it with different formats
        if isinstance(value, datetime):
            logger.info(f"  As ISO: {value.isoformat()}")
            logger.info(f"  Hour: {value.hour}, Minute: {value.minute}")

# Now load with pandas to see how pandas interprets the dates
logger.info("\nLoading Excel file with pandas")
df = pd.read_excel(excel_file, sheet_name='Jobs PlanningDetails-Draft', header=3)

# Find the START_DATE column
start_date_cols = [col for col in df.columns if 'START_DATE' in str(col).strip()]
if start_date_cols:
    start_date_col = start_date_cols[0]
    logger.info(f"Found START_DATE column in pandas: '{start_date_col}'")
    
    # Check the first few values
    sample_values = df[start_date_col].head(10)
    logger.info(f"Sample values from pandas:\n{sample_values}")
    
    # Check the data type
    logger.info(f"Data type: {df[start_date_col].dtype}")
    
    # If it's a datetime type, check the time components
    if pd.api.types.is_datetime64_any_dtype(df[start_date_col]):
        for idx, val in sample_values.items():
            if pd.notna(val):
                logger.info(f"Row {idx}: {val} - Hour: {val.hour}, Minute: {val.minute}")
