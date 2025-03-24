# ingest_data.py | dont edit this line
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import re
from dotenv import load_dotenv
from ingest_config import DATE_COLUMNS_CONFIG, DEFAULT_TIME

# Import the time disparency fix function if available
try:
    from fix_time_disparency import fix_time_disparency
    TIME_DISPARENCY_FIX_AVAILABLE = True
except ImportError:
    TIME_DISPARENCY_FIX_AVAILABLE = False
    logging.warning("fix_time_disparency module not found, falling back to standard date processing")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_job_family(process_code):
    """Extract the job family (e.g., 'CP08-231B') from the process code."""
    process_code = str(process_code).upper()
    match = re.search(r'(.*?)-P\d+', process_code)
    if match:
        family = match.group(1)
        return family
    parts = process_code.split("-P")
    if len(parts) >= 2:
        return parts[0]
    return process_code

def extract_process_number(process_code):
    """Extract the process sequence number (e.g., 1 from 'P01-06') or return 999 if not found."""
    process_code = str(process_code).upper()
    match = re.search(r'P(\d{2})', process_code)  # Match exactly two digits after P
    if match:
        seq = int(match.group(1))
        return seq
    return 999  # Default if parsing fails

def clean_excel_data(df):
    """Clean and prepare Excel data for scheduling."""
    df = df.copy()
    initial_rows = len(df)
    
    logger.info(f"Initial data: {initial_rows} rows")
    
    # Drop completely empty rows
    df = df.dropna(how='all')
    
    # Clean string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        df[col] = df[col].str.replace(r'[^\w\s-]', '', regex=True).str.replace(r'\s+', ' ', regex=True)
    
    # Define rules for numeric columns - updated with new column names
    numeric_cols = {
        'NUMBER_OPERATOR': {'min': 1, 'max': 10, 'default': 1},
        'HOURS_NEED': {'min': 0.1, 'max': 720, 'default': 1},
        'SETTING_HOUR': {'min': 0, 'max': 48, 'default': 0},  # Old name for setup time
        'SETTING_HOURS': {'min': 0, 'max': 48, 'default': 0},  # Setup/changeover time
        'BREAK_HOURS': {'min': 0, 'max': 24, 'default': 0},  # Break time
        'NO_PRODUCTION_HOUR': {'min': 0, 'max': 48, 'default': 0},  # Old name for downtime
        'NO_PROD': {'min': 0, 'max': 48, 'default': 0},  # Renamed from NO_PRODUCTION_HOUR
        'PRIORITY': {'min': 1, 'max': 5, 'default': 3},  # Priority
        'JOB_QUANTITY': {'min': 1, 'max': 1000000, 'default': 1000},  # New column
        'EXPECT_OUTPUT_PER_HOUR': {'min': 0.1, 'max': 10000, 'default': 100},  # New column
        'ACCUMULATED_DAILY_OUTPUT': {'min': 0, 'max': 1000000, 'default': 0},  # New column
        'BALANCE_QUANTITY': {'min': 0, 'max': 1000000, 'default': 0},  # New column
        'BAL_HR': {'min': 0, 'max': 1000, 'default': 0}  # Buffer hours
    }
    
    # Process numeric columns with validation
    for col, rules in numeric_cols.items():
        if col in df.columns:
            # First convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN values with default
            df[col] = df[col].fillna(rules['default'])
            
            # Apply min/max constraints
            df.loc[df[col] < rules['min'], col] = rules['default'] if rules['default'] is not None else rules['min']
            df.loc[df[col] > rules['max'], col] = rules['max']
            
            logger.info(f"Processed {col}: {df[col].notna().sum()} valid values")
    
    # Ensure PRIORITY has a valid value (since we saw 0 valid values in the logs)
    if 'PRIORITY' in df.columns:
        df['PRIORITY'] = df['PRIORITY'].fillna(3).astype(int)
        logger.info(f"Ensured PRIORITY is set with default value 3: {df['PRIORITY'].value_counts().to_dict()}")
    
    # Calculate HOURS_NEED if not provided but JOB_QUANTITY and EXPECT_OUTPUT_PER_HOUR are available
    if 'HOURS_NEED' in df.columns and 'JOB_QUANTITY' in df.columns and 'EXPECT_OUTPUT_PER_HOUR' in df.columns:
        mask = (df['JOB_QUANTITY'].notna() & df['EXPECT_OUTPUT_PER_HOUR'].notna() & (df['EXPECT_OUTPUT_PER_HOUR'] > 0) & df['HOURS_NEED'].isna())
        if mask.any():
            df.loc[mask, 'HOURS_NEED'] = df.loc[mask, 'JOB_QUANTITY'] / df.loc[mask, 'EXPECT_OUTPUT_PER_HOUR']
            logger.info(f"Calculated HOURS_NEED for {mask.sum()} rows from JOB_QUANTITY/EXPECT_OUTPUT_PER_HOUR")
    
    # Calculate BALANCE_QUANTITY if not provided
    if 'BALANCE_QUANTITY' in df.columns and 'JOB_QUANTITY' in df.columns and 'ACCUMULATED_DAILY_OUTPUT' in df.columns:
        mask = (df['JOB_QUANTITY'].notna() & df['ACCUMULATED_DAILY_OUTPUT'].notna() & df['BALANCE_QUANTITY'].isna())
        if mask.any():
            df.loc[mask, 'BALANCE_QUANTITY'] = df.loc[mask, 'JOB_QUANTITY'] - df.loc[mask, 'ACCUMULATED_DAILY_OUTPUT']
            logger.info(f"Calculated BALANCE_QUANTITY for {mask.sum()} rows")
    
    # Enforce required fields
    if 'JOB' in df.columns:
        df = df[df['JOB'].notna() & (df['JOB'].str.len() > 0)]
    if 'PROCESS_CODE' in df.columns:
        df = df[df['PROCESS_CODE'].notna() & (df['PROCESS_CODE'].str.len() > 0)]
    
    # Process date columns - handle column names with or without trailing spaces
    base_date_cols = ['LCD_DATE', 'PLANNED_START', 'PLANNED_END', 'LATEST_COMPLETION_DATE', 'START_DATE']
    date_cols = []
    col_mapping = {}  # Map from base column name to actual column name
    
    # Find actual column names that match base names (handling trailing spaces)
    for base_col in base_date_cols:
        matching_cols = [col for col in df.columns if col.strip() == base_col]
        if matching_cols:
            date_cols.extend(matching_cols)
            col_mapping[base_col] = matching_cols[0]  # Map base name to actual column name with spaces
        elif base_col in df.columns:
            date_cols.append(base_col)
            col_mapping[base_col] = base_col
    
    logger.info(f"Date columns found in Excel: {date_cols}")
    logger.info(f"Column mapping: {col_mapping}")
    
    for col in date_cols:
        # Try multiple date formats - IMPORTANT: Added formats with time components first
        # This ensures we capture the time part properly, especially for START_DATE
        date_formats = [
            '%d/%m/%y %H:%M',     # DD/MM/YY HH:MM (prioritize this format for dates with times)
            '%d/%m/%Y %H:%M',     # DD/MM/YYYY HH:MM
            '%Y-%m-%d %H:%M',     # YYYY-MM-DD HH:MM
            '%m/%d/%Y %H:%M',     # MM/DD/YYYY HH:MM
            '%Y-%m-%dT%H:%M:%S.000Z',  # ISO format with timezone
            '%d/%m/%y',           # DD/MM/YY (without time)
            '%Y-%m-%d',           # YYYY-MM-DD (without time)
            '%m/%d/%Y',           # MM/DD/YYYY (without time)
            '%d-%m-%Y',           # DD-MM-YYYY (without time)
            '%Y/%m/%d'            # YYYY/MM/DD (without time)
        ]
        
        # Special handling for START_DATE to preserve time component
        if col.strip() == 'START_DATE':
            logger.info(f"Special handling for START_DATE column '{col}' to preserve time component")
            # Check if the column actually has time information
            sample_values = df[col].dropna().head(5).astype(str).tolist()
            logger.info(f"Sample raw START_DATE values: {sample_values}")
            
        success = False
        for fmt in date_formats:
            try:
                # For START_DATE, we'll be extra careful to preserve the time component
                temp_dates = pd.to_datetime(df[col], format=fmt, errors='coerce')
                if temp_dates.notna().sum() > 0:
                    df[col] = temp_dates
                    valid_count = df[col].notna().sum()
                    logger.info(f"Converted {col} using format {fmt}: {valid_count} valid dates")
                    
                    # Log a sample of the parsed dates with their time components for debugging
                    if col.strip() == 'START_DATE':
                        sample_dates = df[df[col].notna()][col].head(3)
                        for idx, date_val in sample_dates.items():
                            logger.info(f"  Sample {col} value: {date_val.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    success = True
                    break
            except Exception as e:
                logger.debug(f"Error with format {fmt} for {col}: {e}")
                continue
        
        if not success:
            logger.warning(f"Could not parse any dates in {col} column with standard formats")
            # Try a more flexible approach as a last resort
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"Parsed {col} using pandas flexible parser: {df[col].notna().sum()} valid dates")
            except Exception as e:
                logger.error(f"Failed to parse {col} column: {e}")
    
    # Remove duplicates on key columns - updated with new column names
    key_cols = [col for col in ['JOB', 'PROCESS_CODE', 'RSC_LOCATION'] if col in df.columns]
    if key_cols:
        pre_dedup = len(df)
        df = df.drop_duplicates(subset=key_cols)
        logger.info(f"Removed {pre_dedup - len(df)} duplicate rows based on {key_cols}")
    
    # Ensure machine location exists
    if 'RSC_LOCATION' in df.columns:
        df = df[df['RSC_LOCATION'].notna() & (df['RSC_LOCATION'].str.len() > 0)]
    elif 'MACHINE_ID' in df.columns:  # Backward compatibility
        df = df[df['MACHINE_ID'].notna() & (df['MACHINE_ID'].str.len() > 0)]
    
    # Calculate processing time from hours directly
    if 'HOURS_NEED' in df.columns:
        df['processing_time'] = (df['HOURS_NEED'] * 3600).astype(int)
        logger.info(f"Calculated processing_time from HOURS_NEED: {df['processing_time'].mean():.2f} seconds average")
    
    # Add setup time if available - check both old and new column names
    if 'SETTING_HOURS' in df.columns:  # New column name
        df['setup_time'] = (df['SETTING_HOURS'] * 3600).astype(int)
    elif 'SETTING_HOUR' in df.columns:  # Old column name
        df['setup_time'] = (df['SETTING_HOUR'] * 3600).astype(int)
    else:
        df['setup_time'] = 0
    
    # Add downtime if available - check both old and new column names
    if 'NO_PROD' in df.columns:  # New column name
        df['downtime'] = (df['NO_PROD'] * 3600).astype(int)
    elif 'NO_PRODUCTION_HOUR' in df.columns:  # Old column name
        df['downtime'] = (df['NO_PRODUCTION_HOUR'] * 3600).astype(int)
    else:
        df['downtime'] = 0
    
    # Add break time if available
    if 'BREAK_HOURS' in df.columns:
        df['break_time'] = (df['BREAK_HOURS'] * 3600).astype(int)
    else:
        df['break_time'] = 0
        
    # Total time is processing + setup + downtime + break time
    df['total_time'] = df['processing_time'] + df['setup_time'] + df['downtime'] + df['break_time']
    
    removed_rows = initial_rows - len(df)
    logger.info(f"Data cleaning: {len(df)} rows after cleaning (removed {removed_rows} invalid rows)")
    if removed_rows > 0:
        logger.info("Reasons for removal:")
        logger.info("- Missing or invalid job/process codes")
        logger.info("- Zero or negative processing times")
        logger.info("- Missing machine locations")
        logger.info("- Invalid dates")
    
    return df

def convert_to_epoch(df, columns, base_date=None):
    """Convert date columns to epoch timestamps (seconds since Unix epoch)."""
    df = df.copy()
    if base_date is None:
        base_date = datetime.now()
    
    if isinstance(base_date, str):
        base_date = pd.to_datetime(base_date).replace(tzinfo=None)
    elif isinstance(base_date, pd.Timestamp):
        base_date = base_date.to_pydatetime().replace(tzinfo=None)
    
    # Default due date (30 days from now)
    default_due_date = int((datetime.now() + pd.Timedelta(days=30)).timestamp())
    
    # Find the actual column names that match the base names (handling trailing spaces)
    actual_columns = {}
    col_mapping = {}  # Create a column mapping for use within this function
    for base_col in columns:
        matching_cols = [col for col in df.columns if col.strip() == base_col]
        if matching_cols:
            actual_columns[base_col] = matching_cols[0]
        elif base_col in df.columns:
            actual_columns[base_col] = base_col
    
    logger.info(f"Found date columns: {list(actual_columns.items())}")
    
    for base_col, col in actual_columns.items():
        epoch_col = f"epoch_{base_col.lower().replace(' ', '_').replace('-', '_')}"
        
        # Special handling for LCD_DATE - prioritize this as the first column
        if base_col == 'LCD_DATE':
            logger.info(f"Processing LCD_DATE column ('{col}') with special attention")
            epoch_col = 'LCD_DATE_EPOCH'  # Renamed to match expectation
            
            # Try with multiple date formats
            date_formats = ['%d/%m/%y', '%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']
            success = False
            for fmt in date_formats:
                try:
                    df[epoch_col] = pd.to_datetime(df[col], format=fmt, errors='coerce').apply(
                        lambda x: int(x.timestamp()) if pd.notna(x) else None
                    )
                    valid_count = df[epoch_col].notna().sum()
                    logger.info(f"LCD_DATE with format {fmt}: {valid_count} valid dates")
                    if valid_count > 0:
                        success = True
                        break
                except Exception as e:
                    logger.warning(f"Error with format {fmt}: {e}")
            
            # If we couldn't parse any dates, set default
            if not success or df[epoch_col].isna().all():
                logger.warning(f"No valid dates in LCD_DATE - using default (30 days from now)")
                df[epoch_col] = default_due_date
        
        # Special handling for date columns with time configuration
        elif base_col in DATE_COLUMNS_CONFIG:
            actual_col = col_mapping.get(base_col, col)  # Get the actual column name with spaces
            hour, minute = DATE_COLUMNS_CONFIG[base_col]
            logger.info(f"Processing {base_col} column ('{actual_col}') with time set to {hour:02d}:{minute:02d}")
            epoch_col = f"{base_col}_EPOCH"
            
            # Log raw values for debugging
            sample_values = df[actual_col].dropna().head(5).astype(str).tolist()
            logger.info(f"Sample raw {base_col} values: {sample_values}")
            
            # Try with multiple date formats - prioritize formats with time components
            date_formats = [
                '%d/%m/%y %H:%M',     # DD/MM/YY HH:MM
                '%d/%m/%Y %H:%M',     # DD/MM/YYYY HH:MM
                '%Y-%m-%d %H:%M',     # YYYY-MM-DD HH:MM
                '%m/%d/%Y %H:%M',     # MM/DD/YYYY HH:MM
                '%Y-%m-%dT%H:%M:%S',  # ISO format
                '%d/%m/%y',           # DD/MM/YY (without time)
                '%d/%m/%Y',           # DD/MM/YYYY (without time)
                '%Y-%m-%d',           # YYYY-MM-DD (without time)
                '%m/%d/%Y',           # MM/DD/YYYY (without time)
                '%d-%m-%Y'            # DD-MM-YYYY (without time)
            ]
            
            # First try to directly parse Excel's datetime values and set the configured time
            try:
                # Try to use pandas' native datetime handling first
                if pd.api.types.is_datetime64_any_dtype(df[actual_col]):
                    # Get the target time from configuration
                    hour, minute = DATE_COLUMNS_CONFIG[base_col]
                    
                    logger.info(f"{base_col} column is already in datetime format, setting time to {hour:02d}:{minute:02d}")
                    
                    # Create a new column with the original dates
                    df['original_dates'] = df[actual_col].copy()
                    
                    # Apply the specified time to all datetime values - use direct datetime manipulation
                    # to avoid timezone issues
                    for idx, date_val in df['original_dates'].items():
                        if pd.notna(date_val):
                            # Create a new datetime with the date components but set hour to specified time
                            new_date = datetime(date_val.year, date_val.month, date_val.day, hour, minute, 0)
                            # Convert to epoch timestamp
                            df.loc[idx, epoch_col] = int(new_date.timestamp())
                            
                            # Log for debugging
                            process = df.loc[idx, 'PROCESS_CODE'] if 'PROCESS_CODE' in df.columns else f"Row {idx}"
                            logger.info(f"  Set {base_col} for {process}: {new_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    valid_count = df[epoch_col].notna().sum()
                    if valid_count > 0:
                        logger.info(f"Set {valid_count} datetime values to {hour:02d}:{minute:02d} using direct datetime creation")
                        # Drop the temporary column
                        df = df.drop('original_dates', axis=1)
                        continue
            except Exception as e:
                logger.debug(f"Error setting time for {base_col}: {e}")
            
            # If direct datetime preservation didn't work, try parsing with formats
            success = False
            for fmt in date_formats:
                try:
                    # Convert to datetime first, preserving time component
                    dates = pd.to_datetime(df[actual_col], format=fmt, errors='coerce')
                    # Then convert to epoch timestamps
                    df[epoch_col] = dates.apply(lambda x: int(x.timestamp()) if pd.notna(x) else None)
                    valid_count = df[epoch_col].notna().sum()
                    logger.info(f"{base_col} with format {fmt}: {valid_count} valid dates")
                    if valid_count > 0:
                        success = True
                        # Log each date for debugging with full time component
                        for idx, timestamp in df[df[epoch_col].notna()][epoch_col].items():
                            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                            process = df.loc[idx, 'PROCESS_CODE'] if 'PROCESS_CODE' in df.columns else f"Row {idx}"
                            logger.info(f"  Found {base_col} for {process}: {date_str}")
                        break
                except Exception as e:
                    logger.debug(f"Error with format {fmt}: {e}")
            
            # If standard formats fail, try pandas flexible parser as a last resort
            if not success:
                try:
                    # Use pandas flexible parser
                    dates = pd.to_datetime(df[actual_col], errors='coerce')
                    
                    # For columns with specific time configuration, set the configured time
                    if base_col in DATE_COLUMNS_CONFIG:
                        hour, minute = DATE_COLUMNS_CONFIG[base_col]
                        # Apply time setting to each valid date
                        for idx, date_val in dates.items():
                            if pd.notna(date_val):
                                # Set the time component to the configured value
                                new_date = datetime(date_val.year, date_val.month, date_val.day, hour, minute, 0)
                                df.loc[idx, epoch_col] = int(new_date.timestamp())
                    else:
                        # Just convert to timestamps without altering time
                        df[epoch_col] = dates.apply(lambda x: int(x.timestamp()) if pd.notna(x) else None)
                    
                    valid_count = df[epoch_col].notna().sum()
                    
                    if valid_count > 0:
                        logger.info(f"{base_col} with flexible parser: {valid_count} valid dates")
                        # Log the parsed dates
                        for idx, timestamp in df[df[epoch_col].notna()][epoch_col].items():
                            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                            process = df.loc[idx, 'PROCESS_CODE'] if 'PROCESS_CODE' in df.columns else f"Row {idx}"
                            logger.info(f"  Found {base_col} for {process}: {date_str} (flexible parser)")
                    else:
                        logger.warning(f"Could not parse any {base_col} values")
                except Exception as e:
                    logger.error(f"Error with flexible parser for {base_col}: {e}")
                    
            # As a last resort, try to manually parse the time component
            if not success:
                try:
                    # Try to extract time information directly from string values
                    time_pattern = r'(\d{1,2}):(\d{2})'
                    
                    # First get the dates without time
                    base_dates = pd.to_datetime(df[actual_col], errors='coerce')
                    if base_dates.notna().sum() > 0:
                        logger.info(f"Got {base_dates.notna().sum()} base dates without time")
                        
                        # Now try to extract time components
                        df[epoch_col] = None
                        for idx, date_val in base_dates.items():
                            if pd.isna(date_val):
                                continue
                                
                            # Get the original string value
                            orig_val = str(df.loc[idx, actual_col])
                            time_match = re.search(time_pattern, orig_val)
                            
                            if time_match:
                                hour = int(time_match.group(1))
                                minute = int(time_match.group(2))
                                # Create a new datetime with the extracted time
                                adjusted_date = date_val.replace(hour=hour, minute=minute)
                                df.loc[idx, epoch_col] = int(adjusted_date.timestamp())
                                process = df.loc[idx, 'PROCESS_CODE'] if 'PROCESS_CODE' in df.columns else f"Row {idx}"
                                logger.info(f"  Manually extracted time for {process}: {adjusted_date.strftime('%Y-%m-%d %H:%M:%S')}")
                            else:
                                # Set time to 13:00 as per requirement - use direct datetime creation
                                # to avoid timezone issues
                                new_date = datetime(date_val.year, date_val.month, date_val.day, 13, 0, 0)
                                df.loc[idx, epoch_col] = int(new_date.timestamp())
                                process = df.loc[idx, 'PROCESS_CODE'] if 'PROCESS_CODE' in df.columns else f"Row {idx}"
                                logger.info(f"  Applied 13:00 time for {process}: {new_date.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        valid_count = df[epoch_col].notna().sum()
                        if valid_count > 0:
                            logger.info(f"Manually processed {valid_count} START_DATE values with time set to 13:00")
                except Exception as e:
                    logger.error(f"Error with manual time extraction for START_DATE: {e}")
        
        # Regular date column handling
        else:
            try:
                if df[col].dtype in ['object', 'string'] or pd.api.types.is_datetime64_any_dtype(df[col]):
                    # Try with multiple date formats
                    date_formats = ['%d/%m/%y', '%Y-%m-%dT%H:%M:%S.000Z', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
                    success = False
                    
                    for fmt in date_formats:
                        try:
                            df[epoch_col] = pd.to_datetime(df[col], format=fmt, errors='coerce').apply(
                                lambda x: int(x.timestamp()) if pd.notna(x) else None
                            )
                            if df[epoch_col].notna().sum() > 0:
                                success = True
                                logger.info(f"Converted '{col}' using format {fmt}")
                                break
                        except:
                            continue
                    
                    if not success:
                        # Fall back to flexible parsing
                        df[epoch_col] = pd.to_datetime(df[col], errors='coerce').apply(
                            lambda x: int(x.timestamp()) if pd.notna(x) else None
                        )
                
                # Handle special cases for dates
                if col in ['LCD_DATE'] and df[epoch_col].isna().all():
                    logger.warning(f"No valid data in '{col}' - setting default due dates (30 days from now)")
                    df[epoch_col] = default_due_date
                elif col in ['PLANNED_START', 'PLANNED_END', 'LATEST_COMPLETION_DATE'] and df[epoch_col].isna().all():
                    logger.warning(f"No valid data in '{col}' - these will be computed by the scheduler")
                
                valid_count = df[epoch_col].notna().sum()
                logger.info(f"Converted '{col}' to '{epoch_col}': {valid_count} valid timestamps")
            except Exception as e:
                logger.error(f"Error converting column '{col}' to epoch: {e}")
                if col in ['LCD_DATE']:
                    logger.warning(f"Using default due dates for '{col}'")
                    df[epoch_col] = default_due_date
                elif col in ['PLANNED_START', 'PLANNED_END', 'LATEST_COMPLETION_DATE']:
                    logger.warning(f"Could not convert '{col}' to timestamp - will be computed by scheduler")
                    df[epoch_col] = None
                else:
                    df[epoch_col] = None
    
    return df

def load_job_data(file_path):
    """
    Load and process job data from Excel file.

    Args:
        file_path (str): Path to Excel file containing job data

    Returns:
        pd.DataFrame: Processed DataFrame with job data
    """
    if not os.path.exists(file_path):
        error_msg = f"Excel file not found at {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # Explicitly use the "Jobs PlanningDetails-Draft" sheet and header at row 4
        df = pd.read_excel(file_path, sheet_name="Jobs PlanningDetails-Draft", header=4)
        logger.info(f"Successfully loaded Excel from 'Jobs PlanningDetails-Draft' sheet with header at row 4")
        
        # If that fails, try alternative approaches
        if len(df.columns) < 5:
            logger.warning("Sheet or header row may be incorrect. Attempting to find the correct sheet and header...")
            # Try different sheets and header rows
            for sheet in pd.ExcelFile(file_path).sheet_names:
                for header_row in [4, 0, 1, 2, 3]:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet, header=header_row)
                        # Check for necessary columns - updated to include new column names
                        required_columns = ['PROCESS_CODE']
                        location_columns = ['RSC_LOCATION', 'MACHINE_ID']  # Try both old and new names
                        job_columns = ['JOB', 'JOBCODE', 'RSC_CODE']       # Try all possible job column names
                        
                        # Check if we have at least process code and one location column and one job column
                        has_process = 'PROCESS_CODE' in df.columns
                        has_location = any(col in df.columns for col in location_columns)
                        has_job = any(col in df.columns for col in job_columns)
                        
                        if has_process and has_location and has_job and len(df.columns) >= 5:
                            logger.info(f"Found data in sheet '{sheet}' with header at row {header_row}")
                            break
                    except:
                        continue
                else:
                    continue
                break
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise
    
    logger.info(f"Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
    logger.info("Columns in the Excel file: " + ", ".join(df.columns.tolist()))
    
    # Clean and prepare the data
    df = clean_excel_data(df)
    
    # Convert date columns to epoch timestamps
    date_columns = ['LCD_DATE', 'PLANNED_START', 'PLANNED_END', 'LATEST_COMPLETION_DATE', 'START_DATE']
    df = convert_to_epoch(df, date_columns)
    
    # Find START_DATE column with or without trailing space
    start_date_columns = [col for col in df.columns if col.strip() == 'START_DATE']
    start_date_col = start_date_columns[0] if start_date_columns else None
    if start_date_col:
        logger.info(f"Found START_DATE column: '{start_date_col}'")
    else:
        logger.warning("No START_DATE column found in the data")
    
    # Find PLANNED_START column with or without trailing space
    planned_start_columns = [col for col in df.columns if col.strip() == 'PLANNED_START']
    planned_start_col = planned_start_columns[0] if planned_start_columns else None
    if planned_start_col:
        logger.info(f"Found PLANNED_START column: '{planned_start_col}'")
    
    # Define column mapping to standardize column names - following exact mydata.xlsx sequence
    column_mapping = {
        'LCD_DATE': 'LCD_DATE',
        'JOB': 'JOB',
        'PROCESS_CODE': 'PROCESS_CODE',
        'RSC_LOCATION': 'RSC_LOCATION',  # Previously MACHINE_ID
        'RSC_CODE': 'RSC_CODE',  # Previously JOB_CODE
        'NUMBER_OPERATOR': 'NUMBER_OPERATOR',
        'JOB_QUANTITY': 'JOB_QUANTITY',
        'EXPECT_OUTPUT_PER_HOUR': 'EXPECT_OUTPUT_PER_HOUR',
        'PRIORITY': 'PRIORITY',
        'HOURS_NEED': 'HOURS_NEED',
        'SETTING_HOURS': 'SETTING_HOURS',  # Previously SETTING_HOUR
        'BREAK_HOURS': 'BREAK_HOURS',
        'NO_PROD': 'NO_PROD',  # Previously NO_PRODUCTION_HOUR
        'START_DATE': start_date_col if start_date_col else 'START_DATE',
        'ACCUMULATED_DAILY_OUTPUT': 'ACCUMULATED_DAILY_OUTPUT',
        'BALANCE_QUANTITY': 'BALANCE_QUANTITY',
        'START_TIME': 'START_TIME',  # Output field - calculated after scheduling
        'END_TIME': 'END_TIME',      # Output field - calculated after scheduling
        'BAL_HR': 'BAL_HR',          # Output field - calculated after scheduling (BALANCE_HOUR)
        'BUFFER_STATUS': 'BUFFER_STATUS'  # Output field - calculated after scheduling
    }
    
    # Add epoch date mappings
    column_mapping.update({
        'LCD_DATE_EPOCH': 'LCD_DATE_EPOCH',
        'START_DATE_EPOCH': 'START_DATE_EPOCH'
    })
    
    # Check for missing columns and handle column name compatibility
    missing_columns = []
    for standard_col, actual_col in column_mapping.items():
        # Skip if this is an output field (not needed in input)
        if standard_col in ['START_TIME', 'END_TIME', 'BAL_HR', 'BUFFER_STATUS']:
            continue
            
        # Skip epoch fields (handled separately)
        if standard_col.endswith('_EPOCH'):
            continue
            
        # First try the new column name
        if actual_col in df.columns:
            continue
            
        # For updated column names, try the old name as fallback
        if standard_col == 'RSC_LOCATION' and 'MACHINE_ID' in df.columns:
            column_mapping[standard_col] = 'MACHINE_ID'
            continue
        elif standard_col == 'RSC_CODE' and 'JOB_CODE' in df.columns:
            column_mapping[standard_col] = 'JOB_CODE'
            continue
        elif standard_col == 'RSC_CODE' and 'JOBCODE' in df.columns:
            column_mapping[standard_col] = 'JOBCODE'
            continue
        elif standard_col == 'SETTING_HOURS' and 'SETTING_HOUR' in df.columns:
            column_mapping[standard_col] = 'SETTING_HOUR'
            continue
        elif standard_col == 'NO_PROD' and 'NO_PRODUCTION_HOUR' in df.columns:
            column_mapping[standard_col] = 'NO_PRODUCTION_HOUR'
            continue
        elif standard_col == 'START_DATE' and any(col.strip() == 'START_DATE' for col in df.columns):
            # Handle START_DATE with spaces
            column_mapping[standard_col] = next(col for col in df.columns if col.strip() == 'START_DATE')
            continue
            
        # If column is truly missing and not an optional column, add to missing list
        if actual_col not in df.columns and standard_col not in ['PRIORITY', 'BREAK_HOURS', 'ACCUMULATED_DAILY_OUTPUT', 'BALANCE_QUANTITY']:
            missing_columns.append(f"{standard_col} (looking for '{actual_col}')")
    
    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}")
    
    # Create standardized DataFrame
    data = pd.DataFrame()
    
    # First, add all required columns, handling missing values properly
    for standard_col, source_col in column_mapping.items():
        # Skip if source column doesn't exist
        if source_col not in df.columns:
            continue
            
        # Copy the values
        data[standard_col] = df[source_col].copy()
    
    # Ensure epoch fields are copied
    if 'LCD_DATE_EPOCH' in df.columns and 'LCD_DATE_EPOCH' not in data.columns:
        data['LCD_DATE_EPOCH'] = df['LCD_DATE_EPOCH']
    if 'START_DATE_EPOCH' in df.columns and 'START_DATE_EPOCH' not in data.columns:
        data['START_DATE_EPOCH'] = df['START_DATE_EPOCH']
    
    # Ensure required columns have default values
    data['NUMBER_OPERATOR'] = data.get('NUMBER_OPERATOR', pd.Series()).fillna(1).astype(int)
    
    # Check if HOURS_NEED exists, and is any value is nan, fill with calculated values from JOB_QUANTITY/EXPECT_OUTPUT_PER_HOUR
    if 'HOURS_NEED' in data.columns:
        if 'JOB_QUANTITY' in data.columns and 'EXPECT_OUTPUT_PER_HOUR' in data.columns:
            mask = data['HOURS_NEED'].isna() & data['JOB_QUANTITY'].notna() & data['EXPECT_OUTPUT_PER_HOUR'].notna() & (data['EXPECT_OUTPUT_PER_HOUR'] > 0)
            if mask.any():
                data.loc[mask, 'HOURS_NEED'] = data.loc[mask, 'JOB_QUANTITY'] / data.loc[mask, 'EXPECT_OUTPUT_PER_HOUR']
        data['HOURS_NEED'] = data['HOURS_NEED'].fillna(1.0).astype(float)
    else:
        data['HOURS_NEED'] = 1.0
    
    # Ensure other columns have valid values
    data['SETTING_HOURS'] = data.get('SETTING_HOURS', data.get('SETUP_TIME', pd.Series())).fillna(0).astype(float)
    data['BREAK_HOURS'] = data.get('BREAK_HOURS', pd.Series()).fillna(0).astype(float)
    data['NO_PROD'] = data.get('NO_PROD', data.get('NO_PRODUCTION_HOUR', pd.Series())).fillna(0).astype(float)
    
    # Ensure PRIORITY has a valid value
    data['PRIORITY'] = data.get('PRIORITY', pd.Series()).fillna(3).astype(int)
    
    # Ensure quantity fields have valid values
    data['JOB_QUANTITY'] = data.get('JOB_QUANTITY', pd.Series()).fillna(1000).astype(int)
    data['EXPECT_OUTPUT_PER_HOUR'] = data.get('EXPECT_OUTPUT_PER_HOUR', pd.Series()).fillna(100).astype(float)
    data['ACCUMULATED_DAILY_OUTPUT'] = data.get('ACCUMULATED_DAILY_OUTPUT', pd.Series()).fillna(0).astype(int)
    
    # Calculate BALANCE_QUANTITY if not provided
    if 'BALANCE_QUANTITY' not in data.columns or data['BALANCE_QUANTITY'].isna().any():
        data['BALANCE_QUANTITY'] = data['JOB_QUANTITY'] - data['ACCUMULATED_DAILY_OUTPUT']
    data['BALANCE_QUANTITY'] = data['BALANCE_QUANTITY'].fillna(data['JOB_QUANTITY']).astype(int)
    
    # Ensure LCD_DATE_EPOCH is valid
    if 'LCD_DATE_EPOCH' not in data.columns or data['LCD_DATE_EPOCH'].isna().any():
        default_due_date = int((datetime.now() + pd.Timedelta(days=30)).timestamp())
        data['LCD_DATE_EPOCH'] = data.get('LCD_DATE_EPOCH', pd.Series()).fillna(default_due_date).astype(int)
    
    # Check if START_DATE_EPOCH has been set
    if 'START_DATE_EPOCH' in data.columns and data['START_DATE_EPOCH'].notna().any():
        earliest_dates = data[data['START_DATE_EPOCH'].notna()]
        logger.info(f"Found {len(earliest_dates)} rows with START_DATE constraints:")
        
        # Group jobs with START_DATE by family to check for consistency
        family_data = {}
        for idx, row in earliest_dates.iterrows():
            if pd.notna(row['PROCESS_CODE']):
                family = extract_job_family(row['PROCESS_CODE'])
                if family not in family_data:
                    family_data[family] = []
                
                proc_num = extract_process_number(row['PROCESS_CODE'])
                date_str = datetime.fromtimestamp(row['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
                machine_id = row.get('RSC_LOCATION', row.get('MACHINE_ID', 'Unknown'))  # Try both old and new column names
                family_data[family].append((proc_num, row['PROCESS_CODE'], row['START_DATE_EPOCH'], date_str))
                
                logger.info(f"  Process {row['PROCESS_CODE']} (Seq {proc_num}) on {machine_id}: Will start at {date_str}")
        
        # Log families with multiple START_DATE constraints
        for family, processes in family_data.items():
            if len(processes) > 1:
                logger.info(f"Family {family} has {len(processes)} processes with START_DATE constraints:")
                for proc_num, process_code, _, date_str in sorted(processes, key=lambda x: x[0]):
                    logger.info(f"    Process {process_code} (Seq {proc_num}): {date_str}")
    else:
        logger.warning("No START_DATE constraints were found or parsed correctly")
    
    # Planning columns - these can be None/NaN as they will be computed by the scheduler
    data['PLANNED_START'] = data.get('PLANNED_START', pd.Series(dtype='object'))
    data['PLANNED_END'] = data.get('PLANNED_END', pd.Series(dtype='object'))
    data['LATEST_COMPLETION_DATE'] = data.get('LATEST_COMPLETION_DATE', pd.Series(dtype='object'))
    
    # Calculate processing time in seconds directly from HOURS_NEED
    # Ensure HOURS_NEED is not NaN and is positive
    valid_hours = data['HOURS_NEED'].fillna(0).clip(lower=0.1)
    data['processing_time'] = (valid_hours * 3600).astype(int)
    
    # Include setup time, break time, and downtime in calculation
    data['setup_time'] = (data['SETTING_HOURS'].fillna(0) * 3600).astype(int)
    data['break_time'] = (data['BREAK_HOURS'].fillna(0) * 3600).astype(int)
    data['downtime'] = (data['NO_PROD'].fillna(0) * 3600).astype(int)
    
    # Total time includes processing, setup, break time, and downtime
    data['total_time'] = data['processing_time'] + data['setup_time'] + data['break_time'] + data['downtime']
    
    # Log DataFrame details
    logger.info(f"DataFrame shape: {data.shape}")
    logger.info(f"DataFrame columns: {data.columns.tolist()}")
    logger.info(f"Sample data:\n{data.head(2)}")
    
    return data

def load_jobs_planning_data(file_path=None):
    """
    Load and process job planning data from Excel file.

    Args:
        file_path (str, optional): Path to Excel file containing job data. 
                                  If None, will try to load from environment variable.

    Returns:
        tuple: (jobs, machines, setup_times)
            - jobs: List of job dictionaries
            - machines: List of machine IDs
            - setup_times: Dictionary of setup times between processes
    """
    # Use environment variable if file_path not provided
    if file_path is None:
        # Load environment variables
        load_dotenv()
        file_path = os.getenv('file_path')
        if not file_path:
            logger.error("No file path provided and no 'file_path' found in environment variables")
            raise ValueError("No file path provided and no 'file_path' found in environment variables")
    
    # Log the file path being used
    logger.info(f"Loading job data from: {file_path}")
    
    # Check if we can use the time disparency fix to preserve time components
    df_with_epochs = None
    if TIME_DISPARENCY_FIX_AVAILABLE:
        try:
            # Process the file with our specialized time handling
            logger.info("Using fix_time_disparency to preserve time components in dates")
            df_with_epochs = fix_time_disparency(file_path)
            # Verify we have the epoch columns
            epoch_cols = [col for col in df_with_epochs.columns if col.startswith('epoch_')]
            logger.info(f"Found {len(epoch_cols)} epoch columns: {epoch_cols}")
        except Exception as e:
            logger.error(f"Error using fix_time_disparency: {e}")
            logger.warning("Falling back to standard date processing")
            df_with_epochs = None
    
    # Load and process the data into a DataFrame
    df = load_job_data(file_path)
    
    # Group jobs by family to prepare for dependency-aware processing
    family_jobs = {}
    for _, row in df.iterrows():
        if pd.notna(row['PROCESS_CODE']):
            family = extract_job_family(row['PROCESS_CODE'])
            if family not in family_jobs:
                family_jobs[family] = []
            
            family_jobs[family].append({
                'PROCESS_CODE': row['PROCESS_CODE'],
                'seq_num': extract_process_number(row['PROCESS_CODE']),
                'START_DATE_EPOCH': row.get('START_DATE_EPOCH'),
                'row': row
            })
    
    # Convert DataFrame to list of job dictionaries
    jobs = []
    current_time = int(datetime.now().timestamp())
    for _, row in df.iterrows():
        if pd.isna(row['PROCESS_CODE']):
            logger.warning(f"Skipping row: PROCESS_CODE={row.get('PROCESS_CODE', 'NaN')}")
            continue
            
        # Check for machine ID in both old and new column names
        machine_id = None
        if 'RSC_LOCATION' in row and pd.notna(row['RSC_LOCATION']):
            machine_id = row['RSC_LOCATION']
        elif 'MACHINE_ID' in row and pd.notna(row['MACHINE_ID']):
            machine_id = row['MACHINE_ID']
            
        if machine_id is None:
            logger.warning(f"Skipping row: Missing machine ID for PROCESS_CODE={row.get('PROCESS_CODE', 'NaN')}")
            continue
        
        # Derive job code from process code if missing, using the appropriate column names
        job_code = None
        if 'RSC_CODE' in row and pd.notna(row['RSC_CODE']):
            job_code = row['RSC_CODE']
        elif 'JOB_CODE' in row and pd.notna(row['JOB_CODE']):
            job_code = row['JOB_CODE']
        elif 'JOBCODE' in row and pd.notna(row['JOBCODE']):
            job_code = row['JOBCODE']
        else:
            # Extract from PROCESS_CODE if all else fails
            job_code = row['PROCESS_CODE'].split('-P')[0] if '-P' in row['PROCESS_CODE'] else row['PROCESS_CODE']
        
        # Get the job name if available
        job_name = row['JOB'] if 'JOB' in row and pd.notna(row['JOB']) else job_code
        
        # Handle due date - ensure it's a valid integer
        due_seconds = row.get('LCD_DATE_EPOCH', 0)
        if pd.isna(due_seconds) or due_seconds == 0:
            due_seconds = int((datetime.now() + pd.Timedelta(days=30)).timestamp())
        else:
            due_seconds = int(due_seconds)
            
        # Ensure processing time is valid
        proc_time = max(int(row.get('processing_time', 3600)), 1)  # Minimum 1 second
        due_seconds = max(current_time + proc_time, due_seconds)
        
        # Calculate priority - ensure it's a valid integer
        base_priority = 3  # Default
        if 'PRIORITY' in row and pd.notna(row['PRIORITY']):
            try:
                base_priority = int(row['PRIORITY'])
            except:
                logger.warning(f"Invalid PRIORITY value {row['PRIORITY']} for {row['PROCESS_CODE']}, using default 3")
                
        base_priority = max(1, min(5, base_priority))
        days_remaining = (due_seconds - current_time) / (24 * 3600)
        if days_remaining <= 5 and base_priority > 1:
            base_priority -= 1
        final_priority = max(1, min(5, int(base_priority)))
        
        # Get user-defined start date if it exists
        user_start_time = None
        
        # First check for epoch timestamps from fix_time_disparency
        if df_with_epochs is not None:
            # Try to find matching row in df_with_epochs
            match_found = False
            for _, epoch_row in df_with_epochs.iterrows():
                if (epoch_row.get('PROCESS_CODE') == row.get('PROCESS_CODE') and 
                    epoch_row.get('JOB') == row.get('JOB') and 
                    epoch_row.get('RSC_LOCATION') == row.get('RSC_LOCATION')):
                    # Found matching row in epoch dataframe
                    start_date_cols = [col for col in epoch_row.index if 'start_date' in col.lower() and 'epoch' in col.lower()]
                    if start_date_cols and pd.notna(epoch_row[start_date_cols[0]]):
                        user_start_time = int(epoch_row[start_date_cols[0]])
                        start_date = datetime.fromtimestamp(user_start_time).strftime('%Y-%m-%d %H:%M')
                        logger.info(f"Found time-preserving epoch timestamp for {row['PROCESS_CODE']}: {start_date}")
                        match_found = True
                        break
        
        # Fall back to standard approach if no match found
        if user_start_time is None and 'START_DATE_EPOCH' in row and pd.notna(row['START_DATE_EPOCH']):
            try:
                user_start_time = int(row['START_DATE_EPOCH'])
                start_date = datetime.fromtimestamp(user_start_time).strftime('%Y-%m-%d %H:%M')
                logger.info(f"Job {row['PROCESS_CODE']} has user-defined start date: {start_date}")
            except:
                logger.warning(f"Invalid START_DATE_EPOCH value for {row['PROCESS_CODE']}")
        
        # Process numerical fields - ensure they're valid numbers
        def safe_int(value, default=0):
            try:
                if pd.isna(value):
                    return default
                return int(value)
            except:
                return default
                
        def safe_float(value, default=0.0):
            try:
                if pd.isna(value):
                    return default
                return float(value)
            except:
                return default
        
        # Create job dictionary with all available fields - updated with new column names
        job = {
            'JOB': job_name,
            'JOB_CODE': job_code,
            'RSC_CODE': job_code,  # Store in both fields for compatibility
            'PROCESS_CODE': row['PROCESS_CODE'],
            'MACHINE_ID': machine_id,
            'RSC_LOCATION': machine_id,  # Store in both fields for compatibility
            'processing_time': proc_time,
            'LCD_DATE_EPOCH': due_seconds,
            'PRIORITY': final_priority,
            'NUMBER_OPERATOR': safe_int(row.get('NUMBER_OPERATOR'), 1),
            # Add new columns with safe conversion
            'JOB_QUANTITY': safe_int(row.get('JOB_QUANTITY'), 1000),
            'EXPECT_OUTPUT_PER_HOUR': safe_float(row.get('EXPECT_OUTPUT_PER_HOUR'), 100.0),
            'ACCUMULATED_DAILY_OUTPUT': safe_int(row.get('ACCUMULATED_DAILY_OUTPUT'), 0),
            'BALANCE_QUANTITY': safe_int(row.get('BALANCE_QUANTITY'), 
                safe_int(row.get('JOB_QUANTITY', 1000)) - safe_int(row.get('ACCUMULATED_DAILY_OUTPUT', 0))),
            # Add scheduling fields that might be populated from the Excel
            'PLANNED_START': row.get('PLANNED_START'),
            'PLANNED_END': row.get('PLANNED_END'),
            'LATEST_COMPLETION_DATE': row.get('LATEST_COMPLETION_DATE'),
            # Add setup time, break time, and downtime with safe conversion
            'setup_time': safe_int(row.get('setup_time'), 0),
            'break_time': safe_int(row.get('break_time'), 0),
            'downtime': safe_int(row.get('downtime'), 0),
            'total_time': safe_int(row.get('total_time'), proc_time)
        }
        
        # Add user-defined start date if it exists
        if user_start_time is not None:
            job['START_DATE_EPOCH'] = user_start_time
        
        jobs.append(job)
    
    # Extract unique machines - check both old and new column names
    machines = []
    if 'MACHINE_ID' in df.columns:
        machines.extend(df['MACHINE_ID'].dropna().unique())
    if 'RSC_LOCATION' in df.columns:
        machines.extend(df['RSC_LOCATION'].dropna().unique())
    machines = sorted(list(set([m for m in machines if pd.notna(m) and str(m).strip()])))  # Remove duplicates and empty values
    
    # Generate setup times (placeholder, adjust as needed)
    process_codes = [p for p in df['PROCESS_CODE'].dropna().unique()]
    setup_times = {p1: {p2: 0 for p2 in process_codes} for p1 in process_codes}  # No setup times for now
    
    # Log jobs with START_DATE constraints by family
    current_time = int(datetime.now().timestamp())
    future_start_date_jobs = [job for job in jobs if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] > current_time]
    
    if future_start_date_jobs:
        logger.info(f"Found {len(future_start_date_jobs)} jobs with future START_DATE constraints")
        
        # Group by family for dependency analysis
        family_constraints = {}
        for job in future_start_date_jobs:
            family = extract_job_family(job['PROCESS_CODE'])
            if family not in family_constraints:
                family_constraints[family] = []
            family_constraints[family].append(job)
        
        # Log constraints by family
        for family, constrained_jobs in family_constraints.items():
            logger.info(f"Family {family} has {len(constrained_jobs)} jobs with START_DATE constraints:")
            for job in sorted(constrained_jobs, key=lambda x: extract_process_number(x['PROCESS_CODE'])):
                start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
                logger.info(f"  Job {job['PROCESS_CODE']} must start at {start_date}")
    
    logger.info(f"Generated {len(jobs)} valid jobs, {len(machines)} machines, and setup times for {len(process_codes)} processes")
    return jobs, machines, setup_times

if __name__ == "__main__":
    # Test the function using environment variable for file path
    load_dotenv()
    file_path = os.getenv('file_path')
    
    if not file_path:
        print("Error: No 'file_path' found in environment variables.")
        exit(1)
        
    print(f"Using file path from environment: {file_path}")
    jobs, machines, setup_times = load_jobs_planning_data()
    
    # Print summary information
    print(f"Loaded {len(jobs)} jobs and {len(machines)} machines")
    
    # Count jobs with START_DATE constraints
    current_time = int(datetime.now().timestamp())
    start_date_jobs = [job for job in jobs if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None]
    future_start_date_jobs = [job for job in start_date_jobs if job['START_DATE_EPOCH'] > current_time]
    
    print(f"Total jobs with START_DATE: {len(start_date_jobs)}")
    print(f"Jobs with future START_DATE: {len(future_start_date_jobs)}")
    
    # Group by family to show dependencies
    if future_start_date_jobs:
        family_jobs = {}
        for job in future_start_date_jobs:
            family = extract_job_family(job['PROCESS_CODE'])
            if family not in family_jobs:
                family_jobs[family] = []
            family_jobs[family].append(job)
        
        print("\nJobs with START_DATE constraints by family:")
        for family, jobs_list in family_jobs.items():
            print(f"\nFamily: {family}")
            for job in sorted(jobs_list, key=lambda x: extract_process_number(x['PROCESS_CODE'])):
                start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
                print(f"  {job['PROCESS_CODE']} (Seq {extract_process_number(job['PROCESS_CODE'])}): {start_date}")