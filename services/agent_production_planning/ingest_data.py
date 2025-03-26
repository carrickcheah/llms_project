"""
Configuration for data ingestion from Excel.
This allows for configurable date column handling without hardcoding.
"""
import pandas as pd
import numpy as np
import logging
import re
import os
from datetime import datetime, timedelta
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("production_scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# No hardcoded values - all times will come from Excel directly
DATE_COLUMNS_CONFIG = {}

# Default time for date columns not specifically configured
DEFAULT_TIME = (8, 0)  # Default to 8am

def get_column_time(column_name):
    """
    Get the configured time for a column name, handling trailing spaces.
    
    Args:
        column_name (str): The column name, possibly with trailing spaces
        
    Returns:
        tuple: (hour, minute) tuple from configuration or default
    """
    # Strip any trailing spaces to match with our configuration
    clean_name = column_name.strip() if isinstance(column_name, str) else column_name
    
    # Return the configured time or default
    return DATE_COLUMNS_CONFIG.get(clean_name, DEFAULT_TIME)

def detect_date_format(date_value):
    """
    Detect the format of a date string and return a formatter function.
    
    Args:
        date_value: The date value to analyze
        
    Returns:
        function: A formatter function for converting the date string to datetime
        bool: Whether time component is included in the date format
    """
    if isinstance(date_value, datetime):
        # Already a datetime object, check if it has a time component other than midnight
        has_time = date_value.hour != 0 or date_value.minute != 0 or date_value.second != 0
        return lambda x: x if isinstance(x, datetime) else pd.to_datetime(x), has_time
    
    if not isinstance(date_value, str):
        # Not a string, let pandas handle it
        return lambda x: pd.to_datetime(x), False
    
    # Look for common date patterns

    # Format: 31/12/2023 23:59 (European format with time, no seconds)
    if re.match(r'\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}', date_value):
        return lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M'), True
    
    # Format: 12/31/2023 23:59:59 (US format with time)
    if re.match(r'\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}', date_value):
        # Try to distinguish between US and European format
        parts = date_value.split()[0].split('/')
        if int(parts[0]) <= 12:  # Might be month in US format
            try:
                return lambda x: pd.to_datetime(x, format='%m/%d/%Y %H:%M:%S'), True
            except:
                return lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M:%S'), True
    
    # Format: 2023-12-31 (ISO format without time)
    if re.match(r'\d{4}-\d{2}-\d{2}$', date_value):
        return lambda x: pd.to_datetime(x, format='%Y-%m-%d'), False
    
    # Format: 31/12/2023 (European format without time)
    if re.match(r'\d{2}/\d{2}/\d{4}$', date_value):
        return lambda x: pd.to_datetime(x, format='%d/%m/%Y'), False
    
    # Default: Let pandas try to figure it out
    return lambda x: pd.to_datetime(x), False

def convert_column_to_dates(df, column_name, base_col=None):
    """
    Convert a column to datetime format, handling multiple formats.
    Also adds an _EPOCH column with Unix timestamp.
    
    Args:
        df (DataFrame): The pandas DataFrame
        column_name (str): The name of the column to convert
        base_col (str): Optional base column name for custom time configuration
        
    Returns:
        DataFrame: The updated DataFrame
    """
    # Clean up column name for epoch field (remove trailing spaces)
    clean_epoch_col = column_name.strip() if isinstance(column_name, str) else column_name
    if column_name not in df.columns:
        logger.warning(f"Column '{column_name}' not found in DataFrame")
        return df
    
    # Handle cases where the column is all NaN
    if df[column_name].isna().all():
        logger.warning(f"Column '{column_name}' contains only NaN values")
        # Add empty epoch column
        df[f"{column_name}_EPOCH"] = np.nan
        return df
    
    # Find the first non-null value
    sample_value = df[column_name].dropna().iloc[0] if not df[column_name].dropna().empty else None
    if sample_value is None:
        logger.warning(f"Column '{column_name}' contains no non-null values")
        # Add empty epoch column
        df[f"{column_name}_EPOCH"] = np.nan
        return df
    
    # Detect date format and convert using appropriate formatter
    formatter, has_time = detect_date_format(sample_value)
    try:
        # Convert column to datetime
        df[column_name] = df[column_name].apply(lambda x: formatter(x) if pd.notna(x) else np.nan)
    except Exception as e:
        logger.error(f"Error converting column '{column_name}' to datetime: {e}")
        # Try pandas default parser
        try:
            df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
            logger.info(f"Successfully converted '{column_name}' using pandas default parser")
        except Exception as e2:
            logger.error(f"Failed to convert '{column_name}' using pandas default parser: {e2}")
            # Add empty epoch column
            df[f"{column_name}_EPOCH"] = np.nan
            return df
    
    # Apply time component if needed
    # Option 1: Always use time from the original data if it exists
    if has_time:
        logger.info(f"Column '{column_name}' already has time component, preserving original times")
    # Option 2: For dates without time or with midnight default time, don't modify - keep as is
    else:
        # Get the configured time for this column - but DON'T apply it, just log it
        if base_col is None:
            base_col = column_name
            
        # IMPORTANT: We're not modifying times anymore, just logging for debugging
        hour, minute = get_column_time(base_col)
        logger.info(f"Column '{column_name}' has no time component. Original values will be preserved.")
        
        # We leave all dates as they come from Excel, even if they don't have a time component
        # This ensures we never impose any hardcoded times
    
    # Create epoch timestamp column for all dates
    # Use cleaned column name (no trailing spaces) for the EPOCH field
    epoch_col_name = f"{clean_epoch_col}_EPOCH"
    
    # Convert to epoch timestamp ensuring Singapore timezone (UTC+8)
    # Excel datetimes are timezone-naive, so we need to localize them to SGT
    import pytz
    singapore_tz = pytz.timezone('Asia/Singapore')
    
    def convert_to_sg_timestamp(dt):
        if pd.notna(dt):
            # Localize the naive datetime to Singapore time
            # This assumes the Excel times are already in Singapore local time
            # We create a timezone-aware datetime and then get its UNIX timestamp
            sg_dt = singapore_tz.localize(dt.to_pydatetime(), is_dst=None)
            return int(sg_dt.timestamp())
        return None
        
    df[epoch_col_name] = df[column_name].apply(convert_to_sg_timestamp)
    
    # Create a standardized version of the field (no spaces)
    standard_epoch_name = epoch_col_name.replace(" ", "")
    if standard_epoch_name != epoch_col_name:
        # Also create the standardized version of the column for compatibility
        # Ensure it's converted to int to avoid floating point issues with NaN
        df[standard_epoch_name] = df[epoch_col_name].fillna(-1).astype('int64').replace(-1, None)
        logger.info(f"Created standardized epoch field '{standard_epoch_name}' for column '{column_name}'")
    
    # If entire column is empty/NaN, log it clearly
    if df[epoch_col_name].isna().all():
        logger.info(f"Column '{column_name}' has no valid date values, all EPOCH values are NaN")
    
    return df



def load_jobs_planning_data(excel_file):
    """
    Load job planning data from an Excel file.
    This handles multiple date columns with configurable time settings.
    Adds UNIQUE_JOB_ID as JOB + PROCESS_CODE for each job.
    
    Args:
        excel_file (str): Path to the Excel file
        
    Returns:
        tuple: (jobs, machines, setup_times)
    """
    logger.info(f"Loading job planning data from {excel_file}")
    
    try:
        if not os.path.exists(excel_file):
            raise FileNotFoundError(f"Excel file not found: {excel_file}")
        
        try:
            logger.info("Trying to read Excel with header at row 0 (Excel row 1)")
            df = pd.read_excel(excel_file, header=0)
            if not df.empty and all(isinstance(col, str) for col in df.columns):
                logger.info("Successfully read Excel file with header at row 0")
            else:
                logger.info("Header not found at row 0, falling back to row 4 (Excel row 5)")
                df = pd.read_excel(excel_file, header=4)
        except Exception as e:
            logger.warning(f"Error reading with header at row 0: {e}")
            logger.info("Falling back to header at row 4 (Excel row 5)")
            df = pd.read_excel(excel_file, header=4)
        
        required_cols = ['JOB', 'PROCESS_CODE', 'PRIORITY', 'RSC_LOCATION']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Required columns are missing: {missing_cols}")
        
        df = df[df['JOB'].notna() & df['PROCESS_CODE'].notna()]
        
        # Create UNIQUE_JOB_ID as JOB + PROCESS_CODE
        df['UNIQUE_JOB_ID'] = df['JOB'].astype(str) + '_' + df['PROCESS_CODE'].astype(str)
        
        duplicates = df.duplicated(subset=['UNIQUE_JOB_ID']).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate jobs based on UNIQUE_JOB_ID")
            df = df.drop_duplicates(subset=['UNIQUE_JOB_ID'], keep='first')
            logger.info(f"Removed duplicates, keeping first occurrence. Remaining: {len(df)} jobs")
        
        if 'COMPLETED' in df.columns:
            df = df[~df['COMPLETED'].fillna(False).astype(bool)]
            logger.info(f"Filtered out completed jobs. Remaining: {len(df)} jobs")
            
        date_columns = [col for col in df.columns if any(date_key in col.upper() for date_key in ['DATE', 'DUE', 'TARGET', 'COMPLETION'])]
        logger.info(f"Found {len(date_columns)} potential date columns: {date_columns}")
        
        for col in date_columns:
            if df[col].isna().all():
                logger.info(f"Skipping empty date column '{col}'")
                continue
            convert_column_to_dates(df, col)
        
        if 'RSC_LOCATION' in df.columns:
            machines = df['RSC_LOCATION'].dropna().unique().tolist()
        else:
            machines = []
            logger.warning("No machine column found (RSC_LOCATION)")
        
        if 'HOURS_NEED' in df.columns:
            logger.info("Using HOURS_NEED from Excel for job durations")
            df['processing_time'] = df['HOURS_NEED'] * 3600
        elif 'PROCESSING_TIME_HR' in df.columns:
            df['processing_time'] = df['PROCESSING_TIME_HR'] * 3600
        elif 'PROC_TIME_HR' in df.columns:
            df['processing_time'] = df['PROC_TIME_HR'] * 3600
        elif 'PROCESSING_TIME_MIN' in df.columns:
            df['processing_time'] = df['PROCESSING_TIME_MIN'] * 60
        else:
            df['processing_time'] = 3600
            logger.warning("No processing time column found, defaulting to 1 hour per job")
        
        df.loc[df['processing_time'] <= 0, 'processing_time'] = 3600
        
        jobs = df.to_dict('records')
        setup_times = {}
        
        logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines")
        return jobs, machines, setup_times
        
    except Exception as e:
        logger.error(f"Error loading jobs planning data: {e}")
        raise