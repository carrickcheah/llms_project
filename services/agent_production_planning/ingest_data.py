###############################################
# ingest_data.py | Never edit this line
# There are 4 main functions:
# 1. get_column_time
# 2. detect_date_format
# 3. convert_column_to_dates
# 4. load_jobs_planning_data
###############################################

import pandas as pd
import numpy as np
import logging
import re
import os
from datetime import datetime, timedelta
import pytz
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
def get_column_time(column_name):
    """
    Get the time from Excel data directly. No default values.
    
    Args:
        column_name (str): The column name, possibly with trailing spaces
        
    Returns:
        None: Since we don't use hardcoded times anymore
    """
    # We don't use any hardcoded or default times anymore
    # All times must come from Excel directly
    return None

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

    # Format: 31/12/23 (European format with 2-digit year - dd/mm/yy)
    if re.match(r'\d{1,2}/\d{1,2}/\d{2}$', date_value):
        return lambda x: pd.to_datetime(x, format='%d/%m/%y'), False
    
    # Format: 31/12/2023 23:59 (European format with time, no seconds)
    if re.match(r'\d{1,2}/\d{1,2}/\d{4}\s\d{1,2}:\d{2}', date_value):
        return lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M'), True
    
    # Format: 12/31/2023 23:59:59 (US format with time) or 15/5/2025 1:00:00 (European format with time, single digits)
    if re.match(r'\d{1,2}/\d{1,2}/\d{4}\s\d{1,2}:\d{2}:\d{2}', date_value):
        # Try to distinguish between US and European format
        parts = date_value.split()[0].split('/')
        if int(parts[0]) <= 12:  # Might be month in US format
            try:
                return lambda x: pd.to_datetime(x, format='%m/%d/%Y %H:%M:%S'), True
            except:
                return lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M:%S'), True
        else:
            return lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M:%S'), True
    
    # Format: 2023-12-31 (ISO format without time)
    if re.match(r'\d{4}-\d{2}-\d{2}$', date_value):
        return lambda x: pd.to_datetime(x, format='%Y-%m-%d'), False
    
    # Format: 31/12/2023 or 15/5/2025 (European format without time, possibly with single digits)
    if re.match(r'\d{1,2}/\d{1,2}/\d{4}$', date_value):
        return lambda x: pd.to_datetime(x, format='%d/%m/%Y'), False
    
    # Default: Let pandas try to figure it out
    return lambda x: pd.to_datetime(x), False

def convert_column_to_dates(df, column_name, base_col=None):
    """
    Convert a column to datetime format, handling multiple formats.
    Also adds an _EPOCH column with Unix timestamp.
    All times must come directly from Excel, no defaults.
    
    Args:
        df (DataFrame): The pandas DataFrame
        column_name (str): The name of the column to convert
        base_col (str): Optional base column name (not used anymore since we don't use defaults)
        
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
    
    # Log whether time component exists in the data
    if has_time:
        logger.info(f"Column '{column_name}' has time component from Excel data")
    else:
        logger.info(f"Column '{column_name}' has no time component in Excel data")
    
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



def load_jobs_planning_data(file_path):
    """
    Load job planning data from an Excel file or CSV file.
    This handles multiple date columns with configurable time settings.
    Adds UNIQUE_JOB_ID as JOB + PROCESS_CODE for each job.
    Filters jobs based on MATERIAL_ARRIVAL date - only processes jobs where materials have arrived.
    
    Args:
        file_path (str): Path to the Excel (.xlsx) or CSV (.csv) file
        
    Returns:
        tuple: (jobs, machines, setup_times)
    """
    logger.info(f"Loading job planning data from {file_path}")
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type based on extension
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            logger.info("Detected CSV file format")
            try:
                # First try with default settings
                df = pd.read_csv(file_path)
                if df.empty or not all(isinstance(col, str) for col in df.columns):
                    # Try with different encoding if needed
                    df = pd.read_csv(file_path, encoding='latin1')
                logger.info("Successfully read CSV file")
            except Exception as e:
                logger.error(f"Error reading CSV file: {e}")
                raise
        elif file_extension in ['.xlsx', '.xls']:
            logger.info(f"Detected Excel file format: {file_extension}")
            try:
                logger.info("Trying to read Excel with header at row 0 (Excel row 1)")
                df = pd.read_excel(file_path, header=0)
                if not df.empty and all(isinstance(col, str) for col in df.columns):
                    logger.info("Successfully read Excel file with header at row 0")
                else:
                    logger.info("Header not found at row 0, falling back to row 4 (Excel row 5)")
                    df = pd.read_excel(file_path, header=4)
            except Exception as e:
                logger.warning(f"Error reading with header at row 0: {e}")
                logger.info("Falling back to header at row 4 (Excel row 5)")
                df = pd.read_excel(file_path, header=4)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: .csv, .xlsx, .xls")
        
        required_cols = ['JOB', 'PROCESS_CODE', 'PRIORITY', 'RSC_CODE', 'HOURS_NEED', 'SETTING_HOURS', 'BREAK_HOURS', 'NO_PROD', 'LCD_DATE', 'MATERIAL_ARRIVAL']
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
            
        date_columns = [col for col in df.columns if any(date_key in col.upper() for date_key in ['DATE', 'DUE', 'TARGET', 'COMPLETION', 'ARRIVAL'])]
        logger.info(f"Found {len(date_columns)} potential date columns: {date_columns}")
        
        for col in date_columns:
            if df[col].isna().all():
                logger.info(f"Skipping empty date column '{col}'")
                continue
            convert_column_to_dates(df, col)
        
        # Filter jobs based on MATERIAL_ARRIVAL date
        if 'MATERIAL_ARRIVAL' in df.columns:
            logger.info("Processing MATERIAL_ARRIVAL date to filter jobs")
            # Get current time in Singapore timezone
            singapore_tz = pytz.timezone('Asia/Singapore')
            current_time = datetime.now(singapore_tz).replace(tzinfo=None)
            
            # Count jobs before filtering
            total_jobs = len(df)
            
            # Filter out jobs where material hasn't arrived yet
            df = df[df['MATERIAL_ARRIVAL'].notna() & (df['MATERIAL_ARRIVAL'] <= current_time)]
            
            filtered_jobs = total_jobs - len(df)
            logger.info(f"Filtered out {filtered_jobs} jobs where materials haven't arrived yet. Remaining: {len(df)} jobs")
            
            if len(df) == 0:
                logger.warning("No jobs with arrived materials found! Check your MATERIAL_ARRIVAL dates.")
        else:
            logger.error("MATERIAL_ARRIVAL column is required but not found. Cannot process jobs without material arrival information.")
            raise ValueError("MATERIAL_ARRIVAL column is required but not found")
        
        if 'RSC_CODE' in df.columns:
            machines = df['RSC_CODE'].dropna().unique().tolist()
        else:
            machines = []
            logger.warning("No machine column found (RSC_CODE)")
        
        # Process HOURS_NEED for job processing time
        if 'HOURS_NEED' in df.columns:
            logger.info("Using HOURS_NEED from Excel for job durations")
            # Check for missing values
            missing_hours = df['HOURS_NEED'].isna().sum()
            if missing_hours > 0:
                logger.info(f"Missing HOURS_NEED values for {missing_hours} jobs")
            df['processing_time'] = df['HOURS_NEED'] * 3600
            # Set a minimum processing time but don't default missing values
            df.loc[(df['processing_time'] <= 0) & df['processing_time'].notna(), 'processing_time'] = 3600
            
            # Keep NUMBER_OPERATOR for display purposes but don't use it to adjust processing times
            if 'NUMBER_OPERATOR' in df.columns:
                logger.info("Loading NUMBER_OPERATOR from Excel for display only (not adjusting job durations)")
                # Fill missing values with 1 for display purposes
                df['NUMBER_OPERATOR'] = df['NUMBER_OPERATOR'].fillna(1)
                # Ensure NUMBER_OPERATOR is at least 1
                df.loc[df['NUMBER_OPERATOR'] < 1, 'NUMBER_OPERATOR'] = 1
        
        # Process SETTING_HOURS for setup times
        if 'SETTING_HOURS' in df.columns:
            logger.info("Using SETTING_HOURS from Excel for machine setup times")
            # Check for missing values
            missing_setting = df['SETTING_HOURS'].isna().sum()
            if missing_setting > 0:
                logger.info(f"Missing SETTING_HOURS values for {missing_setting} jobs")
            df['setup_time'] = df['SETTING_HOURS'] * 3600
        else:
            logger.info("SETTING_HOURS column required but not found")
            df['setup_time'] = None
        

        # Process BREAK_HOURS for breaks during production
        if 'BREAK_HOURS' in df.columns:
            logger.info("Using BREAK_HOURS from Excel for production breaks")
            # Check for missing values
            missing_break = df['BREAK_HOURS'].isna().sum()
            if missing_break > 0:
                logger.info(f"Missing BREAK_HOURS values for {missing_break} jobs")
            df['break_time'] = df['BREAK_HOURS'] * 3600
        else:
            logger.info("BREAK_HOURS column required but not found")
            df['break_time'] = None
        
        # Process NO_PROD for non-production hours
        if 'NO_PROD' in df.columns:
            logger.info("Using NO_PROD from Excel for non-production hours")
            # Check for missing values
            missing_no_prod = df['NO_PROD'].isna().sum()
            if missing_no_prod > 0:
                logger.info(f"Missing NO_PROD values for {missing_no_prod} jobs")
            df['no_prod_time'] = df['NO_PROD'] * 3600
        else:
            logger.info("NO_PROD column required but not found")
            df['no_prod_time'] = None
        
        # HOURS_NEED is now mandatory, so we've removed JOB_QUANTITY and EXPECT_OUTPUT_PER_HOUR processing
        # as they're redundant when HOURS_NEED is provided directly
        
        # Check that all mandatory LCD_DATE fields are present
        if 'LCD_DATE' in df.columns:
            missing_lcd_date = df['LCD_DATE'].isna().sum()
            if missing_lcd_date > 0:
                logger.info(f"Missing LCD_DATE values for {missing_lcd_date} jobs")
        else:
            logger.info("LCD_DATE column required but not found")
            
        # Process PLAN_DATE column if present (keep original format for display in chart_two.py)
        if 'PLAN_DATE' in df.columns:
            logger.info("Found PLAN_DATE column in Excel, keeping in original format for display")
            # Do not convert to datetime - keep as is for chart_two.py
        else:
            logger.info("PLAN_DATE column not found, this is optional")
        
        # Convert to dictionary records
        jobs = df.to_dict('records')
        
        # Create setup times dictionary for OR-Tools
        setup_times = {}
        if 'SETTING_HOURS' in df.columns and df['SETTING_HOURS'].notna().any():
            # Group by machine to create setup times matrix
            for machine in machines:
                machine_jobs = df[df['RSC_CODE'] == machine]
                if len(machine_jobs) > 1:
                    # For each job transition on the same machine, record setup time
                    for _, job1 in machine_jobs.iterrows():
                        for _, job2 in machine_jobs.iterrows():
                            if job1['UNIQUE_JOB_ID'] != job2['UNIQUE_JOB_ID']:
                                job1_id = job1['UNIQUE_JOB_ID']
                                job2_id = job2['UNIQUE_JOB_ID']
                                # Use job2's setup time as the transition time
                                setup_time_sec = int(job2['setup_time'])
                                if setup_time_sec > 0:
                                    setup_times[(job1_id, job2_id)] = setup_time_sec
        
        logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines")
        if setup_times:
            logger.info(f"Created {len(setup_times)} setup time entries from SETTING_HOURS")
        return jobs, machines, setup_times
        
    except Exception as e:
        logger.error(f"Error loading jobs planning data: {e}")
        raise

if __name__ == "__main__":
    # Set up test configuration
    # Load environment variables from .env
    load_dotenv()
    
    # Try to get file path from environment variable, default to mydata.xlsx if not found
    file_path = os.getenv('file_path')

    
    # Check if a file path was provided as a command line argument (this overrides env variable)
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    # Configure logging to see what's happening
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test loading the data
        logger.info(f"Testing data ingestion with file: {file_path}")
        jobs, machines, setup_times = load_jobs_planning_data(file_path)
        
        # Print basic information about the loaded data
        logger.info(f"Successfully loaded data:")
        logger.info(f"Number of jobs: {len(jobs)}")
        logger.info(f"Number of machines: {len(machines)}")
        logger.info(f"Number of setup time entries: {len(setup_times)}")
        
        # Print sample data from the first job
        if jobs:
            logger.info("\nSample job data:")
            sample_job = jobs[0]
            for key, value in sample_job.items():
                logger.info(f"{key}: {value}")
            
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise