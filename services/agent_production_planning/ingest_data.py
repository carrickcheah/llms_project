# ingest_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    # Define rules for numeric columns
    numeric_cols = {
        'NUMBER_OPERATOR': {'min': 1, 'max': 10, 'default': 1},
        'HOURS_NEED': {'min': 0.1, 'max': 720, 'default': None},
        'SETTING_HOUR': {'min': 0, 'max': 48, 'default': 0},  # Setup/changeover time
        'NO_PRODUCTION_HOUR': {'min': 0, 'max': 48, 'default': 0},  # Downtime
        'PRIORTY': {'min': 1, 'max': 5, 'default': 4}  # Updated: Changed from PRIORITY to PRIORTY
    }
    
    # Process numeric columns with validation
    for col, rules in numeric_cols.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] < rules['min'], col] = rules['default'] if rules['default'] is not None else rules['min']
            df.loc[df[col] > rules['max'], col] = rules['max']
            logger.info(f"Processed {col}: {df[col].notna().sum()} valid values")
    
    # Enforce required fields
    if 'JOBCODE' in df.columns:
        df = df[df['JOBCODE'].notna() & (df['JOBCODE'].str.len() > 0)]
    if 'PROCESS_CODE' in df.columns:
        df = df[df['PROCESS_CODE'].notna() & (df['PROCESS_CODE'].str.len() > 0)]
    
    # Process date columns
    date_cols = ['LCD_DATE', 'PLANNED_START ', 'PLANNED_END', 'LATEST_COMPLETION_DATE', 'CUT_Q']
    for col in date_cols:
        if col in df.columns:
            # Try multiple date formats - IMPORTANT: Added '%d/%m/%y' first for DD/MM/YY format
            for fmt in ['%d/%m/%y', '%Y-%m-%dT%H:%M:%S.000Z', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']:
                try:
                    df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                    if df[col].notna().sum() > 0:
                        logger.info(f"Converted {col} using format {fmt}: {df[col].notna().sum()} valid dates")
                        break
                except:
                    continue
    
    # Remove duplicates on key columns
    key_cols = [col for col in ['JOBCODE', 'PROCESS_CODE', 'MACHINE_ID'] if col in df.columns]
    if key_cols:
        pre_dedup = len(df)
        df = df.drop_duplicates(subset=key_cols)
        logger.info(f"Removed {pre_dedup - len(df)} duplicate rows based on {key_cols}")
    
    # Ensure machine ID exists
    if 'MACHINE_ID' in df.columns:
        df = df[df['MACHINE_ID'].notna() & (df['MACHINE_ID'].str.len() > 0)]
    
    # Calculate processing time from hours directly
    if 'HOURS_NEED' in df.columns:
        df['processing_time'] = (df['HOURS_NEED'] * 3600).astype(int)
    
    # Add setup time if available
    if 'SETTING_HOUR' in df.columns:
        df['setup_time'] = (df['SETTING_HOUR'] * 3600).astype(int)
    else:
        df['setup_time'] = 0
    
    # Add downtime if available
    if 'NO_PRODUCTION_HOUR' in df.columns:
        df['downtime'] = (df['NO_PRODUCTION_HOUR'] * 3600).astype(int)
    else:
        df['downtime'] = 0
        
    # Total time is processing + setup + downtime
    df['total_time'] = df['processing_time'] + df['setup_time'] + df['downtime']
    
    removed_rows = initial_rows - len(df)
    logger.info(f"Data cleaning: {len(df)} rows after cleaning (removed {removed_rows} invalid rows)")
    if removed_rows > 0:
        logger.info("Reasons for removal:")
        logger.info("- Missing or invalid job/process codes")
        logger.info("- Zero or negative processing times")
        logger.info("- Missing machine IDs")
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
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame. Skipping.")
            continue
        
        epoch_col = f"epoch_{col.lower().replace(' ', '_').replace('-', '_')}"
        try:
            if df[col].dtype in ['object', 'string'] or pd.api.types.is_datetime64_any_dtype(df[col]):
                # Try with multiple date formats - IMPORTANT: Added '%d/%m/%y' first for DD/MM/YY format
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
            
            # Special handling for CUT_Q column - very important!
            if col == 'CUT_Q':
                logger.info(f"Processing CUT_Q column with special attention")
                epoch_col = 'epoch_cut_q'
                
                # Try direct parsing with our specific format first
                try:
                    # Try with DD/MM/YY format explicitly (like "21/3/25")
                    df[epoch_col] = pd.to_datetime(df[col], format='%d/%m/%y', errors='coerce').apply(
                        lambda x: int(x.timestamp()) if pd.notna(x) else None
                    )
                    valid_dates = df[epoch_col].notna().sum()
                    logger.info(f"CUT_Q direct parsing with '%d/%m/%y' format: {valid_dates} valid dates")
                    
                    # If some dates were successfully parsed, log them for verification
                    if valid_dates > 0:
                        for idx, row in df.iterrows():
                            if pd.notna(row[epoch_col]):
                                original = row[col]
                                parsed = datetime.fromtimestamp(row[epoch_col])
                                logger.info(f"CUT_Q: '{original}' parsed as {parsed.strftime('%Y-%m-%d %H:%M')}")
                except Exception as e:
                    logger.error(f"Error in direct CUT_Q parsing: {e}")
                
                # If no valid dates were parsed with the direct method, try multiple formats
                if df[epoch_col].isna().all():
                    logger.warning("Direct parsing of CUT_Q failed, trying multiple formats...")
                    date_formats = ['%d/%m/%y', '%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']
                    
                    for fmt in date_formats:
                        try:
                            temp_col = f"temp_{fmt.replace('%', '').replace('/', '_')}"
                            df[temp_col] = pd.to_datetime(df[col], format=fmt, errors='coerce').apply(
                                lambda x: int(x.timestamp()) if pd.notna(x) else None
                            )
                            valid_count = df[temp_col].notna().sum()
                            logger.info(f"CUT_Q with format {fmt}: {valid_count} valid dates")
                            
                            if valid_count > 0:
                                df[epoch_col] = df[temp_col]
                                logger.info(f"Using format {fmt} for CUT_Q dates")
                                # Log sample of parsed dates for verification
                                for idx, row in df.head().iterrows():
                                    if pd.notna(row[temp_col]):
                                        logger.info(f"Sample: '{row[col]}' → {datetime.fromtimestamp(row[temp_col]).strftime('%Y-%m-%d')}")
                                break
                        except Exception as e:
                            logger.warning(f"Error with format {fmt}: {e}")

                # Log the final results
                valid_cut_q = df[epoch_col].notna().sum()
                logger.info(f"Final CUT_Q parsing results: {valid_cut_q} valid dates out of {len(df)} rows")
                
                # For any remaining unparsed CUT_Q values, attempt additional methods
                if valid_cut_q < len(df[df[col].notna()]):
                    unparsed_count = len(df[df[col].notna()]) - valid_cut_q
                    logger.warning(f"Could not parse {unparsed_count} CUT_Q dates with standard methods")
                    
                    # Log the problematic values
                    problem_values = df[df[col].notna() & df[epoch_col].isna()][col].unique()
                    logger.warning(f"Problematic CUT_Q values: {problem_values}")
            
            # Handle special cases for dates
            if col in ['LCD_DATE'] and df[epoch_col].isna().all():
                logger.warning(f"No valid data in '{col}' - setting default due dates (30 days from now)")
                df[epoch_col] = default_due_date
            elif col in ['PLANNED_START ', 'PLANNED_END', 'LATEST_COMPLETION_DATE'] and df[epoch_col].isna().all():
                logger.warning(f"No valid data in '{col}' - these will be computed by the scheduler")
            
            valid_count = df[epoch_col].notna().sum()
            logger.info(f"Converted '{col}' to '{epoch_col}': {valid_count} valid timestamps")
        except Exception as e:
            logger.error(f"Error converting column '{col}' to epoch: {e}")
            if col in ['LCD_DATE']:
                logger.warning(f"Using default due dates for '{col}'")
                df[epoch_col] = default_due_date
            elif col in ['PLANNED_START ', 'PLANNED_END', 'LATEST_COMPLETION_DATE']:
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
                        if len(df.columns) >= 5 and set(['JOBCODE', 'PROCESS_CODE', 'MACHINE_ID']).issubset(df.columns):
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
    date_columns = ['LCD_DATE', 'PLANNED_START ', 'PLANNED_END', 'LATEST_COMPLETION_DATE', 'CUT_Q']
    df = convert_to_epoch(df, date_columns)
    
    # Define column mapping to standardize column names
    # Updated: Modified mappings to match actual column names in your Excel file
    column_mapping = {
        'DUE_DATE_TIME': 'epoch_lcd_date',
        'JOB_CODE': 'JOBCODE',
        'PROCESS_CODE': 'PROCESS_CODE',
        'MACHINE_ID': 'MACHINE_ID',
        'NUMBER_OPERATOR': 'NUMBER_OPERATOR',
        'DURATION_IN_HOUR': 'HOURS_NEED',
        'SETUP_TIME': 'SETTING_HOUR',
        'DOWNTIME': 'NO_PRODUCTION_HOUR',
        'PRIORITY': 'PRIORTY',
        'PLANNED_START': 'PLANNED_START ',  # Note the space after START in the column name
        'PLANNED_END': 'PLANNED_END',
        'LATEST_COMPLETION_DATE': 'LATEST_COMPLETION_DATE',
        'EARLIEST_START_TIME': 'epoch_cut_q'  # From CUT_Q column
    }
    
    # Check for missing columns
    missing_columns = [f"{expected_col} (looking for '{actual_col}')" 
                      for expected_col, actual_col in column_mapping.items() 
                      if actual_col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}")
    
    # Create standardized DataFrame
    data = pd.DataFrame()
    for standard_col, source_col in column_mapping.items():
        data[standard_col] = df[source_col] if source_col in df.columns else pd.Series()
    
    # Ensure required columns have default values
    data['NUMBER_OPERATOR'] = data['NUMBER_OPERATOR'].fillna(1).astype(int)
    data['DURATION_IN_HOUR'] = data['DURATION_IN_HOUR'].fillna(0).astype(float)
    data['SETUP_TIME'] = data.get('SETUP_TIME', pd.Series()).fillna(0).astype(float)
    data['DOWNTIME'] = data.get('DOWNTIME', pd.Series()).fillna(0).astype(float)
    data['PRIORITY'] = data['PRIORITY'].fillna(3).astype(int)
    data['DUE_DATE_TIME'] = data['DUE_DATE_TIME'].fillna(int((datetime.now() + pd.Timedelta(days=30)).timestamp())).astype(int)
    
    # Check if EARLIEST_START_TIME has been set from CUT_Q
    if 'EARLIEST_START_TIME' in data.columns and data['EARLIEST_START_TIME'].notna().any():
        earliest_dates = data[data['EARLIEST_START_TIME'].notna()]
        logger.info(f"Found {len(earliest_dates)} rows with CUT_Q (earliest start) constraints:")
        for idx, row in earliest_dates.iterrows():
            date_str = datetime.fromtimestamp(row['EARLIEST_START_TIME']).strftime('%Y-%m-%d %H:%M')
            logger.info(f"  Process {row['PROCESS_CODE']} on {row['MACHINE_ID']}: Cannot start before {date_str}")
    else:
        logger.warning("No CUT_Q constraints were found or parsed correctly")
    

    # Planning columns - these can be None/NaN as they will be computed by the scheduler
    data['PLANNED_START'] = data.get('PLANNED_START', pd.Series(dtype='object'))
    data['PLANNED_END'] = data.get('PLANNED_END', pd.Series(dtype='object'))
    data['LATEST_COMPLETION_DATE'] = data.get('LATEST_COMPLETION_DATE', pd.Series(dtype='object'))
    
    # Calculate processing time in seconds directly from HOURS_NEED
    data['processing_time'] = (pd.to_numeric(data['DURATION_IN_HOUR'], errors='coerce').fillna(0) * 3600).astype(int)
    
    # Include setup time and downtime in total processing time if available
    if 'SETUP_TIME' in data.columns:
        data['setup_time'] = (pd.to_numeric(data['SETUP_TIME'], errors='coerce').fillna(0) * 3600).astype(int)
    else:
        data['setup_time'] = 0
        
    if 'DOWNTIME' in data.columns:
        data['downtime'] = (pd.to_numeric(data['DOWNTIME'], errors='coerce').fillna(0) * 3600).astype(int)
    else:
        data['downtime'] = 0
    
    # Total time includes processing, setup, and downtime
    data['total_time'] = data['processing_time'] + data['setup_time'] + data['downtime']
    
    # Log DataFrame details
    logger.info(f"DataFrame shape: {data.shape}")
    logger.info(f"DataFrame columns: {data.columns.tolist()}")
    logger.info(f"Sample data:\n{data.head(2)}")
    
    return data

def load_jobs_planning_data(file_path):
    """
    Load and process job planning data from Excel file.

    Args:
        file_path (str): Path to Excel file containing job data

    Returns:
        tuple: (jobs, machines, setup_times)
            - jobs: List of job dictionaries
            - machines: List of machine IDs
            - setup_times: Dictionary of setup times between processes
    """
    # Load and process the data into a DataFrame
    df = load_job_data(file_path)
    
    # Convert DataFrame to list of job dictionaries
    jobs = []
    current_time = int(datetime.now().timestamp())
    for _, row in df.iterrows():
        if pd.isna(row['PROCESS_CODE']) or pd.isna(row['MACHINE_ID']):
            logger.warning(f"Skipping row: PROCESS_CODE={row.get('PROCESS_CODE', 'NaN')}, MACHINE_ID={row.get('MACHINE_ID', 'NaN')}")
            continue
        
        # Derive job code from process code if missing
        job_code = row['JOB_CODE'] if pd.notna(row['JOB_CODE']) else row['PROCESS_CODE'].split('-P')[0] if '-P' in row['PROCESS_CODE'] else row['PROCESS_CODE']
        
        # Handle due date
        due_seconds = row['DUE_DATE_TIME']
        proc_time = row['processing_time']
        due_seconds = max(current_time + proc_time, int(due_seconds))
        
        # Calculate priority
        base_priority = int(row['PRIORITY'])
        base_priority = max(1, min(5, base_priority))
        days_remaining = (due_seconds - current_time) / (24 * 3600)
        if days_remaining <= 5 and base_priority > 1:
            base_priority -= 1
        final_priority = max(1, min(5, int(base_priority)))
        
        # Get earliest start time from CUT_Q if it exists
        earliest_start_time = None
        if 'EARLIEST_START_TIME' in row and pd.notna(row['EARLIEST_START_TIME']):
            earliest_start_time = int(row['EARLIEST_START_TIME'])
            cut_q_date = datetime.fromtimestamp(earliest_start_time).strftime('%Y-%m-%d %H:%M')
            logger.info(f"Job {row['PROCESS_CODE']} has CUT_Q date: {cut_q_date}")
        
        # Create job dictionary
        job = {
            'JOB_CODE': job_code,
            'PROCESS_CODE': row['PROCESS_CODE'],
            'MACHINE_ID': row['MACHINE_ID'],
            'processing_time': proc_time,
            'DUE_DATE_TIME': due_seconds,
            'PRIORITY': final_priority,
            'NUMBER_OPERATOR': int(row['NUMBER_OPERATOR']),
            # Add scheduling fields that might be populated from the Excel
            'PLANNED_START': row.get('PLANNED_START'),
            'PLANNED_END': row.get('PLANNED_END'),
            'LATEST_COMPLETION_DATE': row.get('LATEST_COMPLETION_DATE'),
            # Add setup time and downtime if available
            'setup_time': row.get('setup_time', 0),
            'downtime': row.get('downtime', 0),
            'total_time': row.get('total_time', proc_time)
        }
        
        # Add earliest start time (CUT_Q) if it exists
        if earliest_start_time is not None:
            job['EARLIEST_START_TIME'] = earliest_start_time
        
        jobs.append(job)
    
    # Extract unique machines
    machines = sorted(df['MACHINE_ID'].dropna().unique())
    
    # Generate setup times (placeholder, adjust as needed)
    process_codes = [p for p in df['PROCESS_CODE'].dropna().unique()]
    setup_times = {p1: {p2: 0 for p2 in process_codes} for p1 in process_codes}  # No setup times for now
    
    # Count jobs with CUT_Q constraints
    cut_q_jobs = [job for job in jobs if 'EARLIEST_START_TIME' in job and job['EARLIEST_START_TIME'] > current_time]
    if cut_q_jobs:
        logger.info(f"Found {len(cut_q_jobs)} jobs with CUT_Q (earliest start) constraints")
        for job in cut_q_jobs:
            cut_q_date = datetime.fromtimestamp(job['EARLIEST_START_TIME']).strftime('%Y-%m-%d %H:%M')
            logger.info(f"  Job {job['PROCESS_CODE']} must start on or after {cut_q_date}")
    
    logger.info(f"Generated {len(jobs)} valid jobs, {len(machines)} machines, and setup times for {len(process_codes)} processes")
    return jobs, machines, setup_times

if __name__ == "__main__":
    # Test the function
    file_path = "mydata.xlsx"
    jobs, machines, setup_times = load_jobs_planning_data(file_path)
    print("Jobs:", jobs)
    print("Machines:", machines)
    print("Setup Times:", setup_times)