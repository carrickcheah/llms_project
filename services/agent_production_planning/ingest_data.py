# ingest_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import re

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
    
    # Define rules for numeric columns
    numeric_cols = {
        'NUMBER_OPERATOR': {'min': 1, 'max': 10, 'default': 1},
        'HOURS_NEED': {'min': 0.1, 'max': 720, 'default': None},
        'SETTING_HOUR': {'min': 0, 'max': 48, 'default': 0},  # Setup/changeover time
        'NO_PRODUCTION_HOUR': {'min': 0, 'max': 48, 'default': 0},  # Downtime
        'PRIORTY': {'min': 1, 'max': 5, 'default': 3}  # Updated: Changed to PRIORTY with default 3
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
    
    # Process date columns - handle column names with or without trailing spaces
    base_date_cols = ['LCD_DATE', 'PLANNED_START', 'PLANNED_END', 'LATEST_COMPLETION_DATE', 'START_DATE']
    date_cols = []
    
    # Find actual column names that match base names (handling trailing spaces)
    for base_col in base_date_cols:
        matching_cols = [col for col in df.columns if col.strip() == base_col]
        if matching_cols:
            date_cols.extend(matching_cols)
        elif base_col in df.columns:
            date_cols.append(base_col)
    
    logger.info(f"Date columns found in Excel: {date_cols}")
    
    for col in date_cols:
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
    
    # Find the actual column names that match the base names (handling trailing spaces)
    actual_columns = {}
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
            for fmt in date_formats:
                try:
                    df[epoch_col] = pd.to_datetime(df[col], format=fmt, errors='coerce').apply(
                        lambda x: int(x.timestamp()) if pd.notna(x) else None
                    )
                    valid_count = df[epoch_col].notna().sum()
                    logger.info(f"LCD_DATE with format {fmt}: {valid_count} valid dates")
                    if valid_count > 0:
                        break
                except Exception as e:
                    logger.warning(f"Error with format {fmt}: {e}")
        
        # Special handling for START_DATE
        elif base_col == 'START_DATE':
            logger.info(f"Processing START_DATE column ('{col}')")
            epoch_col = 'START_DATE_EPOCH'
            
            # Try with multiple date formats
            date_formats = ['%d/%m/%y', '%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y']
            for fmt in date_formats:
                try:
                    # Convert to datetime first
                    dates = pd.to_datetime(df[col], format=fmt, errors='coerce')
                    # Then convert to epoch timestamps
                    df[epoch_col] = dates.apply(lambda x: int(x.timestamp()) if pd.notna(x) else None)
                    valid_count = df[epoch_col].notna().sum()
                    logger.info(f"START_DATE with format {fmt}: {valid_count} valid dates")
                    if valid_count > 0:
                        # Log each START_DATE for debugging
                        for idx, timestamp in df[df[epoch_col].notna()][epoch_col].items():
                            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                            process = df.loc[idx, 'PROCESS_CODE'] if 'PROCESS_CODE' in df.columns else f"Row {idx}"
                            logger.info(f"  Found START_DATE for {process}: {date_str}")
                        break
                except Exception as e:
                    logger.warning(f"Error with format {fmt}: {e}")
        
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
    date_columns = ['LCD_DATE', 'PLANNED_START', 'PLANNED_END', 'LATEST_COMPLETION_DATE', 'START_DATE']
    df = convert_to_epoch(df, date_columns)
    
    # Find START_DATE column with or without trailing space
    start_date_col = next((col for col in df.columns if col.strip() == 'START_DATE'), None)
    if start_date_col:
        logger.info(f"Found START_DATE column: '{start_date_col}'")
    else:
        logger.warning("No START_DATE column found in the data")
    
    # Find PLANNED_START column with or without trailing space
    planned_start_col = next((col for col in df.columns if col.strip() == 'PLANNED_START'), None)
    if planned_start_col:
        logger.info(f"Found PLANNED_START column: '{planned_start_col}'")
    
    # Define column mapping to standardize column names
    column_mapping = {
        'LCD_DATE_EPOCH': 'LCD_DATE_EPOCH',  # Keep original name
        'START_DATE_EPOCH': 'START_DATE_EPOCH',  # Keep original name
        'JOB_CODE': 'JOBCODE',
        'PROCESS_CODE': 'PROCESS_CODE',
        'MACHINE_ID': 'MACHINE_ID',
        'NUMBER_OPERATOR': 'NUMBER_OPERATOR',
        'DURATION_IN_HOUR': 'HOURS_NEED',
        'SETUP_TIME': 'SETTING_HOUR',
        'DOWNTIME': 'NO_PRODUCTION_HOUR',
        'PRIORITY': 'PRIORITY',  # Map to PRIORITY
        'PLANNED_START': planned_start_col if planned_start_col else 'PLANNED_START',
        'PLANNED_END': 'PLANNED_END',
        'LATEST_COMPLETION_DATE': 'LATEST_COMPLETION_DATE'
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
    data['LCD_DATE_EPOCH'] = data['LCD_DATE_EPOCH'].fillna(int((datetime.now() + pd.Timedelta(days=30)).timestamp())).astype(int)
    
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
                family_data[family].append((proc_num, row['PROCESS_CODE'], row['START_DATE_EPOCH'], date_str))
                
                logger.info(f"  Process {row['PROCESS_CODE']} (Seq {proc_num}) on {row['MACHINE_ID']}: Will start at {date_str}")
        
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
        if pd.isna(row['PROCESS_CODE']) or pd.isna(row['MACHINE_ID']):
            logger.warning(f"Skipping row: PROCESS_CODE={row.get('PROCESS_CODE', 'NaN')}, MACHINE_ID={row.get('MACHINE_ID', 'NaN')}")
            continue
        
        # Derive job code from process code if missing
        job_code = row['JOB_CODE'] if pd.notna(row['JOB_CODE']) else row['PROCESS_CODE'].split('-P')[0] if '-P' in row['PROCESS_CODE'] else row['PROCESS_CODE']
        
        # Handle due date
        due_seconds = row['LCD_DATE_EPOCH']
        proc_time = row['processing_time']
        due_seconds = max(current_time + proc_time, int(due_seconds))
        
        # Calculate priority
        base_priority = int(row['PRIORITY'])
        base_priority = max(1, min(5, base_priority))
        days_remaining = (due_seconds - current_time) / (24 * 3600)
        if days_remaining <= 5 and base_priority > 1:
            base_priority -= 1
        final_priority = max(1, min(5, int(base_priority)))
        
        # Get user-defined start date if it exists
        user_start_time = None
        if 'START_DATE_EPOCH' in row and pd.notna(row['START_DATE_EPOCH']):
            user_start_time = int(row['START_DATE_EPOCH'])
            start_date = datetime.fromtimestamp(user_start_time).strftime('%Y-%m-%d %H:%M')
            logger.info(f"Job {row['PROCESS_CODE']} has user-defined start date: {start_date}")
        
        # Create job dictionary
        job = {
            'JOB_CODE': job_code,
            'PROCESS_CODE': row['PROCESS_CODE'],
            'MACHINE_ID': row['MACHINE_ID'],
            'processing_time': proc_time,
            'LCD_DATE_EPOCH': due_seconds,  # Use LCD_DATE_EPOCH instead of DUE_DATE_TIME
            'PRIORITY': final_priority,  # Keep as simple numeric value
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
        
        # Add user-defined start date if it exists
        if user_start_time is not None:
            job['START_DATE_EPOCH'] = user_start_time
        
        jobs.append(job)
    
    # Extract unique machines
    machines = sorted(df['MACHINE_ID'].dropna().unique())
    
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
    # Test the function
    file_path = "/Users/carrickcheah/llms_project/services/agent_production_planning/mydata.xlsx"
    jobs, machines, setup_times = load_jobs_planning_data(file_path)
    
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