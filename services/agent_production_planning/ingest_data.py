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
        'OUTPUT_PER_HOUR': {'min': 0.1, 'max': 10000, 'default': 100},
        'HOURS NEEDED': {'min': 0.1, 'max': 720, 'default': None},
        'PRIORITY': {'min': 1, 'max': 5, 'default': 4}
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
    date_cols = ['LCD_DATE', 'PLANNED_START', 'PLANNED_END', 'LATEST COMPLETION DATE']
    for col in date_cols:
        if col in df.columns:
            # Try multiple date formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']:
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
    
    # Calculate processing time from hours
    if 'HOURS NEEDED' in df.columns and 'OUTPUT_PER_HOUR' in df.columns:
        df.loc[df['OUTPUT_PER_HOUR'] <= 0, 'OUTPUT_PER_HOUR'] = 100
        df['processing_time'] = df['HOURS NEEDED'] * 3600
    
    # Add auto-generated PLANNED_END if PLANNED_START is available but PLANNED_END is not
    if 'PLANNED_START' in df.columns and 'HOURS NEEDED' in df.columns:
        if 'PLANNED_END' not in df.columns:
            df['PLANNED_END'] = None
        mask = df['PLANNED_START'].notna() & df['HOURS NEEDED'].notna() & df['PLANNED_END'].isna()
        if mask.any():
            df.loc[mask, 'PLANNED_END'] = df.loc[mask].apply(
                lambda row: row['PLANNED_START'] + timedelta(hours=float(row['HOURS NEEDED'])), 
                axis=1
            )
            logger.info(f"Auto-generated PLANNED_END for {mask.sum()} rows")
    
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
    """Convert date/time/hour columns to epoch timestamps (seconds since Unix epoch)."""
    df = df.copy()
    if base_date is None:
        base_date = datetime.now()
    
    if isinstance(base_date, str):
        base_date = pd.to_datetime(base_date).replace(tzinfo=None)
    elif isinstance(base_date, pd.Timestamp):
        base_date = base_date.to_pydatetime().replace(tzinfo=None)
    
    # Default dates for missing values (30 days from now for due dates)
    default_due_date = int((datetime.now() + pd.Timedelta(days=30)).timestamp())
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame. Skipping.")
            continue
        
        epoch_col = f"epoch_{col.lower().replace(' ', '_').replace('-', '_')}"
        try:
            # For date columns that should be datetime objects
            if df[col].dtype in ['object', 'string'] or pd.api.types.is_datetime64_any_dtype(df[col]):
                # Try with multiple date formats
                date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
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
            # For numeric columns (hours) that should be added to base date
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[epoch_col] = df[col].apply(
                    lambda x: int(base_date.timestamp() + x * 3600) if pd.notna(x) else None
                )
            else:
                logger.warning(f"Column '{col}' has unsupported type {df[col].dtype}. Skipping.")
                continue
                
            # Handle special cases for required fields
            if col in ['LATEST COMPLETION DATE', 'LCD_DATE'] and df[epoch_col].isna().all():
                logger.warning(f"No valid data in '{col}' - setting default due dates (30 days from now)")
                df[epoch_col] = default_due_date
            
            valid_count = df[epoch_col].notna().sum()
            logger.info(f"Converted '{col}' to '{epoch_col}': {valid_count} valid timestamps")
        except Exception as e:
            logger.error(f"Error converting column '{col}' to epoch: {e}")
            # For due date columns, use default values if conversion fails
            if col in ['LATEST COMPLETION DATE', 'LCD_DATE']:
                logger.warning(f"Using default due dates for '{col}'")
                df[epoch_col] = default_due_date
            else:
                df[epoch_col] = None
    
    return df

def load_jobs_planning_data(file_path):
    """
    Load and process job planning data from Excel file.
    
    Args:
        file_path: Path to Excel file containing job data
        
    Returns:
        Tuple of (jobs, machines, setup_times)
    """
    if not os.path.exists(file_path):
        error_msg = f"Excel file not found at {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # Try different header rows to handle various formats
        for header_row in [4, 0, 1, 2, 3]:
            try:
                df = pd.read_excel(file_path, header=header_row)
                if len(df.columns) >= 5:  # Check if we have enough columns
                    logger.info(f"Successfully loaded Excel with header at row {header_row}")
                    break
            except:
                continue
        else:
            # If no header row works, try without header
            df = pd.read_excel(file_path, header=None)
            # Rename columns to default names
            df.columns = [f"Column_{i}" for i in range(len(df.columns))]
            logger.warning("Could not detect header row, using default column names")
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise
    
    logger.info(f"Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")
    logger.info("Columns in the Excel file: " + ", ".join(df.columns.tolist()))
    
    # Clean and prepare the data
    df = clean_excel_data(df)
    
    # Convert date columns to epoch timestamps
    date_columns = ['LCD_DATE', 'PLANNED_START', 'PLANNED_END', 'LATEST COMPLETION DATE']
    df = convert_to_epoch(df, date_columns)
    
    # Define column mapping to standardize column names
    column_mapping = {
        'DUE_DATE_TIME': 'epoch_lcd_date',
        'JOB_CODE': 'JOBCODE',
        'PROCESS_CODE': 'PROCESS_CODE',
        'MACHINE_ID': 'MACHINE_ID',
        'NUMBER_OPERATOR': 'NUMBER_OPERATOR',
        'OUTPUT_PER_HOUR': 'OUTPUT_PER_HOUR',
        'DURATION_IN_HOUR': 'HOURS NEEDED',
        'START_TIME': 'epoch_planned_start',
        'END_TIME': 'epoch_planned_end',
        'LATEST_COMPLETION_TIME': 'epoch_latest_completion_date',
        'PRIORITY': 'PRIORITY'
    }
    
    # Check for missing columns
    missing_columns = [f"{expected_col} (looking for '{actual_col}')" 
                      for expected_col, actual_col in column_mapping.items() 
                      if actual_col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}")
    
    # Create standardized dataframe
    data = pd.DataFrame()
    data['DUE_DATE_TIME'] = df['epoch_lcd_date'] if 'epoch_lcd_date' in df.columns else pd.Series()
    data['JOB_CODE'] = df['JOBCODE'] if 'JOBCODE' in df.columns else pd.Series()
    data['PROCESS_CODE'] = df['PROCESS_CODE'] if 'PROCESS_CODE' in df.columns else pd.Series()
    data['MACHINE_ID'] = df['MACHINE_ID'] if 'MACHINE_ID' in df.columns else pd.Series()
    data['NUMBER_OPERATOR'] = df['NUMBER_OPERATOR'] if 'NUMBER_OPERATOR' in df.columns else pd.Series(1)
    data['OUTPUT_PER_HOUR'] = df['OUTPUT_PER_HOUR'] if 'OUTPUT_PER_HOUR' in df.columns else pd.Series(0)
    data['DURATION_IN_HOUR'] = df['HOURS NEEDED'] if 'HOURS NEEDED' in df.columns else pd.Series(0)
    data['START_TIME'] = df['epoch_planned_start'] if 'epoch_planned_start' in df.columns else pd.Series()
    data['END_TIME'] = df['epoch_planned_end'] if 'epoch_planned_end' in df.columns else pd.Series()
    data['LATEST_COMPLETION_TIME'] = df['epoch_latest_completion_date'] if 'epoch_latest_completion_date' in df.columns else pd.Series()
    data['PRIORITY'] = df['PRIORITY'] if 'PRIORITY' in df.columns else pd.Series(3)
    
    # Calculate processing time in seconds
    data['processing_time'] = (pd.to_numeric(data['DURATION_IN_HOUR'], errors='coerce').fillna(0) * 3600).astype(int)
    
    logger.info(f"DataFrame shape: {data.shape}")
    logger.info(f"DataFrame columns: {data.columns.tolist()}")
    logger.info(f"Sample data:\n{data.head(2)}")
    
    # Convert dataframe to job tuples
    jobs = []
    current_time = int(datetime.now().timestamp())
    for _, row in data.iterrows():
        if pd.isna(row['PROCESS_CODE']) or pd.isna(row['MACHINE_ID']):
            logger.warning(f"Skipping row: PROCESS_CODE={row.get('PROCESS_CODE', 'NaN')}, MACHINE_ID={row.get('MACHINE_ID', 'NaN')}")
            continue
        
        # Derive job code from process code if missing
        job_code = row['JOB_CODE'] if pd.notna(row['JOB_CODE']) else row['PROCESS_CODE'].split('-P')[0] if '-P' in row['PROCESS_CODE'] else row['PROCESS_CODE']
        
        # Handle dates and times
        due_seconds = row['LATEST_COMPLETION_TIME'] if pd.notna(row['LATEST_COMPLETION_TIME']) else current_time + 30 * 24 * 3600
        proc_time = row['processing_time']
        due_seconds = max(current_time + proc_time, int(due_seconds))
        
        # Handle planned start and end times
        start_seconds = row['START_TIME'] if pd.notna(row['START_TIME']) else current_time
        start_seconds = max(current_time, int(start_seconds))
        
        end_seconds = row['END_TIME'] if pd.notna(row['END_TIME']) else start_seconds + proc_time
        
        # Calculate priority
        base_priority = int(row['PRIORITY']) if pd.notna(row['PRIORITY']) else 3
        base_priority = max(1, min(5, base_priority))
        
        # Adjust priority for urgent jobs
        days_remaining = (due_seconds - current_time) / (24 * 3600)
        if days_remaining <= 5 and base_priority > 1:
            base_priority -= 1
        final_priority = max(1, min(5, int(base_priority)))
        
        # Create job tuple with all available information
        job = (
            job_code,               # Job name
            row['PROCESS_CODE'],    # Process code
            row['MACHINE_ID'],      # Machine ID
            proc_time,              # Processing time in seconds
            due_seconds,            # Due time in epoch seconds
            final_priority,         # Priority (1-5, 1 is highest)
            start_seconds,          # Planned start time
            int(row['NUMBER_OPERATOR']) if pd.notna(row['NUMBER_OPERATOR']) else 1,  # Number of operators
            float(row['OUTPUT_PER_HOUR']) if pd.notna(row['OUTPUT_PER_HOUR']) else 100.0,  # Output rate
            end_seconds             # Planned end time
        )
        
        jobs.append(job)
    
    # Extract unique machines
    machines_df = data[['MACHINE_ID']].drop_duplicates()
    machines = [(i, m, 0) for i, m in enumerate(machines_df['MACHINE_ID']) if pd.notna(m)]
    
    # Generate setup times matrix
    process_codes = [p for p in data['PROCESS_CODE'].unique() if pd.notna(p)]
    setup_times = {p1: {p2: 10 if p1 == p2 else 30 for p2 in process_codes} for p1 in process_codes}
    
    logger.info(f"Generated {len(jobs)} valid jobs, {len(machines)} machines, and setup times for {len(process_codes)} processes")
    return jobs, machines, setup_times