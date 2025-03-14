# ingest_data.py
import pandas as pd
import numpy as np
from datetime import datetime
import os

def clean_excel_data(df):
    """Clean and prepare Excel data for scheduling."""
    df = df.copy()
    initial_rows = len(df)
    
    df = df.dropna(how='all')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        df[col] = df[col].str.replace(r'[^\w\s-]', '', regex=True).str.replace(r'\s+', ' ', regex=True)
    
    numeric_cols = {
        'NUMBER_OPERATOR': {'min': 1, 'max': 10, 'default': 1},
        'OUTPUT_PER_HOUR': {'min': 0.1, 'max': 10000, 'default': 100},
        'HOURS NEEDED': {'min': 0.1, 'max': 720, 'default': None},
        'PRIORITY': {'min': 1, 'max': 5, 'default': 4}
    }
    
    for col, rules in numeric_cols.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] < rules['min'], col] = rules['default'] if rules['default'] is not None else rules['min']
            df.loc[df[col] > rules['max'], col] = rules['max']
    
    if 'JOBCODE' in df.columns:
        df = df[df['JOBCODE'].notna() & (df['JOBCODE'].str.len() > 0)]
    if 'PROCESS_CODE' in df.columns:
        df = df[df['PROCESS_CODE'].notna() & (df['PROCESS_CODE'].str.len() > 0)]
    
    date_cols = ['LCD_DATE', 'PLANNED_START ', 'PLANNED_END', 'LATEST COMPLETION DATE']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
    
    key_cols = [col for col in ['JOBCODE', 'PROCESS_CODE', 'MACHINE_ID'] if col in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols)
    
    if 'MACHINE_ID' in df.columns:
        df = df[df['MACHINE_ID'].notna() & (df['MACHINE_ID'].str.len() > 0)]
    
    if 'HOURS NEEDED' in df.columns and 'OUTPUT_PER_HOUR' in df.columns:
        df.loc[df['OUTPUT_PER_HOUR'] <= 0, 'OUTPUT_PER_HOUR'] = 100
        df['processing_time'] = df['HOURS NEEDED'] * 3600
    
    removed_rows = initial_rows - len(df)
    print(f"Data cleaning: {len(df)} rows after cleaning (removed {removed_rows} invalid rows)")
    if removed_rows > 0:
        print("Reasons for removal:")
        print("- Missing or invalid job/process codes")
        print("- Zero or negative processing times")
        print("- Missing machine IDs")
        print("- Invalid dates")
    
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
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
            continue
        
        epoch_col = f"epoch_{col.lower().replace(' ', '_')}"
        try:
            # For date columns that should be datetime objects
            if df[col].dtype in ['object', 'string'] or pd.api.types.is_datetime64_any_dtype(df[col]):
                # Try with specific format first (for consistency)
                try:
                    df[epoch_col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce').apply(
                        lambda x: int(x.timestamp()) if pd.notna(x) else None
                    )
                except:
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
                print(f"Warning: Column '{col}' has unsupported type {df[col].dtype}. Skipping.")
                continue
                
            # Handle special cases for required fields
            if col == 'LATEST COMPLETION DATE' and df[epoch_col].isna().all():
                print(f"No valid data in '{col}' - setting default due dates (30 days from now)")
                df[epoch_col] = default_due_date
            
            valid_count = df[epoch_col].notna().sum()
            print(f"Converted '{col}' to '{epoch_col}': {valid_count} valid timestamps")
        except Exception as e:
            print(f"Error converting column '{col}' to epoch: {e}")
            # For 'LATEST COMPLETION DATE', use default values if conversion fails
            if col == 'LATEST COMPLETION DATE':
                print(f"Using default due dates for '{col}'")
                df[epoch_col] = default_due_date
            else:
                df[epoch_col] = None
    
    return df

def load_jobs_planning_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found at {file_path}")
    
    df = pd.read_excel(file_path, header=4)
    print("Columns in the Excel file:", df.columns.tolist())
    
    df = clean_excel_data(df)
    date_columns = ['LCD_DATE', 'PLANNED_START ', 'PLANNED_END', 'LATEST COMPLETION DATE']
    df = convert_to_epoch(df, date_columns)
    
    column_mapping = {
        'DUE_DATE_TIME': 'epoch_lcd_date',
        'JOB_CODE': 'JOBCODE',
        'PROCESS_CODE': 'PROCESS_CODE',
        'MACHINE_ID': 'MACHINE_ID',
        'NUMBER_OPERATOR': 'NUMBER_OPERATOR',
        'OUTPUT_PER_HOUR': 'OUTPUT_PER_HOUR',
        'DURATION_IN_HOUR': 'HOURS NEEDED',
        'START_TIME': 'epoch_planned_start_',
        'LATEST_COMPLETION_TIME': 'epoch_latest_completion_date',
        'PRIORITY': 'PRIORITY'
    }
    
    missing_columns = [f"{expected_col} (looking for '{actual_col}')" 
                      for expected_col, actual_col in column_mapping.items() 
                      if actual_col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    
    data = pd.DataFrame()
    data['DUE_DATE_TIME'] = df['epoch_lcd_date'] if 'epoch_lcd_date' in df.columns else pd.Series()
    data['JOB_CODE'] = df['JOBCODE'] if 'JOBCODE' in df.columns else pd.Series()
    data['PROCESS_CODE'] = df['PROCESS_CODE'] if 'PROCESS_CODE' in df.columns else pd.Series()
    data['MACHINE_ID'] = df['MACHINE_ID'] if 'MACHINE_ID' in df.columns else pd.Series()
    data['NUMBER_OPERATOR'] = df['NUMBER_OPERATOR'] if 'NUMBER_OPERATOR' in df.columns else pd.Series(1)
    data['OUTPUT_PER_HOUR'] = df['OUTPUT_PER_HOUR'] if 'OUTPUT_PER_HOUR' in df.columns else pd.Series(0)
    data['DURATION_IN_HOUR'] = df['HOURS NEEDED'] if 'HOURS NEEDED' in df.columns else pd.Series(0)
    data['START_TIME'] = df['epoch_planned_start_'] if 'epoch_planned_start_' in df.columns else pd.Series()
    data['LATEST_COMPLETION_TIME'] = df['epoch_latest_completion_date'] if 'epoch_latest_completion_date' in df.columns else pd.Series()
    data['PRIORITY'] = df['PRIORITY'] if 'PRIORITY' in df.columns else pd.Series(3)
    
    data['processing_time'] = (pd.to_numeric(data['DURATION_IN_HOUR'], errors='coerce').fillna(0) * 3600).astype(int)
    
    print(f"DataFrame shape: {data.shape}")
    print(f"DataFrame columns: {data.columns.tolist()}")
    print(f"Sample data:\n{data.head(2)}")
    
    jobs = []
    current_time = int(datetime.now().timestamp())
    for _, row in data.iterrows():
        if pd.isna(row['PROCESS_CODE']) or pd.isna(row['MACHINE_ID']):
            print(f"Skipping row: PROCESS_CODE={row.get('PROCESS_CODE', 'NaN')}, MACHINE_ID={row.get('MACHINE_ID', 'NaN')}")
            continue
        
        job_code = row['JOB_CODE'] if pd.notna(row['JOB_CODE']) else row['PROCESS_CODE'].split('-P')[0] if '-P' in row['PROCESS_CODE'] else row['PROCESS_CODE']
        
        due_seconds = row['LATEST_COMPLETION_TIME'] if pd.notna(row['LATEST_COMPLETION_TIME']) else current_time + 30 * 24 * 3600
        proc_time = row['processing_time']
        due_seconds = max(current_time + proc_time, int(due_seconds))
        start_seconds = row['START_TIME'] if pd.notna(row['START_TIME']) else current_time
        start_seconds = max(current_time, int(start_seconds))
        
        base_priority = int(row['PRIORITY']) if pd.notna(row['PRIORITY']) else 3
        base_priority = max(1, min(5, base_priority))
        
        days_remaining = (due_seconds - current_time) / (24 * 3600)
        if days_remaining <= 5 and base_priority > 1:
            base_priority -= 1
        final_priority = max(1, min(5, int(base_priority)))
        
        jobs.append((
            job_code,
            row['PROCESS_CODE'],
            row['MACHINE_ID'],
            proc_time,
            due_seconds,
            final_priority,
            start_seconds
        ))
    
    machines_df = data[['MACHINE_ID']].drop_duplicates()
    machines = [(i, m, 0) for i, m in enumerate(machines_df['MACHINE_ID']) if pd.notna(m)]
    process_codes = [p for p in data['PROCESS_CODE'].unique() if pd.notna(p)]
    setup_times = {p1: {p2: 10 if p1 == p2 else 30 for p2 in process_codes} for p1 in process_codes}
    
    return jobs, machines, setup_times