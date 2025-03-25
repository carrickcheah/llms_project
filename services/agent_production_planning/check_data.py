from ingest_data import load_jobs_planning_data
import datetime
import pandas as pd
import numpy as np

def is_valid_timestamp(value):
    """Check if a value is a valid timestamp (not None, nan, etc.)"""
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return True

# Load the data
df = pd.read_excel('mydata.xlsx')
jobs, machines, setup_times = load_jobs_planning_data('mydata.xlsx')

print("START_DATE values from Excel:")
for idx, val in enumerate(df['START_DATE '].head(10)):
    print(f"{idx}: {val} ({type(val).__name__})")

print("\nConverted values in jobs:")
for idx, job in enumerate(jobs[:10]):
    if 'START_DATE ' in job or 'START_DATE_EPOCH' in job or 'START_DATE _EPOCH' in job:
        print(f"Job {idx} {job['PROCESS_CODE']}:")
        if 'START_DATE ' in job:
            print(f"  START_DATE : {job['START_DATE ']} ({type(job['START_DATE ']).__name__})")
        if 'START_DATE _EPOCH' in job:
            print(f"  START_DATE _EPOCH: {job['START_DATE _EPOCH']} ({type(job['START_DATE _EPOCH']).__name__})")
        if 'START_DATE_EPOCH' in job:
            print(f"  START_DATE_EPOCH: {job['START_DATE_EPOCH']} ({type(job['START_DATE_EPOCH']).__name__})")
            
print("\nChecking for NaN values:")
for idx, job in enumerate(jobs[:10]):
    for key, value in job.items():
        if '_EPOCH' in key and isinstance(value, float) and np.isnan(value):
            print(f"Job {idx} {job['PROCESS_CODE']}: {key} = {value} (NaN detected)")
    
        if '_EPOCH' in key and is_valid_timestamp(value):
            try:
                dt = datetime.datetime.fromtimestamp(value)
                print(f"Job {idx} {job['PROCESS_CODE']}: {key} = {value} → {dt}")
            except Exception as e:
                print(f"Job {idx} {job['PROCESS_CODE']}: {key} = {value} ERROR: {e}")