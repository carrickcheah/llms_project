"""
Functions for handling setup times and schedule time adjustments.
"""
import pandas as pd
from datetime import datetime
import re
from loguru import logger

def get_buffer_status(buffer_hours):
    """Get status category based on buffer hours."""
    if buffer_hours < 8:
        return "Critical"
    elif buffer_hours < 24:
        return "Warning"
    elif buffer_hours < 72:
        return "Caution"
    else:
        return "OK"

def extract_job_family(unique_job_id):
    """
    Extract the job family (e.g., 'CP08-231B' from 'JOB_CP08-231B-P01-06') from the UNIQUE_JOB_ID.
    UNIQUE_JOB_ID is in the format JOB_PROCESS_CODE.
    """
    try:
        process_code = unique_job_id.split('_', 1)[1]  # Split on first underscore to get PROCESS_CODE
    except IndexError:
        logger.warning(f"Could not extract PROCESS_CODE from UNIQUE_JOB_ID {unique_job_id}")
        return unique_job_id

    process_code = str(process_code).upper()
    match = re.search(r'(.*?)-P\d+', process_code)
    if match:
        family = match.group(1)
        logger.debug(f"Extracted family {family} from {unique_job_id}")
        return family
    parts = process_code.split("-P")
    if len(parts) >= 2:
        family = parts[0]
        logger.debug(f"Extracted family {family} from {unique_job_id} (using split)")
        return family
    logger.warning(f"Could not extract family from {unique_job_id}, using full code")
    return process_code

def extract_process_number(unique_job_id):
    """
    Extract the process sequence number (e.g., 1 from 'P01-06' in 'JOB_P01-06') or return 999 if not found.
    UNIQUE_JOB_ID is in the format JOB_PROCESS_CODE.
    """
    try:
        process_code = unique_job_id.split('_', 1)[1]  # Split on first underscore to get PROCESS_CODE
    except IndexError:
        logger.warning(f"Could not extract PROCESS_CODE from UNIQUE_JOB_ID {unique_job_id}")
        return 999

    match = re.search(r'P(\d{2})', str(process_code).upper())  # Match exactly two digits after P
    if match:
        seq = int(match.group(1))
        return seq
    return 999  # Default if parsing fails

def add_schedule_times_and_buffer(jobs, schedule):
    """
    Add schedule times (START_TIME and END_TIME) to job dictionaries and
    calculate the buffer time (BAL_HR) between job completion and deadline.
    Also adjusts times for dependent processes to maintain proper sequence.
    
    Args:
        jobs (list): List of job dictionaries, each with UNIQUE_JOB_ID
        schedule (dict): Schedule as {machine: [(unique_job_id, start, end, priority), ...]}
        
    Returns:
        list: Updated jobs list with START_TIME, END_TIME, and BAL_HR added
    """
    times = {}
    for machine, tasks in schedule.items():
        for task in tasks:
            # Handle both old format (4-tuple) and new format (5-tuple with additional params)
            if len(task) >= 5:
                unique_job_id, start, end, _, additional_params = task
            else:
                unique_job_id, start, end, _ = task
                additional_params = {}
                
            times[unique_job_id] = (start, end)
            
            # Store additional parameters in the job object if available
            if additional_params:
                for job in jobs:
                    if job.get('UNIQUE_JOB_ID') == unique_job_id:
                        if 'setup_time' in additional_params and additional_params['setup_time'] > 0:
                            job['ACTUAL_SETUP_TIME'] = additional_params['setup_time']
                        if 'break_time' in additional_params and additional_params['break_time'] > 0:
                            job['ACTUAL_BREAK_TIME'] = additional_params['break_time']
                        if 'no_prod_time' in additional_params and additional_params['no_prod_time'] > 0:
                            job['ACTUAL_NO_PROD_TIME'] = additional_params['no_prod_time']
    
    family_processes = {}
    for job in jobs:
        unique_job_id = job['UNIQUE_JOB_ID']
        family = extract_job_family(unique_job_id)
        seq_num = extract_process_number(unique_job_id)
        
        if family not in family_processes:
            family_processes[family] = []
        
        family_processes[family].append((seq_num, unique_job_id, job))
    
    for family in family_processes:
        family_processes[family].sort(key=lambda x: x[0])
    
    family_time_shifts = {}
    for family, processes in family_processes.items():
        for seq_num, unique_job_id, job in processes:
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and not pd.isna(job['START_DATE_EPOCH']) and unique_job_id in times:
                scheduled_start = times[unique_job_id][0]
                
                requested_start = None
                if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None:
                    if isinstance(job['START_DATE_EPOCH'], float) and not pd.isna(job['START_DATE_EPOCH']):
                        requested_start = job['START_DATE_EPOCH']
                    elif not isinstance(job['START_DATE_EPOCH'], float):
                        requested_start = job['START_DATE_EPOCH']
                        
                if requested_start is None and 'START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None:
                    if isinstance(job['START_DATE _EPOCH'], float) and not pd.isna(job['START_DATE _EPOCH']):
                        requested_start = job['START_DATE _EPOCH']
                    elif not isinstance(job['START_DATE _EPOCH'], float):
                        requested_start = job['START_DATE _EPOCH']
                        
                time_shift = None
                if requested_start is not None:
                    time_shift = scheduled_start - requested_start
                    if family not in family_time_shifts or abs(time_shift) > abs(family_time_shifts[family]):
                        family_time_shifts[family] = time_shift
                
                if time_shift is not None:
                    logger.info(f"Family {family} has START_DATE constraint for {unique_job_id}: " 
                              f"shift={time_shift/3600:.1f} hours")
                else:
                    logger.info(f"Family {family} has START_DATE constraint for {unique_job_id}, but no valid time shift calculated")
    
    job_adjustments = {}
    for family, time_shift in family_time_shifts.items():
        if abs(time_shift) < 60:
            continue
            
        logger.info(f"Applying time shift of {time_shift/3600:.1f} hours to family {family} for visualization")
        
        for seq_num, unique_job_id, job in family_processes[family]:
            if unique_job_id in times:
                original_start, original_end = times[unique_job_id]
                
                if time_shift is None or (isinstance(time_shift, float) and (pd.isna(time_shift) or not pd.notna(time_shift))):
                    logger.warning(f"Skipping time shift for {unique_job_id} due to invalid shift value: {time_shift}")
                    continue
                    
                try:
                    adjusted_start = original_start - time_shift
                    adjusted_end = original_end - time_shift
                    job_adjustments[unique_job_id] = (adjusted_start, adjusted_end)
                    logger.info(f"  Adjusted {unique_job_id}: START={datetime.fromtimestamp(adjusted_start).strftime('%Y-%m-%d %H:%M')}, "
                              f"END={datetime.fromtimestamp(adjusted_end).strftime('%Y-%m-%d %H:%M')}")
                except Exception as e:
                    logger.warning(f"Error adjusting time for {unique_job_id}: {e}")
    
    for job in jobs:
        unique_job_id = job['UNIQUE_JOB_ID']
        if unique_job_id in times:
            original_start, original_end = times[unique_job_id]
            due_time = job.get('LCD_DATE_EPOCH', 0)
            
            if unique_job_id in job_adjustments:
                job_start, job_end = job_adjustments[unique_job_id]
            else:
                job_start = original_start
                job_end = original_end
            
            start_date_epoch = None
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and not pd.isna(job['START_DATE_EPOCH']):
                start_date_epoch = job['START_DATE_EPOCH']
            elif 'START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None and not pd.isna(job['START_DATE _EPOCH']):
                start_date_epoch = job['START_DATE _EPOCH']
                
            if start_date_epoch is not None:
                job_start = start_date_epoch
                duration = original_end - original_start
                job_end = job_start + duration
                start_date = datetime.fromtimestamp(start_date_epoch).strftime('%Y-%m-%d %H:%M')
                logger.info(f"Setting START_TIME to match START_DATE ({start_date}) for job {unique_job_id}")
            
            if job_start is not None and not pd.isna(job_start) and isinstance(job_start, (int, float)):
                job['START_TIME'] = job_start
            else:
                job['START_TIME'] = int(datetime.now().timestamp())
                logger.warning(f"Set default START_TIME for job {unique_job_id} due to invalid value: {job_start}")
                
            if job_end is not None and not pd.isna(job_end) and isinstance(job_end, (int, float)):
                job['END_TIME'] = job_end
            else:
                job['END_TIME'] = job['START_TIME'] + 3600
                logger.warning(f"Set default END_TIME for job {unique_job_id} due to invalid value: {job_end}")
            
            job_start = job['START_TIME'] 
            job_end = job['END_TIME']
            
            valid_due_time = False
            if due_time is not None and not pd.isna(due_time) and isinstance(due_time, (int, float)):
                current_time = int(datetime.now().timestamp())
                if job_end <= due_time <= (current_time + 365 * 24 * 3600):
                    buffer_seconds = max(0, due_time - job_end)
                    buffer_hours = buffer_seconds / 3600
                    valid_due_time = True
                elif due_time < job_end:
                    buffer_seconds = 0
                    buffer_hours = 0
                    valid_due_time = True
                    logger.warning(f"Job {unique_job_id} will be LATE! Due at {datetime.fromtimestamp(due_time).strftime('%Y-%m-%d %H:%M')} but ends at {datetime.fromtimestamp(job_end).strftime('%Y-%m-%d %H:%M')}")
                else:
                    logger.warning(f"Due date for {unique_job_id} is too far in future ({datetime.fromtimestamp(due_time).strftime('%Y-%m-%d %H:%M')}), might be incorrect")
            
            if not valid_due_time:
                buffer_seconds = 24 * 3600
                buffer_hours = 24.0
                logger.warning(f"Set default BAL_HR for job {unique_job_id} due to invalid LCD_DATE_EPOCH: {due_time}")
            
            if buffer_hours > 720:
                logger.info(f"Job {unique_job_id} has a large buffer of {buffer_hours:.1f} hours ({buffer_hours/24:.1f} days)")
            
            job['BAL_HR'] = buffer_hours
            job['BUFFER_STATUS'] = get_buffer_status(buffer_hours)
    
    return jobs
