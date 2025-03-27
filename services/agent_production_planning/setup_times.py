"""
Functions for handling setup times and schedule time adjustments.

This module manages the critical timing aspects of the production planning system:
1. Standardized field access for start date epochs (handling name inconsistencies)  
2. Schedule time calculations and adjustments based on constraints
3. Buffer time calculations between job completion and deadlines
4. Schedule visualization preparation

The overall workflow is:
- Extract scheduled times from the optimized solution
- Group jobs by family and process sequence
- Apply time shifts based on START_DATE constraints
- Calculate buffer hours between job completion and deadline
- Categorize buffer status for visualization
"""
import pandas as pd
from datetime import datetime
import re
from loguru import logger


def get_start_date_epoch(job):
    """
    Standardized accessor for START_DATE_EPOCH field that handles both naming variants.
    Always use this function instead of accessing START_DATE_EPOCH or START_DATE _EPOCH directly.
    
    Args:
        job (dict): Job dictionary
        
    Returns:
        int/float/None: The start date epoch value or None if not present/valid
    """
    # Check standard field name first
    if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and not pd.isna(job['START_DATE_EPOCH']):
        return job['START_DATE_EPOCH']
    
    # Check alternate field name (with space)
    if 'START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None and not pd.isna(job['START_DATE _EPOCH']):
        return job['START_DATE _EPOCH']
    
    return None


def is_valid_timestamp(timestamp):
    """
    Check if a timestamp is valid for calculations.
    
    Args:
        timestamp: Value to check
        
    Returns:
        bool: True if timestamp is valid, False otherwise
    """
    return (timestamp is not None and 
            not pd.isna(timestamp) and 
            isinstance(timestamp, (int, float)))

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
    
    This is a central function in the production planning workflow that:
    1. Extracts the scheduled start/end times for each job from the optimizer solution
    2. Groups jobs by family to handle related processes
    3. Applies time shift adjustments based on START_DATE constraints
    4. Calculates buffer hours between job completion and deadline
    5. Adds status indicators for buffer visualization
    
    Each job's UNIQUE_JOB_ID follows a pattern that includes process number information,
    which is used to identify related jobs that must be processed in sequence.
    
    Args:
        jobs (list): List of job dictionaries, each with UNIQUE_JOB_ID
        schedule (dict): Schedule as {machine: [(unique_job_id, start, end, priority), ...]}
        
    Returns:
        list: Updated jobs list with START_TIME, END_TIME, and BAL_HR added
    """
    # STEP 1: Extract job start/end times from the scheduling solution and store in a dictionary
    # This dictionary will serve as our reference for all subsequent time calculations
    times = {}
    for machine, tasks in schedule.items():
        for task in tasks:
            # Handle both old format (4-tuple) and new format (5-tuple with additional params)
            # The task format depends on the scheduler's output version
            if len(task) >= 5:
                unique_job_id, start, end, _, additional_params = task
            else:
                unique_job_id, start, end, _ = task
                additional_params = {}
                
            # Store the start and end times for each job
            times[unique_job_id] = (start, end)
            
            # Store additional timing information in the job object if available
            # These are actual values from the scheduler that may differ from estimates
            if additional_params:
                for job in jobs:
                    if job.get('UNIQUE_JOB_ID') == unique_job_id:
                        if 'setup_time' in additional_params and additional_params['setup_time'] > 0:
                            job['ACTUAL_SETUP_TIME'] = additional_params['setup_time']  # Machine setup time
                        if 'break_time' in additional_params and additional_params['break_time'] > 0:
                            job['ACTUAL_BREAK_TIME'] = additional_params['break_time']  # Production breaks
                        if 'no_prod_time' in additional_params and additional_params['no_prod_time'] > 0:
                            job['ACTUAL_NO_PROD_TIME'] = additional_params['no_prod_time']  # Non-production hours
    
    # STEP 2: Group jobs by family and sequence number
    # Jobs within the same family are related and often have sequential dependencies
    # We organize them to ensure proper time adjustments across the production sequence
    family_processes = {}
    for job in jobs:
        unique_job_id = job['UNIQUE_JOB_ID']
        # Extract the job family code from the job ID (e.g., 'CP88-888' from 'JOST888001_CP88-888-P01-08')
        family = extract_job_family(unique_job_id)
        # Extract the process sequence number (e.g., '01' from 'P01-08')
        seq_num = extract_process_number(unique_job_id)
        
        # Initialize family group if this is the first job from this family
        if family not in family_processes:
            family_processes[family] = []
        
        # Store the job with its sequence number and ID for later processing
        family_processes[family].append((seq_num, unique_job_id, job))
    
    # Sort jobs within each family by their sequence number to maintain proper order
    for family in family_processes:
        family_processes[family].sort(key=lambda x: x[0])
    
    # STEP 3: Calculate time shifts needed to honor START_DATE constraints
    # Some jobs have fixed start date requirements that must be respected
    # We calculate how much to shift job times to align with these constraints
    family_time_shifts = {}  # Store the maximum time shift needed for each job family
    
    for family, processes in family_processes.items():
        for seq_num, unique_job_id, job in processes:
            # Use helper function to get START_DATE_EPOCH consistently
            # This handles both field name formats: 'START_DATE_EPOCH' and 'START_DATE _EPOCH'
            start_date_epoch = get_start_date_epoch(job)
            
            # Check if this job has a start date constraint and is included in the schedule
            if start_date_epoch is not None and unique_job_id in times:
                # Get the time when this job is currently scheduled to start
                scheduled_start = times[unique_job_id][0]
                # Get the required start time from the constraint
                requested_start = start_date_epoch
                        
                # Calculate how much we need to shift the job to meet the constraint
                # A positive shift means the job is scheduled too late and needs to move earlier
                # A negative shift means the job is scheduled too early and needs to move later
                time_shift = None
                if requested_start is not None:
                    time_shift = scheduled_start - requested_start
                    
                    # For each family, we keep track of the largest shift needed
                    # This ensures we move all related jobs by the same amount to maintain sequence
                    if family not in family_time_shifts or abs(time_shift) > abs(family_time_shifts[family]):
                        family_time_shifts[family] = time_shift
                
                # Log information about the constraint and calculated shift
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
            
            # Use helper function to get START_DATE_EPOCH value regardless of field name format
            start_date_epoch = get_start_date_epoch(job)
                
            if start_date_epoch is not None:
                job_start = start_date_epoch
                duration = original_end - original_start
                job_end = job_start + duration
                start_date = datetime.fromtimestamp(start_date_epoch).strftime('%Y-%m-%d %H:%M')
                logger.info(f"Setting START_TIME to match START_DATE ({start_date}) for job {unique_job_id}")
            
            # Validate start time using helper function
            if is_valid_timestamp(job_start):
                job['START_TIME'] = job_start
            else:
                job['START_TIME'] = int(datetime.now().timestamp())
                logger.warning(f"Set default START_TIME for job {unique_job_id} due to invalid value: {job_start}")
                
            # Validate end time using helper function
            if is_valid_timestamp(job_end):
                job['END_TIME'] = job_end
            else:
                job['END_TIME'] = job['START_TIME'] + 3600
                logger.warning(f"Set default END_TIME for job {unique_job_id} due to invalid value: {job_end}")
            
            job_start = job['START_TIME'] 
            job_end = job['END_TIME']
            
            valid_due_time = False
            # Validate due time using our helper function
            if is_valid_timestamp(due_time):
                current_time = int(datetime.now().timestamp())
                one_year_future = current_time + 365 * 24 * 3600
                
                # Check if due time is within reasonable range (between job end and one year from now)
                if job_end <= due_time <= one_year_future:
                    buffer_seconds = max(0, due_time - job_end)
                    buffer_hours = buffer_seconds / 3600
                    valid_due_time = True
                # Job will be late
                elif due_time < job_end:
                    buffer_seconds = 0
                    buffer_hours = 0
                    valid_due_time = True
                    due_date_str = datetime.fromtimestamp(due_time).strftime('%Y-%m-%d %H:%M')
                    end_date_str = datetime.fromtimestamp(job_end).strftime('%Y-%m-%d %H:%M')
                    logger.warning(f"Job {unique_job_id} will be LATE! Due at {due_date_str} but ends at {end_date_str}")
                # Due time is too far in the future
                else:
                    due_date_str = datetime.fromtimestamp(due_time).strftime('%Y-%m-%d %H:%M')
                    logger.warning(f"Due date for {unique_job_id} is too far in future ({due_date_str}), might be incorrect")
            
            if not valid_due_time:
                buffer_seconds = 24 * 3600
                buffer_hours = 24.0
                logger.warning(f"Set default BAL_HR for job {unique_job_id} due to invalid LCD_DATE_EPOCH: {due_time}")
            
            if buffer_hours > 720:
                logger.info(f"Job {unique_job_id} has a large buffer of {buffer_hours:.1f} hours ({buffer_hours/24:.1f} days)")
            
            job['BAL_HR'] = buffer_hours
            job['BUFFER_STATUS'] = get_buffer_status(buffer_hours)
    
    return jobs
