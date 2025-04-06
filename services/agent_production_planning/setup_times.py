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
import sys

# Configure loguru logger to only write to production_scheduler.log file
# Remove default handlers
logger.remove()
# Add file handler with WARNING level
logger.add("production_scheduler.log", level="WARNING", format="{time} | {level} | {name}:{function}:{line} - {message}")
# No console handler to suppress output

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
    """Get status category based on buffer hours.
    
    Args:
        buffer_hours (float): The buffer time in hours between job completion and deadline.
            Negative values indicate the job will be late by that many hours.
    
    Returns:
        str: Status category for visualization:
            "Late" - Job will be late (negative buffer)
            "Critical" - Less than 8 hours buffer
            "Warning" - Less than 24 hours buffer
            "Caution" - Less than 72 hours buffer
            "OK" - 72 hours or more buffer
    """
    if buffer_hours < 0:
        return "Late"
    elif buffer_hours < 8:
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
        return family
    parts = process_code.split("-P")
    if len(parts) >= 2:
        family = parts[0]
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
    
    # STEP 4: Apply time shifts to jobs to meet fixed start date constraints
    # We adjust the visualization times for all jobs in a family to maintain dependencies
    # This is critical for the Gantt chart visualization to accurately represent the schedule
    job_adjustments = {}  # Will store adjusted times for jobs that need to be shifted
    
    for family, time_shift in family_time_shifts.items():
        # Skip negligible time shifts (less than 1 minute)
        if abs(time_shift) < 60:
            continue
            
        logger.info(f"Applying time shift of {time_shift/3600:.1f} hours to family {family} for visualization")
        
        # Process all jobs in this family
        for seq_num, unique_job_id, job in family_processes[family]:
            if unique_job_id in times:
                # Get the original scheduled times from the optimizer
                original_start, original_end = times[unique_job_id]
                
                # Validate the time shift value before applying
                if time_shift is None or (isinstance(time_shift, float) and (pd.isna(time_shift) or not pd.notna(time_shift))):
                    logger.warning(f"Skipping time shift for {unique_job_id} due to invalid shift value: {time_shift}")
                    continue
                    
                try:
                    # Apply the time shift - subtract because positive shift means job is scheduled too late
                    # and we need to move it earlier
                    adjusted_start = original_start - time_shift
                    adjusted_end = original_end - time_shift
                    
                    # Store the adjusted times for this job
                    job_adjustments[unique_job_id] = (adjusted_start, adjusted_end)
                    
                    # Log the adjusted times for debugging and verification
                    start_str = datetime.fromtimestamp(adjusted_start).strftime('%Y-%m-%d %H:%M')
                    end_str = datetime.fromtimestamp(adjusted_end).strftime('%Y-%m-%d %H:%M')
                    logger.info(f"  Adjusted {unique_job_id}: START={start_str}, END={end_str}")
                except Exception as e:
                    logger.warning(f"Error adjusting time for {unique_job_id}: {e}")
    
    # STEP 5: Update each job with its final scheduling times
    # Either from the adjusted times (if family was shifted) or original optimizer times
    # Also prioritize explicit START_DATE constraints over calculated times
    for job in jobs:
        unique_job_id = job['UNIQUE_JOB_ID']
        if unique_job_id in times:
            # Get the original times from the optimizer solution
            original_start, original_end = times[unique_job_id]
            # Get the job's deadline from LCD_DATE_EPOCH field (or 0 if not available)
            due_time = job.get('LCD_DATE_EPOCH', 0)
            
            # Use adjusted times if available (from time shifts), otherwise use original times
            if unique_job_id in job_adjustments:
                job_start, job_end = job_adjustments[unique_job_id]
            else:
                job_start = original_start
                job_end = original_end
            
            # START_DATE constraints take absolute priority over calculated times
            # Use helper function to get START_DATE_EPOCH value regardless of field name format
            start_date_epoch = get_start_date_epoch(job)
                
            if start_date_epoch is not None:
                # Override the start time with the explicit constraint
                job_start = start_date_epoch
                # Maintain the same duration as originally calculated
                duration = original_end - original_start
                job_end = job_start + duration
                # Log this override for debugging purposes
                start_date = datetime.fromtimestamp(start_date_epoch).strftime('%Y-%m-%d %H:%M')
                logger.info(f"Setting START_TIME to match START_DATE ({start_date}) for job {unique_job_id}")
            
            # Ensure we have valid timestamps before updating the job
            # Validate start time using helper function
            if is_valid_timestamp(job_start):
                job['START_TIME'] = job_start
            else:
                # Use current time as a fallback for invalid start times
                job['START_TIME'] = int(datetime.now().timestamp())
                logger.warning(f"Set default START_TIME for job {unique_job_id} due to invalid value: {job_start}")
                
            # Validate end time using helper function
            if is_valid_timestamp(job_end):
                job['END_TIME'] = job_end
            else:
                # Add a default 1-hour duration for invalid end times
                job['END_TIME'] = job['START_TIME'] + 3600
                logger.warning(f"Set default END_TIME for job {unique_job_id} due to invalid value: {job_end}")
            
            # STEP 6: Calculate buffer time between job completion and deadline
            # Re-get the final start/end times from the job dictionary
            job_start = job['START_TIME'] 
            job_end = job['END_TIME']
            
            # Default to invalid due time (will be set to valid if checks pass)
            valid_due_time = False
            
            # Validate due time using our helper function
            if is_valid_timestamp(due_time):
                current_time = int(datetime.now().timestamp())
                one_year_future = current_time + 365 * 24 * 3600  # One year from now is reasonable max
                
                # CASE 1: Due time is in reasonable future range after job completion
                # This is the normal case - job will complete before deadline with some buffer
                if job_end <= due_time <= one_year_future:
                    # Calculate buffer in seconds, then convert to hours for display
                    buffer_seconds = max(0, due_time - job_end)
                    buffer_hours = buffer_seconds / 3600
                    valid_due_time = True
                
                # CASE 2: Job is scheduled to complete after its deadline
                # This is a problem case that needs user attention - job will be late
                elif due_time < job_end:
                    # Calculate negative buffer to show how many hours the job is over the deadline
                    buffer_seconds = due_time - job_end  # This will be negative
                    buffer_hours = buffer_seconds / 3600  # Negative hours
                    valid_due_time = True
                    # Format timestamps for readable logging
                    due_date_str = datetime.fromtimestamp(due_time).strftime('%Y-%m-%d %H:%M')
                    end_date_str = datetime.fromtimestamp(job_end).strftime('%Y-%m-%d %H:%M')
                    logger.warning(f"Job {unique_job_id} will be LATE! Due at {due_date_str} but ends at {end_date_str}, BAL_HR={buffer_hours:.1f}")
                
                # CASE 3: Due time is unreasonably far in the future (might be data error)
                else:
                    due_date_str = datetime.fromtimestamp(due_time).strftime('%Y-%m-%d %H:%M')
                    logger.warning(f"Due date for {unique_job_id} is too far in future ({due_date_str}), might be incorrect")
            
            # Handle case where due time is invalid or missing
            if not valid_due_time:
                # Default to 24 hours buffer for invalid/missing due dates
                buffer_seconds = 24 * 3600
                buffer_hours = 24.0
                logger.warning(f"Set default BAL_HR for job {unique_job_id} due to invalid LCD_DATE_EPOCH: {due_time}")
            
            # Log excessively large buffers (might indicate scheduling inefficiency or data errors)
            if buffer_hours > 720:  # More than 30 days buffer
                logger.info(f"Job {unique_job_id} has a large buffer of {buffer_hours:.1f} hours ({buffer_hours/24:.1f} days)")
            
            # STEP 7: Store buffer information and status for visualization
            # Add buffer hours to job dictionary for Gantt chart visualization
            job['BAL_HR'] = buffer_hours
            # Add status category based on buffer size (Critical, Warning, Caution, OK)
            job['BUFFER_STATUS'] = get_buffer_status(buffer_hours)
    
    return jobs
