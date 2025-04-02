# greedy.py | dont edit this line
import logging
from datetime import datetime
import re
from dotenv import load_dotenv
import os
import time
import random
from collections import defaultdict
from time_utils import (
    epoch_to_datetime, 
    datetime_to_epoch,
    format_datetime_for_display,
    epoch_to_relative_hours,
    relative_hours_to_epoch
)

# Configure logging (standalone or align with project setup)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_process_number(unique_job_id):
    """
    Extract the process sequence number (e.g., 1 from 'P01-06' in 'JOB_P01-06') or return 999 if not found.
    UNIQUE_JOB_ID is in the format JOB_PROCESS_CODE.
    """
    # Extract the PROCESS_CODE part from UNIQUE_JOB_ID (after the underscore)
    try:
        process_code = unique_job_id.split('_', 1)[1]  # Split on first underscore to get PROCESS_CODE
    except IndexError:
        logger.warning(f"Could not extract PROCESS_CODE from UNIQUE_JOB_ID {unique_job_id}")
        return 999

    match = re.search(r'P(\d{2})', str(process_code).upper())  # Match exactly two digits after P
    if match:
        seq = int(match.group(1))
        return seq
    return 999

def extract_job_family(unique_job_id):
    """
    Extract the job family (e.g., 'CP08-231B' from 'JOB_CP08-231B-P01-06') from the UNIQUE_JOB_ID.
    UNIQUE_JOB_ID is in the format JOB_PROCESS_CODE.
    """
    # Extract the PROCESS_CODE part from UNIQUE_JOB_ID
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

def greedy_schedule(jobs, machines, setup_times=None, enforce_sequence=True, max_operators=0):
    """
    Create a schedule using a greedy algorithm.
    
    Args:
        jobs: List of job dictionaries
        machines: List of machine IDs
        setup_times: Dictionary of setup times between processes
        enforce_sequence: Whether to enforce process sequence dependencies
        max_operators: Maximum number of operators available at any time
        
    Returns:
        Dictionary with machine IDs as keys and lists of scheduled jobs as values
    """
    start_time = time.time()
    logger.info(f"Creating schedule using greedy algorithm for {len(jobs)} jobs on {len(machines)} machines")
    logger.info(f"Using max_operators={max_operators}")
    
    if not jobs:
        logger.warning("No jobs to schedule")
        return {}
        
    if not machines:
        logger.warning("No machines available")
        return {}
    
    # Use relative time instead of epoch to avoid large numbers
    current_time = datetime_to_epoch(datetime.now())
    current_rel_hours = epoch_to_relative_hours(current_time)
    
    # Create a dictionary to track machine availability
    machine_available_time = {machine: current_time for machine in machines}
    
    # Dictionary to track the max end time for each job family
    job_family_end_times = {}
    
    # Track operator usage over time
    operators_in_use = defaultdict(int)  # time_point -> number of operators
    
    # Sort jobs by priority (higher priority first) and then by due date (earlier due date first)
    sorted_jobs = sorted(jobs, key=lambda x: (
        -x.get('PRIORITY', 0),  # Higher priority first
        x.get('LCD_DATE_EPOCH', float('inf'))  # Earlier due date first
    ))
    
    # Jobs with START_DATE constraints should be scheduled first
    start_date_jobs = [job for job in sorted_jobs if 
                      job.get('START_DATE_EPOCH') is not None and 
                      job.get('START_DATE_EPOCH') > current_time]
    
    if start_date_jobs:
        logger.info(f"Found {len(start_date_jobs)} jobs with future START_DATE constraints")
        for job in start_date_jobs:
            start_date_epoch = job.get('START_DATE_EPOCH')
            job_id = job.get('UNIQUE_JOB_ID', 'unknown')
            
            # Use formatter for display
            start_date_dt = epoch_to_datetime(start_date_epoch)
            start_date_str = format_datetime_for_display(start_date_dt) if start_date_dt else "INVALID DATE"
            
            logger.info(f"Job {job_id} must start at {start_date_str}")
    
    # Move jobs with START_DATE constraints to the front of the list
    sorted_jobs = sorted(sorted_jobs, key=lambda x: (
        0 if x.get('START_DATE_EPOCH') is not None and x.get('START_DATE_EPOCH') > current_time else 1,
        -x.get('PRIORITY', 0),
        x.get('LCD_DATE_EPOCH', float('inf'))
    ))
    
    # Create schedule dictionary
    schedule = {machine: [] for machine in machines}
    
    # Group jobs by family for sequence enforcement
    job_families = defaultdict(list)
    if enforce_sequence:
        for job in sorted_jobs:
            family = extract_job_family(job['UNIQUE_JOB_ID'])
            if family:
                process_num = extract_process_number(job['UNIQUE_JOB_ID'])
                if process_num != 999:  # Only add if we got a valid process number
                    job_families[family].append((process_num, job))
                
        # Sort each family's jobs by process number and log
        for family, jobs_in_family in job_families.items():
            # Sort by process number, ensuring proper numerical ordering
            jobs_in_family.sort(key=lambda x: int(x[0]))  # Convert process_num to int for sorting
            logger.info(f"Job family {family} has {len(jobs_in_family)} processes in sequence")
            
            # Update the sorted_jobs list to maintain sequence order
            family_job_ids = [j[1]['UNIQUE_JOB_ID'] for j in jobs_in_family]
            logger.info(f"Job family {family} sequence: {family_job_ids}")
            
            # Update job_family_end_times with initial values
            job_family_end_times[family] = current_time

    # Schedule jobs
    scheduled_jobs = set()
    unscheduled_jobs = []
    
    for job in sorted_jobs:
        if 'UNIQUE_JOB_ID' not in job:
            logger.warning(f"Job has no UNIQUE_JOB_ID: {job}")
            continue
            
        job_id = job['UNIQUE_JOB_ID']
        machine_id = job.get('RSC_CODE')
        
        if not machine_id:
            logger.warning(f"Job {job_id} has no machine assignment")
            unscheduled_jobs.append(job)
            continue
            
        if machine_id not in machines:
            logger.warning(f"Job {job_id} assigned to unknown machine {machine_id}")
            unscheduled_jobs.append(job)
            continue
            
        if not job.get('processing_time'):
            if 'HOURS_NEED' in job and job['HOURS_NEED'] is not None:
                job['processing_time'] = job['HOURS_NEED'] * 3600
            else:
                logger.warning(f"Job {job_id} has no processing time")
                job['processing_time'] = 3600  # Default 1 hour
        
        processing_time = job['processing_time']
        
        # Get job family information for sequence enforcement
        job_family = job.get('JOB')
        process_code = job.get('PROCESS_CODE')
        
        # Handle START_DATE constraints
        start_date_epoch = job.get('START_DATE_EPOCH')
        earliest_start = machine_available_time[machine_id]
        
        if start_date_epoch is not None and start_date_epoch > current_time:
            # This job has a specific start date/time requirement
            earliest_start = max(earliest_start, start_date_epoch)
            
            # Use formatter for display
            start_date_dt = epoch_to_datetime(start_date_epoch)
            start_date_str = format_datetime_for_display(start_date_dt) if start_date_dt else "INVALID DATE"
            
            logger.info(f"Job {job_id} must start exactly at {start_date_str}")
        
        # Check sequence constraints
        if enforce_sequence:
            family = extract_job_family(job_id)
            if family:
                process_num = extract_process_number(job_id)
                if process_num != 999:
                    # Find the previous process in this family
                    prev_jobs = [(p, j) for p, j in job_families[family] if int(p) < int(process_num)]
                    if prev_jobs:
                        # Get the most recent previous process
                        prev_process = max(prev_jobs, key=lambda x: int(x[0]))
                        prev_job = prev_process[1]
                        prev_job_id = prev_job['UNIQUE_JOB_ID']
                        
                        # Check if previous job is scheduled
                        prev_scheduled = False
                        prev_end_time = current_time
                        for m in schedule:
                            for scheduled_id, start, end, *_ in schedule[m]:
                                if scheduled_id == prev_job_id:
                                    prev_scheduled = True
                                    prev_end_time = end
                                    break
                            if prev_scheduled:
                                break
                        
                        if not prev_scheduled:
                            logger.info(f"Cannot schedule {job_id} yet - previous process {prev_job_id} not scheduled")
                            continue
                        
                        # Update earliest start time based on previous process
                        earliest_start = max(earliest_start, prev_end_time)
                        logger.info(f"Enforcing zero-gap sequence: {prev_job_id} -> {job_id}, start = {format_datetime_for_display(epoch_to_datetime(earliest_start))}")
                        
                        # Update machine available time to maintain sequence
                        if earliest_start > machine_available_time[machine_id]:
                            logger.info(f"Machine {machine_id} will be idle from {format_datetime_for_display(epoch_to_datetime(current_time))} to {format_datetime_for_display(epoch_to_datetime(earliest_start))} to maintain sequence")
        
        # Check operator constraints if specified
        if max_operators > 0:
            # Find a time slot with available operators
            while True:
                # Check if we can start at earliest_start without exceeding operator limit
                can_schedule = True
                
                # Convert to relative hours for better handling
                start_rel = epoch_to_relative_hours(earliest_start)
                end_rel = epoch_to_relative_hours(earliest_start + processing_time)
                
                # Check hourly granularity for simplicity
                for hour in range(int(start_rel), int(end_rel) + 1):
                    if operators_in_use[hour] >= max_operators:
                        can_schedule = False
                        earliest_start = relative_hours_to_epoch(hour + 1)  # Try next hour
                        break
                
                if can_schedule:
                    break
        
        # Schedule the job
        start_time_epoch = earliest_start
        end_time_epoch = start_time_epoch + processing_time
        
        # Update operator usage if needed
        if max_operators > 0:
            # Convert to relative hours for better handling
            start_rel = epoch_to_relative_hours(start_time_epoch)
            end_rel = epoch_to_relative_hours(end_time_epoch)
            
            for hour in range(int(start_rel), int(end_rel) + 1):
                operators_in_use[hour] += 1
                if operators_in_use[hour] > max_operators:
                    logger.warning(f"Operator constraint exceeded at hour {hour}: {operators_in_use[hour]}/{max_operators}")
        
        # Update machine availability
        machine_available_time[machine_id] = end_time_epoch
        
        # Update job family end time for sequence enforcement
        if job_family:
            job_family_end_times[job_family] = max(
                job_family_end_times.get(job_family, 0),
                end_time_epoch
            )
        
        # Format times for logging
        start_dt = epoch_to_datetime(start_time_epoch)
        end_dt = epoch_to_datetime(end_time_epoch)
        
        start_str = format_datetime_for_display(start_dt) if start_dt else "INVALID DATE"
        end_str = format_datetime_for_display(end_dt) if end_dt else "INVALID DATE"
        
        # Add job to schedule
        schedule[machine_id].append((job_id, start_time_epoch, end_time_epoch, job.get('PRIORITY', 0)))
        
        # Mark job as scheduled and update job with timing information
        scheduled_jobs.add(job_id)
        job['START_TIME'] = start_time_epoch
        job['END_TIME'] = end_time_epoch
        
        logger.info(f"Scheduled {job_id} on {machine_id}: {start_str} to {end_str}")
    
    # Log scheduling statistics
    elapsed = time.time() - start_time
    total_scheduled = len(scheduled_jobs)
    total_unscheduled = len(unscheduled_jobs)
    
    logger.info(f"Greedy scheduling complete in {elapsed:.2f} seconds")
    logger.info(f"Scheduled {total_scheduled} jobs, {total_unscheduled} jobs could not be scheduled")
    
    # Calculate machines utilized
    machines_utilized = sum(1 for m, jobs in schedule.items() if jobs)
    logger.info(f"Utilized {machines_utilized}/{len(machines)} machines")
    
    # Get the latest completion time
    if any(schedule.values()):
        latest_end = max(end for m, jobs in schedule.items() if jobs for _, _, end, _ in jobs)
        latest_end_dt = epoch_to_datetime(latest_end)
        latest_end_str = format_datetime_for_display(latest_end_dt) if latest_end_dt else "INVALID DATE"
        logger.info(f"All jobs will complete by {latest_end_str}")
    
    # Check and report on START_DATE constraints
    start_date_job_ids = {job.get('UNIQUE_JOB_ID') for job in start_date_jobs}
    scheduled_start_dates = {}
    
    for machine, jobs_on_machine in schedule.items():
        for job_id, start, _, _ in jobs_on_machine:
            if job_id in start_date_job_ids:
                scheduled_start_dates[job_id] = start
    
    # Check if all START_DATE constraints were respected
    violated_constraints = []
    for job in start_date_jobs:
        job_id = job.get('UNIQUE_JOB_ID')
        if job_id in scheduled_start_dates:
            requested_start = job.get('START_DATE_EPOCH')
            actual_start = scheduled_start_dates[job_id]
            
            if requested_start != actual_start:
                # Format for display
                requested_dt = epoch_to_datetime(requested_start)
                actual_dt = epoch_to_datetime(actual_start)
                
                requested_str = format_datetime_for_display(requested_dt) if requested_dt else "INVALID DATE"
                actual_str = format_datetime_for_display(actual_dt) if actual_dt else "INVALID DATE"
                
                violated_constraints.append((job_id, requested_str, actual_str))
    
    if violated_constraints:
        logger.warning(f"Found {len(violated_constraints)} violated START_DATE constraints:")
        for job_id, requested, actual in violated_constraints:
            logger.warning(f"Job {job_id}: requested {requested}, scheduled at {actual}")
    else:
        logger.info("All START_DATE constraints were respected")
    
    # Check operator constraints
    if max_operators > 0:
        max_operators_used = max(operators_in_use.values()) if operators_in_use else 0
        logger.info(f"Maximum operators used at any time: {max_operators_used}/{max_operators}")
        
        if max_operators_used > max_operators:
            logger.warning(f"Operator constraint exceeded: used {max_operators_used}/{max_operators}")
            # Could implement more sophisticated handling here
    
    return schedule

if __name__ == "__main__":
    from ingest_data import load_jobs_planning_data
    
    load_dotenv()
    file_path = os.getenv('file_path')
    
    if not file_path:
        logger.error("No file_path found in environment variables.")
        exit(1)
        
    try:
        jobs, machines, setup_times = load_jobs_planning_data(file_path)
        logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines from {file_path}")
        schedule = greedy_schedule(jobs, machines, setup_times, enforce_sequence=True)
        logger.info(f"Generated schedule with {sum(len(jobs_list) for jobs_list in schedule.values())} scheduled tasks")
    except Exception as e:
        logger.error(f"Error testing scheduler with real data: {e}")