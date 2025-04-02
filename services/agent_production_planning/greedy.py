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

def find_best_machine(job, machines, machine_available_time):
    """Helper function to find the best machine for a job"""
    # First check if job has a specific machine requirement
    required_machine = job.get('RSC_CODE')
    if required_machine and required_machine in machines:
        return required_machine
        
    # If no specific machine required, find least loaded compatible machine
    compatible_machines = []
    for machine in machines:
        # Check if machine is compatible with job requirements
        # You can add more compatibility checks here
        compatible_machines.append(machine)
    
    if not compatible_machines:
        logger.warning(f"No compatible machines found for job {job.get('UNIQUE_JOB_ID')}")
        return None
        
    # Return least loaded compatible machine
    return min(compatible_machines, key=lambda m: machine_available_time[m])

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
    
    if not jobs or not machines:
        logger.warning("No jobs or machines available")
        return {}

    # Use relative time instead of epoch to avoid large numbers
    current_time = datetime_to_epoch(datetime.now())
    
    # Create a dictionary to track machine availability
    machine_available_time = {machine: current_time for machine in machines}
    
    # Track operator usage over time
    operators_in_use = defaultdict(int)  # time_point -> number of operators
    
    # Group jobs by family and sort by process number
    job_families = defaultdict(list)
    unassigned_jobs = []
    start_date_jobs = []
    
    # First pass: Group jobs by family and identify start date jobs
    for job in jobs:
        if not job.get('UNIQUE_JOB_ID'):
            continue
            
        # Handle START_DATE jobs first
        if job.get('START_DATE_EPOCH') is not None and job.get('START_DATE_EPOCH') > current_time:
            start_date_jobs.append(job)
            continue
            
        family = extract_job_family(job['UNIQUE_JOB_ID'])
        if family:
            process_num = extract_process_number(job['UNIQUE_JOB_ID'])
            if process_num != 999:
                job_families[family].append((process_num, job))
            else:
                unassigned_jobs.append(job)
        else:
            unassigned_jobs.append(job)
    
    # Sort jobs within each family by process number
    for family in job_families:
        job_families[family].sort(key=lambda x: int(x[0]))
        logger.info(f"Job family {family} sequence: {[j[1]['UNIQUE_JOB_ID'] for j in job_families[family]]}")
    
    # Create schedule dictionary and tracking sets
    schedule = {machine: [] for machine in machines}
    scheduled_jobs = set()
    unscheduled_jobs = []
    
    def can_schedule_job(job, machine_id, start_time_epoch):
        """Check if a job can be scheduled at the given time"""
        if not job.get('processing_time'):
            if 'HOURS_NEED' in job and job['HOURS_NEED'] is not None:
                job['processing_time'] = job['HOURS_NEED'] * 3600
            else:
                job['processing_time'] = 3600
        
        end_time_epoch = start_time_epoch + job['processing_time']
        
        # Check machine availability
        for scheduled_id, scheduled_start, scheduled_end, _ in schedule[machine_id]:
            if not (end_time_epoch <= scheduled_start or start_time_epoch >= scheduled_end):
                return False
                
        # Check operator constraints
        if max_operators > 0:
            start_rel = epoch_to_relative_hours(start_time_epoch)
            end_rel = epoch_to_relative_hours(end_time_epoch)
            
            for hour in range(int(start_rel), int(end_rel) + 1):
                if operators_in_use[hour] >= max_operators:
                    return False
        
        return True
    
    # First schedule START_DATE jobs since they have fixed start times
    for job in start_date_jobs:
        job_id = job['UNIQUE_JOB_ID']
        start_time_epoch = job.get('START_DATE_EPOCH')
        
        if not start_time_epoch:
            continue
            
        machine_id = find_best_machine(job, machines, machine_available_time)
        if not machine_id:
            logger.warning(f"No compatible machine found for START_DATE job {job_id}")
            unscheduled_jobs.append(job)
            continue
            
        if can_schedule_job(job, machine_id, start_time_epoch):
            end_time_epoch = start_time_epoch + job['processing_time']
            schedule[machine_id].append((job_id, start_time_epoch, end_time_epoch, job.get('PRIORITY', 0)))
            scheduled_jobs.add(job_id)
            machine_available_time[machine_id] = max(machine_available_time[machine_id], end_time_epoch)
            
            # Update operator usage
            if max_operators > 0:
                start_rel = epoch_to_relative_hours(start_time_epoch)
                end_rel = epoch_to_relative_hours(end_time_epoch)
                for hour in range(int(start_rel), int(end_rel) + 1):
                    operators_in_use[hour] += 1
            
            logger.info(f"Scheduled START_DATE job {job_id} on {machine_id}: {format_datetime_for_display(epoch_to_datetime(start_time_epoch))} to {format_datetime_for_display(epoch_to_datetime(end_time_epoch))}")
        else:
            unscheduled_jobs.append(job)
            logger.warning(f"Could not schedule START_DATE job {job_id} at required time {format_datetime_for_display(epoch_to_datetime(start_time_epoch))}")
    
    # Then schedule jobs by family in sequence
    for family, jobs_in_family in sorted(job_families.items()):
        last_end_time = current_time
        
        for process_num, job in jobs_in_family:
            job_id = job['UNIQUE_JOB_ID']
            if job_id in scheduled_jobs:
                continue
                
            # Find best machine for this job
            machine_id = find_best_machine(job, machines, machine_available_time)
            if not machine_id:
                logger.warning(f"No compatible machine found for job {job_id}")
                unscheduled_jobs.append(job)
                continue
            
            # Find earliest possible start time
            earliest_start = max(machine_available_time[machine_id], last_end_time)
            
            # Try to schedule the job
            while not can_schedule_job(job, machine_id, earliest_start):
                earliest_start += 3600  # Try next hour
                
                # Prevent infinite loop
                if earliest_start > current_time + (365 * 24 * 3600):  # 1 year limit
                    logger.error(f"Could not find valid time slot for job {job_id}")
                    unscheduled_jobs.append(job)
                    break
            
            if earliest_start <= current_time + (365 * 24 * 3600):
                end_time_epoch = earliest_start + job['processing_time']
                schedule[machine_id].append((job_id, earliest_start, end_time_epoch, job.get('PRIORITY', 0)))
                scheduled_jobs.add(job_id)
                last_end_time = end_time_epoch
                machine_available_time[machine_id] = end_time_epoch
                
                # Update operator usage
                if max_operators > 0:
                    start_rel = epoch_to_relative_hours(earliest_start)
                    end_rel = epoch_to_relative_hours(end_time_epoch)
                    for hour in range(int(start_rel), int(end_rel) + 1):
                        operators_in_use[hour] += 1
                
                logger.info(f"Scheduled job {job_id} on {machine_id}: {format_datetime_for_display(epoch_to_datetime(earliest_start))} to {format_datetime_for_display(epoch_to_datetime(end_time_epoch))}")
    
    # Finally schedule any remaining unassigned jobs
    for job in unassigned_jobs:
        job_id = job['UNIQUE_JOB_ID']
        if job_id in scheduled_jobs:
            continue
            
        # Find best machine for this job
        machine_id = find_best_machine(job, machines, machine_available_time)
        if not machine_id:
            logger.warning(f"No compatible machine found for job {job_id}")
            unscheduled_jobs.append(job)
            continue
            
        earliest_start = machine_available_time[machine_id]
        
        while not can_schedule_job(job, machine_id, earliest_start):
            earliest_start += 3600
            if earliest_start > current_time + (365 * 24 * 3600):
                unscheduled_jobs.append(job)
                break
        
        if earliest_start <= current_time + (365 * 24 * 3600):
            end_time_epoch = earliest_start + job['processing_time']
            schedule[machine_id].append((job_id, earliest_start, end_time_epoch, job.get('PRIORITY', 0)))
            scheduled_jobs.add(job_id)
            machine_available_time[machine_id] = end_time_epoch
            
            if max_operators > 0:
                start_rel = epoch_to_relative_hours(earliest_start)
                end_rel = epoch_to_relative_hours(end_time_epoch)
                for hour in range(int(start_rel), int(end_rel) + 1):
                    operators_in_use[hour] += 1
            
            logger.info(f"Scheduled unassigned job {job_id} on {machine_id}: {format_datetime_for_display(epoch_to_datetime(earliest_start))} to {format_datetime_for_display(epoch_to_datetime(end_time_epoch))}")
    
    # Log scheduling statistics
    elapsed = time.time() - start_time
    total_scheduled = len(scheduled_jobs)
    total_unscheduled = len(unscheduled_jobs)
    
    logger.info(f"Greedy scheduling complete in {elapsed:.2f} seconds")
    logger.info(f"Scheduled {total_scheduled} jobs, {total_unscheduled} jobs could not be scheduled")
    logger.info(f"Utilized {len([m for m in schedule if schedule[m]])} machines")
    
    # Find latest completion time
    latest_end = max(
        max((job[2] for job in jobs_list), default=current_time)
        for jobs_list in schedule.values()
        if jobs_list
    )
    latest_end_dt = epoch_to_datetime(latest_end)
    logger.info(f"All jobs will complete by {format_datetime_for_display(latest_end_dt)}")
    
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