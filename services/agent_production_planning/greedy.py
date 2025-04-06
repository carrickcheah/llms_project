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
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
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

    # Use a stricter regex to extract just the process number part (the digits after P)
    match = re.search(r'-P(\d{2})-', str(process_code).upper())  # Match the pattern -P##-
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
        return family
    parts = process_code.split("-P")
    if len(parts) >= 2:
        family = parts[0]
        return family
    logger.warning(f"Could not extract family from {unique_job_id}, using full code")
    return process_code

def extract_base_job_code(unique_job_id):
    """Extract the base job code from the UNIQUE_JOB_ID"""
    family = extract_job_family(unique_job_id)
    return family.split('-')[0]

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
    
    # Create schedule dictionary and tracking sets
    schedule = {machine: [] for machine in machines}
    scheduled_jobs = set()
    unscheduled_jobs = []
    
    # Track family end times to enforce dependencies properly
    family_end_times = defaultdict(lambda: current_time)
    process_end_times = {}  # Track (family, process_num) end times for sequence enforcement
    
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
            
            # Update family end time for dependency tracking
            family = extract_job_family(job_id)
            process_num = extract_process_number(job_id)
            family_end_times[family] = max(family_end_times[family], end_time_epoch)
            process_end_times[(family, process_num)] = end_time_epoch
            logger.info(f"Scheduled START_DATE job {job_id} on {machine_id}: {format_datetime_for_display(epoch_to_datetime(start_time_epoch))} to {format_datetime_for_display(epoch_to_datetime(end_time_epoch))}")
        else:
            unscheduled_jobs.append(job)
            logger.warning(f"Could not schedule START_DATE job {job_id} at required time {format_datetime_for_display(epoch_to_datetime(start_time_epoch))}")
    
    # Collect all jobs into a single list to schedule by priority
    all_jobs = []
    for family, jobs_in_family in job_families.items():
        all_jobs.extend([(family, process_num, job) for process_num, job in jobs_in_family])
    
    # Sort all jobs by priority first, then process number
    all_jobs.sort(key=lambda x: (x[2]['PRIORITY'], x[1]))
    logger.info(f"Processing {len(all_jobs)} jobs in priority order, filling vacant machines")
    
    # Process all jobs in priority order, but fill vacant machines when possible
    for family, process_num, job in all_jobs:
        job_id = job['UNIQUE_JOB_ID'] 
        if job_id in scheduled_jobs:
            continue
        
        # Check family dependencies
        min_start_time = current_time
        dependencies_met = True
        
        # Define special process rules
        special_processes = {
            5: {"allow_cross_family": True, "can_skip_dependencies": True}
            # Can be expanded with other process numbers that need special handling
        }
        
        # Enforce process sequence - ensure ALL previous processes in the family are completed
        if enforce_sequence and process_num > 1:
            # Removed debug print
            
            base_code = extract_base_job_code(job_id)
            job_prefix = job_id.split('_')[0]  # Extract the job prefix (e.g., JOST888888)
            
            # Calculate max process number dynamically instead of hard-coding
            max_process = max([p for _, p, _ in all_jobs]) if all_jobs else process_num
            
            # Special handling for processes with cross-family dependencies
            if process_num in special_processes and special_processes[process_num]["allow_cross_family"]:
                # Look for any related family that might contain dependencies for this job
                # First, check if there are any previous processes in the current family
                current_family_has_deps = False
                for p in range(1, process_num):
                    if (family, p) in process_end_times:
                        current_family_has_deps = True
                        break
                
                # If current family doesn't have deps, check for related families with the same job prefix
                if not current_family_has_deps:
                    related_families = set()
                    for f, p, j in all_jobs:
                        if j['UNIQUE_JOB_ID'].split('_')[0] == job_prefix and p < process_num:
                            related_families.add(f)
                    
                    # For each related family, check if it has previous processes
                    if related_families:
                        # Removed debug print
                        
                        # Try each related family to see if it has the needed dependencies
                        for alt_family in related_families:
                            all_prev_complete = True
                            latest_end_time = current_time
                            
                            # Removed debug print
                            
                            # Check if all previous processes in related family are complete
                            for prev_process in range(1, process_num):
                                prev_process_end = process_end_times.get((alt_family, prev_process))
                                if prev_process_end is None:
                                    # Check if process exists but isn't scheduled yet
                                    process_exists = False
                                    for rf, rp, rj in all_jobs:
                                        if rf == alt_family and rp == prev_process:
                                            process_exists = True
                                            break
                                    
                                    if process_exists:
                                        all_prev_complete = False
                                        # Removed debug print
                                        break
                                else:
                                    # Process exists and is scheduled, track its end time
                                    latest_end_time = max(latest_end_time, prev_process_end)
                            
                            if all_prev_complete:
                                # Removed debug print
                                dependencies_met = True
                                min_start_time = latest_end_time
                                # Skip regular dependency checks
                                continue
                        
                        # If we tried all related families but couldn't find proper dependencies, defer
                        if not dependencies_met:
                            # Removed debug print
                            unscheduled_jobs.append(job)
                            continue
                    
                    # If no related families with previous processes or no dependencies are met
                    # AND this process is allowed to skip dependencies, schedule it anyway
                    if not dependencies_met and not related_families and special_processes[process_num]["can_skip_dependencies"]:
                        # Removed debug print
                        dependencies_met = True
                        min_start_time = current_time
                        # Skip the regular dependency checks
                        continue
            
            # Standard dependency checking for regular cases
            # Check ALL previous processes, not just the immediate predecessor
            for prev_process in range(1, process_num):
                prev_process_end = process_end_times.get((family, prev_process))
                if prev_process_end is None:
                    # Previous process hasn't been scheduled yet - check if it exists at all
                    prev_exists = False
                    
                    # Look for the previous process only within the same base code
                    for f, p, j in all_jobs:
                        if extract_base_job_code(j['UNIQUE_JOB_ID']) == base_code and p == prev_process:
                            prev_exists = True
                            # Check if it has a START_DATE constraint
                            if j.get('START_DATE_EPOCH') is not None:
                                # If previous job has a START_DATE, we must wait until after it would finish
                                start_date = j.get('START_DATE_EPOCH')
                                process_time = j.get('processing_time', 3600)
                                estimated_end_time = start_date + process_time
                                # Removed debug print
                                min_start_time = max(min_start_time, estimated_end_time)
                            break
                    
                    if not prev_exists:
                        # Previous process doesn't exist at all
                        # Removed debug print
                        
                        # Check material status - if a previous process is missing due to material issues, don't schedule later processes
                        if "MATERIAL_ARRIVAL" in job_id or (process_num > 1 and process_num <= max_process):
                            # Removed debug print
                            dependencies_met = False
                            break
                        else:
                            # Only skip dependencies for special cases not related to material issues
                            # Removed debug print
                            dependencies_met = True
                            break
                    elif min_start_time <= current_time:
                        # Previous process exists but isn't scheduled yet
                        # Removed debug print
                        dependencies_met = False
                        break
                else:
                    # Previous process exists and has been scheduled, must wait for it
                    # Removed debug print
                    min_start_time = max(min_start_time, prev_process_end)
        
        # If dependencies aren't met, defer this job for later
        if not dependencies_met:
            unscheduled_jobs.append(job)
            continue
        
        # Find best machine for this job
        machine_id = find_best_machine(job, machines, machine_available_time)
        if not machine_id:
            logger.warning(f"No compatible machine found for job {job_id}")
            unscheduled_jobs.append(job)
            continue
        
        # Find earliest possible start time - honor both machine availability and dependencies
        # Look for earliest vacant slot on this machine
        machine_slots = schedule[machine_id]
        
        # Start with the earliest possible time considering dependencies
        earliest_start = min_start_time
        
        # Check if this job has a START_DATE constraint to honor
        if job.get('START_DATE_EPOCH') is not None and job.get('START_DATE_EPOCH') > earliest_start:
            earliest_start = job.get('START_DATE_EPOCH')
            logger.info(f"Job {job_id} has START_DATE constraint, adjusted start time to {format_datetime_for_display(epoch_to_datetime(earliest_start))}")
        
        # Try to find the earliest vacant slot on this machine
        if machine_slots:
            # Sort slots by start time
            machine_slots.sort(key=lambda x: x[1])
            
            # Check if we can schedule before the first job
            first_job_start = machine_slots[0][1]
            if earliest_start + job['processing_time'] <= first_job_start and can_schedule_job(job, machine_id, earliest_start):
                # We can schedule at the earliest start time
                pass
            else:
                # Try to find a gap between scheduled jobs
                found_slot = False
                for i in range(len(machine_slots) - 1):
                    current_end = machine_slots[i][2]
                    next_start = machine_slots[i + 1][1]
                    
                    # Ensure we respect the minimum start time
                    slot_start = max(current_end, earliest_start)
                    
                    # Check if job fits in this gap
                    if slot_start + job['processing_time'] <= next_start and can_schedule_job(job, machine_id, slot_start):
                        earliest_start = slot_start
                        found_slot = True
                        break
                
                # If no gap found, try after the last job
                if not found_slot:
                    last_job_end = machine_slots[-1][2]
                    earliest_start = max(last_job_end, earliest_start)
        
        # Try to schedule the job at the identified slot
        if can_schedule_job(job, machine_id, earliest_start):
            end_time_epoch = earliest_start + job['processing_time']
            schedule[machine_id].append((job_id, earliest_start, end_time_epoch, job.get('PRIORITY', 0)))
            scheduled_jobs.add(job_id)
            
            # Update process and family end times for dependency tracking
            family_end_times[family] = max(family_end_times[family], end_time_epoch)
            process_end_times[(family, process_num)] = end_time_epoch
            
            # Update operator usage
            if max_operators > 0:
                start_rel = epoch_to_relative_hours(earliest_start)
                end_rel = epoch_to_relative_hours(end_time_epoch)
                for hour in range(int(start_rel), int(end_rel) + 1):
                    operators_in_use[hour] += 1
            
            logger.info(f"Scheduled job {job_id} (priority {job.get('PRIORITY')}) on {machine_id}: {format_datetime_for_display(epoch_to_datetime(earliest_start))} to {format_datetime_for_display(epoch_to_datetime(end_time_epoch))}")
        else:
            # Couldn't find a valid slot - try increasing time until we find one
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
                
                # Update process and family end times for dependency tracking
                family_end_times[family] = max(family_end_times[family], end_time_epoch)
                process_end_times[(family, process_num)] = end_time_epoch
                
                # Update operator usage
                if max_operators > 0:
                    start_rel = epoch_to_relative_hours(earliest_start)
                    end_rel = epoch_to_relative_hours(end_time_epoch)
                    for hour in range(int(start_rel), int(end_rel) + 1):
                        operators_in_use[hour] += 1
                
                logger.info(f"Scheduled job {job_id} (priority {job.get('PRIORITY')}) on {machine_id}: {format_datetime_for_display(epoch_to_datetime(earliest_start))} to {format_datetime_for_display(epoch_to_datetime(end_time_epoch))}")
    
    # Finally schedule any remaining unassigned jobs
    unassigned_jobs.sort(key=lambda x: (x['PRIORITY'], x.get('LCD_DATE_EPOCH', float('inf'))))
    
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
            
        # Find earliest vacant slot on this machine
        machine_slots = schedule[machine_id]
        earliest_start = current_time
        
        # Similar vacant slot finding logic as above
        if machine_slots:
            machine_slots.sort(key=lambda x: x[1])
            
            # Check if we can schedule before the first job
            first_job_start = machine_slots[0][1]
            if earliest_start + job['processing_time'] <= first_job_start and can_schedule_job(job, machine_id, earliest_start):
                # We can schedule at the earliest start time
                pass
            else:
                # Try to find a gap between scheduled jobs
                found_slot = False
                for i in range(len(machine_slots) - 1):
                    current_end = machine_slots[i][2]
                    next_start = machine_slots[i + 1][1]
                    
                    # Check if job fits in this gap
                    slot_start = max(current_end, earliest_start)
                    if slot_start + job['processing_time'] <= next_start and can_schedule_job(job, machine_id, slot_start):
                        earliest_start = slot_start
                        found_slot = True
                        break
                
                # If no gap found, try after the last job
                if not found_slot:
                    last_job_end = machine_slots[-1][2]
                    earliest_start = max(last_job_end, earliest_start)
        
        # Try to schedule the job
        if can_schedule_job(job, machine_id, earliest_start):
            end_time_epoch = earliest_start + job['processing_time']
            schedule[machine_id].append((job_id, earliest_start, end_time_epoch, job.get('PRIORITY', 0)))
            scheduled_jobs.add(job_id)
            
            # Update operator usage
            if max_operators > 0:
                start_rel = epoch_to_relative_hours(earliest_start)
                end_rel = epoch_to_relative_hours(end_time_epoch)
                for hour in range(int(start_rel), int(end_rel) + 1):
                    operators_in_use[hour] += 1
            
            logger.info(f"Scheduled unassigned job {job_id} on {machine_id}: {format_datetime_for_display(epoch_to_datetime(earliest_start))} to {format_datetime_for_display(epoch_to_datetime(end_time_epoch))}")
        else:
            # If we couldn't find a valid slot, try increasing the time
            while not can_schedule_job(job, machine_id, earliest_start):
                earliest_start += 3600
                if earliest_start > current_time + (365 * 24 * 3600):
                    unscheduled_jobs.append(job)
                    break
            
            if earliest_start <= current_time + (365 * 24 * 3600):
                end_time_epoch = earliest_start + job['processing_time']
                schedule[machine_id].append((job_id, earliest_start, end_time_epoch, job.get('PRIORITY', 0)))
                scheduled_jobs.add(job_id)
                
                # Update operator usage
                if max_operators > 0:
                    start_rel = epoch_to_relative_hours(earliest_start)
                    end_rel = epoch_to_relative_hours(end_time_epoch)
                    for hour in range(int(start_rel), int(end_rel) + 1):
                        operators_in_use[hour] += 1
                
                logger.info(f"Scheduled unassigned job {job_id} on {machine_id}: {format_datetime_for_display(epoch_to_datetime(earliest_start))} to {format_datetime_for_display(epoch_to_datetime(end_time_epoch))}")
    
    # Sort the schedule on each machine by start time
    for machine_id in schedule:
        schedule[machine_id].sort(key=lambda x: x[1])
    
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
    
    print("Starting greedy scheduler...")
    print(f"Loading job data from: {file_path}")
        
    try:
        jobs, machines, setup_times = load_jobs_planning_data(file_path)
        print(f"Loaded {len(jobs)} jobs and {len(machines)} machines from {file_path}")
        schedule = greedy_schedule(jobs, machines, setup_times, enforce_sequence=True)
        print(f"Generated schedule with {sum(len(jobs_list) for jobs_list in schedule.values())} scheduled tasks")
        
        # Print the CP11 jobs for debugging
        for job in jobs:
            if 'CP11-111' in job.get('UNIQUE_JOB_ID', ''):
                start_date = job.get('START_DATE_EPOCH')
                if start_date:
                    print(f"Job {job['UNIQUE_JOB_ID']} has START_DATE: {epoch_to_datetime(start_date)}")
        
    except Exception as e:
        print(f"Error testing scheduler with real data: {e}")
        raise