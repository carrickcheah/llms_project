# sch_jobs.py | dont edit this line
# Constraint Programming (CP-SAT) solver for production scheduling

from ortools.sat.python import cp_model
from datetime import datetime
import logging
import time
import re
from dotenv import load_dotenv
import os
import math
from collections import defaultdict
from time_utils import (
    epoch_to_relative_hours,
    relative_hours_to_epoch,
    epoch_to_datetime,
    datetime_to_epoch,
    format_datetime_for_display
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

    match = re.search(r'P(\d{2})-\d+', str(process_code).upper())
    if match:
        seq = int(match.group(1))
        return seq
    return 999

def extract_job_family(unique_job_id, job_id=None):
    """
    Extract the job family (e.g., 'CP08-231B' from 'JOB_CP08-231B-P01-06') from the UNIQUE_JOB_ID.
    If job_id is provided, it will be included in the family to distinguish between
    different jobs that share the same process code pattern.
    UNIQUE_JOB_ID is in the format JOB_PROCESS_CODE.
    """
    try:
        process_code = unique_job_id.split('_', 1)[1]  # Split on first underscore to get PROCESS_CODE
    except IndexError:
        logger.warning(f"Could not extract PROCESS_CODE from UNIQUE_JOB_ID {unique_job_id}")
        if job_id:
            return f"{unique_job_id}_{job_id}"
        return unique_job_id

    process_code = str(process_code).upper()
    match = re.search(r'(.*?)-P\d+', process_code)
    if match:
        family = match.group(1)
        logger.debug(f"Extracted family {family} from {unique_job_id}")
        if job_id:
            return f"{family}_{job_id}"
        return family
    parts = process_code.split("-P")
    if len(parts) >= 2:
        family = parts[0]
        logger.debug(f"Extracted family {family} from {unique_job_id} (using split)")
        if job_id:
            return f"{family}_{job_id}"
        return family
    logger.warning(f"Could not extract family from {unique_job_id}, using full code")
    if job_id:
        return f"{process_code}_{job_id}"
    return process_code

def schedule_jobs(jobs, machines, setup_times=None, enforce_sequence=True, time_limit_seconds=300, max_operators=0):
    """
    Schedule jobs using the CP-SAT solver.
    
    Args:
        jobs: List of job dictionaries, each with a unique ID, machine ID, processing time, and LCD date.
        machines: List of available machines (IDs).
        setup_times: Dictionary of setup times between processes.
        enforce_sequence: Whether to enforce process sequence dependencies.
        time_limit_seconds: Time limit for the solver in seconds.
        max_operators: Maximum number of operators available (0 means unlimited).
    
    Returns:
        Dictionary with machine IDs as keys and lists of scheduled jobs as values.
        Each job is represented as (unique_job_id, start_time, end_time, priority).
    """
    if not jobs:
        logging.warning("No jobs to schedule.")
        return {}
    
    if not machines:
        logging.warning("No machines available for scheduling.")
        return {}
    
    logging.info(f"Scheduling {len(jobs)} jobs on {len(machines)} machines with CP-SAT solver")
    logging.info(f"Using max_operators={max_operators}")
    
    model = cp_model.CpModel()
    
    # Convert jobs to a more convenient format and calculate horizon
    # Use relative hours from reference time instead of absolute epoch
    current_time = datetime_to_epoch(datetime.now())
    horizon_start = current_time
    horizon_end = current_time
    
    for job in jobs:
        if 'LCD_DATE_EPOCH' in job and job['LCD_DATE_EPOCH']:
            horizon_end = max(horizon_end, job['LCD_DATE_EPOCH'])
        
        if 'processing_time' not in job or job['processing_time'] is None:
            if 'HOURS_NEED' in job and job['HOURS_NEED'] is not None:
                job['processing_time'] = job['HOURS_NEED'] * 3600
            else:
                logging.warning(f"Job {job.get('UNIQUE_JOB_ID', 'unknown')} has no processing time!")
                job['processing_time'] = 3600  # Default of 1 hour
    
    # Calculate horizon in relative hours to avoid very large numbers
    horizon_start_hours = epoch_to_relative_hours(horizon_start)
    horizon_end_hours = epoch_to_relative_hours(horizon_end)
    
    logging.info(f"Planning horizon: {horizon_start_hours:.1f} to {horizon_end_hours:.1f} relative hours")
    logging.info(f"Horizon span: {horizon_end_hours - horizon_start_hours:.1f} hours")
    
    # Limit the number of time points to prevent memory issues
    MAX_TIME_POINTS = 1000
    time_granularity = 1  # Default to hourly granularity
    
    # If the horizon is too large, adjust the granularity
    if horizon_end_hours - horizon_start_hours > MAX_TIME_POINTS:
        time_granularity = math.ceil((horizon_end_hours - horizon_start_hours) / MAX_TIME_POINTS)
        logging.warning(f"Horizon too large ({horizon_end_hours - horizon_start_hours:.1f} hours), "
                      f"adjusting granularity to {time_granularity}-hour intervals")
    
    # Create time points at the specified granularity
    time_points = list(range(
        int(horizon_start_hours), 
        int(horizon_end_hours) + 1, 
        time_granularity
    ))
    
    logging.info(f"Created {len(time_points)} time points at {time_granularity}-hour granularity")
    
    # Create variables for each job
    all_jobs = {}  # Dictionary mapping job ID to (start_var, end_var, interval_var)
    job_variables = []
    machine_to_intervals = defaultdict(list)  # Dictionary mapping machine ID to list of interval variables
    
    # Process sequence handling
    job_families = defaultdict(list)
    if enforce_sequence:
        for job in jobs:
            job_code = job.get('JOB')
            if job_code:
                job_families[job_code].append(job)
        
        for family_id, family_jobs in job_families.items():
            if len(family_jobs) > 1:
                # Sort by process code to establish sequence
                family_jobs.sort(key=lambda j: j.get('PROCESS_CODE', 0))
                logging.info(f"Job family {family_id} has {len(family_jobs)} processes in sequence")
    
    # Create variables for each job
    for job in jobs:
        unique_job_id = job.get('UNIQUE_JOB_ID')
        if not unique_job_id:
            logging.warning(f"Job has no UNIQUE_JOB_ID field: {job}")
            continue
        
        machine_id = job.get('RSC_CODE')
        if not machine_id:
            logging.warning(f"Job {unique_job_id} has no machine assignment (RSC_CODE)")
            continue
        
        if machine_id not in machines:
            logging.warning(f"Job {unique_job_id} assigned to unknown machine {machine_id}")
            continue
        
        proc_time = job.get('processing_time')
        if not proc_time or proc_time <= 0:
            logging.warning(f"Job {unique_job_id} has invalid processing time: {proc_time}")
            proc_time = 3600  # Default to 1 hour
        
        # Convert times to relative hours
        start_date_epoch = job.get('START_DATE_EPOCH')
        start_min = epoch_to_relative_hours(current_time)
        
        if start_date_epoch is not None:
            # If START_DATE_EPOCH is specified, job must start exactly at that time
            start_rel_hours = epoch_to_relative_hours(start_date_epoch)
            
            # Format for logging
            start_date_dt = epoch_to_datetime(start_date_epoch)
            start_date_str = format_datetime_for_display(start_date_dt) if start_date_dt else "INVALID DATE"
            
            logging.info(f"Job {unique_job_id} must start exactly at {start_date_str} (relative hour {start_rel_hours:.1f})")
            start_min = start_rel_hours
            start_max = start_rel_hours
        else:
            # Otherwise, job can start anytime after current time
            start_max = epoch_to_relative_hours(horizon_end)
        
        # Convert job durations from seconds to hours for variable creation
        proc_time_hours = proc_time / 3600
        
        # Create variables for the job
        start_var = model.NewIntVar(
            int(start_min), 
            int(start_max), 
            f'start_{unique_job_id}'
        )
        
        end_var = model.NewIntVar(
            int(start_min + proc_time_hours), 
            int(start_max + proc_time_hours), 
            f'end_{unique_job_id}'
        )
        
        interval_var = model.NewIntervalVar(
            start_var, 
            model.NewIntVar(int(proc_time_hours), int(proc_time_hours), f'duration_{unique_job_id}'), 
            end_var, 
            f'interval_{unique_job_id}'
        )
        
        # Store the variables for later use
        all_jobs[unique_job_id] = (start_var, end_var, interval_var)
        job_variables.append((unique_job_id, start_var, end_var, interval_var, machine_id, proc_time_hours))
        machine_to_intervals[machine_id].append(interval_var)
    
    # Enforce no overlap constraints for each machine
    for machine_id, intervals in machine_to_intervals.items():
        if intervals:
            model.AddNoOverlap(intervals)
            logging.info(f"Added no overlap constraint for machine {machine_id} ({len(intervals)} jobs)")
    
    # Enforce process sequence dependencies if required
    if enforce_sequence:
        for job_code, family_jobs in job_families.items():
            if len(family_jobs) <= 1:
                continue
                
            # Sort jobs by process code
            family_jobs.sort(key=lambda j: j.get('PROCESS_CODE', 0))
            
            # Add sequence constraints
            for i in range(len(family_jobs) - 1):
                job1 = family_jobs[i]
                job2 = family_jobs[i + 1]
                
                job1_id = job1.get('UNIQUE_JOB_ID')
                job2_id = job2.get('UNIQUE_JOB_ID')
                
                if job1_id in all_jobs and job2_id in all_jobs:
                    _, end_var1, _ = all_jobs[job1_id]
                    start_var2, _, _ = all_jobs[job2_id]
                    
                    model.Add(start_var2 >= end_var1)
                    logging.info(f"Added sequence constraint: {job1_id} must finish before {job2_id}")
    
    # Enforce operator constraints if specified
    if max_operators > 0:
        logging.info(f"Adding operator constraints: max_operators={max_operators}")
        
        # For each time point, track which jobs are running
        for t in time_points:
            # Create Boolean variables for whether each job is active at time t
            active_jobs = []
            
            for job_id, start_var, end_var, _, _, _ in job_variables:
                is_active = model.NewBoolVar(f'active_{job_id}_at_{t}')
                
                # Job is active if start_var <= t < end_var
                model.Add(start_var <= t).OnlyEnforceIf(is_active)
                model.Add(end_var > t).OnlyEnforceIf(is_active)
                
                # Set is_active to false if the above condition is not met
                model.Add(start_var > t).OnlyEnforceIf(is_active.Not())
                model.Add(end_var <= t).OnlyEnforceIf(is_active.Not())
                
                active_jobs.append(is_active)
            
            # Sum the number of active jobs at this time point
            if active_jobs:
                model.Add(sum(active_jobs) <= max_operators)
                logging.info(f"Added operator constraint at time {t}: max {max_operators} operators")
    
    # Minimize makespan (maximum completion time)
    makespan = model.NewIntVar(0, 10000, 'makespan')
    for job_id, job_tuple in all_jobs.items():
        _, end_var, _ = job_tuple
        model.Add(makespan >= end_var)
    
    # Set up the solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    
    # Solve the model to minimize makespan
    model.Minimize(makespan)
    status = solver.Solve(model)
    
    # Check if a solution was found
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        logging.info(f"CP-SAT solver found {'optimal' if status == cp_model.OPTIMAL else 'feasible'} solution")
        
        # Retrieve and return the schedule
        schedule = defaultdict(list)
        for job_id, (start_var, end_var, _), machine_id, proc_time_hours in [
            (job_id, all_jobs[job_id], machine_id, proc_time_hours) 
            for job_id, _, _, _, machine_id, proc_time_hours in job_variables
        ]:
            # Convert relative hours back to epoch times for output
            start_rel = solver.Value(start_var)
            end_rel = solver.Value(end_var)
            
            start_epoch = relative_hours_to_epoch(start_rel)
            end_epoch = relative_hours_to_epoch(end_rel)
            
            # For logging and debugging
            start_dt = epoch_to_datetime(start_epoch)
            end_dt = epoch_to_datetime(end_epoch)
            
            schedule[machine_id].append((job_id, start_epoch, end_epoch, 0))
            
            # Format for logging
            start_str = format_datetime_for_display(start_dt) if start_dt else "INVALID DATE"
            end_str = format_datetime_for_display(end_dt) if end_dt else "INVALID DATE"
            
            logging.info(f"Scheduled {job_id} on {machine_id}: {start_str} to {end_str} ({proc_time_hours:.1f} hours)")
            
            # Update the job object with the scheduled times
            for job in jobs:
                if job.get('UNIQUE_JOB_ID') == job_id:
                    job['START_TIME'] = start_epoch
                    job['END_TIME'] = end_epoch
                    break
        
        return dict(schedule)
    else:
        logging.error(f"CP-SAT solver failed to find a solution. Status: {solver.StatusName(status)}")
        return None

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
        
        schedule = schedule_jobs(jobs, machines, setup_times, enforce_sequence=True)
        
        if schedule:
            total_jobs = sum(len(jobs_list) for jobs_list in schedule.values())
            logger.info(f"Generated schedule with {total_jobs} tasks across {len(schedule)} machines")
            
            for job in jobs:
                if 'UNIQUE_JOB_ID' not in job:
                    continue
                if 'START_TIME' not in job or 'END_TIME' not in job:
                    unique_job_id = job['UNIQUE_JOB_ID']
                    logger.warning(f"Job {unique_job_id} is missing START_TIME or END_TIME. Adding them now.")
                    for machine, tasks in schedule.items():
                        for task_id, start, end, _ in tasks:
                            if task_id == unique_job_id:
                                job['START_TIME'] = start
                                job['END_TIME'] = end
                                break
            
            start_date_jobs = [job for job in jobs if 
                              ('START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and job['START_DATE_EPOCH'] > int(datetime.now().timestamp())) or
                              ('START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None and job['START_DATE _EPOCH'] > int(datetime.now().timestamp()))]
            
            if start_date_jobs:
                logger.info(f"Summary of {len(start_date_jobs)} jobs with START_DATE constraints:")
                for job in start_date_jobs:
                    unique_job_id = job['UNIQUE_JOB_ID']
                    resource_location = job.get('RSC_CODE', 'Unknown')
                    requested_time = job.get('START_DATE_EPOCH', job.get('START_DATE _EPOCH'))
                    start_time = job.get('START_TIME')
                    
                    if start_time == requested_time:
                        logger.info(f"  ✅ Job {unique_job_id} scheduled exactly at START_DATE={datetime.fromtimestamp(requested_time).strftime('%Y-%m-%d %H:%M')}")
                    else:
                        logger.warning(f"  ❌ Job {unique_job_id} scheduled at {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M')} " 
                                     f"instead of requested {datetime.fromtimestamp(requested_time).strftime('%Y-%m-%d %H:%M')}")
        else:
            logger.error("Failed to generate valid schedule")
    except Exception as e:
        logger.error(f"Error testing scheduler with real data: {e}")