# schedule_jobs.py | dont edit this line
# Constraint Programming (CP-SAT) solver for production scheduling

from ortools.sat.python import cp_model
from datetime import datetime
import logging
import time
import re
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_process_number(process_code):
    """
    Extract the process sequence number (e.g., 1 from 'P01-06') or return 999 if not found.
    """
    match = re.search(r'P(\d{2})-\d+', str(process_code).upper())  # Match exactly two digits after P
    if match:
        seq = int(match.group(1))
        return seq
    return 999  # Default if parsing fails

def extract_job_family(process_code, job_id=None):
    """
    Extract the job family (e.g., 'CP08-231B') from the process code.
    If job_id is provided, it will be included in the family to distinguish between
    different jobs that share the same process code pattern.
    """
    process_code = str(process_code).upper()
    match = re.search(r'(.*?)-P\d+', process_code)
    if match:
        family = match.group(1)
        logger.debug(f"Extracted family {family} from {process_code}")
        # If a job_id is provided, include it in the family
        if job_id:
            return f"{family}_{job_id}"
        return family
    parts = process_code.split("-P")
    if len(parts) >= 2:
        family = parts[0]
        logger.debug(f"Extracted family {family} from {process_code} (using split)")
        # If a job_id is provided, include it in the family
        if job_id:
            return f"{family}_{job_id}"
        return family
    logger.warning(f"Could not extract family from {process_code}, using full code")
    # If a job_id is provided, include it in the family
    if job_id:
        return f"{process_code}_{job_id}"
    return process_code

def schedule_jobs(jobs, machines, setup_times=None, enforce_sequence=True, time_limit_seconds=300):
    """
    Advanced scheduling function using CP-SAT solver with priority-based optimization
    and process sequence dependencies.

    Args:
        jobs (list): List of job dictionaries
        machines (list): List of machine IDs
        setup_times (dict): Dictionary mapping (process_code1, process_code2) to setup duration
        enforce_sequence (bool): Whether to enforce process sequence dependencies
        time_limit_seconds (int): Maximum time limit for the solver in seconds

    Returns:
        dict: Schedule as {machine: [(process_code, start, end, priority), ...]}
    """
    start_time = time.time()
    current_time = int(datetime.now().timestamp())

    # Determine horizon (end of scheduling period)
    horizon_end = int(current_time + max(job.get('LCD_DATE_EPOCH', current_time + 2592000) for job in jobs) + max(job['processing_time'] for job in jobs))
    horizon = horizon_end - current_time

    # Create CP-SAT model
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.log_search_progress = True
    solver.parameters.num_search_workers = 16
    
    # Use a more relaxed linearization level to improve solver flexibility
    solver.parameters.linearization_level = 2

    # Check for jobs with START_DATE constraints (both formats)
    start_date_jobs = [job for job in jobs if 
                      (job.get('START_DATE_EPOCH', current_time) > current_time) or 
                      (job.get('START_DATE _EPOCH', current_time) > current_time)]
    if start_date_jobs:
        logger.info(f"Found {len(start_date_jobs)} jobs with START_DATE constraints for CP-SAT solver:")
        for job in start_date_jobs:
            # Get the START_DATE_EPOCH value (from either format of field name)
            start_date_epoch = job.get('START_DATE_EPOCH')
            if start_date_epoch is None:
                start_date_epoch = job.get('START_DATE _EPOCH')
            start_date = datetime.fromtimestamp(start_date_epoch).strftime('%Y-%m-%d %H:%M')
            # Get resource location (try both old and new column names)
            resource_location = job.get('RSC_LOCATION', job.get('MACHINE_ID', 'Unknown'))
            logger.info(f"  Job {job['PROCESS_CODE']} on {resource_location}: MUST start EXACTLY at {start_date}")
        logger.info("START_DATE constraints will be strictly enforced in the model")

    # Step 1: Organize jobs by families and process numbers, including JOB ID to differentiate
    family_jobs = {}
    for job in jobs:
        # Include JOB ID in family to differentiate jobs with same process code pattern
        job_id = job.get('JOB', '')
        family = extract_job_family(job['PROCESS_CODE'], job_id)
        if family not in family_jobs:
            family_jobs[family] = []
        family_jobs[family].append(job)

    # Sort processes within each family by sequence number
    for family in family_jobs:
        family_jobs[family].sort(key=lambda x: extract_process_number(x['PROCESS_CODE']))

    # Step 2: Find families with START_DATE constraints and calculate time shifts
    family_constraints = {}
    for family, family_job_list in family_jobs.items():
        for job in family_job_list:
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] > current_time:
                # Record the family as having a time constraint
                if family not in family_constraints:
                    family_constraints[family] = []
                family_constraints[family].append(job)

    # Log family constraints
    for family, constrained_jobs in family_constraints.items():
        logger.info(f"Family {family} has {len(constrained_jobs)} jobs with START_DATE constraints")
        for job in constrained_jobs:
            start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
            logger.info(f"  Job {job['PROCESS_CODE']} must start at {start_date}")

    # Variables for each job
    start_vars = {}
    end_vars = {}
    intervals = {}
    for job in jobs:
        job_id = job['PROCESS_CODE']
        # Get resource location (try both old and new column names)
        machine_id = job.get('RSC_LOCATION', job.get('MACHINE_ID'))
        duration = job['processing_time']
        due_time = job.get('LCD_DATE_EPOCH', horizon_end - duration)
        priority = job['PRIORITY']

        # Get user-defined start date if exists (handle both column formats)
        has_start_date = (('START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None) or 
                         ('START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None))
        user_start_time = job.get('START_DATE_EPOCH', job.get('START_DATE _EPOCH', current_time))

        # Validate duration
        if not isinstance(duration, (int, float)) or duration <= 0:
            logger.error(f"Invalid duration for job {job_id}: {duration}")
            raise ValueError(f"Invalid duration for job {job_id}")

        # Define start and end variables with proper domains
        # MODIFIED: For future START_DATE jobs, set exact start time with narrow domain
        if has_start_date and user_start_time > current_time:
            # Fixed start time - must start EXACTLY at START_DATE
            # Ensure it's an integer for the CP-SAT solver
            int_start_time = int(user_start_time)
            start_var = model.NewIntVar(int_start_time, int_start_time, f'start_{machine_id}_{job_id}')
            logger.info(f"Job {job_id} must start EXACTLY at {datetime.fromtimestamp(int_start_time).strftime('%Y-%m-%d %H:%M')}")
            # Store START_TIME equal to START_DATE for visualization
            job['START_TIME'] = int_start_time
        else:
            # Regular job with flexible start time
            earliest_start = int(current_time)  # All other jobs start from current time
            start_var = model.NewIntVar(earliest_start, horizon_end, f'start_{machine_id}_{job_id}')
            
        # Ensure end_time domain is integer
        end_time = int(horizon_end + duration)
        end_var = model.NewIntVar(current_time, end_time, f'end_{machine_id}_{job_id}')
        
        # Ensure duration is integer for interval var
        int_duration = int(duration)
        interval_var = model.NewIntervalVar(start_var, int_duration, end_var, f'interval_{machine_id}_{job_id}')

        start_vars[(job_id, machine_id)] = start_var
        end_vars[(job_id, machine_id)] = end_var
        intervals[(job_id, machine_id)] = interval_var

        # Constraint: end = start + duration (using integer duration)
        model.Add(end_var == start_var + int_duration)

    # Add machine no-overlap constraints with duplicate job detection
    for machine in machines:
        # Create a set of unique job signatures to prevent duplicates
        unique_job_signatures = set()
        machine_intervals = []
        
        for job in jobs:
            if job.get('RSC_LOCATION') == machine or job.get('MACHINE_ID') == machine:
                job_id = job['PROCESS_CODE']
                interval_key = (job_id, job.get('RSC_LOCATION', job.get('MACHINE_ID')))
                
                # Create a unique signature using JOB, PROCESS_CODE, RSC_LOCATION, and RSC_CODE
                job_signature = (
                    job.get('JOB', ''),
                    job['PROCESS_CODE'],
                    job.get('RSC_LOCATION', job.get('MACHINE_ID', '')),
                    job.get('RSC_CODE', '')
                )
                
                # Only add the interval if this exact job hasn't been seen before
                if job_signature not in unique_job_signatures:
                    unique_job_signatures.add(job_signature)
                    machine_intervals.append(intervals[interval_key])
                else:
                    logger.warning(f"Skipping duplicate job {job_id} on machine {machine} with signature {job_signature} to prevent constraint conflicts")
        
        if machine_intervals:
            model.AddNoOverlap(machine_intervals)

    # Add sequence constraints if enforced
    if enforce_sequence:
        logger.info("Enforcing process sequence dependencies (P01->P02->P03)...")
        added_constraints = 0
        for family, family_job_list in family_jobs.items():
            sorted_jobs = sorted(family_job_list, key=lambda x: extract_process_number(x['PROCESS_CODE']))
            for i in range(len(sorted_jobs) - 1):
                job1 = sorted_jobs[i]
                job2 = sorted_jobs[i + 1]
                machine_id1 = job1.get('RSC_LOCATION', job1.get('MACHINE_ID'))
                machine_id2 = job2.get('RSC_LOCATION', job2.get('MACHINE_ID'))
                
                # Check for START_DATE constraints that might conflict with sequence
                job1_start_time = job1.get('START_DATE_EPOCH', 0)
                job2_start_time = job2.get('START_DATE_EPOCH', 0)
                
                # If both jobs have fixed start times and they'd create an infeasibility
                if job1_start_time > 0 and job2_start_time > 0 and job1_start_time >= job2_start_time:
                    logger.warning(f"Potential sequence conflict: {job1['PROCESS_CODE']} (starts {datetime.fromtimestamp(job1_start_time).strftime('%Y-%m-%d %H:%M')}) " 
                                  f"should finish before {job2['PROCESS_CODE']} (starts {datetime.fromtimestamp(job2_start_time).strftime('%Y-%m-%d %H:%M')})")
                    # Skip this constraint to prevent infeasibility
                    logger.warning(f"Skipping sequence constraint to avoid infeasibility")
                    continue
                
                # Verify this isn't a self-dependency (same process code)
                if job1['PROCESS_CODE'] == job2['PROCESS_CODE']:
                    logger.warning(f"Skipping invalid self-dependency for {job1['PROCESS_CODE']}")
                    continue
                
                # Get process numbers for verification
                proc1 = extract_process_number(job1['PROCESS_CODE'])
                proc2 = extract_process_number(job2['PROCESS_CODE'])
                
                # Double-check process numbers are sequential
                if proc2 != proc1 + 1:
                    logger.warning(f"Process numbers not sequential: {job1['PROCESS_CODE']} ({proc1}) → {job2['PROCESS_CODE']} ({proc2})")
                
                # For logging, create signatures to show the full context
                job1_signature = (
                    job1.get('JOB', ''),
                    job1['PROCESS_CODE'],
                    job1.get('RSC_LOCATION', job1.get('MACHINE_ID', '')),
                    job1.get('RSC_CODE', '')
                )
                
                job2_signature = (
                    job2.get('JOB', ''),
                    job2['PROCESS_CODE'],
                    job2.get('RSC_LOCATION', job2.get('MACHINE_ID', '')),
                    job2.get('RSC_CODE', '')
                )
                
                # Add the constraint normally
                model.Add(end_vars[(job1['PROCESS_CODE'], machine_id1)] <= 
                         start_vars[(job2['PROCESS_CODE'], machine_id2)])
                added_constraints += 1
                logger.info(f"Added sequence constraint: {job1['PROCESS_CODE']} must finish before {job2['PROCESS_CODE']} starts")
        logger.info(f"Added {added_constraints} explicit sequence constraints")

    # Objective: Minimize makespan and delays
    makespan = model.NewIntVar(current_time, horizon_end, 'makespan')
    for machine in machines:
        machine_ends = [end_vars[(job['PROCESS_CODE'], job.get('RSC_LOCATION', job.get('MACHINE_ID')))] 
                      for job in jobs 
                      if job.get('RSC_LOCATION') == machine or job.get('MACHINE_ID') == machine]
        if machine_ends:
            model.AddMaxEquality(makespan, machine_ends)

    # Add delay minimization (weighted by priority)
    objective_terms = [makespan]  # Start with makespan
    for job in jobs:
        job_id = job['PROCESS_CODE']
        machine_id = job.get('RSC_LOCATION', job.get('MACHINE_ID'))
        due_time = job.get('LCD_DATE_EPOCH', 0)
        priority = job['PRIORITY']
        if due_time > 0:
            delay = model.NewIntVar(0, horizon_end - due_time, f'delay_{machine_id}_{job_id}')
            model.AddMaxEquality(delay, [0, end_vars[(job_id, machine_id)] - due_time])
            priority_weight = 6 - priority  # Higher priority (1) gets higher weight (5)
            objective_terms.append(delay * priority_weight * 10)

    if objective_terms:
        model.Minimize(sum(objective_terms))

    # Solve the model
    logger.info(f"CP-SAT Model created with {len(jobs)} jobs")
    logger.info(f"Objective components: makespan, {len(jobs)} priority-weighted delays")
    logger.info(f"Starting CP-SAT solver with {time_limit_seconds} seconds time limit")

    status = solver.Solve(model)
    solve_time = time.time() - start_time

    if status == cp_model.OPTIMAL:
        logger.info(f"Optimal solution found in {solve_time:.2f} seconds")
    elif status == cp_model.FEASIBLE:
        logger.info(f"Feasible solution found in {solve_time:.2f} seconds")
    elif status == cp_model.INFEASIBLE:
        logger.error("Problem is infeasible with CP-SAT solver")
        logger.warning("Falling back to greedy scheduler due to constraint conflicts")
        
        # Import here to avoid circular imports
        from greedy import greedy_schedule
        
        # Try with greedy scheduler which is more flexible with constraints
        return greedy_schedule(jobs, machines, setup_times)
    else:
        logger.error(f"No solution found. Status: {solver.StatusName(status)}")
        return {}

    # Build the schedule
    schedule = {}
    total_jobs = 0
    for job in jobs:
        job_id = job['PROCESS_CODE']
        machine_id = job.get('RSC_LOCATION', job.get('MACHINE_ID'))
        start = solver.Value(start_vars[(job_id, machine_id)])
        end = solver.Value(end_vars[(job_id, machine_id)])
        priority = job['PRIORITY']
        total_jobs += 1

        # For jobs with START_DATE, verify that the schedule honors the constraint
        start_date_epoch = job.get('START_DATE_EPOCH', job.get('START_DATE _EPOCH'))
        if start_date_epoch is not None and start_date_epoch > current_time:
            if start != start_date_epoch:
                logger.warning(f"⚠️ Job {job_id} scheduled at {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')} " 
                             f"instead of requested START_DATE={datetime.fromtimestamp(start_date_epoch).strftime('%Y-%m-%d %H:%M')}")
            else:
                logger.info(f"✅ Job {job_id} scheduled exactly at requested START_DATE")
        
        # Store actual start and end times for all jobs
        job['START_TIME'] = start
        job['END_TIME'] = end
        
        if machine_id not in schedule:
            schedule[machine_id] = []
        schedule[machine_id].append((job_id, start, end, priority))

    # Sort jobs by start time for each machine
    for machine in schedule:
        schedule[machine].sort(key=lambda x: x[1])

    # Log statistics
    logger.info(f"Scheduled {sum(len(jobs) for jobs in schedule.values())} jobs on {len(schedule)} machines")
    logger.info(f"Total jobs scheduled: {total_jobs}")

    # Step 3: Calculate family time shifts based on scheduled vs. requested times
    family_time_shifts = {}
    scheduled_times = {}
    
    # First collect all scheduled times
    for machine, tasks in schedule.items():
        for job_id, start, end, _ in tasks:
            scheduled_times[job_id] = (start, end)
    
    # Calculate time shifts needed for each family
    for family, constrained_jobs in family_constraints.items():
        time_shifts = []
        for job in constrained_jobs:
            job_id = job['PROCESS_CODE']
            if job_id in scheduled_times:
                scheduled_start = scheduled_times[job_id][0]
                requested_start = job['START_DATE_EPOCH']
                time_shift = scheduled_start - requested_start
                time_shifts.append(time_shift)
        
        if time_shifts:
            # Use the maximum time shift for consistent family adjustment
            family_time_shifts[family] = max(time_shifts, key=abs)
            logger.info(f"Family {family} needs time shift of {family_time_shifts[family]/3600:.1f} hours")
    
    # Step 4: Store time shifts in job dictionaries for visualization
    for family, time_shift in family_time_shifts.items():
        if abs(time_shift) < 60:  # Skip shifts less than a minute
            continue
            
        for job in family_jobs[family]:
            job['family_time_shift'] = time_shift
            logger.info(f"Added time shift of {time_shift/3600:.1f} hours to job {job['PROCESS_CODE']}")

    return schedule

if __name__ == "__main__":
    from ingest_data import load_jobs_planning_data
    
    # Load environment variables
    load_dotenv()
    file_path = os.getenv('file_path')
    
    if not file_path:
        logger.error("No file_path found in environment variables.")
        exit(1)
        
    try:
        # Load real data
        jobs, machines, setup_times = load_jobs_planning_data(file_path)
        logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines from {file_path}")
        
        # Create schedule using real data
        schedule = schedule_jobs(jobs, machines, setup_times, enforce_sequence=True)
        
        # Output statistics about the schedule
        if schedule:
            total_jobs = sum(len(jobs_list) for jobs_list in schedule.values())
            logger.info(f"Generated schedule with {total_jobs} tasks across {len(schedule)} machines")
            
            # Verify all jobs have START_TIME and END_TIME values
            for job in jobs:
                if 'START_TIME' not in job or 'END_TIME' not in job:
                    job_id = job['PROCESS_CODE']
                    logger.warning(f"Job {job_id} is missing START_TIME or END_TIME. Adding them now.")
                    for machine, tasks in schedule.items():
                        for task_id, start, end, _ in tasks:
                            if task_id == job_id:
                                job['START_TIME'] = start
                                job['END_TIME'] = end
                                break
            
            # Count jobs with START_DATE that were scheduled exactly as requested
            start_date_jobs = [job for job in jobs if 
                              ('START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and job['START_DATE_EPOCH'] > int(datetime.now().timestamp())) or
                              ('START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None and job['START_DATE _EPOCH'] > int(datetime.now().timestamp()))]
            
            if start_date_jobs:
                logger.info(f"Summary of {len(start_date_jobs)} jobs with START_DATE constraints:")
                for job in start_date_jobs:
                    job_id = job['PROCESS_CODE']
                    resource_location = job.get('RSC_LOCATION', job.get('MACHINE_ID', 'Unknown'))
                    requested_time = job.get('START_DATE_EPOCH', job.get('START_DATE _EPOCH'))
                    start_time = job.get('START_TIME')
                    
                    if start_time == requested_time:
                        logger.info(f"  ✅ Job {job_id} scheduled exactly at START_DATE={datetime.fromtimestamp(requested_time).strftime('%Y-%m-%d %H:%M')}")
                    else:
                        logger.warning(f"  ❌ Job {job_id} scheduled at {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M')} " 
                                     f"instead of requested {datetime.fromtimestamp(requested_time).strftime('%Y-%m-%d %H:%M')}")
        else:
            logger.error("Failed to generate valid schedule")
    except Exception as e:
        logger.error(f"Error testing scheduler with real data: {e}")