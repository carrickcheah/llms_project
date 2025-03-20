# schedule_jobs.py
# Constraint Programming (CP-SAT) solver for production scheduling

from ortools.sat.python import cp_model
from datetime import datetime
import logging
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_process_number(process_code):
    """
    Extract the process sequence number (e.g., 1 from 'P01-06') or return 999 if not found.
    """
    print(f"Extracting sequence from: {process_code}")
    match = re.search(r'P(\d{2})-\d+', str(process_code).upper())  # Match exactly two digits after P
    if match:
        seq = int(match.group(1))
        print(f"Extracted sequence: {seq}")
        return seq
    print("No match found, returning 999")
    return 999  # Default if parsing fails

def extract_job_family(process_code):
    """
    Extract the job family (e.g., 'CP08-231B') from the process code.
    """
    process_code = str(process_code).upper()
    match = re.search(r'(.*?)-P\d+', process_code)
    if match:
        family = match.group(1)
        logger.debug(f"Extracted family {family} from {process_code}")
        return family
    parts = process_code.split("-P")
    if len(parts) >= 2:
        family = parts[0]
        logger.debug(f"Extracted family {family} from {process_code} (using split)")
        return family
    logger.warning(f"Could not extract family from {process_code}, using full code")
    return process_code

def schedule_jobs(jobs, machines, setup_times=None, enforce_sequence=True, time_limit_seconds=300):
    """
    Advanced scheduling function using CP-SAT solver with priority-based optimization
    and process sequence dependencies.

    Args:
        jobs (list): List of job dictionaries (e.g., {'PROCESS_CODE': '...', 'MACHINE_ID': '...', 'processing_time': ..., 'LCD_DATE_EPOCH': ..., 'PRIORITY': ...})
        machines (list): List of machine IDs
        setup_times (dict): Dictionary mapping (process_code1, process_code2) to setup duration (default: None)
        enforce_sequence (bool): Whether to enforce process sequence dependencies
        time_limit_seconds (int): Maximum time limit for the solver in seconds

    Returns:
        dict: Schedule as {machine: [(process_code, start, end, priority), ...]}
    """
    start_time = time.time()
    current_time = int(datetime.now().timestamp())

    # Determine horizon (end of scheduling period)
    horizon_end = current_time + max(job.get('LCD_DATE_EPOCH', current_time + 2592000) for job in jobs) + max(job['processing_time'] for job in jobs)
    horizon = horizon_end - current_time

    # Create CP-SAT model
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.log_search_progress = True
    solver.parameters.num_search_workers = 16

    # Check for jobs with START_DATE constraints
    start_date_jobs = [job for job in jobs if job.get('START_DATE_EPOCH', current_time) > current_time]
    if start_date_jobs:
        logger.info(f"Found {len(start_date_jobs)} jobs with START_DATE constraints for CP-SAT solver:")
        for job in start_date_jobs:
            start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
            logger.info(f"  Job {job['PROCESS_CODE']} on {job['MACHINE_ID']}: Must start on or after {start_date}")

    # Variables for each job
    start_vars = {}
    end_vars = {}
    intervals = {}
    for job in jobs:
        job_id = job['PROCESS_CODE']
        machine_id = job['MACHINE_ID']
        duration = job['processing_time']
        due_time = job.get('LCD_DATE_EPOCH', horizon_end - duration)
        priority = job['PRIORITY']

        # Get user-defined start date if exists
        user_start_time = job.get('START_DATE_EPOCH', current_time)

        # Validate duration
        if not isinstance(duration, (int, float)) or duration <= 0:
            logger.error(f"Invalid duration for job {job_id}: {duration}")
            raise ValueError(f"Invalid duration for job {job_id}")

        # Define start and end variables with proper domains
        # Use max of current_time and user_start_time for earliest possible start
        earliest_start = max(current_time, user_start_time)
        start_var = model.NewIntVar(earliest_start, horizon_end, f'start_{machine_id}_{job_id}')
        end_var = model.NewIntVar(earliest_start, horizon_end + duration, f'end_{machine_id}_{job_id}')
        interval_var = model.NewIntervalVar(start_var, duration, end_var, f'interval_{machine_id}_{job_id}')

        start_vars[(job_id, machine_id)] = start_var
        end_vars[(job_id, machine_id)] = end_var
        intervals[(job_id, machine_id)] = interval_var

        # Constraint: end = start + duration
        model.Add(end_var == start_var + duration)
        
        # If there's a user-defined start date, add a constraint for it
        if user_start_time > current_time:
            model.Add(start_var >= user_start_time)
            logger.info(f"Added START_DATE constraint for {job_id}: must start on or after {datetime.fromtimestamp(user_start_time).strftime('%Y-%m-%d %H:%M')}")

    # Add machine no-overlap constraints
    for machine in machines:
        machine_intervals = [intervals[(job['PROCESS_CODE'], job['MACHINE_ID'])] for job in jobs if job['MACHINE_ID'] == machine]
        if machine_intervals:
            model.AddNoOverlap(machine_intervals)

    # Add sequence constraints if enforced
    if enforce_sequence:
        logger.info("Enforcing process sequence dependencies (P01->P02->P03)...")
        family_jobs = {}
        for job in jobs:
            family = extract_job_family(job['PROCESS_CODE'])
            if family not in family_jobs:
                family_jobs[family] = []
            family_jobs[family].append(job)

        added_constraints = 0
        for family, family_jobs_list in family_jobs.items():
            sorted_jobs = sorted(family_jobs_list, key=lambda x: extract_process_number(x['PROCESS_CODE']))
            for i in range(len(sorted_jobs) - 1):
                job1 = sorted_jobs[i]
                job2 = sorted_jobs[i + 1]
                model.Add(end_vars[(job1['PROCESS_CODE'], job1['MACHINE_ID'])] <= 
                         start_vars[(job2['PROCESS_CODE'], job2['MACHINE_ID'])])
                added_constraints += 1
                logger.info(f"Added sequence constraint: {job1['PROCESS_CODE']} (seq {extract_process_number(job1['PROCESS_CODE'])}) "
                           f"must finish before {job2['PROCESS_CODE']} (seq {extract_process_number(job2['PROCESS_CODE'])}) starts")
        logger.info(f"Added {added_constraints} explicit sequence constraints")

    # Objective: Minimize makespan and delays
    makespan = model.NewIntVar(current_time, horizon_end, 'makespan')
    for machine in machines:
        machine_ends = [end_vars[(job['PROCESS_CODE'], job['MACHINE_ID'])] for job in jobs if job['MACHINE_ID'] == machine]
        if machine_ends:
            model.AddMaxEquality(makespan, machine_ends)

    # Add delay minimization (weighted by priority)
    objective_terms = [makespan]  # Start with makespan
    for job in jobs:
        job_id = job['PROCESS_CODE']
        machine_id = job['MACHINE_ID']
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
        logger.error("Problem is infeasible")
        return {}
    else:
        logger.error(f"No solution found. Status: {solver.StatusName(status)}")
        return {}

    # Build the schedule
    schedule = {}
    total_jobs = 0
    for job in jobs:
        job_id = job['PROCESS_CODE']
        machine_id = job['MACHINE_ID']
        start = solver.Value(start_vars[(job_id, machine_id)])
        end = solver.Value(end_vars[(job_id, machine_id)])
        priority = job['PRIORITY']
        duration_hours = (end - start) / 3600
        total_jobs += 1

        if machine_id not in schedule:
            schedule[machine_id] = []
        schedule[machine_id].append((job_id, start, end, priority))

    # Sort jobs by start time for each machine
    for machine in schedule:
        schedule[machine].sort(key=lambda x: x[1])

    # Log statistics
    logger.info(f"Scheduled {sum(len(jobs) for jobs in schedule.values())} jobs on {len(schedule)} machines")
    logger.info(f"Total jobs scheduled: {total_jobs}")

    return schedule

if __name__ == "__main__":
    # Example usage for testing
    jobs = [
        {'PROCESS_CODE': 'CP08-231B-P01-06', 'MACHINE_ID': 'WS01', 'processing_time': 20000, 'LCD_DATE_EPOCH': 1744848000, 'PRIORITY': 2},
        {'PROCESS_CODE': 'CP08-231B-P02-06', 'MACHINE_ID': 'PP23-060T', 'processing_time': 43636, 'LCD_DATE_EPOCH': 1744848000, 'PRIORITY': 2},
        {'PROCESS_CODE': 'CP08-231B-P03-06', 'MACHINE_ID': 'JIG-HAND BEND', 'processing_time': 60000, 'LCD_DATE_EPOCH': 1744848000, 'PRIORITY': 2},
    ]
    machines = ['WS01', 'PP23-060T', 'JIG-HAND BEND']
    schedule = schedule_jobs(jobs, machines, enforce_sequence=True)
    print(schedule)