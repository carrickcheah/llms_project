# sch_jobs.py | dont edit this line
# Constraint Programming (CP-SAT) solver for production scheduling

from ortools.sat.python import cp_model
import ortools
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

    match = re.search(r'P(\d{2})', str(process_code).upper())  # Match exactly two digits after P
    if match:
        seq = int(match.group(1))
        return seq
    return 999  # Default if parsing fails

def extract_job_family(unique_job_id, job_id=None):
    """
    Extract the job family (e.g., 'CP33-333' from 'JOST333333_CP33-333-P01-02') from the UNIQUE_JOB_ID.
    If job_id is provided, it will be included in the family to distinguish between
    different jobs that share the same process code pattern.
    UNIQUE_JOB_ID is in the format PREFIX_FAMILY-PROCESS.
    """
    try:
        # Split on first underscore to get the part after the prefix
        process_code = unique_job_id.split('_', 1)[1] if '_' in unique_job_id else unique_job_id
    except IndexError:
        logger.warning(f"Could not extract PROCESS_CODE from UNIQUE_JOB_ID {unique_job_id}")
        if job_id:
            return f"{unique_job_id}_{job_id}"
        return unique_job_id

    process_code = str(process_code).upper()
    # First try to match everything up to -P followed by digits
    match = re.search(r'(.*?)-P\d+', process_code)
    if match:
        family = match.group(1)
        logger.debug(f"Extracted family {family} from {unique_job_id}")
        if job_id:
            return f"{family}_{job_id}"
        return family
    
    # If that fails, try splitting on -P
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

def schedule_jobs(jobs, machines, setup_times=None, enforce_sequence=True, time_limit_seconds=300, max_operators=None):
    """
    Schedule a set of jobs on machines using CP-SAT solver.
    
    Args:
        jobs (list): List of job dictionaries
        machines (list): List of machine names
        setup_times (dict, optional): Dictionary of setup times
        enforce_sequence (bool, optional): Whether to enforce sequence constraints
        time_limit_seconds (int, optional): Time limit for solver in seconds
        max_operators (int, optional): Maximum number of operators allowed at any time
        
    Returns:
        dict: Dictionary of scheduled jobs with start and end times
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Using CP-SAT solver to schedule {len(jobs)} jobs on {len(machines)} machines")
    
    # Get reference time for converting relative to absolute times
    reference_time = int(datetime.now().timestamp())
    
    # Create the model
    model = cp_model.CpModel()
    
    # Define the horizon - use relative hours from now
    horizon = int(max([job['HOURS_NEED'] for job in jobs]) * 2)  # Convert to integer 
    
    # Create job variables and intervals
    all_tasks = {}
    operators_used = defaultdict(list)  # Operators needed at each time point
    
    # Create a list to store all start, end, and interval variables
    all_starts = []
    all_ends = []
    all_intervals = []
    
    # Dictionary to cache job intervals on machines
    jobs_on_machine = defaultdict(list)
    
    # Create a list to add sequence constraints softly
    sequence_penalties = []
    max_sequence_penalty = 1000000  # High penalty for violating sequence constraints
    
    # Separate dictionary to store sequence constraints for visualization purposes
    job_dependencies = defaultdict(list)
    
    # Track jobs with constraints for visualization
    start_date_processes = {}
    
    # Dictionary to track known due dates for tardiness calculations
    jobs_with_due_dates = {}
    
    # Initialize start time preferences dictionary
    start_time_preferences = {}
    
    # Dictionary to track job hours for sequence validation
    job_hours = {}
    
    # Process all jobs to handle constraints
    for job in jobs:
        job_id = job['UNIQUE_JOB_ID']
        job_hours[job_id] = job['HOURS_NEED']
        
        # Check if the job has a machine assignment
        if not job.get('RSC_CODE') or job['RSC_CODE'] not in machines:
            logger.warning(f"Job {job_id} has no machine assignment, skipping")
            continue
            
        machine = job['RSC_CODE']
        
        # Convert hours to integer for solver
        hours_need = int(math.ceil(job['HOURS_NEED']))  # Ensure hours is an integer
        
        # Check if the job has a due date
        if 'LCD_DATE_EPOCH' in job and job['LCD_DATE_EPOCH']:
            # Convert absolute due date to relative hours from reference time
            due_date_rel = epoch_to_relative_hours(job['LCD_DATE_EPOCH'])
            due_date_rel_int = int(due_date_rel)  # Convert to integer for CP-SAT
            jobs_with_due_dates[job_id] = due_date_rel_int  # Store due date for tardiness calculation
            logger.info(f"Added due date for job {job_id}: {due_date_rel_int} hours from reference")
        
        # Check for START_DATE constraint (special fixed starting time)
        if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH']:
            # Convert absolute start date to relative hours from reference time
            start_date_rel = epoch_to_relative_hours(job['START_DATE_EPOCH'])
            start_date_rel_int = int(start_date_rel)  # Convert to integer
            
            # Store as a preference rather than a hard constraint
            start_time_preferences[job_id] = start_date_rel_int
            logger.info(f"Added start time preference for job {job_id}: {start_date_rel_int} hours from reference")
            
            # Store for visualization
            start_date_processes[job_id] = job['START_DATE_EPOCH']
        
        # Create start and end variables
        start_var = model.NewIntVar(0, horizon, f'start_{job_id}')
        end_var = model.NewIntVar(0, horizon + hours_need, f'end_{job_id}')
        
        # Record all start and end variables
        all_starts.append(start_var)
        all_ends.append(end_var)
        
        # Create interval variable
        interval_var = model.NewIntervalVar(
            start_var, hours_need, end_var, f'interval_{job_id}'
        )
        
        # Record the interval
        all_intervals.append(interval_var)
        
        # Store task info
        all_tasks[job_id] = {
            'start': start_var,
            'end': end_var,
            'interval': interval_var,
            'machine': machine,
            'hours': hours_need,
            'job': job  # Store original job data
        }
        
        # Add to machine-specific list
        jobs_on_machine[machine].append(interval_var)
        
        # Track operators needed for this job
        if 'OPERATORS' in job and job['OPERATORS']:
            try:
                num_operators = int(job['OPERATORS'])
                
                # Create a flattened range for operator counting
                for t in range(horizon):
                    # Create literal for if job is active at time t
                    is_active = model.NewBoolVar(f'is_{job_id}_active_at_{t}')
                    
                    # Model start <= t < end
                    # We'll use implications instead of hard constraints
                    
                    # If start <= t then use a bool var
                    start_before_t = model.NewBoolVar(f'start_{job_id}_before_{t}')
                    model.Add(start_var <= t).OnlyEnforceIf(start_before_t)
                    model.Add(start_var > t).OnlyEnforceIf(start_before_t.Not())
                    
                    # If t < end then use another bool var  
                    t_before_end = model.NewBoolVar(f't_before_end_{job_id}_{t}')
                    model.Add(t < end_var).OnlyEnforceIf(t_before_end)
                    model.Add(t >= end_var).OnlyEnforceIf(t_before_end.Not())
                    
                    # is_active is true if both conditions are met
                    model.AddBoolAnd([start_before_t, t_before_end]).OnlyEnforceIf(is_active)
                    model.AddBoolOr([start_before_t.Not(), t_before_end.Not()]).OnlyEnforceIf(is_active.Not())
                    
                    operators_used[t].append((is_active, num_operators))
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid operator count for job {job_id}: {job['OPERATORS']}. Error: {e}")
    
    # Add sequence constraints if needed
    if enforce_sequence:
        # Group jobs by family to enforce process sequence
        job_families = defaultdict(list)
        
        for job in jobs:
            job_id = job['UNIQUE_JOB_ID']
            if job_id not in all_tasks:
                continue  # Skip jobs that weren't added
                
            family = extract_job_family(job_id)
            if family:
                process_num = extract_process_number(job_id)
                if process_num != 999:  # Only add if we got a valid process number
                    job_families[family].append((process_num, job_id))
        
        # Sort each family's jobs by process number
        for family, jobs_in_family in job_families.items():
            # Sort by process number, ensuring proper numerical ordering
            jobs_in_family.sort(key=lambda x: int(x[0]))  # Convert process_num to int for sorting
            
            logger.info(f"Job family {family} has {len(jobs_in_family)} processes in sequence: {[j[1] for j in jobs_in_family]}")
            
            # Add constraints between consecutive jobs
            for i in range(len(jobs_in_family) - 1):
                current_job_id = jobs_in_family[i][1]  # Get job_id from tuple
                next_job_id = jobs_in_family[i + 1][1]  # Get next job_id from tuple
                
                if current_job_id in all_tasks and next_job_id in all_tasks:
                    # Add hard constraint for sequence
                    model.Add(all_tasks[current_job_id]['end'] <= all_tasks[next_job_id]['start'])
                    
                    # Record the dependency for visualization
                    job_dependencies[current_job_id].append(next_job_id)
                    
                    logger.info(f"Added hard sequence constraint: {current_job_id} (P{jobs_in_family[i][0]:02d}) must finish before {next_job_id} (P{jobs_in_family[i+1][0]:02d})")

    # Create non-overlapping constraints for each machine
    for machine, intervals in jobs_on_machine.items():
        if intervals:  # Only add constraint if there are intervals
            model.AddNoOverlap(intervals)
    
    # Add operator constraints if max_operators is specified
    if max_operators is not None:
        for t, ops_list in operators_used.items():
            if ops_list:
                # Create expressions for operator count at time t
                op_vars = []
                op_coeffs = []
                
                for is_active, num_ops in ops_list:
                    op_vars.append(is_active)
                    op_coeffs.append(num_ops)
                
                # Use a bool var to indicate if constraint is satisfied
                constraint_var = model.NewBoolVar(f'op_constraint_{t}')
                model.Add(sum(op_vars[i] * op_coeffs[i] for i in range(len(op_vars))) <= max_operators).OnlyEnforceIf(constraint_var)
                
                # Create soft constraint with high penalty
                op_penalty = model.NewBoolVar(f'op_penalty_{t}')
                model.AddBoolOr([constraint_var]).OnlyEnforceIf(op_penalty.Not())
                model.AddBoolOr([constraint_var.Not()]).OnlyEnforceIf(op_penalty)
                
                # Add to sequence penalties list to be used in objective
                sequence_penalties.append(op_penalty)
                
                logger.info(f"Added operator constraint at time {t}: max {max_operators} operators")
    
    # Objective: Minimize makespan and tardiness
    # Create variables for gap minimization between dependent jobs
    total_gap = model.NewIntVar(0, horizon * len(job_dependencies), 'total_gap')
    gap_vars = []
    
    # Add gap between dependent jobs (from same family)
    for job_id, deps in job_dependencies.items():
        for dep_id in deps:
            gap_var = model.NewIntVar(0, horizon, f'gap_{job_id}_{dep_id}')
            model.Add(gap_var == all_tasks[dep_id]['start'] - all_tasks[job_id]['end'])
            gap_vars.append(gap_var)
    
    # Sum all gaps
    if gap_vars:
        model.Add(total_gap == sum(gap_vars))
        logger.info(f"Created objective to minimize {len(gap_vars)} sequence gaps")
    else:
        model.Add(total_gap == 0)
    
    # Create tardiness variables for jobs with due dates
    total_tardiness = model.NewIntVar(0, horizon * len(jobs_with_due_dates), 'total_tardiness')
    tardiness_vars = []
    
    for job_id, due_date_rel in jobs_with_due_dates.items():
        if job_id in all_tasks:
            # Create tardiness variable - max(0, end - due_date)
            tardiness = model.NewIntVar(0, horizon, f'tardiness_{job_id}')
            
            # We use a BoolVar to model the condition (end > due_date)
            is_late = model.NewBoolVar(f'is_late_{job_id}')
            model.Add(all_tasks[job_id]['end'] > due_date_rel).OnlyEnforceIf(is_late)
            model.Add(all_tasks[job_id]['end'] <= due_date_rel).OnlyEnforceIf(is_late.Not())
            
            # If late, tardiness = end - due_date, else 0
            model.Add(tardiness == all_tasks[job_id]['end'] - due_date_rel).OnlyEnforceIf(is_late)
            model.Add(tardiness == 0).OnlyEnforceIf(is_late.Not())
            
            # Add piecewise linear penalty for tardiness to prioritize reducing larger delays
            # Break into segments of increasing penalty with more aggressive scaling
            segments = [12, 24, 48, 96]  # Hours thresholds (reduced first segment)
            penalties = [2, 4, 8, 16, 32]  # Doubled penalties for each segment
            
            # Add critical lateness penalty for jobs over 96 hours late
            critical_lateness = model.NewBoolVar(f'critical_lateness_{job_id}')
            model.Add(tardiness > 96).OnlyEnforceIf(critical_lateness)
            model.Add(tardiness <= 96).OnlyEnforceIf(critical_lateness.Not())
            
            # Add very high penalty for critical lateness
            critical_penalty = model.NewIntVar(0, horizon, f'critical_penalty_{job_id}')
            model.Add(critical_penalty == tardiness * 64).OnlyEnforceIf(critical_lateness)  # 64x penalty for critical lateness
            model.Add(critical_penalty == 0).OnlyEnforceIf(critical_lateness.Not())
            
            segment_vars = []
            for i, (lower, upper) in enumerate(zip([0] + segments, segments + [horizon])):
                # Create variable for this segment
                segment = model.NewIntVar(0, upper - lower, f'segment_{job_id}_{i}')
                
                # Add constraint that segment can only be used if tardiness is in range
                use_segment = model.NewBoolVar(f'use_segment_{job_id}_{i}')
                model.Add(tardiness > lower).OnlyEnforceIf(use_segment)
                model.Add(tardiness <= upper).OnlyEnforceIf(use_segment)
                
                # Segment value is min(tardiness - lower, upper - lower)
                model.Add(segment <= tardiness - lower).OnlyEnforceIf(use_segment)
                model.Add(segment <= upper - lower).OnlyEnforceIf(use_segment)
                model.Add(segment == 0).OnlyEnforceIf(use_segment.Not())
                
                # Add weighted segment to tardiness vars
                segment_vars.append(segment * penalties[i])
            
            tardiness_vars.extend(segment_vars)
            
            # Add early start incentive
            early_start = model.NewIntVar(0, horizon, f'early_start_{job_id}')
            model.Add(early_start == due_date_rel - all_tasks[job_id]['start'])
            tardiness_vars.append(early_start)  # Small bonus for starting early
            
            logger.info(f"Added tardiness penalty for job {job_id} (due at relative hour {due_date_rel})")
    
    # Sum all tardiness
    if tardiness_vars:
        model.Add(total_tardiness == sum(tardiness_vars))
        logger.info(f"Created objective to minimize tardiness for {len(tardiness_vars)} jobs with due dates")
    else:
        model.Add(total_tardiness == 0)

    # Handle START_DATE constraints as hard constraints
    for job_id, preferred_start in start_time_preferences.items():
        if job_id in all_tasks:
            # Add hard constraint for start time
            model.Add(all_tasks[job_id]['start'] == preferred_start)
            logger.info(f"Added hard START_DATE constraint for job {job_id} (must start at: {preferred_start})")
    
    # Calculate makespan variable (maximum end time)
    makespan = model.NewIntVar(0, horizon * 2, 'makespan')
    model.AddMaxEquality(makespan, all_ends)
    
    # Objective function weights - prioritize sequence and start time constraints
    gap_weight = 1  # Minimal weight on gaps
    tardiness_weight = 1000  # Increased weight on tardiness (was 100)
    makespan_weight = 1  # Minimal weight on makespan
    
    # Define objective with appropriate weights - removed start_dev and sequence penalties since they're now hard constraints
    model.Minimize(
        total_gap * gap_weight +
        total_tardiness * tardiness_weight +
        makespan * makespan_weight
    )
    
    # Create solver and set time limit
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.log_search_progress = True
    solver.parameters.linearization_level = 2  # More aggressive linearization
    solver.parameters.num_search_workers = 8
    
    # Extra parameters to help with feasibility
    solver.parameters.optimize_with_core = True
    solver.parameters.mip_max_bound = 1e9
    
    # Print solver configuration
    logger.info(f"Starting CP-SAT solver v{ortools.__version__}")
    logger.info(f"Parameters: max_time_in_seconds: {time_limit_seconds} log_search_progress: true search_branching: HINT_SEARCH num_search_workers: 8")
    
    # Try to solve
    try:
        status = solver.Solve(model)
    except Exception as e:
        logger.error(f"Error solving CP-SAT model: {e}")
        return None
    
    # Check if a solution was found
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        logger.info(f"Solution found with status: {status}")
        
        # Extract solution
        scheduled_jobs = {}
        
        for job_id, task in all_tasks.items():
            # Get relative hours from solver
            start_rel = solver.Value(task['start'])
            end_rel = solver.Value(task['end'])
            
            # Convert relative hours to absolute epoch times
            start_time = relative_hours_to_epoch(start_rel)
            end_time = relative_hours_to_epoch(end_rel)
            
            scheduled_jobs[job_id] = {
                'machine': task['machine'],
                'start': start_time,
                'end': end_time,
                'job': task['job']  # Include original job data
            }
            
            logger.info(f"Scheduled {job_id} on {task['machine']}: start={format_datetime_for_display(epoch_to_datetime(start_time))}, end={format_datetime_for_display(epoch_to_datetime(end_time))}")
        
        # Log objective values
        if gap_vars:
            logger.info(f"Total gap: {solver.Value(total_gap)}")
        
        if tardiness_vars:
            logger.info(f"Total tardiness: {solver.Value(total_tardiness)}")
        
        logger.info(f"Makespan: {solver.Value(makespan)}")
        
        # Return the schedule and visualization data
        scheduled_jobs['_metadata'] = {
            'job_dependencies': job_dependencies,
            'start_date_processes': start_date_processes,
            'reference_time': reference_time  # Include reference time for time conversions
        }
        
        return scheduled_jobs
    else:
        if status == cp_model.INFEASIBLE:
            logger.error(f"Solver failed with status: INFEASIBLE")
        elif status == cp_model.MODEL_INVALID:
            logger.error(f"Solver failed with status: MODEL_INVALID")
        elif status == cp_model.UNKNOWN:
            logger.error(f"Solver failed with status: UNKNOWN (time limit reached?)")
        else:
            logger.error(f"Solver failed with status: {status}")
        
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
        logger.error(f"Error in main: {e}")