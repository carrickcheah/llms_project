# sch_jobs.py
# Constraint Programming (CP-SAT) solver for production scheduling

from ortools.sat.python import cp_model
from datetime import datetime
import logging
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_process_sequence(process_code):
    """
    Extract sequence number from process code (e.g., P01 -> 1, P02 -> 2)
    Works with formats like "CP08-231B-P01-06", "CP08-231-P02-06", or "cp08-231-P01"
    """
    # Convert to uppercase for consistency
    process_code = process_code.upper()
    
    # Try to find P followed by 2 digits (like "P01", "P02")
    match = re.search(r'P(\d{2})', process_code)
    if match:
        try:
            # Extract and convert to int - "01" becomes 1, "02" becomes 2, etc.
            sequence_str = match.group(1)
            sequence_num = int(sequence_str)
            logger.info(f"Extracted sequence {sequence_num} from {process_code}")
            return sequence_num
        except ValueError:
            logger.warning(f"Failed to convert sequence number in {process_code}")
            pass
    else:
        logger.warning(f"No sequence number found in {process_code}")
    
    return 999  # Default if parsing fails (high number to put at end)

def extract_job_family(process_code):
    """
    Extract job family code from process code
    E.g., "CP08-231B-P01-06" -> "CP08-231B", "cp08-231-P01" -> "CP08-231"
    """
    # Convert to uppercase for consistency
    process_code = process_code.upper()
    
    # Try to find everything before P followed by digits
    match = re.search(r'(.*?)-P\d+', process_code)
    if match:
        family = match.group(1)
        logger.debug(f"Extracted family {family} from {process_code}")
        return family
    
    # Alternative: split by P and take everything before it
    parts = process_code.split("-P")
    if len(parts) >= 2:
        family = parts[0]
        logger.debug(f"Extracted family {family} from {process_code} (using split)")
        return family
    
    logger.warning(f"Could not extract family from {process_code}, using full code")
    return process_code  # Default to full code if parsing fails

def process_jobs_data(jobs, machines):
    """Process job data for scheduling by removing invalid jobs and adding unique IDs."""
    processed_jobs = []
    machine_jobs = {}
    
    # Initialize machine_jobs
    for m in machines:
        machine_id = m[1]
        machine_jobs[machine_id] = []
        
    # Filter valid jobs and add them to respective machines
    for i, job in enumerate(jobs):
        if len(job) < 6:
            logging.warning(f"Job at position {i} has insufficient data, skipping")
            continue
            
        # Extract key job information
        job_name, process_code, machine, duration, due_time, priority = job[:6]
        start_time = job[6] if len(job) > 6 else None
        
        # Skip invalid jobs
        if not machine or machine not in machine_jobs or duration <= 0:
            logging.warning(f"Job {job_name} has invalid machine or duration, skipping")
            continue
            
        # Add unique job ID
        job_with_id = (i,) + job
        processed_jobs.append(job_with_id)
        machine_jobs[machine].append(job_with_id)
    
    logging.info(f"Processed {len(processed_jobs)} valid jobs for {len(machine_jobs)} machines")
    return processed_jobs, machine_jobs

def preprocess_for_sequence_dependency(jobs):
    """
    Preprocess jobs to ensure proper sequence dependency.
    Sort jobs by family and sequence number and adjust priorities to enforce sequence.
    
    cp08-231-P01 must finish first, then cp08-231-P02 can start, and so on.
    """
    # Extract job families and sequences
    job_data = []
    for job in jobs:
        if len(job) < 6:
            continue
            
        job_name, process_code, machine, duration, due_time, priority = job[:6]
        start_time = job[7] if len(job) > 7 else None
        
        family = extract_job_family(process_code)
        sequence = extract_process_sequence(process_code)
        
        job_data.append({
            'original_job': job,
            'family': family,
            'sequence': sequence,
            'process_code': process_code
        })
    
    # Group by family
    families = {}
    for job in job_data:
        family = job['family']
        if family not in families:
            families[family] = []
        families[family].append(job)
    
    # Sort each family by sequence and enforce the sequence by adjusting priorities
    # Lower sequence jobs get higher priority to ensure they're scheduled first
    for family, jobs in families.items():
        # Sort by sequence number (P01, P02, P03, etc.)
        sorted_jobs = sorted(jobs, key=lambda x: x['sequence'])
        
        # Update priorities to enforce the sequence - P01 gets highest priority
        for i, job in enumerate(sorted_jobs):
            # Set an artificially high priority for earlier processes in sequence
            # Original job tuple: (job_name, process_code, machine, duration, due_time, priority)
            # Modify the priority (index 5) to enforce sequence
            original_job_list = list(job['original_job'])
            if len(original_job_list) > 5:  # Make sure priority field exists
                # Lower number = higher priority; subtract from 100 to maintain original relative priority
                # while ensuring sequence order is respected (P01 = highest priority)
                base_priority = 100 - (len(sorted_jobs) - i) * 10
                original_job_list[5] = min(original_job_list[5], base_priority)
                job['original_job'] = tuple(original_job_list)
        
        # Update the family with sorted and priority-adjusted jobs
        families[family] = sorted_jobs
        
    # Log what we found
    for family, jobs in families.items():
        process_codes = [j['process_code'] for j in jobs]
        logger.info(f"Family {family} process sequence: {', '.join(process_codes)}")
    
    # Return the processed data
    return families

def verify_sequences(schedule):
    """Verify that the sequence constraints are followed in the final schedule."""
    # Collect all jobs across all machines
    all_jobs = []
    for machine, jobs in schedule.items():
        for job in jobs:
            process_code, start, end, priority = job
            all_jobs.append({
                'process_code': process_code,
                'family': extract_job_family(process_code),
                'sequence': extract_process_sequence(process_code),
                'start': start,
                'end': end,
                'machine': machine
            })
    
    # Group by family
    family_jobs = {}
    for job in all_jobs:
        family = job['family']
        if family not in family_jobs:
            family_jobs[family] = []
        family_jobs[family].append(job)
    
    # Check each family for sequence correctness
    sequence_violations = 0
    for family, jobs in family_jobs.items():
        if len(jobs) <= 1:
            continue
            
        # Sort by sequence number
        sorted_by_seq = sorted(jobs, key=lambda x: x['sequence'])
        
        # Check each pair of consecutive processes to ensure end time of earlier process
        # is before start time of later process
        violations = 0
        for i in range(len(sorted_by_seq) - 1):
            job1 = sorted_by_seq[i]
            job2 = sorted_by_seq[i+1]
            
            if job1['end'] > job2['start']:
                violations += 1
                logger.warning(f"Sequence violation: {job1['process_code']} (ends at {datetime.fromtimestamp(job1['end'])}) " +
                              f"finishes after {job2['process_code']} starts (at {datetime.fromtimestamp(job2['start'])})")
        
        sequence_violations += violations
        
        if violations == 0:
            logger.info(f"Family {family}: All sequence constraints satisfied")
        else:
            logger.warning(f"Family {family}: Found {violations} sequence constraint violations")
            logger.warning("Process order: " + ", ".join([j['process_code'] for j in sorted_by_seq]))
            
    if sequence_violations == 0:
        logger.info("All process sequence dependencies satisfied across all job families")
    else:
        logger.warning(f"Found {sequence_violations} sequence constraint violations across all job families")

def schedule_jobs(jobs, machines, setup_times=None, time_limit_seconds=300):
    """
    Advanced scheduling function using CP-SAT solver with priority-based optimization
    and process sequence dependencies.
    
    Args:
        jobs: List of tuples (job_name, process_code, machine, duration, due_time, priority, start_time)
        machines: List of tuples (id, machine_name, capacity)
        setup_times: Dictionary mapping (process_code1, process_code2) to setup duration
        time_limit_seconds: Maximum time limit for the solver in seconds
        
    Returns:
        Dictionary mapping machine IDs to lists of scheduled jobs (process_code, start_time, end_time, priority)
    """
    # Start timing
    start_time = time.time()
    
    # First, preprocess jobs to enforce sequence dependencies
    job_families = preprocess_for_sequence_dependency(jobs)
    
    # Apply very strong sequence dependency enforcement:
    # Modify start times for each job to enforce sequence
    modified_jobs = []
    current_time = int(datetime.now().timestamp())
    
    # Process each family in sequence
    for family, family_jobs in job_families.items():
        last_end_time = current_time
        
        for job_info in family_jobs:
            job = list(job_info['original_job'])  # Convert to list to allow modification
            
            # Set the earliest start time to ensure sequential processing
            if len(job) > 6:
                job[6] = max(job[6] if job[6] is not None else current_time, last_end_time)
            else:
                # If start_time isn't in the job tuple, extend it
                job.append(last_end_time)
            
            # Update last_end_time for next job in sequence
            duration = job[4]
            last_end_time = job[6] + duration + 60  # Add a 60-second buffer between jobs
            
            modified_jobs.append(tuple(job))
            
    # Use the modified jobs for scheduling
    processed_jobs, machine_jobs = process_jobs_data(modified_jobs, machines)
    if not processed_jobs:
        logging.error("No valid jobs to schedule")
        return {}
        
    # Create model
    model = cp_model.CpModel()
    
    # Get parameters for objective scaling
    horizon_end = max(job[5] for job in processed_jobs) + 86400  # Add one day to the latest due date
    horizon = horizon_end - current_time
    
    # Create job variables
    # For each job: start time variable, end time variable
    starts = {}
    ends = {}
    intervals = {}  # For each job
    presences = {}  # For optional jobs
    job_delays = {}  # For tracking delays relative to due dates
    early_starts = {}  # For tracking early starts relative to planned start
    
    # Create variables for each job
    for job in processed_jobs:
        job_id, job_name, process_code, machine, duration, due_time, priority = job[0], job[1], job[2], job[3], job[4], job[5], job[6]
        
        # Earliest start time defaults to now
        earliest_start = job[7] if len(job) > 7 and job[7] is not None else current_time
        earliest_start = max(current_time, earliest_start)
        
        # Create variables - ensure we have a valid domain
        suffix = f"_{job_id}_{process_code}"
        starts[job_id] = model.NewIntVar(earliest_start, horizon_end, f"start{suffix}")
        ends[job_id] = model.NewIntVar(earliest_start + duration, horizon_end + duration, f"end{suffix}")
        
        # Create a fixed duration variable
        duration_var = model.NewConstant(duration)
        
        # Create the interval variable
        intervals[job_id] = model.NewIntervalVar(starts[job_id], duration_var, ends[job_id], f"interval{suffix}")
        
        # Add constraint: end = start + duration
        model.Add(ends[job_id] == starts[job_id] + duration)
        
        # Track delay beyond due date (in seconds)
        if due_time > 0:
            # Delay if end time > due time, otherwise 0
            job_delays[job_id] = model.NewIntVar(0, horizon_end - due_time, f"delay{suffix}")
            model.AddMaxEquality(job_delays[job_id], [model.NewConstant(0), ends[job_id] - due_time])
            
        # Track early start (negative of actual start - earliest start)
        if earliest_start > current_time:
            early_starts[job_id] = model.NewIntVar(0, horizon, f"early_start{suffix}")
            model.AddMaxEquality(early_starts[job_id], [model.NewConstant(0), earliest_start - starts[job_id]])
    
    # Create constraints for machine capacity
    for machine_id, jobs_on_machine in machine_jobs.items():
        intervals_for_machine = [intervals[job[0]] for job in jobs_on_machine]
        
        if intervals_for_machine:
            # Ensure no overlap on each machine
            model.AddNoOverlap(intervals_for_machine)
            
            # Add setup times if provided
            if setup_times:
                for i, job1 in enumerate(jobs_on_machine):
                    for j, job2 in enumerate(jobs_on_machine):
                        if i != j:
                            job1_id, job1_name, job1_process = job1[0], job1[1], job1[2]
                            job2_id, job2_name, job2_process = job2[0], job2[1], job2[2]
                            
                            # Check for setup time between these process codes
                            if job1_process in setup_times and job2_process in setup_times[job1_process]:
                                setup_duration = setup_times[job1_process][job2_process]
                                
                                # If there's a non-zero setup time required
                                if setup_duration > 0:
                                    # Boolean indicator if job2 follows job1
                                    suffix = f"_{job1_id}_{job2_id}"
                                    follows = model.NewBoolVar(f"job{job2_id}_follows_job{job1_id}")
                                    
                                    # Either job2 starts after job1 ends + setup_time, or job2 doesn't follow job1
                                    model.Add(starts[job2_id] >= ends[job1_id] + setup_duration).OnlyEnforceIf(follows)
                                    
                                    # Either job1 starts after job2 ends, or job2 follows job1
                                    model.Add(starts[job1_id] >= ends[job2_id]).OnlyEnforceIf(follows.Not())
    
    # Add explicit process sequence constraints again as extra enforcement
    # Group processed_jobs by family and sequence
    family_jobs = {}
    for job in processed_jobs:
        job_id = job[0]
        process_code = job[2]
        
        family = extract_job_family(process_code)
        sequence = extract_process_sequence(process_code)
        
        if family not in family_jobs:
            family_jobs[family] = []
        
        family_jobs[family].append((job_id, process_code, sequence))
    
    # Add strong precedence constraints
    added_constraints = 0
    for family, jobs in family_jobs.items():
        sorted_jobs = sorted(jobs, key=lambda x: x[2])
        
        for i in range(len(sorted_jobs) - 1):
            # Always enforce sequence based on process code numbers
            # If first job is P01 and next is P02, ensure P01 finishes before P02 starts
            pred_id = sorted_jobs[i][0]
            succ_id = sorted_jobs[i+1][0]
            
            # Add hard constraint
            model.Add(ends[pred_id] <= starts[succ_id])
            added_constraints += 1
            
            logger.info(f"Added sequence constraint: {sorted_jobs[i][1]} (seq {sorted_jobs[i][2]}) " +
                       f"must finish before {sorted_jobs[i+1][1]} (seq {sorted_jobs[i+1][2]}) starts")
    
    logger.info(f"Added {added_constraints} explicit sequence constraints")
    
    # Create objective function
    objective_terms = []
    
    # Minimize delays (weighted by priority)
    priority_delay_costs = 0
    for job in processed_jobs:
        job_id, _, _, _, _, due_time, priority = job[0], job[1], job[2], job[3], job[4], job[5], job[6]
        
        if job_id in job_delays and due_time > 0:
            # Scale delay by priority (higher priority = higher cost for delays)
            # Convert priority from 1-5 (where 1 is highest) to weights 5-1
            priority_weight = 6 - priority
            delay_cost = job_delays[job_id] * priority_weight * 10
            objective_terms.append(delay_cost)
            priority_delay_costs += 1
    
    # Minimize makespan (completion time of all jobs)
    makespan = model.NewIntVar(current_time, horizon_end, "makespan")
    for job_id in ends:
        model.Add(ends[job_id] <= makespan)
    
    # Add makespan to objective with a lower weight
    objective_terms.append(makespan)
    
    # Minimize early starts (prefer to start close to planned start dates)
    early_start_costs = 0
    for job_id, early_start in early_starts.items():
        objective_terms.append(early_start)
        early_start_costs += 1
    
    # Set the objective
    if objective_terms:
        model.Minimize(sum(objective_terms))
    
    # Create solver and solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.log_search_progress = True
    
    # Log model stats
    logging.info(f"CP-SAT Model created with {len(processed_jobs)} jobs")
    logging.info(f"Objective components: {priority_delay_costs} priority-weighted delays, makespan, {early_start_costs} early starts")
    
    # Solve the model
    logging.info(f"Starting CP-SAT solver with {time_limit_seconds} seconds time limit")
    status = solver.Solve(model)
    
    # Process results
    solve_time = time.time() - start_time
    if status == cp_model.OPTIMAL:
        logging.info(f"Optimal solution found in {solve_time:.2f} seconds")
    elif status == cp_model.FEASIBLE:
        logging.info(f"Feasible solution found in {solve_time:.2f} seconds")
    elif status == cp_model.INFEASIBLE:
        logging.error("Problem is infeasible")
        return {}
    else:
        logging.error(f"No solution found. Status: {solver.StatusName(status)}")
        return {}
    
    # Build the schedule
    schedule = {}
    total_output = 0
    
    for job_id in starts:
        job = processed_jobs[job_id]
        job_name = job[1]
        process_code = job[2]  # Get the PROCESS_CODE
        machine = job[3]
        priority = job[6]
        output_rate = job[9] if len(job) > 9 else 100
        start = solver.Value(starts[job_id])
        end = solver.Value(ends[job_id])
        
        duration_hours = (end - start) / 3600
        job_output = output_rate * duration_hours
        total_output += job_output
        
        if machine not in schedule:
            schedule[machine] = []
        # Use process_code instead of job_name in the output tuple
        schedule[machine].append((process_code, start, end, priority))
    
    # Sort jobs by start time for each machine
    for machine in schedule:
        schedule[machine].sort(key=lambda x: x[1])
    
    # Log statistics about the schedule
    logging.info(f"Scheduled {sum(len(jobs) for jobs in schedule.values())} jobs on {len(schedule)} machines")
    logging.info(f"Expected total output: {total_output:.2f} units")
    
    # Verify sequence constraints were followed
    verify_sequences(schedule)
    
    return schedule