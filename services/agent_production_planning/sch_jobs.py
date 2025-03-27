# sch_jobs.py | dont edit this line
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

def schedule_jobs(jobs, machines, setup_times=None, enforce_sequence=True, time_limit_seconds=300):
    """
    Advanced scheduling function using CP-SAT solver with priority-based optimization
    and process sequence dependencies.

    Args:
        jobs (list): List of job dictionaries, each with UNIQUE_JOB_ID
        machines (list): List of machine IDs
        setup_times (dict): Dictionary mapping (unique_job_id1, unique_job_id2) to setup duration
        enforce_sequence (bool): Whether to enforce process sequence dependencies
        time_limit_seconds (int): Maximum time limit for the solver in seconds

    Returns:
        dict: Schedule as {machine: [(unique_job_id, start, end, priority), ...]}
    """
    start_time = time.time()
    current_time = int(datetime.now().timestamp())

    horizon_end = int(current_time + max(job.get('LCD_DATE_EPOCH', current_time + 2592000) for job in jobs) + max(job['processing_time'] for job in jobs))
    horizon = horizon_end - current_time

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.log_search_progress = True
    
    import multiprocessing
    available_cpus = multiprocessing.cpu_count()
    optimal_workers = min(available_cpus, 16)
    solver.parameters.num_search_workers = optimal_workers
    logger.info(f"Using {optimal_workers} parallel search workers for CP-SAT solver")
    
    solver.parameters.linearization_level = 2
    
    # Log additional data columns that will be used in scheduling
    additional_columns = []
    if any('setup_time' in job and job['setup_time'] > 0 for job in jobs):
        additional_columns.append('SETTING_HOURS (setup times)')
    if any('break_time' in job and job['break_time'] > 0 for job in jobs):
        additional_columns.append('BREAK_HOURS (production breaks)')
    if any('no_prod_time' in job and job['no_prod_time'] > 0 for job in jobs):
        additional_columns.append('NO_PROD(non-production times)')
        
    if additional_columns:
        logger.info(f"Using additional data columns for advanced scheduling: {', '.join(additional_columns)}")
    else:
        logger.info("No additional data columns found for advanced scheduling features")
    solver.parameters.optimize_with_core = True
    solver.parameters.use_lns = True

    start_date_jobs = [job for job in jobs if 
                      (job.get('START_DATE_EPOCH', current_time) > current_time) or 
                      (job.get('START_DATE _EPOCH', current_time) > current_time)]
    if start_date_jobs:
        logger.info(f"Found {len(start_date_jobs)} jobs with START_DATE constraints for CP-SAT solver:")
        for job in start_date_jobs:
            start_date_epoch = job.get('START_DATE_EPOCH')
            if start_date_epoch is None:
                start_date_epoch = job.get('START_DATE _EPOCH')
            start_date = datetime.fromtimestamp(start_date_epoch).strftime('%Y-%m-%d %H:%M')
            resource_location = job.get('RSC_LOCATION', 'Unknown')
            logger.info(f"  Job {job['UNIQUE_JOB_ID']} on {resource_location}: MUST start EXACTLY at {start_date}")
        logger.info("START_DATE constraints will be strictly enforced in the model")

    family_jobs = {}
    for job in jobs:
        if 'UNIQUE_JOB_ID' not in job:
            logger.error(f"Job missing UNIQUE_JOB_ID: {job}")
            continue
        job_id = job.get('JOB', '')
        family = extract_job_family(job['UNIQUE_JOB_ID'], job_id)
        if family not in family_jobs:
            family_jobs[family] = []
        family_jobs[family].append(job)

    for family in family_jobs:
        family_jobs[family].sort(key=lambda x: extract_process_number(x['UNIQUE_JOB_ID']))

    family_constraints = {}
    for family, family_job_list in family_jobs.items():
        for job in family_job_list:
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] > current_time:
                if family not in family_constraints:
                    family_constraints[family] = []
                family_constraints[family].append(job)

    for family, constrained_jobs in family_constraints.items():
        logger.info(f"Family {family} has {len(constrained_jobs)} jobs with START_DATE constraints")
        for job in constrained_jobs:
            start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
            logger.info(f"  Job {job['UNIQUE_JOB_ID']} must start at {start_date}")

    start_vars = {}
    end_vars = {}
    intervals = {}
    for job in jobs:
        if 'UNIQUE_JOB_ID' not in job:
            continue
        unique_job_id = job['UNIQUE_JOB_ID']
        machine_id = job.get('RSC_LOCATION')
        duration = job['processing_time']
        due_time = job.get('LCD_DATE_EPOCH', horizon_end - duration)
        priority = job['PRIORITY']

        has_start_date = (('START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None) or 
                         ('START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None))
        user_start_time = job.get('START_DATE_EPOCH', job.get('START_DATE _EPOCH', current_time))

        if not isinstance(duration, (int, float)) or duration <= 0:
            logger.error(f"Invalid duration for job {unique_job_id}: {duration}")
            raise ValueError(f"Invalid duration for job {unique_job_id}")

        if has_start_date and user_start_time > current_time:
            int_start_time = int(user_start_time)
            start_var = model.NewIntVar(int_start_time, int_start_time, f'start_{machine_id}_{unique_job_id}')
            logger.info(f"Job {unique_job_id} must start EXACTLY at {datetime.fromtimestamp(int_start_time).strftime('%Y-%m-%d %H:%M')}")
            job['START_TIME'] = int_start_time
        else:
            earliest_start = int(current_time)
            start_var = model.NewIntVar(earliest_start, horizon_end, f'start_{machine_id}_{unique_job_id}')
            
        end_time = int(horizon_end + duration)
        end_var = model.NewIntVar(current_time, end_time, f'end_{machine_id}_{unique_job_id}')
        
        int_duration = int(duration)
        interval_var = model.NewIntervalVar(start_var, int_duration, end_var, f'interval_{machine_id}_{unique_job_id}')

        start_vars[(unique_job_id, machine_id)] = start_var
        end_vars[(unique_job_id, machine_id)] = end_var
        intervals[(unique_job_id, machine_id)] = interval_var

        model.Add(end_var == start_var + int_duration)

    # Handle machine constraints with setup times, BREAK_HOURS, and no-production periods
    for machine in machines:
        unique_job_signatures = set()
        machine_intervals = []
        machine_jobs = []
        
        for job in jobs:
            if 'UNIQUE_JOB_ID' not in job:
                continue
            try:
                if job.get('RSC_LOCATION') == machine:
                    unique_job_id = job['UNIQUE_JOB_ID']
                    interval_key = (unique_job_id, job.get('RSC_LOCATION', 'Unknown'))
                    
                    if interval_key not in intervals:
                        logger.warning(f"Missing interval for job {unique_job_id} on machine {machine}. Skipping.")
                        continue
                    
                    job_signature = (
                        job.get('JOB', ''),
                        job['UNIQUE_JOB_ID'],
                        job.get('RSC_LOCATION', ''),
                        job.get('RSC_CODE', '')
                    )
                    
                    if job_signature not in unique_job_signatures:
                        unique_job_signatures.add(job_signature)
                        machine_intervals.append(intervals[interval_key])
                        machine_jobs.append(job)
                    else:
                        logger.warning(f"Skipping duplicate job {unique_job_id} on machine {machine} with signature {job_signature} to prevent constraint conflicts")
            except Exception as e:
                logger.error(f"Error adding job to machine {machine}: {str(e)}")
                continue
        
        if machine_intervals:
            try:
                # Add basic no-overlap constraint for the machine
                model.AddNoOverlap(machine_intervals)
                logger.info(f"Added no-overlap constraint for machine {machine} with {len(machine_intervals)} jobs")
                
                # Add setup time transitions between jobs when available
                if setup_times:
                    setup_time_pairs = 0
                    for i, job1 in enumerate(machine_jobs):
                        for j, job2 in enumerate(machine_jobs):
                            if i != j:  # Different jobs
                                job1_id = job1['UNIQUE_JOB_ID']
                                job2_id = job2['UNIQUE_JOB_ID']
                                key = (job1_id, job2_id)
                                
                                # If we have setup time for this transition, add a constraint
                                if key in setup_times and setup_times[key] > 0:
                                    setup_duration = setup_times[key]
                                    
                                    # Create literal variable for the pair
                                    job1_precedes_job2 = model.NewBoolVar(f"{job1_id} precedes {job2_id}")
                                    
                                    # If job1 precedes job2, ensure there's enough setup time
                                    job1_machine_key = (job1_id, job1.get('RSC_LOCATION', 'Unknown'))
                                    job2_machine_key = (job2_id, job2.get('RSC_LOCATION', 'Unknown'))
                                    
                                    # job2 starts at least setup_duration after job1 ends
                                    time_diff = start_vars[job2_machine_key] - end_vars[job1_machine_key]
                                    
                                    # If job1 precedes job2, start2 >= end1 + setup_duration
                                    # This is modeled as: job1_precedes_job2 => time_diff >= setup_duration
                                    # 1. If job1_precedes_job2 is True, then time_diff >= setup_duration
                                    # 2. If job1_precedes_job2 is False, this constraint is inactive
                                    model.Add(time_diff >= setup_duration).OnlyEnforceIf(job1_precedes_job2)
                                    
                                    # For any pair of jobs, exactly one precedes the other
                                    # (Only when there are overlapping intervals)
                                    # If job2 precedes job1, no need to enforce setup time here
                                    # (will be handled in another iteration)
                                    setup_time_pairs += 1
                    
                    if setup_time_pairs > 0:
                        logger.info(f"Added {setup_time_pairs} setup time transition constraints on machine {machine}")
                
                # Add break/no-production constraints if any job has these values
                # (This is a simple implementation - assumes breaks are taken between jobs)
                break_time_jobs = [j for j in machine_jobs if j.get('break_time', 0) > 0 or j.get('no_prod_time', 0) > 0]
                if break_time_jobs:
                    logger.info(f"Found {len(break_time_jobs)} jobs on machine {machine} requiring breaks or non-production time")
                    # Could add more complex break scheduling here in future versions
                
            except Exception as e:
                logger.error(f"Failed to add constraints for machine {machine}: {str(e)}")

    if enforce_sequence:
        logger.info("Enforcing process sequence dependencies (P01->P02->P03)...")
        added_constraints = 0
        for family, family_job_list in family_jobs.items():
            sorted_jobs = sorted(family_job_list, key=lambda x: extract_process_number(x['UNIQUE_JOB_ID']))
            for i in range(len(sorted_jobs) - 1):
                job1 = sorted_jobs[i]
                job2 = sorted_jobs[i + 1]
                machine_id1 = job1.get('RSC_LOCATION', 'Unknown')
                machine_id2 = job2.get('RSC_LOCATION', 'Unknown')
                
                job1_start_time = job1.get('START_DATE_EPOCH', 0)
                job2_start_time = job2.get('START_DATE_EPOCH', 0)
                
                if job1_start_time > 0 and job2_start_time > 0 and job1_start_time >= job2_start_time:
                    logger.warning(f"Potential sequence conflict: {job1['UNIQUE_JOB_ID']} (starts {datetime.fromtimestamp(job1_start_time).strftime('%Y-%m-%d %H:%M')}) " 
                                  f"should finish before {job2['UNIQUE_JOB_ID']} (starts {datetime.fromtimestamp(job2_start_time).strftime('%Y-%m-%d %H:%M')})")
                    logger.warning(f"Skipping sequence constraint to avoid infeasibility")
                    continue
                
                if job1['UNIQUE_JOB_ID'] == job2['UNIQUE_JOB_ID']:
                    logger.warning(f"Skipping invalid self-dependency for {job1['UNIQUE_JOB_ID']}")
                    continue
                
                proc1 = extract_process_number(job1['UNIQUE_JOB_ID'])
                proc2 = extract_process_number(job2['UNIQUE_JOB_ID'])
                
                if proc2 != proc1 + 1:
                    logger.warning(f"Process numbers not sequential: {job1['UNIQUE_JOB_ID']} ({proc1}) → {job2['UNIQUE_JOB_ID']} ({proc2})")
                
                job1_signature = (
                    job1.get('JOB', ''),
                    job1['UNIQUE_JOB_ID'],
                    job1.get('RSC_LOCATION', ''),
                    job1.get('RSC_CODE', '')
                )
                
                job2_signature = (
                    job2.get('JOB', ''),
                    job2['UNIQUE_JOB_ID'],
                    job2.get('RSC_LOCATION', ''),
                    job2.get('RSC_CODE', '')
                )
                
                model.Add(end_vars[(job1['UNIQUE_JOB_ID'], machine_id1)] <= 
                         start_vars[(job2['UNIQUE_JOB_ID'], machine_id2)])
                added_constraints += 1
                logger.info(f"Added sequence constraint: {job1['UNIQUE_JOB_ID']} must finish before {job2['UNIQUE_JOB_ID']} starts")
        logger.info(f"Added {added_constraints} explicit sequence constraints")

    makespan = model.NewIntVar(current_time, horizon_end, 'makespan')
    for machine in machines:
        machine_ends = [end_vars[(job['UNIQUE_JOB_ID'], job.get('RSC_LOCATION', 'Unknown'))] 
                      for job in jobs 
                      if job.get('RSC_LOCATION') == machine and 'UNIQUE_JOB_ID' in job]
        if machine_ends:
            model.AddMaxEquality(makespan, machine_ends)

    # Enhanced objective function with production targets and breaks
    objective_terms = [makespan]
    
    # Track additional objective components 
    production_targets_count = 0
    break_time_count = 0
    setup_time_count = 0
    
    for job in jobs:
        if 'UNIQUE_JOB_ID' not in job:
            continue
            
        unique_job_id = job['UNIQUE_JOB_ID']
        machine_id = job.get('RSC_LOCATION')
        due_time = job.get('LCD_DATE_EPOCH', 0)
        priority = job['PRIORITY']
        
        # Basic delay cost (standard objective component)
        if due_time > 0:
            delay = model.NewIntVar(0, horizon_end - due_time, f'delay_{machine_id}_{unique_job_id}')
            model.AddMaxEquality(delay, [0, end_vars[(unique_job_id, machine_id)] - due_time])
            priority_weight = 6 - priority  # Invert priority (higher priority = lower weight)
            objective_terms.append(delay * priority_weight * 10)
        
        # Add production target and efficiency components
        if 'JOB_QUANTITY' in job and job.get('JOB_QUANTITY', 0) > 0 and \
           'EXPECT_OUTPUT_PER_HOUR' in job and job.get('EXPECT_OUTPUT_PER_HOUR', 0) > 0:
            # Create cost for jobs with specific production targets
            # This encourages the scheduler to group similar production runs
            job_quantity = job.get('JOB_QUANTITY', 0)
            output_rate = job.get('EXPECT_OUTPUT_PER_HOUR', 0)
            
            if job_quantity > 0 and output_rate > 0:
                # Small bonus for jobs with efficient production rates
                # This gives slightly higher priority to more efficient jobs
                production_efficiency = min(5, int(output_rate / 10))  # Scale to reasonable value
                production_bonus = production_efficiency * priority
                production_targets_count += 1
                
                # The higher the production_bonus, the more we want to prioritize this job
                # So we subtract it from the objective (minimizing objective value)
                objective_terms.append(-1 * production_bonus)
        
        # Penalize jobs with long setup times slightly to encourage grouping similar job types
        if 'setup_time' in job and job['setup_time'] > 0:
            setup_penalty = min(5, int(job['setup_time'] / 1800))  # Scale setup time (30 min = 1 unit)
            setup_time_count += 1
            objective_terms.append(setup_penalty)
        
        # Account for break times in scheduling optimization
        if 'break_time' in job and job['break_time'] > 0:
            break_penalty = min(3, int(job['break_time'] / 3600))  # Scale break time (1 hour = 1 unit)
            break_time_count += 1
            objective_terms.append(break_penalty)
    
    logger.info(f"Objective function includes: ")
    logger.info(f"  - Basic makespan and delay penalties")
    if production_targets_count > 0:
        logger.info(f"  - Production targets and efficiency bonuses for {production_targets_count} jobs")
    if setup_time_count > 0:
        logger.info(f"  - Setup time penalties for {setup_time_count} jobs")
    if break_time_count > 0:
        logger.info(f"  - Break time penalties for {break_time_count} jobs")
        
    if objective_terms:
        model.Minimize(sum(objective_terms))

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
        from greedy import greedy_schedule
        return greedy_schedule(jobs, machines, setup_times)
    else:
        logger.error(f"No solution found. Status: {solver.StatusName(status)}")
        return {}

    # Create enhanced schedule with additional parameters
    schedule = {}
    total_jobs = 0
    total_setup_time = 0
    total_break_time = 0
    
    for job in jobs:
        if 'UNIQUE_JOB_ID' not in job:
            continue
            
        unique_job_id = job['UNIQUE_JOB_ID']
        machine_id = job.get('RSC_LOCATION')
        start = solver.Value(start_vars[(unique_job_id, machine_id)])
        end = solver.Value(end_vars[(unique_job_id, machine_id)])
        priority = job['PRIORITY']
        total_jobs += 1

        # Handle START_DATE constraints
        start_date_epoch = job.get('START_DATE_EPOCH', job.get('START_DATE _EPOCH'))
        if start_date_epoch is not None and start_date_epoch > current_time:
            if start != start_date_epoch:
                logger.warning(f"⚠️ Job {unique_job_id} scheduled at {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')} " 
                              f"instead of requested START_DATE={datetime.fromtimestamp(start_date_epoch).strftime('%Y-%m-%d %H:%M')}")
            else:
                logger.info(f"✅ Job {unique_job_id} scheduled exactly at requested START_DATE")
        
        # Store setup time, break time, and other additional parameters
        setup_time = job.get('setup_time', 0)
        break_time = job.get('break_time', 0) 
        no_prod_time = job.get('no_prod_time', 0)
        job_quantity = job.get('JOB_QUANTITY')
        output_rate = job.get('EXPECT_OUTPUT_PER_HOUR')
        
        # Track total setup and break times
        if setup_time > 0:
            total_setup_time += setup_time
        if break_time > 0 or no_prod_time > 0:
            total_break_time += break_time + no_prod_time
        
        # Store timing information in the job
        job['START_TIME'] = start
        job['END_TIME'] = end
        
        # Create a dictionary of additional parameters to store with the schedule
        additional_params = {
            'setup_time': setup_time,
            'break_time': break_time,
            'no_prod_time': no_prod_time,
            'job_quantity': job_quantity,
            'output_rate': output_rate
        }
        
        # Add job to schedule with additional parameters
        if machine_id not in schedule:
            schedule[machine_id] = []
        schedule[machine_id].append((unique_job_id, start, end, priority, additional_params))
    
    # Log summary of additional parameters usage
    if total_setup_time > 0:
        logger.info(f"Total setup time in schedule: {total_setup_time/3600:.2f} hours")
    if total_break_time > 0:
        logger.info(f"Total break/no-production time in schedule: {total_break_time/3600:.2f} hours")

    for machine in schedule:
        schedule[machine].sort(key=lambda x: x[1])  # Sort by start time

    logger.info(f"Scheduled {sum(len(jobs) for jobs in schedule.values())} jobs on {len(schedule)} machines")
    logger.info(f"Total jobs scheduled: {total_jobs}")

    # Print schedule summary with additional parameters
    for machine, tasks in schedule.items():
        machine_setup_time = sum(task[4].get('setup_time', 0) for task in tasks)
        machine_break_time = sum(task[4].get('break_time', 0) + task[4].get('no_prod_time', 0) for task in tasks)
        
        if machine_setup_time > 0 or machine_break_time > 0:
            logger.info(f"Machine {machine} summary:")
            if machine_setup_time > 0:
                logger.info(f"  Setup time: {machine_setup_time/3600:.2f} hours")
            if machine_break_time > 0:
                logger.info(f"  Break/No-prod time: {machine_break_time/3600:.2f} hours")

    family_time_shifts = {}
    scheduled_times = {}
    
    # Extract scheduled times from the updated schedule format
    for machine, tasks in schedule.items():
        for task in tasks:
            unique_job_id, start, end = task[0], task[1], task[2]
            scheduled_times[unique_job_id] = (start, end)
    
    for family, constrained_jobs in family_constraints.items():
        time_shifts = []
        for job in constrained_jobs:
            unique_job_id = job['UNIQUE_JOB_ID']
            if unique_job_id in scheduled_times:
                scheduled_start = scheduled_times[unique_job_id][0]
                requested_start = job['START_DATE_EPOCH']
                time_shift = scheduled_start - requested_start
                time_shifts.append(time_shift)
        
        if time_shifts:
            family_time_shifts[family] = max(time_shifts, key=abs)
            logger.info(f"Family {family} needs time shift of {family_time_shifts[family]/3600:.1f} hours")
    
    for family, time_shift in family_time_shifts.items():
        if abs(time_shift) < 60:
            continue
            
        for job in family_jobs[family]:
            job['family_time_shift'] = time_shift
            logger.info(f"Added time shift of {time_shift/3600:.1f} hours to job {job['UNIQUE_JOB_ID']}")

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
                    resource_location = job.get('RSC_LOCATION', 'Unknown')
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