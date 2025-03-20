import logging
from datetime import datetime
import re

# Configure logging (standalone or align with project setup)
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
    return 999

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

def greedy_schedule(jobs, machines, setup_times=None, enforce_sequence=True):
    """
    Create a schedule using an enhanced greedy algorithm with process sequence enforcement,
    considering due dates and earliest start times (START_DATE).

    Args:
        jobs (list): List of job dictionaries
        machines (list): List of machine IDs
        setup_times (dict): Dictionary mapping (process_code1, process_code2) to setup duration
        enforce_sequence (bool): Whether to enforce process sequence dependencies

    Returns:
        dict: Schedule as {machine: [(process_code, start, end, priority), ...]}
    """
    current_time = int(datetime.now().timestamp())
    schedule = {machine: [] for machine in machines}
    machine_availability = {machine: current_time for machine in machines}
    
    # Validate jobs
    for job in jobs:
        if not isinstance(job.get('processing_time', 0), (int, float)) or job.get('processing_time', 0) <= 0:
            logger.error(f"Invalid processing_time for job {job.get('PROCESS_CODE', 'Unknown')}: {job.get('processing_time')}")
            raise ValueError(f"Invalid processing_time for job {job.get('PROCESS_CODE', 'Unknown')}")

    # First, identify jobs with START_DATE constraints
    start_date_jobs = [job for job in jobs if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None]
    future_start_date_jobs = [job for job in start_date_jobs if job['START_DATE_EPOCH'] > current_time]
    
    if future_start_date_jobs:
        logger.info(f"Found {len(future_start_date_jobs)} jobs with future START_DATE constraints")
        for job in future_start_date_jobs:
            job_start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
            logger.info(f"Job {job['PROCESS_CODE']} must start exactly at START_DATE={job_start_date}")
    
    # Sort jobs - prioritize jobs with START_DATE first to ensure they're scheduled at their exact required times
    if enforce_sequence:
        # Group jobs by family
        family_groups = {}
        for job in jobs:
            family = extract_job_family(job['PROCESS_CODE'])
            if family not in family_groups:
                family_groups[family] = []
            family_groups[family].append(job)

        # Sort within each family - but prioritize families with START_DATE constraints
        families_with_start_dates = set()
        for job in future_start_date_jobs:
            families_with_start_dates.add(extract_job_family(job['PROCESS_CODE']))
            
        # Sort families - those with START_DATE constraints first
        sorted_families = sorted(family_groups.keys(), 
                                key=lambda f: (0 if f in families_with_start_dates else 1, f))
        
        sorted_jobs = []
        for family in sorted_families:
            sorted_family_jobs = sorted(family_groups[family], 
                                      key=lambda x: extract_process_number(x['PROCESS_CODE']))
            sorted_jobs.extend(sorted_family_jobs)
    else:
        # Sort jobs: START_DATE jobs first, then priority and due date
        sorted_jobs = sorted(jobs, key=lambda x: (
            0 if ('START_DATE_EPOCH' in x and x['START_DATE_EPOCH'] is not None and x['START_DATE_EPOCH'] > current_time) else 1,
            x.get('START_DATE_EPOCH', float('inf')),
            x['PRIORITY'], 
            x.get('LCD_DATE_EPOCH', float('inf'))
        ))

    # Track the latest end time for each family to enforce sequence
    family_end_times = {}
    
    # Track exact required start times by machine to ensure no conflicts
    required_start_times = {machine: [] for machine in machines}
    for job in future_start_date_jobs:
        machine_id = job['MACHINE_ID']
        start_time = job['START_DATE_EPOCH']
        process_code = job['PROCESS_CODE']
        required_start_times[machine_id].append((process_code, start_time))
    
    # Sort the required start times for each machine
    for machine in required_start_times:
        required_start_times[machine] = sorted(required_start_times[machine], key=lambda x: x[1])
    
    # Process each job in the sorted order
    for job in sorted_jobs:
        job_id = job['PROCESS_CODE']
        machine_id = job['MACHINE_ID']
        duration = job['processing_time']
        due_time = job.get('LCD_DATE_EPOCH', current_time + 30 * 24 * 3600)
        family = extract_job_family(job_id)
        
        # 1. Current time constraint
        time_constraint = current_time
        
        # 2. Machine availability constraint
        machine_constraint = machine_availability[machine_id]
        
        # 3. Sequence constraint (if applicable)
        sequence_constraint = family_end_times.get(family, current_time) if enforce_sequence else current_time
        
        # 4. START_DATE constraint - MODIFIED FOR EXACT TIMING
        has_start_date = False
        start_date_constraint = current_time
        
        if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None:
            start_date_constraint = job['START_DATE_EPOCH']
            has_start_date = True
            
            # For jobs with START_DATE in the future, enforce EXACT start time
            if start_date_constraint > current_time:
                start_date_str = datetime.fromtimestamp(start_date_constraint).strftime('%Y-%m-%d %H:%M')
                logger.info(f"ENFORCING EXACT START_DATE for job {job_id}: {start_date_str}")
                
                # For ALL jobs with START_DATE constraint, force the exact start time
                if machine_constraint > start_date_constraint:
                    # Check if we need to delay to respect sequence constraints
                    if sequence_constraint > start_date_constraint:
                        logger.info(f"⚠️ Cannot start {job_id} at exactly START_DATE={start_date_str} due to sequence constraints")
                        # In this case, we use sequence_constraint as our start time
                    else:
                        # Make machine available exactly at START_DATE
                        logger.info(f"🔄 Adjusting machine availability for {job_id} to ensure START_DATE={start_date_str}")
                        machine_constraint = start_date_constraint
        
        # Find the most restrictive constraint
        constraints = {
            "Current Time": time_constraint,
            "Machine Availability": machine_constraint,
            "Sequence Dependency": sequence_constraint,
            "START_DATE (User-defined)": start_date_constraint if has_start_date else current_time
        }
        
        # MODIFIED: If we have a START_DATE constraint, prefer using it exactly
        # unless sequence constraints make that impossible
        if has_start_date and start_date_constraint > current_time:
            # If sequence allows, use the exact START_DATE
            if sequence_constraint <= start_date_constraint:
                earliest_start = start_date_constraint
                active_constraint = "START_DATE (User-defined)"
            else:
                # We can't respect START_DATE due to sequence constraints
                earliest_start = sequence_constraint
                active_constraint = "Sequence Dependency"
        else:
            # Standard constraint resolution for jobs without START_DATE
            earliest_start = max(constraints.values())
            active_constraint = [name for name, value in constraints.items() if value == earliest_start][0]
        
        # Log which constraint is controlling this job's start time
        if earliest_start > current_time:
            active_date = datetime.fromtimestamp(earliest_start).strftime('%Y-%m-%d %H:%M')
            logger.info(f"Job {job_id} start time ({active_date}) determined by: {active_constraint}")
            
            if has_start_date and active_constraint != "START_DATE (User-defined)":
                original_date = datetime.fromtimestamp(start_date_constraint).strftime('%Y-%m-%d %H:%M')
                logger.info(f"⚠️ Job {job_id} START_DATE={original_date} couldn't be honored exactly")
        
        # Schedule the job
        start = earliest_start
        end = start + duration

        # Apply setup time if provided, but respect START_DATE if specified
        if setup_times and job_id in setup_times and schedule[machine_id]:
            last_job = schedule[machine_id][-1]
            last_job_end = last_job[2]
            setup_duration = setup_times[job_id].get(last_job[0], 0)
            
            # Determine if this is a job with START_DATE that must be honored
            must_honor_start_date = has_start_date and start_date_constraint > current_time and start == start_date_constraint
            
            if start < last_job_end + setup_duration:
                if must_honor_start_date:
                    logger.info(f"⚠️ Skipping setup time for job {job_id} to ensure exact START_DATE constraint is met")
                else:
                    old_start = start
                    start = last_job_end + setup_duration
                    end = start + duration
                    logger.info(f"Applied setup time: Job {job_id} start delayed from {datetime.fromtimestamp(old_start).strftime('%Y-%m-%d %H:%M')} to {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')}")

        # For jobs with START_DATE, store START_TIME = START_DATE_EPOCH for visualization
        if has_start_date and start_date_constraint > current_time:
            # This ensures that chart_two.py will display START_TIME = START_DATE
            job['START_TIME'] = start_date_constraint
        
        # Schedule the job
        schedule[machine_id].append((job_id, start, end, job['PRIORITY']))
        machine_availability[machine_id] = end
        if enforce_sequence:
            family_end_times[family] = end
        
        # Log the final scheduling decision
        logger.info(f"✅ Scheduled job {job.get('JOB_CODE', job_id)} ({job_id}) on {machine_id}: "
                   f"{datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')} to {datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')}")

    logger.info(f"Enhanced greedy schedule created with {sum(len(tasks) for tasks in schedule.values())} jobs")
    
    return schedule

if __name__ == "__main__":
    # Use real data from file_path
    from ingest_data import load_jobs_planning_data
    
    file_path = "/Users/carrickcheah/llms_project/services/agent_production_planning/mydata.xlsx"
    try:
        jobs, machines, setup_times = load_jobs_planning_data(file_path)
        logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines from {file_path}")
        schedule = greedy_schedule(jobs, machines, setup_times, enforce_sequence=True)
        logger.info(f"Generated schedule with {sum(len(jobs_list) for jobs_list in schedule.values())} scheduled tasks")
    except Exception as e:
        logger.error(f"Error testing scheduler with real data: {e}")