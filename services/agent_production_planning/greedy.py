# greedy.py | dont edit this line
import logging
from datetime import datetime
import re
from dotenv import load_dotenv
import os

# Configure logging (standalone or align with project setup)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_process_number(process_code):
    """
    Extract the process sequence number (e.g., 1 from 'P01-06') or return 999 if not found.
    """
    match = re.search(r'P(\d{2})', str(process_code).upper())  # Match exactly two digits after P
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

    # Step 1: Group jobs by family and process number
    family_groups = {}
    for job in jobs:
        family = extract_job_family(job['PROCESS_CODE'])
        if family not in family_groups:
            family_groups[family] = []
        family_groups[family].append(job)

    # Sort processes within each family
    for family in family_groups:
        family_groups[family].sort(key=lambda x: extract_process_number(x['PROCESS_CODE']))

    # Identify families with START_DATE constraints
    start_date_families = set()
    start_date_jobs = []
    
    for job in jobs:
        if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and job['START_DATE_EPOCH'] > current_time:
            family = extract_job_family(job['PROCESS_CODE'])
            start_date_families.add(family)
            start_date_jobs.append(job)
            job_start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
            logger.info(f"Job {job['PROCESS_CODE']} must start exactly at START_DATE={job_start_date}")
    
    if start_date_jobs:
        logger.info(f"Found {len(start_date_jobs)} jobs with future START_DATE constraints in {len(start_date_families)} families")
    
    # Create a mapping of process codes to process sequence numbers
    process_to_seq = {}
    for job in jobs:
        process_code = job['PROCESS_CODE']
        process_to_seq[process_code] = extract_process_number(process_code)
    
    # Sort families - prioritize families with START_DATE constraints first
    sorted_families = sorted(family_groups.keys(), key=lambda f: (0 if f in start_date_families else 1, f))
    
    # Create a flattened list of sorted jobs
    sorted_jobs = []
    for family in sorted_families:
        sorted_jobs.extend(family_groups[family])
    
    # Track the latest end time for each family to enforce sequence
    family_end_times = {}
    
    # Process each job in the sorted order
    for job in sorted_jobs:
        job_id = job['PROCESS_CODE']
        # Use RSC_LOCATION if available, fall back to MACHINE_ID for compatibility
        machine_id = job.get('RSC_LOCATION', job.get('MACHINE_ID'))
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
                
                # For jobs with START_DATE constraint, force the exact start time
                if machine_constraint > start_date_constraint:
                    # Make machine available exactly at START_DATE
                    logger.info(f"🔄 Adjusting machine availability for {job_id} to ensure START_DATE={start_date_str}")
                    machine_constraint = start_date_constraint
                
                # FIX: Prioritize START_DATE over sequence constraints
                # Even if sequence constraint would delay the start time, we'll use the exact START_DATE
                if sequence_constraint > start_date_constraint:
                    logger.info(f"⚠️ START_DATE for {job_id} conflicts with sequence constraints")
                    logger.info(f"🔄 Prioritizing START_DATE over sequence constraint as requested")
                    # We'll still record the sequence violation for reporting
                    job['sequence_violation'] = True
        
        # Find the most restrictive constraint
        constraints = {
            "Current Time": time_constraint,
            "Machine Availability": machine_constraint,
            "Sequence Dependency": sequence_constraint,
            "START_DATE (User-defined)": start_date_constraint if has_start_date else current_time
        }
        
        # MODIFIED: If we have a START_DATE constraint, ALWAYS use it exactly
        # even if it conflicts with sequence constraints
        if has_start_date and start_date_constraint > current_time:
            earliest_start = start_date_constraint
            active_constraint = "START_DATE (User-defined)"
        else:
            # Standard constraint resolution for jobs without START_DATE
            earliest_start = max(constraints.values())
            active_constraint = [name for name, value in constraints.items() if value == earliest_start][0]
        
        # Log which constraint is controlling this job's start time
        if earliest_start > current_time:
            active_date = datetime.fromtimestamp(earliest_start).strftime('%Y-%m-%d %H:%M')
            logger.info(f"Job {job_id} start time ({active_date}) determined by: {active_constraint}")
        
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
        
        # Schedule the job
        schedule[machine_id].append((job_id, start, end, job['PRIORITY']))
        machine_availability[machine_id] = end
        
        # Handle sequence dependencies differently for jobs with START_DATE constraints
        if enforce_sequence:
            if has_start_date and start_date_constraint > current_time and job.get('sequence_violation', False):
                # Don't update family_end_times for jobs with START_DATE that violated sequence
                # This prevents cascading violations - only this one job will have its exact time honored
                logger.info(f"Not updating family end time for {job_id} due to sequence violation")
            else:
                family_end_times[family] = end
        
        # Log the final scheduling decision
        job_name = job.get('JOB', job.get('JOB_CODE', job_id))
        logger.info(f"✅ Scheduled job {job_name} ({job_id}) on {machine_id}: "
                   f"{datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')} to {datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')}")

    # Calculate time shifts needed for families with START_DATE constraints
    family_time_shifts = {}
    scheduled_times = {}
    
    # First collect all scheduled times
    for machine, tasks in schedule.items():
        for job_id, start, end, _ in tasks:
            scheduled_times[job_id] = (start, end)
    
    # After all jobs are scheduled, calculate time shifts to represent the logical view of the schedule
    # This will show what the schedule would look like if all sequence dependencies were respected
    for family in start_date_families:
        family_has_violation = False
        for job in family_groups.get(family, []):
            if job.get('sequence_violation', False):
                family_has_violation = True
                break
        
        if family_has_violation:
            # Calculate appropriate time shift for visualization
            # This makes the visualization reflect the logical schedule with dependencies
            # even though the actual execution schedule honors the START_DATE constraints
            logger.info(f"Family {family} has sequence violations due to START_DATE constraints")
            
            # Find the process with START_DATE constraint
            for job in family_groups.get(family, []):
                job_id = job['PROCESS_CODE']
                if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and job_id in scheduled_times:
                    # Using 0 for time shift ensures the START_TIME shown to user matches START_DATE
                    # This visually represents our honoring of the START_DATE constraint
                    family_time_shifts[family] = 0
                    job['family_time_shift'] = 0
                    job['START_TIME'] = job['START_DATE_EPOCH']
                    
                    logger.info(f"Setting START_TIME to equal START_DATE for {job_id} for visualization")
    
    # Identify time shifts for each family with START_DATE constraints
    for family in start_date_families:
        if family not in family_time_shifts:  # Skip if already processed due to violations
            for job in family_groups.get(family, []):
                job_id = job['PROCESS_CODE']
                if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] > current_time and job_id in scheduled_times:
                    scheduled_start = scheduled_times[job_id][0]
                    requested_start = job['START_DATE_EPOCH']
                    time_shift = scheduled_start - requested_start
                    
                    # If significant time shift, record it
                    if abs(time_shift) > 60:  # More than a minute
                        if family not in family_time_shifts or abs(time_shift) > abs(family_time_shifts[family]):
                            family_time_shifts[family] = time_shift
                            logger.info(f"Family {family} has time shift of {time_shift/3600:.1f} hours based on {job_id}")
    
    # Store time shifts in job dictionaries for visualization
    for family, time_shift in family_time_shifts.items():
        logger.info(f"Adding time shift of {time_shift/3600:.1f} hours to all jobs in family {family}")
        for job in family_groups.get(family, []):
            job['family_time_shift'] = time_shift
    
    # For jobs with START_DATE, make sure START_TIME = START_DATE_EPOCH for visualization
    for job in jobs:
        if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None:
            # Always set START_TIME equal to START_DATE_EPOCH for jobs with this constraint
            job['START_TIME'] = job['START_DATE_EPOCH']
    
    logger.info(f"Enhanced greedy schedule created with {sum(len(tasks) for tasks in schedule.values())} jobs")
    
    return schedule

if __name__ == "__main__":
    # Use real data from file_path
    from ingest_data import load_jobs_planning_data
    
    # Load environment variables
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