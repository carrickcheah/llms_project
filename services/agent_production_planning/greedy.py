# greedy.py | dont edit this line
import logging
from datetime import datetime
import re
from dotenv import load_dotenv
import os

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

def greedy_schedule(jobs, machines, setup_times=None, enforce_sequence=True, max_operators=0):
    """
    Create a schedule using an enhanced greedy algorithm with process sequence enforcement,
    considering due dates and earliest start times (START_DATE).

    Args:
        jobs (list): List of job dictionaries, each with UNIQUE_JOB_ID
        machines (list): List of machine IDs
        setup_times (dict): Dictionary mapping (unique_job_id1, unique_job_id2) to setup duration
        enforce_sequence (bool): Whether to enforce process sequence dependencies
        max_operators (int): Maximum number of operators available at any time (0 means no limit)

    Returns:
        dict: Schedule as {machine: [(unique_job_id, start, end, priority), ...]}
    """
    current_time = int(datetime.now().timestamp())
    schedule = {machine: [] for machine in machines}
    machine_availability = {machine: current_time for machine in machines}
    
    # Track operator usage over time if max_operators is set
    if max_operators > 0:
        logger.info(f"Enforcing maximum operator constraint: {max_operators} operators")
        # Dictionary to track how many operators are in use at any given time point
        operator_usage = {}  # {time_point: number_of_operators_in_use}
    else:
        logger.info("No maximum operator constraint enforced")
    
    # Validate jobs and fix any issues
    for job in jobs:
        # Check for UNIQUE_JOB_ID
        if 'UNIQUE_JOB_ID' not in job:
            logger.error(f"Job missing UNIQUE_JOB_ID: {job}")
            continue

        # Check for valid processing_time
        if not isinstance(job.get('processing_time', 0), (int, float)) or job.get('processing_time', 0) <= 0:
            logger.warning(f"Invalid processing_time for job {job['UNIQUE_JOB_ID']}: {job.get('processing_time')}")
            job['processing_time'] = 3600  # Default to 1 hour
            logger.info(f"Set default processing_time of 3600 seconds (1 hour) for job {job['UNIQUE_JOB_ID']}")
        
        # Ensure machine ID is available
        if not job.get('RSC_CODE') and not job.get('MACHINE_ID'):
            logger.warning(f"No machine assignment for job {job['UNIQUE_JOB_ID']}")
            continue
        
        # Ensure PRIORITY is valid
        if not isinstance(job.get('PRIORITY'), (int, float)) or not 1 <= job.get('PRIORITY', 3) <= 5:
            logger.warning(f"Invalid priority for job {job['UNIQUE_JOB_ID']}: {job.get('PRIORITY')}")
            job['PRIORITY'] = 3  # Default to medium priority

    # Step 1: Group jobs by family and process number
    family_groups = {}
    for job in jobs:
        if 'UNIQUE_JOB_ID' not in job:
            continue
        family = extract_job_family(job['UNIQUE_JOB_ID'])
        if family not in family_groups:
            family_groups[family] = []
        family_groups[family].append(job)

    # Sort processes within each family
    for family in family_groups:
        family_groups[family].sort(key=lambda x: extract_process_number(x['UNIQUE_JOB_ID']))

    # Identify families with START_DATE constraints
    start_date_families = set()
    start_date_jobs = []
    
    for job in jobs:
        if 'UNIQUE_JOB_ID' not in job:
            continue
        if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and job['START_DATE_EPOCH'] > current_time:
            family = extract_job_family(job['UNIQUE_JOB_ID'])
            start_date_families.add(family)
            start_date_jobs.append(job)
            job_start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
            logger.info(f"Job {job['UNIQUE_JOB_ID']} must start exactly at START_DATE={job_start_date}")
    
    if start_date_jobs:
        logger.info(f"Found {len(start_date_jobs)} jobs with future START_DATE constraints in {len(start_date_families)} families")
    
    # Create a mapping of unique_job_ids to process sequence numbers
    process_to_seq = {}
    for job in jobs:
        if 'UNIQUE_JOB_ID' not in job:
            continue
        unique_job_id = job['UNIQUE_JOB_ID']
        process_to_seq[unique_job_id] = extract_process_number(unique_job_id)
    
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
        if 'UNIQUE_JOB_ID' not in job:
            continue
        unique_job_id = job['UNIQUE_JOB_ID']
        machine_id = job.get('RSC_CODE', job.get('MACHINE_ID'))
        duration = job['processing_time']
        due_time = job.get('LCD_DATE_EPOCH', current_time + 30 * 24 * 3600)
        family = extract_job_family(unique_job_id)
        
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
            
            if start_date_constraint > current_time:
                start_date_str = datetime.fromtimestamp(start_date_constraint).strftime('%Y-%m-%d %H:%M')
                logger.info(f"ENFORCING EXACT START_DATE for job {unique_job_id}: {start_date_str}")
                
                if machine_constraint > start_date_constraint:
                    logger.info(f"🔄 Adjusting machine availability for {unique_job_id} to ensure START_DATE={start_date_str}")
                    machine_constraint = start_date_constraint
                
                if sequence_constraint > start_date_constraint:
                    logger.info(f"⚠️ START_DATE for {unique_job_id} conflicts with sequence constraints")
                    logger.info(f"🔄 Prioritizing START_DATE over sequence constraint as requested")
                    job['sequence_violation'] = True
        
        # Find the most restrictive constraint
        constraints = {
            "Current Time": time_constraint,
            "Machine Availability": machine_constraint,
            "Sequence Dependency": sequence_constraint,
            "START_DATE (User-defined)": start_date_constraint if has_start_date else current_time
        }
        
        if has_start_date and start_date_constraint > current_time:
            earliest_start = start_date_constraint
            active_constraint = "START_DATE (User-defined)"
        else:
            earliest_start = max(constraints.values())
            active_constraint = [name for name, value in constraints.items() if value == earliest_start][0]
            
        end_time = earliest_start + duration
        num_operators = job.get('NUMBER_OPERATOR', 1)
        
        # Operator constraint check if max_operators is set
        if max_operators > 0 and num_operators > 0:
            # Find a suitable time slot where we don't exceed max_operators
            proposed_start = earliest_start
            can_schedule = False
            
            # Keep trying later start times until we find a feasible slot or hit some limit
            max_attempts = 10  # Limit the search to prevent infinite loops
            attempt = 0
            
            while attempt < max_attempts:
                # Check if adding this job would exceed operator limit at any point
                operator_conflict = False
                
                # For each time point where this job would be active
                for t in range(int(proposed_start), int(proposed_start + duration), 3600):  # Check hourly points
                    current_operators = operator_usage.get(t, 0)
                    if current_operators + num_operators > max_operators:
                        operator_conflict = True
                        logger.info(f"Operator constraint violation at {datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M')}: "
                                    f"Need {num_operators} operators, but only {max_operators - current_operators} available")
                        break
                
                if not operator_conflict:
                    # Found a feasible slot
                    can_schedule = True
                    earliest_start = proposed_start
                    end_time = earliest_start + duration
                    active_constraint = "Operator Availability"
                    break
                
                # Try next feasible time slot (1 hour later)
                proposed_start += 3600
                attempt += 1
            
            if not can_schedule and attempt >= max_attempts:
                logger.warning(f"Could not find feasible time slot for job {unique_job_id} after {max_attempts} attempts due to operator constraints")
                # We'll still schedule it, but note the constraint violation
                job['operator_constraint_violation'] = True
        
        if earliest_start > current_time:
            active_date = datetime.fromtimestamp(earliest_start).strftime('%Y-%m-%d %H:%M')
            logger.info(f"Job {unique_job_id} start time ({active_date}) determined by: {active_constraint}")
        
        # Schedule the job
        start = earliest_start
        end = start + duration

        # Apply setup time if provided, but respect START_DATE if specified
        if setup_times and unique_job_id in setup_times and schedule[machine_id]:
            last_job = schedule[machine_id][-1]
            last_job_end = last_job[2]
            setup_duration = setup_times[unique_job_id].get(last_job[0], 0)
            
            must_honor_start_date = has_start_date and start_date_constraint > current_time and start == start_date_constraint
            
            if start < last_job_end + setup_duration:
                if must_honor_start_date:
                    logger.info(f"⚠️ Skipping setup time for job {unique_job_id} to ensure exact START_DATE constraint is met")
                else:
                    old_start = start
                    start = last_job_end + setup_duration
                    end = start + duration
                    logger.info(f"Applied setup time: Job {unique_job_id} start delayed from {datetime.fromtimestamp(old_start).strftime('%Y-%m-%d %H:%M')} to {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')}")
        
        # Schedule the job
        schedule[machine_id].append((unique_job_id, start, end, job['PRIORITY']))
        machine_availability[machine_id] = end
        
        if enforce_sequence:
            if has_start_date and start_date_constraint > current_time and job.get('sequence_violation', False):
                logger.info(f"Not updating family end time for {unique_job_id} due to sequence violation")
            else:
                family_end_times[family] = end
        
        # Log the final scheduling decision
        job_name = job.get('JOB', job.get('JOB_CODE', unique_job_id))
        logger.info(f"✅ Scheduled job {job_name} ({unique_job_id}) on {machine_id}: "
                   f"{datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')} to {datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')}")

        # Determine setup time if any
        setup_duration = 0
        if setup_times and machine_id in schedule and schedule[machine_id]:
            last_job_id = schedule[machine_id][-1][0]
            setup_key = (last_job_id, unique_job_id)
            if setup_key in setup_times:
                setup_duration = setup_times[setup_key]
                logger.debug(f"Setup time: {setup_duration/60:.1f} min for transition from {last_job_id} to {unique_job_id}")
        
        # Calculate final start and end times
        start_time = earliest_start
        end_time = start_time + duration
        
        # Update operator usage if max_operators is set
        if max_operators > 0 and job.get('NUMBER_OPERATOR', 0) > 0:
            num_operators = job.get('NUMBER_OPERATOR', 1)
            for t in range(int(start_time), int(end_time), 3600):  # Update hourly
                operator_usage[t] = operator_usage.get(t, 0) + num_operators
                logger.debug(f"Time {datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M')}: {operator_usage[t]}/{max_operators} operators in use")
        
        # Update the machine availability
        machine_availability[machine_id] = end_time

    # Calculate time shifts needed for families with START_DATE constraints
    family_time_shifts = {}
    scheduled_times = {}
    
    for machine, tasks in schedule.items():
        for unique_job_id, start, end, _ in tasks:
            scheduled_times[unique_job_id] = (start, end)
    
    for family in start_date_families:
        family_has_violation = False
        for job in family_groups.get(family, []):
            if job.get('sequence_violation', False):
                family_has_violation = True
                break
        
        if family_has_violation:
            logger.info(f"Family {family} has sequence violations due to START_DATE constraints")
            for job in family_groups.get(family, []):
                unique_job_id = job['UNIQUE_JOB_ID']
                if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and unique_job_id in scheduled_times:
                    family_time_shifts[family] = 0
                    job['family_time_shift'] = 0
                    job['START_TIME'] = job['START_DATE_EPOCH']
                    logger.info(f"Setting START_TIME to equal START_DATE for {unique_job_id} for visualization")
    
    for family in start_date_families:
        if family not in family_time_shifts:
            for job in family_groups.get(family, []):
                unique_job_id = job['UNIQUE_JOB_ID']
                if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] > current_time and unique_job_id in scheduled_times:
                    scheduled_start = scheduled_times[unique_job_id][0]
                    requested_start = job['START_DATE_EPOCH']
                    time_shift = scheduled_start - requested_start
                    
                    if abs(time_shift) > 60:
                        if family not in family_time_shifts or abs(time_shift) > abs(family_time_shifts[family]):
                            family_time_shifts[family] = time_shift
                            logger.info(f"Family {family} has time shift of {time_shift/3600:.1f} hours based on {unique_job_id}")
    
    for family, time_shift in family_time_shifts.items():
        logger.info(f"Adding time shift of {time_shift/3600:.1f} hours to all jobs in family {family}")
        for job in family_groups.get(family, []):
            job['family_time_shift'] = time_shift
    
    for job in jobs:
        if 'UNIQUE_JOB_ID' not in job:
            continue
        if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None:
            job['START_TIME'] = job['START_DATE_EPOCH']
    
    logger.info(f"Enhanced greedy schedule created with {sum(len(tasks) for tasks in schedule.values())} jobs")
    
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