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
    print(f"Extracting sequence from: {process_code}")
    match = re.search(r'P(\d{2})-\d+', str(process_code).upper())  # Match exactly two digits after P
    if match:
        seq = int(match.group(1))
        print(f"Extracted sequence: {seq}")
        return seq
    print("No match found, returning 999")
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
    considering due dates and earliest start times (CUT_Q).

    Args:
        jobs (list): List of job dictionaries (e.g., {'PROCESS_CODE': '...', 'MACHINE_ID': '...', 'processing_time': ..., 'DUE_DATE_TIME': ..., 'PRIORITY': ...})
        machines (list): List of machine IDs
        setup_times (dict): Dictionary mapping (process_code1, process_code2) to setup duration (default: None)
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

    # Check for and log any jobs with CUT_Q (EARLIEST_START_TIME) constraints
    cut_q_jobs = [job for job in jobs if job.get('EARLIEST_START_TIME', current_time) > current_time]
    if cut_q_jobs:
        logger.info(f"Found {len(cut_q_jobs)} jobs with CUT_Q (earliest start) constraints:")
        for job in cut_q_jobs:
            cut_q_date = datetime.fromtimestamp(job['EARLIEST_START_TIME']).strftime('%Y-%m-%d %H:%M')
            logger.info(f"  Job {job['PROCESS_CODE']} (Machine: {job['MACHINE_ID']}): Must start on or after {cut_q_date}")
    else:
        logger.info("No jobs with CUT_Q constraints found in input data")

    # Enforce process sequence dependencies or sort by priority and due date
    if enforce_sequence:
        logger.info("Enforcing process sequence dependencies (P01->P02->P03)...")
        # Group jobs by family
        family_groups = {}
        for job in jobs:
            family = extract_job_family(job['PROCESS_CODE'])
            if family not in family_groups:
                family_groups[family] = []
            family_groups[family].append(job)

        # Sort within each family by sequence
        sorted_jobs = []
        for family, family_jobs in family_groups.items():
            sorted_family_jobs = sorted(family_jobs, key=lambda x: extract_process_number(x['PROCESS_CODE']))
            logger.info(f"Family {family} process sequence: {', '.join(job['PROCESS_CODE'] for job in sorted_family_jobs)}")
            sorted_jobs.extend(sorted_family_jobs)
    else:
        # Sort by priority and due date (earlier due dates get higher priority)
        sorted_jobs = sorted(jobs, key=lambda x: (x['PRIORITY'], x.get('DUE_DATE_TIME', float('inf'))))

    # Track the latest end time for each family to enforce sequence
    family_end_times = {}
    
    # Process each job in the sorted order
    for job in sorted_jobs:
        job_id = job['PROCESS_CODE']
        machine_id = job['MACHINE_ID']
        duration = job['processing_time']
        due_time = job.get('DUE_DATE_TIME', current_time + 30 * 24 * 3600)  # Default to 30 days if not provided
        family = extract_job_family(job_id)
        
        # Get each constraint independently for clarity
        
        # 1. Current time constraint (can't schedule in the past)
        time_constraint = current_time
        
        # 2. Machine availability constraint
        machine_constraint = machine_availability[machine_id]
        
        # 3. Sequence constraint (if applicable)
        sequence_constraint = family_end_times.get(family, current_time) if enforce_sequence else current_time
        
        # 4. CUT_Q constraint (earliest allowed start date from user input)
        cut_q_constraint = job.get('EARLIEST_START_TIME', current_time)
        
        # Find the most restrictive constraint (maximum start time)
        constraints = {
            "Current Time": time_constraint,
            "Machine Availability": machine_constraint,
            "Sequence Dependency": sequence_constraint,
            "CUT_Q (Earliest Start)": cut_q_constraint
        }
        
        # Find the most restrictive constraint
        earliest_start = max(constraints.values())
        active_constraint = [name for name, value in constraints.items() if value == earliest_start][0]
        
        # Log which constraint is controlling this job's start time
        if earliest_start > current_time:
            active_date = datetime.fromtimestamp(earliest_start).strftime('%Y-%m-%d %H:%M')
            logger.info(f"Job {job_id} start time ({active_date}) determined by: {active_constraint}")
            
            # Additional detailed logging for CUT_Q constraints
            if active_constraint == "CUT_Q (Earliest Start)":
                logger.info(f"  🚨 CUT_Q date restriction active for {job_id}: Cannot start before {active_date}")
        
        # Schedule the job
        start = earliest_start
        end = start + duration

        # Apply setup time if provided
        if setup_times and job_id in setup_times and schedule[machine_id]:
            last_job = schedule[machine_id][-1]
            last_job_end = last_job[2]
            setup_duration = setup_times[job_id].get(last_job[0], 0)
            if start < last_job_end + setup_duration:
                old_start = start
                start = last_job_end + setup_duration
                end = start + duration
                logger.info(f"Applied setup time: Job {job_id} start delayed from {datetime.fromtimestamp(old_start).strftime('%Y-%m-%d %H:%M')} to {datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')}")

        # Schedule the job
        schedule[machine_id].append((job_id, start, end, job['PRIORITY']))
        machine_availability[machine_id] = end
        if enforce_sequence:
            family_end_times[family] = end
        
        # Log the final scheduling decision
        logger.info(f"✅ Scheduled job {job.get('JOB_CODE', job_id)} ({job_id}) on {machine_id}: "
                   f"{datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M')} to {datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M')}")

    logger.info(f"Enhanced greedy schedule created with {sum(len(tasks) for tasks in schedule.values())} jobs")
    
    if any(schedule.values()):
        min_start = min((start for tasks in schedule.values() for _, start, _, _ in tasks), default=current_time)
        max_end = max((end for tasks in schedule.values() for _, _, end, _ in tasks), default=current_time)
        schedule_span = max_end - min_start
        
        logger.info(f"Schedule span: {schedule_span/3600:.2f} hours ({datetime.fromtimestamp(min_start).strftime('%Y-%m-%d %H:%M')} to "
                    f"{datetime.fromtimestamp(max_end).strftime('%Y-%m-%d %H:%M')})")
    else:
        logger.warning("No jobs were scheduled!")
    
    return schedule

if __name__ == "__main__":
    # Example usage for testing
    jobs = [
        {'PROCESS_CODE': 'CP08-231B-P01-06', 'MACHINE_ID': 'WS01', 'JOB_CODE': 'JOAW24120317', 'PRIORITY': 2, 'processing_time': 20000, 'DUE_DATE_TIME': 1744848000},
        {'PROCESS_CODE': 'CP08-231B-P02-06', 'MACHINE_ID': 'PP23-060T', 'JOB_CODE': 'JOAW24120317', 'PRIORITY': 2, 'processing_time': 43636, 'DUE_DATE_TIME': 1744848000},
        {'PROCESS_CODE': 'CP08-231B-P03-06', 'MACHINE_ID': 'JIG-HAND BEND', 'JOB_CODE': 'JOAW24120317', 'PRIORITY': 2, 'processing_time': 60000, 'DUE_DATE_TIME': 1744848000},
        {'PROCESS_CODE': 'CP08-231B-P04-06', 'MACHINE_ID': 'PP23-060T', 'JOB_CODE': 'JOAW24120317', 'PRIORITY': 2, 'processing_time': 34325, 'DUE_DATE_TIME': 1744848000},
        {'PROCESS_CODE': 'CP08-231B-P05-06', 'MACHINE_ID': 'PP23', 'JOB_CODE': 'JOAW24120317', 'PRIORITY': 2, 'processing_time': 60000, 'DUE_DATE_TIME': 1744848000},
    ]
    machines = ['WS01', 'PP23-060T', 'JIG-HAND BEND', 'PP23']
    setup_times = {job['PROCESS_CODE']: {prev_job['PROCESS_CODE']: 0 for prev_job in jobs if prev_job != job} for job in jobs}  # Placeholder setup times
    schedule = greedy_schedule(jobs, machines, setup_times, enforce_sequence=True)
    print(schedule)