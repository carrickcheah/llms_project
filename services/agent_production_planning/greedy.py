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
    considering due dates.

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
    for job in sorted_jobs:
        job_id = job['PROCESS_CODE']
        machine_id = job['MACHINE_ID']
        duration = job['processing_time']
        due_time = job.get('DUE_DATE_TIME', current_time + 30 * 24 * 3600)  # Default to 30 days if not provided
        family = extract_job_family(job_id)

        # Get the earliest start time considering sequence dependencies and due time
        earliest_start = max(machine_availability[machine_id], current_time)
        if enforce_sequence and family in family_end_times:
            earliest_start = max(earliest_start, family_end_times[family])
        # Ensure start time respects due date (start before due minus processing time)
        earliest_start = min(earliest_start, due_time - duration)

        # Schedule the job
        start = earliest_start
        end = start + duration

        # Apply setup time if provided
        if setup_times and job_id in setup_times and schedule[machine_id]:
            last_job = schedule[machine_id][-1]
            last_job_end = last_job[2]
            setup_duration = setup_times[job_id].get(last_job[0], 0)
            start = max(start, last_job_end + setup_duration)
            end = start + duration

        # Schedule the job
        schedule[machine_id].append((job_id, start, end, job['PRIORITY']))
        machine_availability[machine_id] = end
        if enforce_sequence:
            family_end_times[family] = end
        logger.info(f"Scheduled priority {job['PRIORITY']} job {job.get('JOB_CODE', job_id)} on {machine_id}: "
                    f"{datetime.fromtimestamp(start)} to {datetime.fromtimestamp(end)}")

    logger.info(f"Enhanced greedy schedule created with {sum(len(tasks) for tasks in schedule.values())} jobs")
    schedule_span = max((end for tasks in schedule.values() for _, _, end, _ in tasks), default=current_time) - \
                    min((start for tasks in schedule.values() for _, start, _, _ in tasks), default=current_time)
    logger.info(f"Schedule span: {schedule_span/3600:.2f} hours ({datetime.fromtimestamp(min((start for tasks in schedule.values() for _, start, _, _ in tasks), default=current_time))} to "
                f"{datetime.fromtimestamp(max((end for tasks in schedule.values() for _, _, end, _ in tasks), default=current_time))})")
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