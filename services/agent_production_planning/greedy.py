# greedy.py (modified to use PROCESS_CODE in output and enforce process sequence)
from datetime import datetime
import re

def extract_process_number(process_code):
    """Extract process number (e.g., 1 from CP08-231B-P01-06 or cp08-231-P01)"""
    # Convert to uppercase for consistency
    process_code = process_code.upper()
    
    # Look for the pattern P followed by 2 digits
    match = re.search(r'P(\d{2})', process_code)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return 999
    return 999

def extract_job_family(process_code):
    """Extract job family code (e.g., CP08-231B from CP08-231B-P01-06)"""
    # Convert to uppercase for consistency
    process_code = process_code.upper()
    
    match = re.search(r'(.*?)-P\d+', process_code)
    if match:
        return match.group(1)
    parts = process_code.split("-P")
    if len(parts) >= 2:
        return parts[0]
    return process_code

def greedy_schedule(jobs, machines, setup_times=None):
    """
    Enhanced greedy scheduling algorithm with machine balancing and setup time consideration.
    Enforces process code sequence dependencies (P01->P02->P03).
    
    Args:
        jobs: List of tuples (job_name, process_code, machine, duration, due_time, priority, start_time)
        machines: List of tuples (id, machine_name, capacity)
        setup_times: Optional dictionary mapping (process_code1, process_code2) to setup duration
    
    Returns:
        Dictionary mapping machine IDs to lists of scheduled jobs (process_code, start_time, end_time, priority)
    """
    print("Using enhanced greedy scheduling algorithm with process sequence enforcement")
    current_time = int(datetime.now().timestamp())
    
    # Filter valid jobs
    valid_jobs = []
    for job in jobs:
        if len(job) < 6:
            continue
            
        # Extract key job information
        job_name = job[0] if len(job) > 0 else ""
        process_code = job[1] if len(job) > 1 else ""
        machine = job[2] if len(job) > 2 else ""
        duration = job[3] if len(job) > 3 else 0
        due_time = job[4] if len(job) > 4 else 0
        priority = job[5] if len(job) > 5 else 5
        earliest_start = job[6] if len(job) > 6 else int(datetime.now().timestamp())
        
        if duration <= 0 or not machine:
            continue
            
        # Extract sequence information
        family = extract_job_family(process_code)
        sequence = extract_process_number(process_code)
        
        # Create expanded job tuple with family and sequence information
        valid_job = (job_name, process_code, machine, duration, due_time, priority, 
                     earliest_start, family, sequence)
        valid_jobs.append(valid_job)
    
    # Group jobs by family
    job_families = {}
    for job in valid_jobs:
        family = job[7]  # family is at index 7
        if family not in job_families:
            job_families[family] = []
        job_families[family].append(job)
    
    # Sort each family by sequence number and set earliest_start times
    # to ensure P01 finishes before P02 can start
    consolidated_jobs = []
    for family, family_jobs in job_families.items():
        # Sort by sequence number
        sorted_jobs = sorted(family_jobs, key=lambda j: j[8])  # sequence is at index 8
        
        # Set cascading earliest start times
        prev_end_time = current_time
        for i, job in enumerate(sorted_jobs):
            job_name, process_code, machine, duration, due_time, priority, _, family, sequence = job
            
            # Set earliest start time to ensure sequence dependency
            earliest_start = max(current_time, prev_end_time)
            
            # Update for next job in sequence
            prev_end_time = earliest_start + duration
            
            # Re-create job tuple with updated earliest_start
            updated_job = (job_name, process_code, machine, duration, due_time, priority, earliest_start)
            consolidated_jobs.append(updated_job)
            
            print(f"Sequencing: {family} - {process_code} (seq {sequence}) can start after {datetime.fromtimestamp(earliest_start)}")
    
    # Prioritize jobs by a weighted score of priority and due date
    # Priority 1 (highest) gets highest weight
    def job_score(job):
        _, _, _, _, due_time, priority, _ = job
        days_to_due = max(0, (due_time - current_time) / (24 * 3600))
        # Priority is 1-5 where 1 is highest, convert to 5-1 where 5 is highest
        priority_weight = 6 - priority 
        urgency = max(1, 30 - days_to_due)  # Higher urgency if fewer days to due date
        return (priority_weight * 10) * urgency  # Weighted score favoring high priority and close due dates
    
    sorted_jobs = sorted(consolidated_jobs, key=job_score, reverse=True)
    
    # Initialize schedule data structures
    machine_end_times = {m[1]: current_time for m in machines if m[1]}
    machine_last_process = {}  # To track setup times
    schedule = {}
    
    # Maximum number of jobs to consider (safety cap)
    max_jobs = min(1000, len(sorted_jobs))
    
    # Schedule jobs one by one
    for job in sorted_jobs[:max_jobs]:
        job_name, process_code, machine, duration, due_time, priority, earliest_start = job
        
        if machine not in machine_end_times:
            continue
        
        # Consider setup time if provided
        setup_time = 0
        if setup_times and process_code and machine in machine_last_process:
            last_process = machine_last_process.get(machine)
            if last_process and last_process in setup_times and process_code in setup_times[last_process]:
                setup_time = setup_times[last_process][process_code]
        
        # Calculate start and end times
        start_time = max(machine_end_times[machine] + setup_time, earliest_start)
        end_time = start_time + duration
        
        # Add to schedule
        if machine not in schedule:
            schedule[machine] = []
        
        # MODIFIED: Use process_code instead of job_name as the first element in the tuple
        schedule[machine].append((process_code, start_time, end_time, priority))
        machine_end_times[machine] = end_time
        machine_last_process[machine] = process_code
        
        # Print debugging information for high-priority jobs
        if priority <= 2:  # Priority 1 or 2
            print(f"Scheduled priority {priority} job {job_name} on {machine}: " +
                  f"{datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}")
    
    # Sort jobs by start time for each machine
    for machine in schedule:
        schedule[machine].sort(key=lambda x: x[1])
    
    total_jobs = sum(len(jobs) for jobs in schedule.values())
    print(f"Enhanced greedy schedule created with {total_jobs} jobs")
    
    # Calculate schedule statistics
    if total_jobs > 0:
        earliest_start = min(job[1] for machine_jobs in schedule.values() for job in machine_jobs)
        latest_end = max(job[2] for machine_jobs in schedule.values() for job in machine_jobs)
        total_span = latest_end - earliest_start
        print(f"Schedule span: {total_span/3600:.2f} hours ({datetime.fromtimestamp(earliest_start)} to {datetime.fromtimestamp(latest_end)})")
    
    return schedule