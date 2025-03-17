# greedy.py
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def greedy_schedule(jobs, machines, setup_times=None):
    """
    Enhanced greedy scheduling algorithm with machine balancing and setup time consideration.
    Supports both standard and extended job formats.
    
    Args:
        jobs: List of tuples:
             Standard format: (job_name, process_code, machine, duration, due_time, priority, start_time)
             Extended format: (job_name, process_code, machine, duration, due_time, priority, 
                              start_time, num_operators, output_rate, planned_end)
        machines: List of tuples (id, machine_name, capacity)
        setup_times: Optional dictionary mapping (process_code1, process_code2) to setup duration
    
    Returns:
        Dictionary mapping machine IDs to lists of scheduled jobs (job_name, start_time, end_time, priority, 
        [planned_start], [planned_end])
    """
    logger.info("Using enhanced greedy scheduling algorithm as fallback")
    current_time = int(datetime.now().timestamp())
    
    # Filter valid jobs
    valid_jobs = []
    for job in jobs:
        # Handle different job tuple formats
        if len(job) >= 10:  # Extended format with planned_end
            job_name, process_code, machine, duration, due_time, priority, start_time, num_operators, output_rate, planned_end = job
        elif len(job) >= 9:  # Extended format without planned_end
            job_name, process_code, machine, duration, due_time, priority, start_time, num_operators, output_rate = job
            planned_end = None
        elif len(job) >= 7:  # Standard format
            job_name, process_code, machine, duration, due_time, priority, start_time = job
            num_operators = 1
            output_rate = 100.0
            planned_end = None
        else:
            logger.warning(f"Invalid job format with {len(job)} elements, skipping: {job}")
            continue
            
        if duration <= 0 or not machine:
            logger.warning(f"Skipping job {job_name} due to invalid duration ({duration}) or missing machine")
            continue
            
        # Store in standardized format
        valid_jobs.append((job_name, process_code, machine, duration, due_time, priority, start_time, 
                          num_operators, output_rate, planned_end))
    
    # Prioritize jobs by a weighted score of priority and due date
    # Priority 1 (highest) gets highest weight
    def job_score(job):
        _, _, _, _, due_time, priority, _, _, _, _ = job
        days_to_due = max(0, (due_time - current_time) / (24 * 3600))
        # Priority is 1-5 where 1 is highest, convert to 5-1 where 5 is highest
        priority_weight = 6 - priority 
        urgency = max(1, 30 - days_to_due)  # Higher urgency if fewer days to due date
        return (priority_weight * 10) * urgency  # Weighted score favoring high priority and close due dates
    
    sorted_jobs = sorted(valid_jobs, key=job_score, reverse=True)
    
    # Initialize schedule data structures
    machine_end_times = {m[1]: current_time for m in machines if m[1]}
    machine_last_process = {}  # To track setup times
    schedule = {}
    
    # Maximum number of jobs to consider (safety cap)
    max_jobs = min(1000, len(sorted_jobs))
    
    # Schedule jobs one by one
    for job in sorted_jobs[:max_jobs]:
        job_name, process_code, machine, duration, due_time, priority, earliest_start, num_operators, output_rate, planned_end = job
        
        if machine not in machine_end_times:
            logger.warning(f"Machine {machine} not found in available machines, skipping job {job_name}")
            continue
        
        # Consider setup time if provided
        setup_time = 0
        if setup_times and process_code and machine in machine_last_process:
            last_process = machine_last_process.get(machine)
            if last_process and last_process in setup_times and process_code in setup_times[last_process]:
                setup_time = setup_times[last_process][process_code]
                logger.debug(f"Added setup time of {setup_time} seconds between {last_process} and {process_code}")
        
        # Calculate start and end times
        start_time = max(machine_end_times[machine] + setup_time, earliest_start)
        end_time = start_time + duration
        
        # Add to schedule
        if machine not in schedule:
            schedule[machine] = []
        
        # Include all available information in the job data
        if planned_end:
            schedule[machine].append((job_name, start_time, end_time, priority, earliest_start, planned_end))
        else:
            schedule[machine].append((job_name, start_time, end_time, priority, earliest_start, end_time))
            
        machine_end_times[machine] = end_time
        machine_last_process[machine] = process_code
        
        # Print debugging information for high-priority jobs
        if priority <= 2:  # Priority 1 or 2
            logger.info(f"Scheduled priority {priority} job {job_name} on {machine}: " +
                      f"{datetime.fromtimestamp(start_time)} to {datetime.fromtimestamp(end_time)}")
    
    # Sort jobs by start time for each machine
    for machine in schedule:
        schedule[machine].sort(key=lambda x: x[1])
    
    total_jobs = sum(len(jobs) for jobs in schedule.values())
    logger.info(f"Enhanced greedy schedule created with {total_jobs} jobs")
    
    # Calculate schedule statistics
    if total_jobs > 0:
        earliest_start = min(job[1] for machine_jobs in schedule.values() for job in machine_jobs)
        latest_end = max(job[2] for machine_jobs in schedule.values() for job in machine_jobs)
        total_span = latest_end - earliest_start
        logger.info(f"Schedule span: {total_span/3600:.2f} hours ({datetime.fromtimestamp(earliest_start)} to {datetime.fromtimestamp(latest_end)})")
        
        # Calculate machine utilization
        machine_loads = {}
        for machine, jobs_list in schedule.items():
            total_duration = sum(job[2] - job[1] for job in jobs_list)
            machine_loads[machine] = total_duration
            
        if machine_loads:
            avg_load = sum(machine_loads.values()) / len(machine_loads)
            most_loaded = max(machine_loads.items(), key=lambda x: x[1])
            logger.info(f"Average machine load: {avg_load/3600:.1f} hours")
            logger.info(f"Most loaded machine: {most_loaded[0]} ({most_loaded[1]/3600:.1f} hours)")
        
        # Calculate due date compliance
        late_jobs = []
        for machine, jobs_list in schedule.items():
            for job_data in jobs_list:
                job_name, start_time, end_time, priority = job_data[:4]
                # Find the corresponding job in sorted_jobs to get the due_time
                for original_job in sorted_jobs:
                    if original_job[0] == job_name and original_job[2] == machine:
                        due_time = original_job[4]
                        if end_time > due_time:
                            late_jobs.append((job_name, machine, priority, (end_time - due_time)/3600))
                        break
        
        if late_jobs:
            logger.warning(f"{len(late_jobs)} jobs will be late:")
            for job_name, machine, priority, hours_late in sorted(late_jobs, key=lambda x: (x[2], -x[3]))[:5]:
                logger.warning(f"  - {job_name} on {machine}: {hours_late:.1f} hours late (Priority {priority})")
            if len(late_jobs) > 5:
                logger.warning(f"  - ... and {len(late_jobs) - 5} more")
    
    return schedule