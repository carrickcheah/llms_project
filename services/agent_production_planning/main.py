# main.py | dont edit this line
import logging
import time
import argparse
import os
from datetime import datetime
import pandas as pd
import re
from dotenv import load_dotenv
from ingest_data import load_jobs_planning_data
from sch_jobs import schedule_jobs
from greedy import greedy_schedule, extract_job_family, extract_process_number
from chart import create_interactive_gantt
from chart_two import export_schedule_html

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def add_schedule_times_and_buffer(jobs, schedule):
    """
    Add schedule times (START_TIME and END_TIME) to job dictionaries and
    calculate the buffer time (BAL_HR) between job completion and deadline.
    Also adjusts times for dependent processes to maintain proper sequence.
    
    Args:
        jobs (list): List of job dictionaries
        schedule (dict): Schedule as {machine: [(process_code, start, end, priority), ...]}
        
    Returns:
        list: Updated jobs list with START_TIME, END_TIME, and BAL_HR added
    """
    # Create a lookup of start and end times from the schedule
    times = {}
    for machine, tasks in schedule.items():
        for task in tasks:
            process_code, start, end, _ = task
            times[process_code] = (start, end)
    
    # Step 1: Create a mapping of job families and their processes in sequence
    family_processes = {}
    for job in jobs:
        process_code = job['PROCESS_CODE']
        family = extract_job_family(process_code)
        seq_num = extract_process_number(process_code)
        
        if family not in family_processes:
            family_processes[family] = []
        
        family_processes[family].append((seq_num, process_code, job))
    
    # Sort processes within each family by sequence number
    for family in family_processes:
        family_processes[family].sort(key=lambda x: x[0])
    
    # Step 2: Identify families that have START_DATE constraints and calculate time shifts
    family_time_shifts = {}
    for family, processes in family_processes.items():
        # Check if any process in this family has a START_DATE
        for seq_num, process_code, job in processes:
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and not pd.isna(job['START_DATE_EPOCH']) and process_code in times:
                # Calculate the time shift (positive for earlier, negative for later)
                scheduled_start = times[process_code][0]
                
                # Get the START_DATE_EPOCH value (checking both fields)
                requested_start = None
                if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None:
                    if isinstance(job['START_DATE_EPOCH'], float) and not pd.isna(job['START_DATE_EPOCH']):
                        requested_start = job['START_DATE_EPOCH']
                    elif not isinstance(job['START_DATE_EPOCH'], float):
                        requested_start = job['START_DATE_EPOCH']
                        
                if requested_start is None and 'START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None:
                    if isinstance(job['START_DATE _EPOCH'], float) and not pd.isna(job['START_DATE _EPOCH']):
                        requested_start = job['START_DATE _EPOCH']
                    elif not isinstance(job['START_DATE _EPOCH'], float):
                        requested_start = job['START_DATE _EPOCH']
                        
                # Only calculate time shift for valid values
                time_shift = None
                if requested_start is not None:
                    time_shift = scheduled_start - requested_start
                    
                    # Store the time shift for this family
                    if family not in family_time_shifts or abs(time_shift) > abs(family_time_shifts[family]):
                        family_time_shifts[family] = time_shift
                
                if time_shift is not None:
                    logger.info(f"Family {family} has START_DATE constraint for {process_code}: " 
                              f"shift={time_shift/3600:.1f} hours")
                else:
                    logger.info(f"Family {family} has START_DATE constraint for {process_code}, but no valid time shift calculated")
    
    # Step 3: Apply time shifts to all processes in affected families
    job_adjustments = {}  # Store adjusted times for each job
    
    for family, time_shift in family_time_shifts.items():
        if abs(time_shift) < 60:  # Skip tiny shifts (less than a minute)
            continue
            
        logger.info(f"Applying time shift of {time_shift/3600:.1f} hours to family {family} for visualization")
        
        for seq_num, process_code, job in family_processes[family]:
            if process_code in times:
                original_start, original_end = times[process_code]
                
                # Apply the time shift, checking for valid values first
                # Skip NaN or invalid values
                if time_shift is None or (isinstance(time_shift, float) and (pd.isna(time_shift) or not pd.notna(time_shift))):
                    logger.warning(f"Skipping time shift for {process_code} due to invalid shift value: {time_shift}")
                    continue
                    
                # Only apply time shift to valid timestamps
                try:
                    adjusted_start = original_start - time_shift
                    adjusted_end = original_end - time_shift
                    
                    # Store the adjusted times
                    job_adjustments[process_code] = (adjusted_start, adjusted_end)
                    
                    logger.info(f"  Adjusted {process_code}: START={datetime.fromtimestamp(adjusted_start).strftime('%Y-%m-%d %H:%M')}, "
                              f"END={datetime.fromtimestamp(adjusted_end).strftime('%Y-%m-%d %H:%M')}")
                except Exception as e:
                    logger.warning(f"Error adjusting time for {process_code}: {e}")
    
    # Step 4: Update job dictionaries with adjusted times
    for job in jobs:
        process_code = job['PROCESS_CODE']
        if process_code in times:
            original_start, original_end = times[process_code]
            due_time = job.get('LCD_DATE_EPOCH', 0)
            
            # Use adjusted times if available, otherwise use original times
            if process_code in job_adjustments:
                job_start, job_end = job_adjustments[process_code]
            else:
                job_start = original_start
                job_end = original_end
            
            # Override with exact START_DATE if specified for any job with this constraint
            # Check both formats of the START_DATE field
            start_date_epoch = None
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and not pd.isna(job['START_DATE_EPOCH']):
                start_date_epoch = job['START_DATE_EPOCH']
            elif 'START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None and not pd.isna(job['START_DATE _EPOCH']):
                start_date_epoch = job['START_DATE _EPOCH']
                
            if start_date_epoch is not None:
                # CRITICAL FIX: Always prioritize START_DATE
                job_start = start_date_epoch
                # Adjust end time to maintain the same duration
                duration = original_end - original_start
                job_end = job_start + duration
                
                # For logging
                start_date = datetime.fromtimestamp(start_date_epoch).strftime('%Y-%m-%d %H:%M')
                logger.info(f"Setting START_TIME to match START_DATE ({start_date}) for job {process_code}")
            
            # Make sure start and end times are valid numbers, not NaN
            if job_start is not None and not pd.isna(job_start) and isinstance(job_start, (int, float)):
                job['START_TIME'] = job_start
            else:
                # If invalid, set a default start time (current time)
                job['START_TIME'] = int(datetime.now().timestamp())
                logger.warning(f"Set default START_TIME for job {process_code} due to invalid value: {job_start}")
                
            if job_end is not None and not pd.isna(job_end) and isinstance(job_end, (int, float)):
                job['END_TIME'] = job_end
            else:
                # If invalid, set default end time (start + 1 hour)
                job['END_TIME'] = job['START_TIME'] + 3600
                logger.warning(f"Set default END_TIME for job {process_code} due to invalid value: {job_end}")
            
            # Recalculate to ensure we have valid values
            job_start = job['START_TIME'] 
            job_end = job['END_TIME']
            
            # Calculate buffer in hours (with improved validation)
            valid_due_time = False
            if due_time is not None and not pd.isna(due_time) and isinstance(due_time, (int, float)):
                # Check if due time is reasonable (not too far in past or future)
                current_time = int(datetime.now().timestamp())
                # Only use due dates that are after the job's end time or at most 1 year in the future
                if job_end <= due_time <= (current_time + 365 * 24 * 3600):
                    buffer_seconds = max(0, due_time - job_end)
                    buffer_hours = buffer_seconds / 3600
                    valid_due_time = True
                elif due_time < job_end:
                    # Due date is before job end - LATE!
                    buffer_seconds = 0
                    buffer_hours = 0
                    valid_due_time = True
                    logger.warning(f"Job {process_code} will be LATE! Due at {datetime.fromtimestamp(due_time).strftime('%Y-%m-%d %H:%M')} but ends at {datetime.fromtimestamp(job_end).strftime('%Y-%m-%d %H:%M')}")
                else:
                    # Due date too far in future, might be incorrect
                    logger.warning(f"Due date for {process_code} is too far in future ({datetime.fromtimestamp(due_time).strftime('%Y-%m-%d %H:%M')}), might be incorrect")
            
            if not valid_due_time:
                # Set a reasonable default buffer for invalid due dates
                buffer_seconds = 24 * 3600  # 24 hours default
                buffer_hours = 24.0
                logger.warning(f"Set default BAL_HR for job {process_code} due to invalid LCD_DATE_EPOCH: {due_time}")
            
            # No capping - return true buffer value regardless of size
            if buffer_hours > 720:  # Just log large values (over 30 days) but don't cap them
                logger.info(f"Job {process_code} has a large buffer of {buffer_hours:.1f} hours ({buffer_hours/24:.1f} days)")
            
            job['BAL_HR'] = buffer_hours
            job['BUFFER_STATUS'] = get_buffer_status(buffer_hours)
    
    return jobs

def get_buffer_status(buffer_hours):
    """Get status category based on buffer hours."""
    if buffer_hours < 8:
        return "Critical"
    elif buffer_hours < 24:
        return "Warning"
    elif buffer_hours < 72:
        return "Caution"
    else:
        return "OK"

def main():
    print("Production Planning Scheduler")

    # Load environment variables
    load_dotenv()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Production Planning Scheduler")
    parser.add_argument("--file", default=os.getenv('file_path'), 
                       help="Path to the Excel file with job data")
    parser.add_argument("--max-jobs", type=int, default=500, help="Maximum number of jobs to schedule (default: 250)")
    parser.add_argument("--force-greedy", action="store_true", help="Force the use of the greedy scheduler")
    parser.add_argument("--output", default="interactive_schedule.html", help="Output file for the Gantt chart")
    parser.add_argument("--enforce-sequence", action="store_true", default=True, help="Enforce process sequence dependencies (default: True)")
    args = parser.parse_args()

    if not args.file:
        logger.error("No file path provided. Set --file argument or file_path in .env file.")
        return

    print(f"Configuration: file={args.file}, max_jobs={args.max_jobs}, force_greedy={args.force_greedy}, "
          f"output={args.output}, enforce_sequence={args.enforce_sequence}")

    # Define current time for consistent timestamp usage
    current_time = int(datetime.now().timestamp())

    # Load job data using load_jobs_planning_data (returns jobs, machines, setup_times)
    logger.info(f"Loading job data from {args.file}...")
    try:
        jobs, machines, setup_times = load_jobs_planning_data(args.file)
    except Exception as e:
        logger.error(f"Failed to load job data: {e}")
        return

    # Sort jobs by LCD_DATE to implement FIFO (First In, First Out)
    jobs.sort(key=lambda job: job.get('LCD_DATE_EPOCH', 0))
    logger.info(f"Sorted {len(jobs)} jobs by LCD_DATE (First In, First Out)")

    # Limit to max_jobs
    if len(jobs) > args.max_jobs:
        logger.warning(f"Number of jobs ({len(jobs)}) exceeds max_jobs ({args.max_jobs}). Limiting to {args.max_jobs} jobs.")
        jobs = jobs[:args.max_jobs]

    logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines")

    # Check for START_DATE values and log them (supporting both field name formats)
    start_date_jobs = [job for job in jobs if 
                      ('START_DATE_EPOCH' in job and job.get('START_DATE_EPOCH') is not None and not pd.isna(job.get('START_DATE_EPOCH')) and job.get('START_DATE_EPOCH') > current_time) or 
                      ('START_DATE _EPOCH' in job and job.get('START_DATE _EPOCH') is not None and not pd.isna(job.get('START_DATE _EPOCH')) and job.get('START_DATE _EPOCH') > current_time)]
    if start_date_jobs:
        logger.info(f"Found {len(start_date_jobs)} jobs with START_DATE constraints:")
        for job in start_date_jobs:
            # Get START_DATE_EPOCH from either field format
            start_date_epoch = None
            if 'START_DATE_EPOCH' in job and job.get('START_DATE_EPOCH') is not None and not pd.isna(job.get('START_DATE_EPOCH')):
                start_date_epoch = job.get('START_DATE_EPOCH')
            elif 'START_DATE _EPOCH' in job and job.get('START_DATE _EPOCH') is not None and not pd.isna(job.get('START_DATE _EPOCH')):
                start_date_epoch = job.get('START_DATE _EPOCH')
                
            start_date = datetime.fromtimestamp(start_date_epoch).strftime('%Y-%m-%d %H:%M') if start_date_epoch is not None else 'INVALID DATE'
            resource_location = job.get('RSC_LOCATION') or job.get('MACHINE_ID')  # Try both column names
            logger.info(f"  Job {job['PROCESS_CODE']} (Resource: {resource_location}): MUST start EXACTLY at {start_date}")
        
        # Ensure START_DATE constraints are strictly enforced
        logger.info("Ensuring START_DATE constraints are enforced...")
        
        # Handle all forms of START_DATE in jobs
        for job in jobs:
            # Check for both field name formats
            start_date_epoch = None
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and not pd.isna(job['START_DATE_EPOCH']):
                start_date_epoch = job['START_DATE_EPOCH']
            elif 'START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None and not pd.isna(job['START_DATE _EPOCH']):
                # Copy to standard format for consistency
                start_date_epoch = job['START_DATE _EPOCH']
                job['START_DATE_EPOCH'] = start_date_epoch
            
            # Only enforce START_DATE constraints if they were actually provided
            if start_date_epoch is not None:
                # Only enforce constraints that are in the future
                if start_date_epoch > current_time:
                    logger.info(f"ENFORCING START_DATE {datetime.fromtimestamp(start_date_epoch).strftime('%Y-%m-%d %H:%M')} for {job['PROCESS_CODE']}")
            else:
                # If START_DATE wasn't provided in input, remove it from job dictionary if present
                if 'START_DATE_EPOCH' in job:
                    del job['START_DATE_EPOCH']
                    logger.debug(f"Removed empty START_DATE_EPOCH for {job['PROCESS_CODE']}")
                if 'START_DATE _EPOCH' in job:
                    del job['START_DATE _EPOCH']
                    logger.debug(f"Removed empty START_DATE _EPOCH for {job['PROCESS_CODE']}")

    # Schedule jobs with improved error handling and performance monitoring
    start_time = time.time()
    schedule = None

    # Step 1: Try CP-SAT solver first (if not forced to use greedy)
    if not args.force_greedy:
        logger.info("Attempting to create schedule with CP-SAT solver...")
        cp_sat_start_time = time.time()
        try:
            # Set a reasonable timeout for CP-SAT based on job count
            # More jobs need more time to solve
            if len(jobs) > 200:
                time_limit = 900  # 15 minutes for large job sets
            elif len(jobs) > 100:
                time_limit = 600  # 10 minutes for medium job sets
            else:
                time_limit = 300  # 5 minutes for small job sets
                
            logger.info(f"Using {time_limit} seconds time limit for CP-SAT solver with {len(jobs)} jobs")
            
            # Validate input data before passing to solver
            valid_jobs = []
            for job in jobs:
                if 'PROCESS_CODE' not in job:
                    logger.warning(f"Job missing PROCESS_CODE field, skipping: {job}")
                    continue
                    
                if not job.get('RSC_LOCATION') and not job.get('MACHINE_ID'):
                    logger.warning(f"Job {job['PROCESS_CODE']} has no machine assignment, skipping")
                    continue
                    
                if not isinstance(job.get('processing_time', 0), (int, float)) or job.get('processing_time', 0) <= 0:
                    logger.warning(f"Job {job['PROCESS_CODE']} has invalid processing time, defaulting to 3600 seconds (1 hour)")
                    job['processing_time'] = 3600
                    
                valid_jobs.append(job)
                
            if len(valid_jobs) < len(jobs):
                logger.warning(f"Filtered out {len(jobs) - len(valid_jobs)} invalid jobs before scheduling")
                
            # Try CP-SAT with validated data
            schedule = schedule_jobs(valid_jobs, machines, setup_times, enforce_sequence=args.enforce_sequence, time_limit_seconds=time_limit)
            
            # Verify schedule is valid and not empty
            if not schedule:
                logger.warning("CP-SAT solver returned None instead of a schedule dictionary")
                schedule = None
            elif not any(schedule.values()):
                logger.warning("CP-SAT solver returned an empty schedule (no jobs scheduled)")
                schedule = None
            else:
                total_jobs = sum(len(jobs_list) for jobs_list in schedule.values())
                cp_sat_time = time.time() - cp_sat_start_time
                logger.info(f"CP-SAT solver successfully scheduled {total_jobs} jobs in {cp_sat_time:.2f} seconds")
        except Exception as e:
            logger.error(f"CP-SAT solver failed: {e}")
            logger.error(f"Stack trace:", exc_info=True)
            schedule = None

    # Step 2: Fall back to greedy scheduler if CP-SAT failed or was skipped
    if not schedule or not any(schedule.values()):
        logger.info("Falling back to greedy scheduler...")
        greedy_start_time = time.time()
        try:
            # Ensure START_DATE constraints are properly passed to the greedy scheduler
            start_date_jobs = [job for job in jobs if 
                              ('START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and not pd.isna(job['START_DATE_EPOCH']) and job['START_DATE_EPOCH'] > current_time) or
                              ('START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None and not pd.isna(job['START_DATE _EPOCH']) and job['START_DATE _EPOCH'] > current_time)]
            
            if start_date_jobs:
                logger.info(f"Passing {len(start_date_jobs)} START_DATE constraints to greedy scheduler:")
                for job in start_date_jobs:
                    # Get START_DATE from either field format
                    start_date_epoch = job.get('START_DATE_EPOCH')
                    if start_date_epoch is None or pd.isna(start_date_epoch):
                        start_date_epoch = job.get('START_DATE _EPOCH')
                        
                    if start_date_epoch is not None and not pd.isna(start_date_epoch):
                        start_date = datetime.fromtimestamp(start_date_epoch).strftime('%Y-%m-%d %H:%M')
                        logger.info(f"  Constraint: {job['PROCESS_CODE']} must start EXACTLY at {start_date}")
            
            # Apply same validation as CP-SAT for consistency
            valid_jobs = []
            for job in jobs:
                if 'PROCESS_CODE' not in job:
                    logger.warning(f"Job missing PROCESS_CODE field, skipping: {job}")
                    continue
                    
                if not job.get('RSC_LOCATION') and not job.get('MACHINE_ID'):
                    logger.warning(f"Job {job['PROCESS_CODE']} has no machine assignment, skipping")
                    continue
                    
                valid_jobs.append(job)
                
            schedule = greedy_schedule(valid_jobs, machines, setup_times, enforce_sequence=args.enforce_sequence)
            
            if schedule and any(schedule.values()):
                total_jobs = sum(len(jobs_list) for jobs_list in schedule.values())
                greedy_time = time.time() - greedy_start_time
                logger.info(f"Greedy scheduler successfully scheduled {total_jobs} jobs in {greedy_time:.2f} seconds")
            else:
                logger.error("Greedy scheduler returned an empty schedule")
        except Exception as e:
            logger.error(f"Greedy scheduler failed: {e}")
            logger.error(f"Stack trace:", exc_info=True)
            schedule = None

    if not schedule or not any(schedule.values()):
        logger.error("Failed to create a valid schedule.")
        return

    # Add schedule times and calculate BAL_HR for each job
    jobs = add_schedule_times_and_buffer(jobs, schedule)
    
    # Generate HTML view of the schedule
    try:
        html_output = os.path.splitext(args.output)[0] + "_view.html"
        html_success = export_schedule_html(jobs, schedule, html_output)
        if not html_success:
            logger.error("Failed to generate HTML schedule view.")
            return
        logger.info(f"HTML schedule view saved to: {os.path.abspath(html_output)}")
    except Exception as e:
        logger.error(f"Error generating HTML schedule view: {e}")
        return

    # Flatten the schedule and compute START_TIME, END_TIME, LATEST_COMPLETION_TIME as outputs
    flat_schedule = []
    for machine, tasks in schedule.items():
        for task in tasks:
            process_code, start, end, priority = task
            
            # Find the corresponding job to get accurate START_TIME and END_TIME (might differ from scheduled times)
            job_entry = next((j for j in jobs if j['PROCESS_CODE'] == process_code), None)
            start_time = job_entry['START_TIME'] if job_entry and 'START_TIME' in job_entry else start
            end_time = job_entry['END_TIME'] if job_entry and 'END_TIME' in job_entry else end
            
            # Get resource location - try both old and new column names
            resource_location = job_entry.get('RSC_LOCATION', job_entry.get('MACHINE_ID', machine))
            
            flat_schedule.append({
                'PROCESS_CODE': process_code,
                'RSC_LOCATION': resource_location,
                'START_TIME': start_time,
                'END_TIME': end_time,
                'PRIORITY': priority
            })

    # Compute LATEST_COMPLETION_TIME (latest end time per process family)
    df_schedule = pd.DataFrame(flat_schedule)
    latest_completion = df_schedule.groupby('PROCESS_CODE')['END_TIME'].max().to_dict()
    for entry in flat_schedule:
        entry['LATEST_COMPLETION_TIME'] = latest_completion[entry['PROCESS_CODE']]

    # Log the schedule with computed times
    logger.info("Computed schedule with START_TIME, END_TIME, and LATEST_COMPLETION_TIME:")
    logger.info(df_schedule.to_string())

    # Generate Gantt chart
    try:
        # Pass the jobs list to the Gantt chart function for buffer display
        success = create_interactive_gantt(schedule, jobs, args.output)
        if not success:
            logger.error("Failed to generate Gantt chart.")
            return
    except Exception as e:
        logger.error(f"Error generating Gantt chart: {e}")
        return

    # Schedule statistics
    total_jobs = sum(len(tasks) for tasks in schedule.values())
    machines_used = len([m for m, tasks in schedule.items() if tasks])
    total_duration = sum((end - start) for machine, tasks in schedule.items() for _, start, end, _ in tasks)
    avg_machine_load = total_duration / len(machines) if machines else 0
    most_loaded_machine = max(schedule.items(), key=lambda x: sum(end - start for _, start, end, _ in x[1]), default=(None, []))[0]
    most_loaded_time = sum(end - start for _, start, end, _ in schedule.get(most_loaded_machine, []))
    schedule_span = max((end for machine, tasks in schedule.items() for _, _, end, _ in tasks), default=current_time) - \
                    min((start for machine, tasks in schedule.items() for _, start, _, _ in tasks), default=current_time)
    
    # Jobs per machine
    jobs_per_machine = {machine: len(tasks) for machine, tasks in schedule.items() if tasks}

    # Buffer statistics
    jobs_with_buffer = [job for job in jobs if 'BAL_HR' in job]
    if jobs_with_buffer:
        avg_buffer = sum(job['BAL_HR'] for job in jobs_with_buffer) / len(jobs_with_buffer)
        min_buffer = min(job['BAL_HR'] for job in jobs_with_buffer)
        max_buffer = max(job['BAL_HR'] for job in jobs_with_buffer)
        critical_jobs = [job for job in jobs_with_buffer if job['BAL_HR'] < 8]
        warning_jobs = [job for job in jobs_with_buffer if 8 <= job['BAL_HR'] < 24]
        
        # Print buffer statistics
        print("\nBuffer statistics:")
        print(f"- Average buffer time: {avg_buffer:.1f} hours")
        print(f"- Min buffer time: {min_buffer:.1f} hours")
        print(f"- Max buffer time: {max_buffer:.1f} hours")
        print(f"- Critical jobs (<8h buffer): {len(critical_jobs)}")
        print(f"- Warning jobs (<24h buffer): {len(warning_jobs)}")
        
        # Print critical jobs
        if critical_jobs:
            print("\nCritical jobs with minimal buffer:")
            for job in sorted(critical_jobs, key=lambda x: x['BAL_HR'])[:5]:  # Show top 5 most critical
                process = job['PROCESS_CODE']
                resource_location = job.get('RSC_LOCATION', job.get('MACHINE_ID', 'Unknown'))
                buffer = job['BAL_HR']
                lcd_epoch = job.get('LCD_DATE_EPOCH', 0)
                # Keep the original date and time from the Excel file
                dt = datetime.fromtimestamp(lcd_epoch)
                due_date = dt.strftime('%Y-%m-%d %H:%M')
                orig_date = ""
                for j in jobs:
                    if j['PROCESS_CODE'] == process:
                        if 'LCD_DATE' in j and pd.notna(j['LCD_DATE']):
                            # Use the original timestamp from Excel
                            if isinstance(j['LCD_DATE'], str):
                                orig_date = f" (original={j['LCD_DATE']})"
                            else:
                                orig_date = f" (original={j['LCD_DATE'].strftime('%Y-%m-%d %H:%M')})"
                print(f"  Debug - Job {process}: LCD_DATE_EPOCH={lcd_epoch}, formatted={due_date}{orig_date}")
                print(f"  {process} on {resource_location}: {buffer:.1f} hours buffer, due {due_date}")

    print("\nSchedule statistics:")
    print(f"- Total jobs scheduled: {total_jobs}")
    print(f"- Machines utilized: {machines_used}/{len(machines)} ({machines_used/len(machines)*100:.1f}%)")
    print(f"- Average machine load: {avg_machine_load/3600:.1f} hours")
    print(f"- Most loaded machine: {most_loaded_machine} ({most_loaded_time/3600:.1f} hours)")
    print(f"- Jobs per machine: {jobs_per_machine}")
    print(f"- All jobs will complete by: {datetime.fromtimestamp(max((end for machine, tasks in schedule.items() for _, _, end, _ in tasks), default=current_time)).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"- Total production time: {total_duration/3600:.1f} hours")
    print(f"- Schedule span: {schedule_span/3600:.1f} hours")
    print(f"Scheduling completed in {time.time() - start_time:.2f} seconds")
    
    # Check for START_DATE constraints and their impact
    start_date_jobs = [job for job in jobs if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and not pd.isna(job['START_DATE_EPOCH'])]
    future_date_jobs = [job for job in start_date_jobs if job['START_DATE_EPOCH'] > current_time]
    
    if start_date_jobs:
        print("\nSTART_DATE constraints:")
        for job in start_date_jobs:
            process = job['PROCESS_CODE']
            resource_location = job.get('RSC_LOCATION', job.get('MACHINE_ID', 'Unknown'))
            start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
            is_future = job['START_DATE_EPOCH'] > current_time
            
            # Find the scheduled start time
            scheduled_start = None
            for tasks in schedule.values():
                for proc_code, start, _, _ in tasks:
                    if proc_code == process:
                        scheduled_start = start
                        break
            
            if scheduled_start:
                scheduled_date = datetime.fromtimestamp(scheduled_start).strftime('%Y-%m-%d %H:%M')
                start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
                
                if is_future:
                    if scheduled_start >= job['START_DATE_EPOCH']:
                        impact = "RESPECTED"
                    else:
                        impact = "VIOLATED"
                    
                    # Also check if START_TIME matches START_DATE for future jobs
                    start_time_matches = job.get('START_TIME', 0) == job['START_DATE_EPOCH']
                    
                    print(f"  {process} on {resource_location}: START_DATE={start_date}, Scheduled={scheduled_date} - {impact}")
                    if not start_time_matches:
                        print(f"    START_TIME doesn't match START_DATE for {process}")
        
        if future_date_jobs:
            violated_constraints = []
            for job in future_date_jobs:
                process = job['PROCESS_CODE']
                # Find the scheduled start time
                for tasks in schedule.values():
                    for proc_code, start, _, _ in tasks:
                        if proc_code == process and start < job['START_DATE_EPOCH']:
                            violated_constraints.append(job)
                            break
            
            if violated_constraints:
                logger.error(f"Found {len(violated_constraints)} violated START_DATE constraints!")
                for job in violated_constraints:
                    process = job['PROCESS_CODE']
                    start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
                    logger.error(f"  VIOLATED: {process} should start EXACTLY at {start_date}")
            else:
                logger.info("All future START_DATE constraints were respected by the scheduler")

    print(f"\nResults saved to:")
    print(f"- Gantt chart: {os.path.abspath(args.output)}")
    print(f"- HTML Schedule View: {os.path.abspath(html_output)}")

if __name__ == "__main__":
    main()