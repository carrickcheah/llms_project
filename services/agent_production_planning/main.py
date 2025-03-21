# main.py | dont edit this line
import logging
import time
import argparse
import os
from datetime import datetime
import pandas as pd
import re
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
    calculate the buffer time (BALANCE_HOUR) between job completion and deadline.
    Also adjusts times for dependent processes to maintain proper sequence.
    
    Args:
        jobs (list): List of job dictionaries
        schedule (dict): Schedule as {machine: [(process_code, start, end, priority), ...]}
        
    Returns:
        list: Updated jobs list with START_TIME, END_TIME, and BALANCE_HOUR added
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
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] and process_code in times:
                # Calculate the time shift (positive for earlier, negative for later)
                scheduled_start = times[process_code][0]
                requested_start = job['START_DATE_EPOCH']
                time_shift = scheduled_start - requested_start
                
                # Store the time shift for this family
                if family not in family_time_shifts or abs(time_shift) > abs(family_time_shifts[family]):
                    family_time_shifts[family] = time_shift
                
                logger.info(f"Family {family} has START_DATE constraint for {process_code}: " 
                          f"shift={time_shift/3600:.1f} hours")
    
    # Step 3: Apply time shifts to all processes in affected families
    job_adjustments = {}  # Store adjusted times for each job
    
    for family, time_shift in family_time_shifts.items():
        if abs(time_shift) < 60:  # Skip tiny shifts (less than a minute)
            continue
            
        logger.info(f"Applying time shift of {time_shift/3600:.1f} hours to all processes in family {family}")
        
        for seq_num, process_code, job in family_processes[family]:
            if process_code in times:
                original_start, original_end = times[process_code]
                
                # Apply the time shift
                adjusted_start = original_start - time_shift
                adjusted_end = original_end - time_shift
                
                # Store the adjusted times
                job_adjustments[process_code] = (adjusted_start, adjusted_end)
                
                logger.info(f"  Adjusted {process_code}: START={datetime.fromtimestamp(adjusted_start).strftime('%Y-%m-%d %H:%M')}, "
                          f"END={datetime.fromtimestamp(adjusted_end).strftime('%Y-%m-%d %H:%M')}")
    
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
            
            # Override with exact START_DATE if specified for first process in family
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH']:
                # Get family and sequence
                family = extract_job_family(process_code)
                seq_num = extract_process_number(process_code)
                
                # Check if this is the first process in the family
                is_first_process = False
                if family in family_processes and len(family_processes[family]) > 0:
                    is_first_process = (family_processes[family][0][1] == process_code)
                
                # For logging
                start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
                logger.info(f"Setting START_TIME to match START_DATE ({start_date}) for job {process_code}")
            
            # Update job with adjusted times
            job['START_TIME'] = job_start
            job['END_TIME'] = job_end
            
            # Calculate buffer in hours
            buffer_seconds = max(0, due_time - job_end)
            buffer_hours = buffer_seconds / 3600
            
            job['BALANCE_HOUR'] = buffer_hours
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

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Production Planning Scheduler")
    parser.add_argument("--file", default="/Users/carrickcheah/llms_project/services/agent_production_planning/mydata.xlsx", 
                       help="Path to the Excel file with job data")
    parser.add_argument("--max-jobs", type=int, default=100, help="Maximum number of jobs to schedule")
    parser.add_argument("--force-greedy", action="store_true", help="Force the use of the greedy scheduler")
    parser.add_argument("--output", default="interactive_schedule.html", help="Output file for the Gantt chart")
    parser.add_argument("--enforce-sequence", action="store_true", default=True, help="Enforce process sequence dependencies (default: True)")
    args = parser.parse_args()

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

    # Limit to max_jobs
    if len(jobs) > args.max_jobs:
        logger.warning(f"Number of jobs ({len(jobs)}) exceeds max_jobs ({args.max_jobs}). Limiting to {args.max_jobs} jobs.")
        jobs = jobs[:args.max_jobs]

    logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines")

    # Check for START_DATE values and log them
    start_date_jobs = [job for job in jobs if job.get('START_DATE_EPOCH', current_time) > current_time]
    if start_date_jobs:
        logger.info(f"Found {len(start_date_jobs)} jobs with START_DATE constraints:")
        for job in start_date_jobs:
            start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
            logger.info(f"  Job {job['PROCESS_CODE']} (Machine: {job['MACHINE_ID']}): MUST start EXACTLY at {start_date}")
        
        # Ensure START_DATE constraints are strictly enforced
        logger.info("Ensuring START_DATE constraints are enforced...")
        
        # Only keep START_DATE_EPOCH when it was actually provided in the input
        # DO NOT set defaults for those without specified constraints
        for job in jobs:
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None:
                # Only enforce constraints that are in the future
                if job['START_DATE_EPOCH'] > current_time:
                    logger.info(f"ENFORCING START_DATE {datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')} for {job['PROCESS_CODE']}")
            else:
                # If START_DATE wasn't provided in input, remove it from job dictionary if present
                if 'START_DATE_EPOCH' in job:
                    del job['START_DATE_EPOCH']
                    logger.debug(f"Removed empty START_DATE_EPOCH for {job['PROCESS_CODE']}")

    # Schedule jobs
    start_time = time.time()
    schedule = None
    if not args.force_greedy:
        logger.info("Attempting to create schedule with CP-SAT solver...")
        try:
            schedule = schedule_jobs(jobs, machines, setup_times, enforce_sequence=args.enforce_sequence, time_limit_seconds=600)
            if not schedule or not any(schedule.values()):
                logger.warning("CP-SAT solver returned an empty schedule.")
                schedule = None
        except Exception as e:
            logger.error(f"CP-SAT solver failed: {e}")
            schedule = None

    if not schedule or not any(schedule.values()):
        logger.info("Falling back to greedy scheduler...")
        try:
            # Ensure START_DATE constraints are properly passed to the greedy scheduler
            start_date_jobs = [job for job in jobs if job.get('START_DATE_EPOCH', current_time) > current_time]
            if start_date_jobs:
                logger.info(f"Passing {len(start_date_jobs)} START_DATE constraints to greedy scheduler:")
                for job in start_date_jobs:
                    start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
                    logger.info(f"  Constraint: {job['PROCESS_CODE']} must start EXACTLY at {start_date}")
            
            schedule = greedy_schedule(jobs, machines, setup_times, enforce_sequence=args.enforce_sequence)
        except Exception as e:
            logger.error(f"Greedy scheduler failed: {e}")
            schedule = None

    if not schedule or not any(schedule.values()):
        logger.error("Failed to create a valid schedule.")
        return

    # Add schedule times and calculate BALANCE_HOUR for each job
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
            
            flat_schedule.append({
                'PROCESS_CODE': process_code,
                'MACHINE_ID': machine,
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
    jobs_with_buffer = [job for job in jobs if 'BALANCE_HOUR' in job]
    if jobs_with_buffer:
        avg_buffer = sum(job['BALANCE_HOUR'] for job in jobs_with_buffer) / len(jobs_with_buffer)
        min_buffer = min(job['BALANCE_HOUR'] for job in jobs_with_buffer)
        max_buffer = max(job['BALANCE_HOUR'] for job in jobs_with_buffer)
        critical_jobs = [job for job in jobs_with_buffer if job['BALANCE_HOUR'] < 8]
        warning_jobs = [job for job in jobs_with_buffer if 8 <= job['BALANCE_HOUR'] < 24]
        
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
            for job in sorted(critical_jobs, key=lambda x: x['BALANCE_HOUR'])[:5]:  # Show top 5 most critical
                process = job['PROCESS_CODE']
                machine = job['MACHINE_ID']
                buffer = job['BALANCE_HOUR']
                due_date = datetime.fromtimestamp(job.get('LCD_DATE_EPOCH', 0)).strftime('%Y-%m-%d %H:%M')
                print(f"  {process} on {machine}: {buffer:.1f} hours buffer, due {due_date}")

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
    start_date_jobs = [job for job in jobs if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None]
    future_date_jobs = [job for job in start_date_jobs if job['START_DATE_EPOCH'] > current_time]
    
    if start_date_jobs:
        print("\nSTART_DATE constraints:")
        for job in start_date_jobs:
            process = job['PROCESS_CODE']
            machine = job['MACHINE_ID']
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
                    
                    print(f"  {process} on {machine}: START_DATE={start_date}, Scheduled={scheduled_date} - {impact}")
                    if not start_time_matches:
                        print(f"    ⚠️ START_TIME doesn't match START_DATE for {process}")
        
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