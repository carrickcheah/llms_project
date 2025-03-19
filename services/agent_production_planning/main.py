import logging
import time
import argparse
import os
from datetime import datetime
import pandas as pd
from ingest_data import load_jobs_planning_data
from sch_jobs import schedule_jobs
from greedy import greedy_schedule
from chart import create_interactive_gantt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def add_balance_hours(jobs, schedule):
    """
    Calculate the buffer time (BALANCE_HOUR) between job completion and deadline.
    
    Args:
        jobs (list): List of job dictionaries
        schedule (dict): Schedule as {machine: [(process_code, start, end, priority), ...]}
        
    Returns:
        list: Updated jobs list with BALANCE_HOUR added
    """
    # Create a lookup of end times from the schedule
    end_times = {}
    for machine, tasks in schedule.items():
        for task in tasks:
            process_code, _, end, _ = task
            end_times[process_code] = end
    
    # Add BALANCE_HOUR to each job
    for job in jobs:
        process_code = job['PROCESS_CODE']
        if process_code in end_times:
            job_end = end_times[process_code]
            due_time = job.get('DUE_DATE_TIME', 0)
            
            # Calculate buffer in hours
            buffer_seconds = max(0, due_time - job_end)
            buffer_hours = buffer_seconds / 3600
            
            # Add to job data
            job['END_TIME'] = job_end
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

def export_schedule_with_buffer(jobs, output_file="schedule_with_buffer.xlsx"):
    """
    Export the schedule with buffer information to an Excel file.
    
    Args:
        jobs (list): List of job dictionaries with buffer information
        output_file (str): Path to save the Excel file
    """
    # Create a DataFrame from the jobs list
    df = pd.DataFrame(jobs)
    
    # Select columns for export
    columns = [
        'JOB_CODE', 'PROCESS_CODE', 'MACHINE_ID', 'PRIORITY',
        'EARLIEST_START_TIME', 'END_TIME', 'DUE_DATE_TIME', 
        'BALANCE_HOUR', 'BUFFER_STATUS'
    ]
    
    # Convert timestamps to datetime
    export_df = pd.DataFrame()
    for col in columns:
        if col in df.columns:
            export_df[col] = df[col]
    
    # Convert timestamp columns to readable dates
    time_columns = ['EARLIEST_START_TIME', 'END_TIME', 'DUE_DATE_TIME']
    for col in time_columns:
        if col in export_df.columns:
            export_df[f"{col}_DATE"] = export_df[col].apply(
                lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else None
            )
    
    # Round buffer hours to 1 decimal place
    if 'BALANCE_HOUR' in export_df.columns:
        export_df['BALANCE_HOUR'] = export_df['BALANCE_HOUR'].round(1)
    
    # Save to Excel
    try:
        export_df.to_excel(output_file, index=False)
        logger.info(f"Schedule with buffer information saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error exporting schedule to Excel: {e}")
        return False

def main():
    print("Production Planning Scheduler")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Production Planning Scheduler")
    parser.add_argument("--file", default="mydata.xlsx", help="Path to the Excel file with job data")
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

    # Check for CUT_Q dates and log them
    cut_q_jobs = [job for job in jobs if job.get('EARLIEST_START_TIME', current_time) > current_time]
    if cut_q_jobs:
        logger.info(f"Found {len(cut_q_jobs)} jobs with CUT_Q (earliest start) constraints:")
        for job in cut_q_jobs:
            cut_q_date = datetime.fromtimestamp(job['EARLIEST_START_TIME']).strftime('%Y-%m-%d %H:%M')
            logger.info(f"  Job {job['PROCESS_CODE']} (Machine: {job['MACHINE_ID']}): Must start on or after {cut_q_date}")

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
            schedule = greedy_schedule(jobs, machines, setup_times, enforce_sequence=args.enforce_sequence)
        except Exception as e:
            logger.error(f"Greedy scheduler failed: {e}")
            schedule = None

    if not schedule or not any(schedule.values()):
        logger.error("Failed to create a valid schedule.")
        return

    # Calculate BALANCE_HOUR for each job
    jobs = add_balance_hours(jobs, schedule)
    
    # Export schedule with buffer information
    export_file = os.path.splitext(args.output)[0] + "_schedule.xlsx"
    export_schedule_with_buffer(jobs, export_file)

    # Flatten the schedule and compute START_TIME, END_TIME, LATEST_COMPLETION_TIME as outputs
    flat_schedule = []
    for machine, tasks in schedule.items():
        for task in tasks:
            process_code, start, end, priority = task
            flat_schedule.append({
                'PROCESS_CODE': process_code,
                'MACHINE_ID': machine,
                'START_TIME': start,  # Computed by scheduler
                'END_TIME': end,      # Computed by scheduler
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
                due_date = datetime.fromtimestamp(job['DUE_DATE_TIME']).strftime('%Y-%m-%d %H:%M')
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
    
    # Check for CUT_Q constraints and their impact
    cut_q_jobs = [job for job in jobs if job.get('EARLIEST_START_TIME', current_time) > current_time]
    if cut_q_jobs:
        print("\nCUT_Q constraints:")
        for job in cut_q_jobs:
            process = job['PROCESS_CODE']
            machine = job['MACHINE_ID']
            cut_q_date = datetime.fromtimestamp(job['EARLIEST_START_TIME']).strftime('%Y-%m-%d %H:%M')
            
            # Find the scheduled start time
            scheduled_start = None
            for tasks in schedule.values():
                for proc_code, start, _, _ in tasks:
                    if proc_code == process:
                        scheduled_start = start
                        break
            
            if scheduled_start:
                scheduled_date = datetime.fromtimestamp(scheduled_start).strftime('%Y-%m-%d %H:%M')
                impact = "RESPECTED" if scheduled_start >= job['EARLIEST_START_TIME'] else "VIOLATED"
                print(f"  {process} on {machine}: CUT_Q={cut_q_date}, Scheduled={scheduled_date} - {impact}")

    print(f"\nResults saved to:")
    print(f"- Gantt chart: {os.path.abspath(args.output)}")
    print(f"- Schedule Excel: {os.path.abspath(export_file)}")

if __name__ == "__main__":
    main()

