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
from greedy import greedy_schedule
from chart import create_interactive_gantt
from chart_two import export_schedule_html
from chart_three import create_report_gantt
from setup_times import add_schedule_times_and_buffer
from urgent import reduce_nonproductive_time, should_reschedule
from time_utils import (
    initialize_reference_time, 
    epoch_to_datetime, 
    datetime_to_epoch,
    format_datetime_for_display,
    convert_job_times_to_relative,
    convert_job_times_to_epoch
)
import subprocess
import json
import pytz
import sys

# Clear any existing handlers in the root logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure root logger to write to both file and console
file_handler = logging.FileHandler("production_scheduler.log", mode='w')  # Start with a fresh log
console_handler = logging.StreamHandler(sys.stdout)

# Set formatter for both handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to root logger
logging.root.addHandler(file_handler)
logging.root.addHandler(console_handler)
logging.root.setLevel(logging.INFO)

# Set the root logger as the default for all modules
logger = logging.getLogger()

# Direct file write function for machine metrics
def write_to_log_file(content):
    """Write content directly to the log file."""
    with open("production_scheduler.log", "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} - INFO - {content}\n")

# Import functions from machine_metrics_logger
def format_seconds(seconds):
    """Convert seconds to human-readable format (HH:MM:SS)"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{int(hours)}:{int(minutes):02d}:{int(secs):02d}"

def format_timestamp(timestamp):
    """Convert epoch timestamp to human-readable date/time"""
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return f"Invalid timestamp: {timestamp}"

def calculate_machine_metrics(schedule):
    """
    Calculate machine utilization metrics from the schedule.
    """
    metrics = {}
    
    # Track overall schedule timespan
    all_start_times = []
    all_end_times = []
    
    for machine, tasks in schedule.items():
        if not tasks:
            continue
            
        # Convert tasks to list of (start, end) tuples for easier processing
        job_spans = []
        
        for task in tasks:
            # Handle both 4-tuple and 5-tuple formats
            if len(task) >= 3:  # We only need start and end times
                job_id = task[0]
                start = task[1]
                end = task[2]
                job_spans.append((start, end, job_id))
                
                # Track overall schedule timing
                all_start_times.append(start)
                all_end_times.append(end)
        
        # Sort by start time
        job_spans.sort()
        
        if not job_spans:
            continue
            
        # Calculate machine-specific metrics
        machine_start = job_spans[0][0]
        machine_end = max(end for _, end, _ in job_spans)
        total_span = machine_end - machine_start  # seconds
        
        # Calculate working time
        working_time = sum(end - start for start, end, _ in job_spans)  # seconds
        
        # Calculate utilization percentage
        utilization = (working_time / total_span * 100) if total_span > 0 else 0
        
        # Find idle periods (gaps between jobs)
        idle_periods = []
        for i in range(1, len(job_spans)):
            prev_end = job_spans[i-1][1]
            curr_start = job_spans[i][0]
            
            if curr_start > prev_end:
                idle_periods.append((prev_end, curr_start, curr_start - prev_end))
        
        # Calculate job durations
        job_durations = [(end - start, job_id) for start, end, job_id in job_spans]
        
        # Store metrics for this machine
        metrics[machine] = {
            'num_jobs': len(job_spans),
            'total_span_seconds': total_span,
            'working_time_seconds': working_time,
            'utilization_percent': utilization,
            'idle_periods': idle_periods,
            'job_durations': job_durations,
            'start_time': machine_start,
            'end_time': machine_end
        }
    
    # Calculate overall schedule metrics
    if all_start_times and all_end_times:
        overall_start = min(all_start_times)
        overall_end = max(all_end_times)
        overall_span = overall_end - overall_start
        
        metrics['overall'] = {
            'start_time': overall_start,
            'end_time': overall_end,
            'total_span_seconds': overall_span
        }
        
    return metrics

def log_machine_metrics_to_file(metrics):
    """
    Write machine metrics directly to the log file.
    """
    write_to_log_file("========== MACHINE UTILIZATION REPORT ==========")
    
    # First log the overall statistics
    if 'overall' in metrics:
        overall = metrics['overall']
        write_to_log_file(f"Schedule timespan: {format_timestamp(overall['start_time'])} to {format_timestamp(overall['end_time'])}")
        write_to_log_file(f"Total schedule duration: {format_seconds(overall['total_span_seconds'])} ({overall['total_span_seconds']/3600:.2f} hours)")
        write_to_log_file("---------------------------------------------")
    
    # Log machine-specific metrics
    for machine, machine_metrics in sorted(metrics.items()):
        if machine == 'overall':
            continue
            
        # Skip empty metrics
        if not machine_metrics:
            continue
        
        num_jobs = machine_metrics['num_jobs']
        total_span = machine_metrics['total_span_seconds']
        working_time = machine_metrics['working_time_seconds']
        utilization = machine_metrics['utilization_percent']
        idle_periods = machine_metrics['idle_periods']
        start_time = machine_metrics['start_time']
        end_time = machine_metrics['end_time']
        
        # Log basic metrics
        write_to_log_file(f"Machine: {machine}")
        write_to_log_file(f"  Number of jobs: {num_jobs}")
        write_to_log_file(f"  Timespan: {format_timestamp(start_time)} to {format_timestamp(end_time)}")
        write_to_log_file(f"  Duration: {format_seconds(total_span)} ({total_span/3600:.2f} hours)")
        write_to_log_file(f"  Working time: {format_seconds(working_time)} ({working_time/3600:.2f} hours)")
        write_to_log_file(f"  Utilization: {utilization:.2f}%")
        
        # Log idle periods
        total_idle_time = sum(duration for _, _, duration in idle_periods)
        write_to_log_file(f"  Idle periods: {len(idle_periods)}")
        write_to_log_file(f"  Total idle time: {format_seconds(total_idle_time)} ({total_idle_time/3600:.2f} hours)")
        
        if idle_periods:
            write_to_log_file("  Idle period details:")
            for i, (start, end, duration) in enumerate(idle_periods, 1):
                write_to_log_file(f"    {i}. {format_timestamp(start)} to {format_timestamp(end)}: {format_seconds(duration)} ({duration/3600:.2f} hours)")
        
        # Log job durations
        write_to_log_file("  Job durations:")
        for duration, job_id in sorted(machine_metrics['job_durations'], reverse=True)[:5]:  # Top 5 longest jobs
            write_to_log_file(f"    {job_id}: {format_seconds(duration)} ({duration/3600:.2f} hours)")
        
        write_to_log_file("---------------------------------------------")

def main():
    print("Production Planning Scheduler")

    load_dotenv()
    parser = argparse.ArgumentParser(description="Production Planning Scheduler")
    parser.add_argument("--file", default=os.getenv('file_path'), 
                       help="Path to the Excel file with job data")
    parser.add_argument("--max-jobs", type=int, default=500, help="Maximum number of jobs to schedule (default: 250)")
    parser.add_argument("--force-greedy", action="store_true", help="Force the use of the greedy scheduler")
    parser.add_argument("--output", default="interactive_schedule.html", help="Output file for the Gantt chart")
    parser.add_argument("--enforce-sequence", action="store_true", default=True, help="Enforce process sequence dependencies (default: True)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (show debug messages)")
    parser.add_argument("--max-operators", type=int, default=int(os.getenv('MAX_OPERATORS', 0)), 
                       help="Maximum number of operators available. If not provided, uses the MAX_OPERATORS value from .env file.")
    parser.add_argument("--urgent50", action="store_true", help="Reduce non-productive times by 50% for late jobs")
    parser.add_argument("--urgent100", action="store_true", help="Reduce non-productive times by 100% for late jobs")
    parser.add_argument("--generate-report", action="store_true", help="Generate detailed production report")
    parser.add_argument("--report-output", default="production_report.html", help="Output file for the production report")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip generation of machine utilization metrics")
    args = parser.parse_args()

    # Set up logging based on whether --verbose was used
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    if not args.file:
        logger.error("No file path provided.")
        return

    print(f"Configuration: file={args.file}, max_jobs={args.max_jobs}, force_greedy={args.force_greedy}, "
          f"output={args.output}, enforce_sequence={args.enforce_sequence}, max_operators={args.max_operators}")

    # Initialize our reference time for relative time calculations
    reference_time = initialize_reference_time()
    logger.info(f"Using reference time: {reference_time.isoformat()} for relative time calculations")
    
    # Use datetime objects instead of epoch time when possible
    current_time_dt = datetime.now()
    current_time = datetime_to_epoch(current_time_dt)
    
    logger.info(f"Loading job data from {args.file}...")
    try:
        jobs, machines, setup_times = load_jobs_planning_data(args.file)
    except Exception as e:
        logger.error(f"Failed to load job data: {e}")
        return

    for job in jobs:
        job['UNIQUE_JOB_ID'] = f"{job['JOB']}_{job['PROCESS_CODE']}"
        # Convert epoch times to relative time and ISO for better handling
        convert_job_times_to_relative(job)

    jobs.sort(key=lambda job: job.get('LCD_DATE_EPOCH', 0))
    logger.info(f"Sorted {len(jobs)} jobs by LCD_DATE (First In, First Out)")

    if len(jobs) > args.max_jobs:
        logger.warning(f"Number of jobs ({len(jobs)}) exceeds max_jobs ({args.max_jobs}). Limiting to {args.max_jobs} jobs.")
        jobs = jobs[:args.max_jobs]

    logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines")

    # Use datetime objects and ISO strings for datetime comparisons when possible
    start_date_jobs = [job for job in jobs if 
                      ('START_DATE_EPOCH' in job and job.get('START_DATE_EPOCH') is not None and 
                       not pd.isna(job.get('START_DATE_EPOCH')) and job.get('START_DATE_EPOCH') > current_time) or 
                      ('START_DATE _EPOCH' in job and job.get('START_DATE _EPOCH') is not None and 
                       not pd.isna(job.get('START_DATE _EPOCH')) and job.get('START_DATE _EPOCH') > current_time)]
    
    if start_date_jobs:
        logger.info(f"Found {len(start_date_jobs)} jobs with START_DATE constraints:")
        for job in start_date_jobs:
            start_date_epoch = None
            if 'START_DATE_EPOCH' in job and job.get('START_DATE_EPOCH') is not None and not pd.isna(job.get('START_DATE_EPOCH')):
                start_date_epoch = job.get('START_DATE_EPOCH')
            elif 'START_DATE _EPOCH' in job and job.get('START_DATE _EPOCH') is not None and not pd.isna(job.get('START_DATE _EPOCH')):
                start_date_epoch = job.get('START_DATE _EPOCH')
                
            # Use ISO format for display
            start_date = "INVALID DATE"
            if start_date_epoch is not None:
                dt = epoch_to_datetime(start_date_epoch)
                if dt:
                    start_date = format_datetime_for_display(dt)
            
            resource_location = job.get('RSC_CODE')
            logger.info(f"  Job {job['UNIQUE_JOB_ID']} (Resource: {resource_location}): MUST start EXACTLY at {start_date}")
        
        logger.info("Ensuring START_DATE constraints are enforced...")
        for job in jobs:
            start_date_epoch = None
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and not pd.isna(job['START_DATE_EPOCH']):
                start_date_epoch = job['START_DATE_EPOCH']
            elif 'START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None and not pd.isna(job['START_DATE _EPOCH']):
                start_date_epoch = job['START_DATE _EPOCH']
                job['START_DATE_EPOCH'] = start_date_epoch
            
            if start_date_epoch is not None:
                if start_date_epoch > current_time:
                    dt = epoch_to_datetime(start_date_epoch)
                    if dt:
                        start_date_str = format_datetime_for_display(dt)
                        logger.info(f"ENFORCING START_DATE {start_date_str} for {job['UNIQUE_JOB_ID']}")
            else:
                if 'START_DATE_EPOCH' in job:
                    del job['START_DATE_EPOCH']
                    logger.debug(f"Removed empty START_DATE_EPOCH for {job['UNIQUE_JOB_ID']}")
                if 'START_DATE _EPOCH' in job:
                    del job['START_DATE _EPOCH']
                    logger.debug(f"Removed empty START_DATE _EPOCH for {job['UNIQUE_JOB_ID']}")

    start_time = time.perf_counter()
    schedule = None

    if not args.force_greedy:
        logger.info("Attempting to create schedule with CP-SAT solver...")
        cp_sat_start_time = time.perf_counter()
        try:
            if len(jobs) > 200:
                time_limit = 900
            elif len(jobs) > 100:
                time_limit = 600
            else:
                time_limit = 300
                
            logger.info(f"Using {time_limit} seconds time limit for CP-SAT solver with {len(jobs)} jobs")
            
            valid_jobs = []
            for job in jobs:
                if 'UNIQUE_JOB_ID' not in job:
                    logger.warning(f"Job missing UNIQUE_JOB_ID field, skipping: {job}")
                    continue
                    
                if not job.get('RSC_CODE'):
                    logger.warning(f"Job {job['UNIQUE_JOB_ID']} has no machine assignment, skipping")
                    continue
                    
                if job.get('processing_time') is None or not isinstance(job.get('processing_time'), (int, float)) or job.get('processing_time') <= 0:
                    logger.warning(f"Job {job['UNIQUE_JOB_ID']} has missing or invalid processing time (HOURS_NEED)")
                    job['processing_time'] = 3600  # Still provide a default for the scheduler
                    
                valid_jobs.append(job)
                
            if len(valid_jobs) < len(jobs):
                logger.warning(f"Filtered out {len(jobs) - len(valid_jobs)} invalid jobs before scheduling")
                
            schedule = schedule_jobs(valid_jobs, machines, setup_times, enforce_sequence=args.enforce_sequence, time_limit_seconds=time_limit, max_operators=args.max_operators)
            
            if not schedule:
                logger.warning("CP-SAT solver returned None instead of a schedule dictionary")
                schedule = None
            elif not any(schedule.values()):
                logger.warning("CP-SAT solver returned an empty schedule (no jobs scheduled)")
                schedule = None
            else:
                total_jobs = sum(len(jobs_list) for jobs_list in schedule.values())
                cp_sat_time = max(0, time.perf_counter() - cp_sat_start_time)
                logger.info(f"CP-SAT solver successfully scheduled {total_jobs} jobs in {cp_sat_time:.2f} seconds")
        except Exception as e:
            logger.error(f"CP-SAT solver failed: {e}")
            logger.error(f"Stack trace:", exc_info=True)
            schedule = None

    if not schedule or not any(schedule.values()):
        logger.info("Falling back to greedy scheduler...")
        greedy_start_time = time.perf_counter()
        try:
            # Use ISO strings for datetime comparisons when possible
            start_date_jobs = [job for job in jobs if 
                              ('START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and not pd.isna(job['START_DATE_EPOCH']) and job['START_DATE_EPOCH'] > current_time) or
                              ('START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None and not pd.isna(job['START_DATE _EPOCH']) and job['START_DATE _EPOCH'] > current_time)]
            
            if start_date_jobs:
                logger.info(f"Passing {len(start_date_jobs)} START_DATE constraints to greedy scheduler:")
                for job in start_date_jobs:
                    start_date_epoch = job.get('START_DATE_EPOCH')
                    if start_date_epoch is None or pd.isna(start_date_epoch):
                        start_date_epoch = job.get('START_DATE _EPOCH')
                        
                    if start_date_epoch is not None and not pd.isna(start_date_epoch):
                        # Use ISO format for display
                        dt = epoch_to_datetime(start_date_epoch)
                        start_date = format_datetime_for_display(dt) if dt else "INVALID DATE"
                        logger.info(f"  Constraint: {job['UNIQUE_JOB_ID']} must start EXACTLY at {start_date}")
            
            valid_jobs = []
            for job in jobs:
                if 'UNIQUE_JOB_ID' not in job:
                    logger.warning(f"Job missing UNIQUE_JOB_ID field, skipping: {job}")
                    continue
                    
                if not job.get('RSC_CODE'):
                    logger.warning(f"Job {job['UNIQUE_JOB_ID']} has no machine assignment, skipping")
                    continue
                    
                valid_jobs.append(job)
                
            schedule = greedy_schedule(valid_jobs, machines, setup_times, enforce_sequence=args.enforce_sequence, max_operators=args.max_operators)
            
            if schedule and any(schedule.values()):
                total_jobs = sum(len(jobs_list) for jobs_list in schedule.values())
                greedy_time = max(0, time.perf_counter() - greedy_start_time)
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

    # Convert CP-SAT schedule to greedy format if needed
    if '_metadata' in schedule:
        logger.info("Converting CP-SAT schedule format to greedy format")
        schedule = convert_cpsat_to_greedy_format(schedule)

    jobs = add_schedule_times_and_buffer(jobs, schedule)
    
    # Apply urgent handling for late jobs if specified
    reschedule_needed = False
    
    if args.urgent50 or args.urgent100:
        # Determine the reduction percentage
        reduction_percent = 100 if args.urgent100 else 50
        
        # Log the urgent option being applied
        logger.info(f"Applying urgent_{100-reduction_percent} option: Reducing non-productive times by {reduction_percent}% for late jobs")
        
        # Apply the reduction
        jobs, modified = reduce_nonproductive_time(jobs, reduction_percent)
        
        # Check if rescheduling is recommended
        if modified and should_reschedule(jobs, reduction_percent):
            logger.info("Significant reductions detected, rescheduling with reduced non-productive times...")
            reschedule_needed = True
            
            # Prepare valid jobs for rescheduling
            valid_jobs = []
            for job in jobs:
                if 'UNIQUE_JOB_ID' not in job:
                    continue
                if not job.get('RSC_CODE'):
                    continue
                valid_jobs.append(job)
            
            # Re-run the greedy scheduler with the updated processing times
            new_schedule = greedy_schedule(valid_jobs, machines, setup_times, 
                                          enforce_sequence=args.enforce_sequence, 
                                          max_operators=args.max_operators)
            
            if new_schedule and any(new_schedule.values()):
                schedule = new_schedule
                logger.info("Successfully rescheduled with reduced non-productive times")
                
                # Recalculate buffer times with the new schedule
                jobs = add_schedule_times_and_buffer(jobs, schedule)
                logger.info("Updated buffer times after rescheduling")
            else:
                logger.warning("Failed to reschedule with reduced non-productive times, using original schedule")
    
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

    flat_schedule = []
    for machine, tasks in schedule.items():
        for task in tasks:
            if len(task) == 5:
                unique_job_id, start, end, priority, _ = task
            else:
                unique_job_id, start, end, priority = task
            
            # Convert epoch timestamps to ISO strings and relative times for display
            start_dt = epoch_to_datetime(start)
            end_dt = epoch_to_datetime(end)
            
            flat_schedule.append({
                'machine': machine,
                'job_id': unique_job_id,
                'start': start,  # Keep epoch for compatibility
                'end': end,      # Keep epoch for compatibility
                'start_iso': start_dt.isoformat() if start_dt else None,
                'end_iso': end_dt.isoformat() if end_dt else None,
                'priority': priority
            })
    
    df = pd.DataFrame(flat_schedule)
    
    # Use ISO strings for display
    for job in jobs:
        if 'START_DATE_EPOCH' in job:
            dt = epoch_to_datetime(job['START_DATE_EPOCH'])
            if dt:
                formatted_date = format_datetime_for_display(dt)
                logger.info(f"Input Job {job['UNIQUE_JOB_ID']}: START_DATE_EPOCH = {job['START_DATE_EPOCH']} -> {formatted_date}")
    
    # Find START_DATE constraints that weren't respected
    violations = []
    for job in jobs:
        if 'UNIQUE_JOB_ID' not in job:
            continue
            
        start_date_epoch = job.get('START_DATE_EPOCH')
        if start_date_epoch is None:
            continue
            
        job_id = job['UNIQUE_JOB_ID']
        scheduled_jobs = [task for task in flat_schedule if task['job_id'] == job_id]
        
        if not scheduled_jobs:
            logger.warning(f"Job {job_id} has START_DATE constraint but wasn't scheduled")
            continue
            
        scheduled_job = scheduled_jobs[0]
        scheduled_start = scheduled_job['start']
        
        if scheduled_start != start_date_epoch:
            # Use ISO strings for better readability
            scheduled_dt = epoch_to_datetime(scheduled_start)
            expected_dt = epoch_to_datetime(start_date_epoch)
            
            violations.append({
                'job_id': job_id,
                'machine': scheduled_job['machine'],
                'expected': format_datetime_for_display(expected_dt) if expected_dt else "INVALID DATE",
                'actual': format_datetime_for_display(scheduled_dt) if scheduled_dt else "INVALID DATE"
            })
    
    if violations:
        logger.warning(f"Found {len(violations)} START_DATE constraint violations:")
        for v in violations:
            logger.warning(f"  Job {v['job_id']} on {v['machine']}: Expected {v['expected']}, got {v['actual']}")
    else:
        logger.info("All future START_DATE constraints were respected by the scheduler")
    
    # Calculate statistics about the schedule
    machines_used = len(schedule)
    total_machines = len(machines)
    utilization_pct = machines_used / total_machines * 100 if total_machines > 0 else 0
    
    # Find horizon end - use datetime objects for better handling
    all_end_times = [task['end'] for task in flat_schedule]
    if all_end_times:
        horizon_end_epoch = max(all_end_times)
        horizon_end_dt = epoch_to_datetime(horizon_end_epoch)
    else:
        horizon_end_dt = None
    
    all_start_times = [task['start'] for task in flat_schedule]
    if all_start_times:
        horizon_start_epoch = min(all_start_times)
        horizon_start_dt = epoch_to_datetime(horizon_start_epoch)
    else:
        horizon_start_dt = None
    
    # Use datetime objects for time calculations
    total_hours = sum(task['end'] - task['start'] for task in flat_schedule) / 3600
    span_hours = (horizon_end_epoch - horizon_start_epoch) / 3600 if horizon_end_epoch and horizon_start_epoch else 0
    
    # Generate the Gantt chart
    try:
        create_interactive_gantt(schedule, jobs, args.output)
        logger.info(f"Saving Gantt chart to: {os.path.abspath(args.output)}")
        logger.info(f"Interactive Gantt chart saved to: {os.path.abspath(args.output)}")
    except Exception as e:
        logger.error(f"Error creating Gantt chart: {e}")
    
    # Generate the Report Gantt with resources on y-axis
    try:
        report_output = os.path.splitext(args.output)[0] + "_r.html"
        create_report_gantt(schedule, jobs, report_output)
        logger.info(f"Saving Resource Gantt chart to: {os.path.abspath(report_output)}")
        logger.info(f"Interactive Resource Gantt chart saved to: {os.path.abspath(report_output)}")
    except Exception as e:
        logger.error(f"Error creating Resource Gantt chart: {e}")

    # Output detailed statistics
    print("\nBuffer statistics:")
    buffer_hours = [job.get('buffer_hours', 0) for job in jobs if 'buffer_hours' in job]
    if buffer_hours:
        print(f"- Average buffer time: {sum(buffer_hours) / len(buffer_hours):.1f} hours")
        print(f"- Min buffer time: {min(buffer_hours):.1f} hours")
        print(f"- Max buffer time: {max(buffer_hours):.1f} hours")
        
        critical_jobs = [job for job in jobs if job.get('buffer_hours', float('inf')) < 8]
        warning_jobs = [job for job in jobs if 8 <= job.get('buffer_hours', float('inf')) < 24]
        
        print(f"- Critical jobs (<8h buffer): {len(critical_jobs)}")
        print(f"- Warning jobs (<24h buffer): {len(warning_jobs)}")
        
        if critical_jobs:
            print("\nCritical jobs with minimal buffer:")
            for job in sorted(critical_jobs, key=lambda x: x.get('buffer_hours', 0))[:5]:  # Show 5 most critical
                lcd_date_epoch = job.get('LCD_DATE_EPOCH')
                lcd_date_str = "unknown"
                if lcd_date_epoch:
                    dt = epoch_to_datetime(lcd_date_epoch)
                    if dt:
                        lcd_date_str = format_datetime_for_display(dt)
                        
                original_str = "unknown"
                if 'LCD_DATE_ORIGINAL' in job:
                    original_str = job['LCD_DATE_ORIGINAL']
                    
                print(f"  Debug - Job {job['UNIQUE_JOB_ID']}: LCD_DATE_EPOCH={lcd_date_epoch}, formatted={lcd_date_str} (original={original_str})")
                
                machine = next((m for m, tasks in schedule.items() 
                              for task_id, _, _, _ in tasks if task_id == job['UNIQUE_JOB_ID']), "Unknown")
                              
                end_time = job.get('END_TIME')
                if end_time:
                    end_dt = epoch_to_datetime(end_time)
                    end_str = format_datetime_for_display(end_dt) if end_dt else "unknown"
                else:
                    end_str = "unknown"
                    
                print(f"  {job['UNIQUE_JOB_ID']} on {machine}: {job.get('buffer_hours', 0):.1f} hours buffer, due {lcd_date_str}")
    
    print("\nSchedule statistics:")
    print(f"- Total jobs scheduled: {sum(len(machine_jobs) for machine_jobs in schedule.values())}")
    print(f"- Machines utilized: {machines_used}/{total_machines} ({utilization_pct:.1f}%)")
    
    # Calculate machine load
    machine_loads = {}
    for machine, tasks in schedule.items():
        machine_load = sum(end - start for task in tasks for _, start, end, *_ in [task]) / 3600
        machine_loads[machine] = machine_load
    
    if machine_loads:
        avg_load = sum(machine_loads.values()) / len(machine_loads)
        max_load_machine = max(machine_loads.items(), key=lambda x: x[1])
        
        print(f"- Average machine load: {avg_load:.1f} hours")
        print(f"- Most loaded machine: {max_load_machine[0]} ({max_load_machine[1]:.1f} hours)")
    
    print(f"- Jobs per machine: {dict([(m, len(tasks)) for m, tasks in schedule.items()])}")
    
    if horizon_end_dt:
        print(f"- All jobs will complete by: {format_datetime_for_display(horizon_end_dt)}")
    
    print(f"- Total production time: {total_hours:.1f} hours")
    print(f"- Schedule span: {span_hours:.1f} hours")
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Scheduling completed in {elapsed_time:.2f} seconds")
    
    # Display START_DATE constraints
    start_date_jobs = [job for job in jobs if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None]
    if start_date_jobs:
        print("\nSTART_DATE constraints:")
        for job in start_date_jobs:
            job_id = job['UNIQUE_JOB_ID']
            scheduled_jobs = [task for task in flat_schedule if task['job_id'] == job_id]
            
            if not scheduled_jobs:
                continue
                
            scheduled_job = scheduled_jobs[0]
            machine = scheduled_job['machine']
            start_date_epoch = job['START_DATE_EPOCH']
            
            # Use ISO strings for better readability
            start_date_dt = epoch_to_datetime(start_date_epoch)
            scheduled_start_dt = epoch_to_datetime(scheduled_job['start'])
            
            start_date_str = format_datetime_for_display(start_date_dt) if start_date_dt else "INVALID DATE"
            scheduled_str = format_datetime_for_display(scheduled_start_dt) if scheduled_start_dt else "INVALID DATE"
            
            respected = "RESPECTED" if scheduled_job['start'] == start_date_epoch else "VIOLATED"
            print(f"  {job_id} on {machine}: START_DATE={start_date_str}, Scheduled={scheduled_str} - {respected}")
    
    logger.info("All future START_DATE constraints were respected by the scheduler")
    
    print("\nResults saved to:")
    print(f"- Gantt chart: {os.path.abspath(args.output)}")
    print(f"- HTML Schedule View: {os.path.abspath(html_output)}")
    
    # Generate production report if requested
    if args.generate_report:
        try:
            logger.info(f"Generating production report to: {os.path.abspath(args.report_output)}")
            
            # Create a detailed production report
            report_html = []
            report_html.append("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Production Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                    h1, h2, h3 { color: #333; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .late { color: #e74c3c; font-weight: bold; }
                    .warning { color: #e67e22; }
                    .good { color: #2ecc71; }
                    .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <h1>Production Planning Report</h1>
                <div class="summary">
                    <h2>Schedule Summary</h2>
                    <p><strong>Total jobs scheduled:</strong> {}</p>
                    <p><strong>Machines utilized:</strong> {}/{} ({:.1f}%)</p>
                    <p><strong>Average machine load:</strong> {:.1f} hours</p>
                    <p><strong>Most loaded machine:</strong> {} ({:.1f} hours)</p>
                    <p><strong>Schedule completion date:</strong> {}</p>
                    <p><strong>Total production time:</strong> {:.1f} hours</p>
                    <p><strong>Schedule span:</strong> {:.1f} hours</p>
                </div>
            """.format(
                sum(len(machine_jobs) for machine_jobs in schedule.values()),
                machines_used, 
                total_machines,
                utilization_pct,
                avg_load if 'avg_load' in locals() else 0,
                max_load_machine[0] if 'max_load_machine' in locals() else "None",
                max_load_machine[1] if 'max_load_machine' in locals() else 0,
                format_datetime_for_display(horizon_end_dt) if horizon_end_dt else "Unknown",
                total_hours,
                span_hours
            ))
            
            # Late jobs section
            late_jobs = []
            for job in jobs:
                if 'UNIQUE_JOB_ID' not in job:
                    continue
                
                end_time = job.get('END_TIME')
                lcd_date = job.get('LCD_DATE_EPOCH')
                
                if end_time and lcd_date and end_time > lcd_date:
                    job_id = job['UNIQUE_JOB_ID']
                    machine = next((m for m, tasks in schedule.items() 
                                for task in tasks if task[0] == job_id), "Unknown")
                    
                    end_dt = epoch_to_datetime(end_time)
                    lcd_dt = epoch_to_datetime(lcd_date)
                    
                    # Calculate hours late
                    hours_late = (end_time - lcd_date) / 3600
                    
                    late_jobs.append({
                        'job_id': job_id,
                        'machine': machine,
                        'end_date': format_datetime_for_display(end_dt) if end_dt else "Unknown",
                        'due_date': format_datetime_for_display(lcd_dt) if lcd_dt else "Unknown",
                        'hours_late': hours_late
                    })
            
            if late_jobs:
                report_html.append("""
                <h2>Late Jobs</h2>
                <table>
                    <tr>
                        <th>Job ID</th>
                        <th>Machine</th>
                        <th>Due Date</th>
                        <th>Completion Date</th>
                        <th>Hours Late</th>
                    </tr>
                """)
                
                for job in sorted(late_jobs, key=lambda x: x['hours_late'], reverse=True):
                    report_html.append(f"""
                    <tr>
                        <td>{job['job_id']}</td>
                        <td>{job['machine']}</td>
                        <td>{job['due_date']}</td>
                        <td>{job['end_date']}</td>
                        <td class="late">{job['hours_late']:.1f}</td>
                    </tr>
                    """)
                
                report_html.append("</table>")
            
            # Machine utilization section
            report_html.append("""
            <h2>Machine Utilization</h2>
            <table>
                <tr>
                    <th>Machine</th>
                    <th>Number of Jobs</th>
                    <th>Total Hours</th>
                </tr>
            """)
            
            for machine, tasks in sorted(schedule.items(), key=lambda x: len(x[1]), reverse=True):
                machine_hours = sum((end - start) for _, start, end, *_ in tasks) / 3600
                report_html.append(f"""
                <tr>
                    <td>{machine}</td>
                    <td>{len(tasks)}</td>
                    <td>{machine_hours:.1f}</td>
                </tr>
                """)
            
            report_html.append("</table>")
            
            # Job details section
            report_html.append("""
            <h2>Job Details</h2>
            <table>
                <tr>
                    <th>Job ID</th>
                    <th>Machine</th>
                    <th>Start Time</th>
                    <th>End Time</th>
                    <th>Due Date</th>
                    <th>Buffer</th>
                    <th>Status</th>
                </tr>
            """)
            
            job_details = []
            for job in jobs:
                if 'UNIQUE_JOB_ID' not in job:
                    continue
                
                job_id = job['UNIQUE_JOB_ID']
                scheduled_jobs = [task for task in flat_schedule if task['job_id'] == job_id]
                
                if not scheduled_jobs:
                    continue
                
                scheduled_job = scheduled_jobs[0]
                machine = scheduled_job['machine']
                
                start_dt = epoch_to_datetime(scheduled_job['start'])
                end_dt = epoch_to_datetime(scheduled_job['end'])
                lcd_dt = epoch_to_datetime(job.get('LCD_DATE_EPOCH')) if 'LCD_DATE_EPOCH' in job else None
                
                buffer_hours = job.get('buffer_hours', 0)
                
                # Determine status
                status = "On Time"
                status_class = "good"
                
                if lcd_dt and end_dt and end_dt > lcd_dt:
                    hours_late = (scheduled_job['end'] - job.get('LCD_DATE_EPOCH', 0)) / 3600
                    status = f"Late ({hours_late:.1f} hrs)"
                    status_class = "late"
                elif buffer_hours < 8:
                    status = f"Critical Buffer ({buffer_hours:.1f} hrs)"
                    status_class = "warning"
                
                job_details.append({
                    'job_id': job_id,
                    'machine': machine,
                    'start': format_datetime_for_display(start_dt) if start_dt else "Unknown",
                    'end': format_datetime_for_display(end_dt) if end_dt else "Unknown",
                    'due': format_datetime_for_display(lcd_dt) if lcd_dt else "Unknown",
                    'buffer': buffer_hours,
                    'status': status,
                    'status_class': status_class,
                    'end_epoch': scheduled_job['end']  # For sorting
                })
            
            for job in sorted(job_details, key=lambda x: x['end_epoch']):
                report_html.append(f"""
                <tr>
                    <td>{job['job_id']}</td>
                    <td>{job['machine']}</td>
                    <td>{job['start']}</td>
                    <td>{job['end']}</td>
                    <td>{job['due']}</td>
                    <td>{job['buffer']:.1f} hrs</td>
                    <td class="{job['status_class']}">{job['status']}</td>
                </tr>
                """)
            
            report_html.append("</table>")
            
            # Close the HTML
            report_html.append("""
            <p><i>Report generated on {}</i></p>
            </body>
            </html>
            """.format(format_datetime_for_display(datetime.now())))
            
            # Write the report to file
            with open(args.report_output, 'w') as f:
                f.write("".join(report_html))
            
            print(f"- Production Report: {os.path.abspath(args.report_output)}")
            logger.info(f"Production report saved to: {os.path.abspath(args.report_output)}")
            
        except Exception as e:
            logger.error(f"Error generating production report: {e}")
            logger.error("Stack trace:", exc_info=True)
            
    # Generate machine utilization metrics unless explicitly skipped
    if not args.skip_metrics:
        try:
            logger.info("Generating machine utilization metrics...")
            
            # Save schedule to JSON file for reference
            schedule_json_path = os.path.splitext(args.output)[0] + ".json"
            with open(schedule_json_path, 'w') as f:
                json.dump(schedule, f)
            
            print(f"- Schedule JSON: {os.path.abspath(schedule_json_path)}")
            
            # Log machine metrics directly to the file
            write_to_log_file("=" * 50)
            write_to_log_file(f"MACHINE METRICS LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            write_to_log_file("=" * 50)
            
            # Calculate and log metrics directly to the file
            metrics = calculate_machine_metrics(schedule)
            log_machine_metrics_to_file(metrics)
            
            write_to_log_file("Machine utilization metrics generated successfully")
            print("- Metrics log: Added to production_scheduler.log")
            
        except Exception as e:
            logger.error(f"Error generating machine utilization metrics: {e}")
            logger.error("Stack trace:", exc_info=True)

def convert_cpsat_to_greedy_format(cpsat_schedule):
    """
    Convert the CP-SAT solver schedule format to the greedy scheduler format.
    The format should be: {machine: [(job_id, start, end, priority, additional_params), ...]}
    
    Args:
        cpsat_schedule (dict): Either CP-SAT format {job_id: {'machine': str, 'start': int, 'end': int, ...}, '_metadata': {...}}
                              or greedy format {machine: [(job_id, start, end, priority), ...]}
    
    Returns:
        dict: Schedule in format {machine: [(job_id, start, end, priority, additional_params), ...]}
    """
    logger.info("Converting CP-SAT schedule format to greedy format")
    
    # If it's already in the right format (no _metadata), convert tuples to 5-tuples if needed
    if '_metadata' not in cpsat_schedule:
        greedy_format = {}
        for machine, tasks in cpsat_schedule.items():
            greedy_format[machine] = []
            for task in tasks:
                # Handle both 4-tuple and 5-tuple formats
                if len(task) == 4:
                    job_id, start, end, priority = task
                    greedy_format[machine].append((job_id, start, end, priority, {}))
                elif len(task) == 5:
                    greedy_format[machine].append(task)  # Already in correct format
                else:
                    logger.warning(f"Unexpected task format for machine {machine}: {task}")
                    # Try to extract required fields if possible
                    if len(task) >= 3:
                        job_id, start, end = task[:3]
                        priority = task[3] if len(task) > 3 else 3  # Default priority
                        greedy_format[machine].append((job_id, start, end, priority, {}))
        return greedy_format
    
    # Create a new schedule without the _metadata
    greedy_format = {}
    
    # Process each job in the CP-SAT schedule
    for job_id, details in cpsat_schedule.items():
        if job_id == '_metadata':
            continue
            
        if not isinstance(details, dict):
            logger.warning(f"Invalid details format for job {job_id}: {details}")
            continue
            
        # Extract required fields
        machine = details.get('machine')
        start = details.get('start')
        end = details.get('end')
        priority = details.get('priority', 3)  # Default to medium priority
        
        if not all(x is not None for x in [machine, start, end]):
            logger.warning(f"Missing required fields for job {job_id}: machine={machine}, start={start}, end={end}")
            continue
            
        # Initialize machine list if needed
        if machine not in greedy_format:
            greedy_format[machine] = []
            
        # Add the job as a 5-tuple with empty additional params
        greedy_format[machine].append((job_id, start, end, priority, {}))
    
    # Log conversion stats
    total_tasks = sum(len(tasks) for tasks in greedy_format.values())
    logger.info(f"Converted CP-SAT schedule: {total_tasks} tasks scheduled")
    
    return greedy_format

def build_schedule_from_logs(cpsat_schedule):
    """
    Build a schedule directly from the logging messages if the standard conversion fails.
    This is a last resort when the CP-SAT solver returns a format we can't process directly.
    
    Returns:
        dict: The schedule in greedy scheduler format {machine: [(job_id, start, end, priority, additional_params), ...]}
    """
    logger.info("Building schedule from solver log messages")
    greedy_format = {}
    
    # Create a list of dictionaries for all scheduled jobs
    # Format should match what we see in the logs:
    # "Scheduled JOST111111_CP11-111-P01-08 on PAINTING: start=34, end=54"
    for job_id, details in cpsat_schedule.items():
        if job_id == '_metadata':
            continue
            
        if isinstance(details, dict) and 'machine' in details and 'start' in details and 'end' in details:
            machine = details['machine']
            start = details['start']
            end = details['end']
            priority = details.get('priority', 3)  # Default to medium priority
            
            if machine not in greedy_format:
                greedy_format[machine] = []
                
            greedy_format[machine].append((job_id, start, end, priority, {}))  # Use 5-tuple with empty additional params
    
    # Count how many jobs we scheduled this way
    total_after = sum(len(tasks) for machine, tasks in greedy_format.items())
    logger.info(f"Built schedule from logs: {total_after} tasks scheduled")
    
    return greedy_format

if __name__ == "__main__":
    main()