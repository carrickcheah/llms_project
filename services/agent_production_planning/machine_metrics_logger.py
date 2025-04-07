#!/usr/bin/env python3
# machine_metrics_logger.py
"""
Machine Utilization and Capacity Metrics Logger

This script analyzes the production schedule and logs detailed machine utilization
and capacity metrics to a log file. It's designed to be run after main.py
has generated a schedule.

Usage:
    python machine_metrics_logger.py --schedule SCHEDULE_FILE --output OUTPUT_FILE [--append-log]
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
import pytz
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set up Singapore timezone
SG_TIMEZONE = pytz.timezone('Asia/Singapore')

def setup_logging(log_file_path, append=False):
    """
    Set up logging configuration.
    
    Args:
        log_file_path (str): Path to the log file
        append (bool): If True, append to existing log file; otherwise create new
    """
    # Create logger
    logger = logging.getLogger('machine_metrics')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Clear any existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler only when needed
    if not append or not os.path.exists(log_file_path):
        file_handler = logging.FileHandler(log_file_path, mode='a' if append else 'w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def load_schedule(schedule_path):
    """
    Load the schedule JSON file and convert it to the expected format.
    
    Args:
        schedule_path (str): Path to the schedule JSON file
        
    Returns:
        dict: Schedule dictionary in the format {machine: [(job_id, start, end, priority, metadata), ...]}
    """
    try:
        with open(schedule_path, 'r') as f:
            schedule_data = json.load(f)
            
        # Validate and convert if necessary
        if isinstance(schedule_data, dict):
            return schedule_data
        else:
            raise ValueError("Invalid schedule format: expected dictionary")
    except Exception as e:
        logging.error(f"Error loading schedule: {e}")
        return {}

def calculate_machine_metrics(schedule):
    """
    Calculate machine utilization metrics from the schedule.
    
    Args:
        schedule (dict): Schedule in the format {machine: [(job_id, start, end, priority, metadata), ...]}
        
    Returns:
        dict: Machine metrics including utilization, idle periods, and job statistics
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

def log_machine_metrics(metrics, logger):
    """
    Log the machine metrics in a readable format.
    
    Args:
        metrics (dict): Machine metrics dictionary
        logger (logging.Logger): Logger instance to use
    """
    logger.info("========== MACHINE UTILIZATION REPORT ==========")
    
    # First log the overall statistics
    if 'overall' in metrics:
        overall = metrics['overall']
        logger.info(f"Schedule timespan: {format_timestamp(overall['start_time'])} to {format_timestamp(overall['end_time'])}")
        logger.info(f"Total schedule duration: {format_seconds(overall['total_span_seconds'])} ({overall['total_span_seconds']/3600:.2f} hours)")
        logger.info("---------------------------------------------")
    
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
        logger.info(f"Machine: {machine}")
        logger.info(f"  Number of jobs: {num_jobs}")
        logger.info(f"  Timespan: {format_timestamp(start_time)} to {format_timestamp(end_time)}")
        logger.info(f"  Duration: {format_seconds(total_span)} ({total_span/3600:.2f} hours)")
        logger.info(f"  Working time: {format_seconds(working_time)} ({working_time/3600:.2f} hours)")
        logger.info(f"  Utilization: {utilization:.2f}%")
        
        # Log idle periods
        total_idle_time = sum(duration for _, _, duration in idle_periods)
        logger.info(f"  Idle periods: {len(idle_periods)}")
        logger.info(f"  Total idle time: {format_seconds(total_idle_time)} ({total_idle_time/3600:.2f} hours)")
        
        if idle_periods:
            logger.info("  Idle period details:")
            for i, (start, end, duration) in enumerate(idle_periods, 1):
                logger.info(f"    {i}. {format_timestamp(start)} to {format_timestamp(end)}: {format_seconds(duration)} ({duration/3600:.2f} hours)")
        
        # Log job durations
        logger.info("  Job durations:")
        for duration, job_id in sorted(machine_metrics['job_durations'], reverse=True)[:5]:  # Top 5 longest jobs
            logger.info(f"    {job_id}: {format_seconds(duration)} ({duration/3600:.2f} hours)")
        
        logger.info("---------------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Analyze machine utilization from production schedule")
    parser.add_argument("--schedule", required=True, help="Path to schedule JSON file")
    parser.add_argument("--output", default="machine_utilization.log", help="Output file for logs")
    parser.add_argument("--append-log", action="store_true", help="Append to existing log file instead of creating new")
    args = parser.parse_args()
    
    # Configure logging
    logger = setup_logging(args.output, append=args.append_log)
    
    # Add a header to clearly identify this log entry if appending
    if args.append_log:
        logger.info("=" * 50)
        logger.info(f"MACHINE METRICS LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 50)
    
    # Load schedule
    logger.info(f"Loading schedule from {args.schedule}")
    schedule = load_schedule(args.schedule)
    
    if not schedule:
        logger.error("No valid schedule found. Exiting.")
        return
    
    # Calculate metrics
    logger.info("Calculating machine utilization metrics...")
    metrics = calculate_machine_metrics(schedule)
    
    # Log the metrics
    log_machine_metrics(metrics, logger)
    
    logger.info("Machine metrics analysis complete.")

if __name__ == "__main__":
    main() 