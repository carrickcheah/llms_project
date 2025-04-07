#!/usr/bin/env python3
# log_machine_metrics.py
"""
Production Scheduler with Machine Metrics Logging

This script runs the main production scheduler and then automatically
generates machine utilization and capacity metrics logs.

Usage:
    python log_machine_metrics.py [args]

Any arguments passed to this script will be forwarded to main.py.
"""

import os
import sys
import time
import subprocess
import argparse
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_main_script(args):
    """Run the main.py script with the given arguments."""
    cmd = ["uv", "run", "python", "main.py"] + args
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Main script completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running main script: {e}")
        if e.stdout:
            logger.error(f"Main script stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Main script stderr: {e.stderr}")
        return False, None

def extract_schedule_path(main_output):
    """Extract the schedule file path from main.py output."""
    # Default schedule path if we can't extract it
    default_path = "schedule.json"
    
    if not main_output:
        return default_path
    
    # Look for the line that contains the schedule path
    for line in main_output.split('\n'):
        if "- Gantt chart:" in line:
            # Extract the path from the line
            parts = line.split("- Gantt chart:")
            if len(parts) > 1:
                path = parts[1].strip()
                # The path might be an absolute path to an HTML file
                # We need to convert it to a JSON file path
                if path.endswith('.html'):
                    path = path.replace('.html', '.json')
                return path
    
    return default_path

def run_metrics_logger(schedule_path, output_dir="@PR"):
    """Run the machine_metrics_logger.py script."""
    # Replace @PR with the appropriate directory
    if output_dir == "@PR":
        output_dir = os.path.dirname(os.path.abspath(schedule_path))
    
    # Create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_log = os.path.join(output_dir, f"machine_utilization_{timestamp}.log")
    output_csv = os.path.join(output_dir, f"machine_metrics_{timestamp}.csv")
    
    cmd = [
        "uv", "run", "python", "machine_metrics_logger.py",
        "--schedule", schedule_path,
        "--output", output_log,
        "--csv", output_csv
    ]
    
    logger.info(f"Running metrics logger: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Metrics logger completed successfully")
        logger.info(f"Machine utilization log saved to: {output_log}")
        logger.info(f"Machine metrics CSV saved to: {output_csv}")
        return True, (output_log, output_csv)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running metrics logger: {e}")
        if e.stdout:
            logger.error(f"Metrics logger stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Metrics logger stderr: {e.stderr}")
        return False, None

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run production scheduler with machine metrics logging")
    parser.add_argument("--output-dir", default="@PR",
                        help="Directory to save metrics logs (default: same as schedule file)")
    parser.add_argument("main_args", nargs="*",
                        help="Arguments to pass to main.py")
    
    args, unknown_args = parser.parse_known_args()
    
    # Combine known and unknown args to pass to main.py
    main_args = args.main_args + unknown_args
    
    # Step 1: Run the main.py script
    success, main_output = run_main_script(main_args)
    
    if not success:
        logger.error("Failed to run main script, exiting.")
        return 1
    
    # Step 2: Extract the schedule file path
    schedule_path = extract_schedule_path(main_output)
    logger.info(f"Extracted schedule path: {schedule_path}")
    
    # Check if schedule file exists
    if not os.path.exists(schedule_path):
        # Try to find a JSON file with a similar name
        dirname = os.path.dirname(schedule_path)
        basename = os.path.basename(schedule_path)
        name_without_ext = os.path.splitext(basename)[0]
        
        potential_paths = [
            os.path.join(dirname, f"{name_without_ext}.json"),
            "schedule.json"
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                schedule_path = path
                logger.info(f"Found alternative schedule path: {schedule_path}")
                break
        else:
            logger.error(f"Schedule file not found: {schedule_path}")
            return 1
    
    # Step 3: Run the metrics logger
    success, log_paths = run_metrics_logger(schedule_path, args.output_dir)
    
    if not success:
        logger.error("Failed to run metrics logger, exiting.")
        return 1
    
    logger.info("Production scheduling and metrics logging completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 