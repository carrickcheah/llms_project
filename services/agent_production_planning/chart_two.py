# chart_two.py | dont edit this line
import os
import re
from datetime import datetime, timedelta
import plotly.figure_factory as ff
import plotly.offline as pyo
import pandas as pd
import plotly.graph_objects as go
import logging
from dotenv import load_dotenv
from ingest_data import load_jobs_planning_data
from greedy import greedy_schedule
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# At the top of the file
SINGAPORE_TZ = pytz.timezone('Asia/Singapore')

def format_date_correctly(epoch_timestamp, is_lcd_date=False):
    """Format an epoch timestamp maintaining exact time values from Excel."""
    default_date = "N/A"
    
    try:
        # More extensive validation to handle NaN values
        if (epoch_timestamp is None or 
            pd.isna(epoch_timestamp) or 
            not isinstance(epoch_timestamp, (int, float)) or 
            isinstance(epoch_timestamp, float) and (pd.isna(epoch_timestamp) or not pd.notna(epoch_timestamp)) or
            (isinstance(epoch_timestamp, (int, float)) and epoch_timestamp <= 0)):
            return default_date
        
        # Get the datetime in Singapore timezone
        date_obj = datetime.fromtimestamp(epoch_timestamp, tz=SINGAPORE_TZ)
        
        # For LCD_DATE column, use special handling for format
        if is_lcd_date:
            # Use the exact format and time from the Excel file
            formatted = date_obj.strftime('%d/%m/%y %H:%M')
        else:
            # For all other dates
            formatted = date_obj.strftime('%Y-%m-%d %H:%M')
        
        # For debug logging
        logger.debug(f"Formatted date for {'LCD_DATE' if is_lcd_date else 'other date'}: {epoch_timestamp} -> {formatted}")
        
        return formatted
    except Exception as e:
        logger.error(f"Error formatting timestamp {epoch_timestamp}: {e}")
        return default_date

def get_buffer_status_color(buffer_hours):
    """
    Get color for buffer status based on hours remaining.
    """
    if buffer_hours < 8:
        return "red"
    elif buffer_hours < 24:
        return "orange"
    elif buffer_hours < 72:
        return "yellow"
    else:
        return "green"

def extract_job_family(unique_job_id):
    """
    Extract the job family (e.g., 'CP08-231B' from 'JOB_CP08-231B-P01-06') from the UNIQUE_JOB_ID.
    UNIQUE_JOB_ID is in the format JOB_PROCESS_CODE.
    """
    try:
        process_code = unique_job_id.split('_', 1)[1]  # Split on first underscore to get PROCESS_CODE
    except IndexError:
        logger.warning(f"Could not extract PROCESS_CODE from UNIQUE_JOB_ID {unique_job_id}")
        return unique_job_id

    process_code = str(process_code).upper()
    match = re.search(r'(.*?)-P\d+', process_code)
    if match:
        family = match.group(1)
        logger.debug(f"Extracted family {family} from {unique_job_id}")
        return family
    parts = process_code.split("-P")
    if len(parts) >= 2:
        family = parts[0]
        logger.debug(f"Extracted family {family} from {unique_job_id} (using split)")
        return family
    logger.warning(f"Could not extract family from {unique_job_id}, using full code")
    return process_code

def extract_process_number(unique_job_id):
    """
    Extract the process sequence number (e.g., 1 from 'P01-06' in 'JOB_P01-06') or return 999 if not found.
    UNIQUE_JOB_ID is in the format JOB_PROCESS_CODE.
    """
    try:
        process_code = unique_job_id.split('_', 1)[1]  # Split on first underscore to get PROCESS_CODE
    except IndexError:
        logger.warning(f"Could not extract PROCESS_CODE from UNIQUE_JOB_ID {unique_job_id}")
        return 999

    match = re.search(r'P(\d{2})', str(process_code).upper())  # Match exactly two digits after P
    if match:
        seq = int(match.group(1))
        return seq
    return 999  # Default if parsing fails

def export_schedule_html(jobs, schedule, output_file='schedule_view.html'):
    """
    Export the schedule to an interactive HTML file.
    
    Args:
        jobs (list): List of job dictionaries with buffer information
        schedule (dict): Schedule as {machine: [(unique_job_id, start, end, priority), ...]} or
                        {machine: [(unique_job_id, start, end, priority, additional_params), ...]}
        output_file (str): Path to save the HTML file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Step 1: Create a mapping of job families and their processes in sequence
        family_processes = {}
        for job in jobs:
            unique_job_id = job['UNIQUE_JOB_ID']
            family = extract_job_family(unique_job_id)
            seq_num = extract_process_number(unique_job_id)
            
            if family not in family_processes:
                family_processes[family] = []
            
            family_processes[family].append((seq_num, unique_job_id, job))
        
        # Sort processes within each family by sequence number
        for family in family_processes:
            family_processes[family].sort(key=lambda x: x[0])
        
        # Create a lookup of original end and start times from the schedule
        end_times = {}
        start_times = {}
        for machine, tasks in schedule.items():
            for task in tasks:
                # Handle both 4-tuple and 5-tuple formats
                if len(task) >= 4:  # Both formats have at least 4 elements
                    unique_job_id, start, end = task[:3]  # First 3 elements are the same in both formats
                    end_times[unique_job_id] = end
                    start_times[unique_job_id] = start
        
        # Step 2: Apply time shifts from jobs to their visualization times
        adjusted_times = {}
        current_time = int(datetime.now().timestamp())
        
        # FIRST PRIORITY: EXPLICIT START_DATE OVERRIDES
        for job in jobs:
            unique_job_id = job['UNIQUE_JOB_ID']
            # Ensure all unique_job_ids have entries in the times dictionaries
            if unique_job_id not in start_times and unique_job_id not in end_times:
                logger.warning(f"Job {unique_job_id} missing from schedule, will not be visualized properly")
                continue
                
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] and unique_job_id in start_times:
                # For START_DATE jobs, always use exactly the requested START_DATE for visualization
                original_start = start_times[unique_job_id]
                original_end = end_times[unique_job_id]
                duration = original_end - original_start
                
                # Set the visualization start time exactly to START_DATE
                requested_start = job['START_DATE_EPOCH']
                adjusted_start = requested_start  # Use exactly the START_DATE
                adjusted_end = adjusted_start + duration
                
                # Store the adjusted times
                adjusted_times[unique_job_id] = (adjusted_start, adjusted_end)
                
                start_date_str = format_date_correctly(adjusted_start)
                logger.info(f"Job {unique_job_id} visualization using exact START_DATE: {start_date_str}")
        
        # SECOND PRIORITY: FAMILY-WIDE TIME SHIFTS        
        for family, processes in family_processes.items():
            # Check if any job in this family has a time shift
            family_time_shift = 0
            for seq_num, unique_job_id, job in processes:
                if 'family_time_shift' in job and abs(job['family_time_shift']) > 60:  # More than a minute
                    family_time_shift = job['family_time_shift']
                    break
            
            # If family has time shift, apply to all processes in family
            if family_time_shift != 0:
                logger.info(f"Applying time shift of {family_time_shift/3600:.1f} hours to family {family} for visualization")
                for seq_num, unique_job_id, job in processes:
                    # Skip if already processed by START_DATE override
                    if unique_job_id in adjusted_times:
                        continue
                        
                    if unique_job_id in start_times and unique_job_id in end_times:
                        original_start = start_times[unique_job_id]
                        original_end = end_times[unique_job_id]
                        
                        # Apply the time shift
                        adjusted_start = original_start - family_time_shift
                        adjusted_end = original_end - family_time_shift
                        
                        # Store the adjusted times
                        adjusted_times[unique_job_id] = (adjusted_start, adjusted_end)
                        
                        logger.info(f"Adjusted {unique_job_id}: START={format_date_correctly(adjusted_start)}, "
                                  f"END={format_date_correctly(adjusted_end)}")
        
        # Step 3: Process each job and create schedule data
        schedule_data = []
        
        for job in jobs:
            unique_job_id = job['UNIQUE_JOB_ID']
            if unique_job_id in end_times:
                # Get original scheduled times
                original_start = start_times[unique_job_id]
                original_end = end_times[unique_job_id]
                due_time = job.get('LCD_DATE_EPOCH', 0)
                
                # Use adjusted times if available, otherwise use original times
                if unique_job_id in adjusted_times:
                    job_start, job_end = adjusted_times[unique_job_id]
                else:
                    job_start, job_end = original_start, original_end
                
                # Override with exact START_DATE if specified
                if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH']:
                    # For display purposes
                    user_start_date = format_date_correctly(job['START_DATE_EPOCH'])
                    
                    # Find the family this job belongs to
                    family = extract_job_family(unique_job_id)
                    seq_num = extract_process_number(unique_job_id)
                    
                    # Check if this is the first process in the family
                    is_first_process = False
                    if family in family_processes and len(family_processes[family]) > 0:
                        is_first_process = (family_processes[family][0][1] == unique_job_id)
                    
                    # For jobs with START_DATE constraints, prioritize using the exact date
                    # IMPORTANT: We store the START_DATE_EPOCH value for display in the START_DATE column
                    # but we still want to calculate and show START_TIME and END_TIME values
                    # Check both formats of START_DATE_EPOCH (with and without space)
                    start_date_epoch = None
                    if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH']:
                        start_date_epoch = job['START_DATE_EPOCH']
                    elif 'START_DATE _EPOCH' in job and job['START_DATE _EPOCH']:
                        start_date_epoch = job['START_DATE _EPOCH']
                        
                    if start_date_epoch and start_date_epoch > current_time:
                        job_start = start_date_epoch
                        # Adjust end time to maintain the same duration
                        original_duration = original_end - original_start
                        job_end = job_start + original_duration
                        logger.info(f"Using START_DATE={user_start_date} for {unique_job_id} in visualization")
                
                # Format the dates for display
                # Always format job_start and job_end for display even if START_DATE is provided
                job_start_date = format_date_correctly(job_start)
                end_date = format_date_correctly(job_end)
                due_date = format_date_correctly(due_time, is_lcd_date=True)
                
                # IMPORTANT: Also update the original dictionaries to ensure data is preserved
                # This ensures that all jobs (not just START_DATE jobs) have proper START_TIME and END_TIME
                start_times[unique_job_id] = job_start
                end_times[unique_job_id] = job_end
                
                # Get START_DATE for display
                user_start_date = ""
                # Try multiple variations of the START_DATE field
                if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH']:
                    user_start_date = format_date_correctly(job['START_DATE_EPOCH'])
                # Also check for START_DATE _EPOCH (with space)
                elif 'START_DATE _EPOCH' in job and job['START_DATE _EPOCH']:
                    user_start_date = format_date_correctly(job['START_DATE _EPOCH'])
                
                # Calculate duration and buffer
                # Calculate HOURS_NEED based on JOB_QUANTITY and EXPECT_OUTPUT_PER_HOUR
                job_quantity = job.get('JOB_QUANTITY', 0)
                expect_output = job.get('EXPECT_OUTPUT_PER_HOUR', 0)
                
                # Validate job start and end times before calculations to prevent NaN
                valid_times = (job_start is not None and not pd.isna(job_start) and 
                               job_end is not None and not pd.isna(job_end) and
                               isinstance(job_start, (int, float)) and 
                               isinstance(job_end, (int, float)))
                
                # Get HOURS_NEED directly from the Excel data
                if 'HOURS_NEED' in job and job['HOURS_NEED'] is not None and not pd.isna(job['HOURS_NEED']) and isinstance(job['HOURS_NEED'], (int, float)):
                    # Use the HOURS_NEED value directly from Excel
                    hours_need = job['HOURS_NEED']
                    logger.info(f"Using HOURS_NEED={hours_need} directly from Excel for {unique_job_id}")
                # Fallback only if HOURS_NEED is not available in Excel
                elif valid_times:
                    # Fallback to scheduled duration if HOURS_NEED not found
                    duration_seconds = job_end - job_start
                    hours_need = duration_seconds / 3600
                    logger.info(f"HOURS_NEED not found in Excel for {unique_job_id}, using calculated duration: {hours_need:.2f} hours")
                

                
                # Calculate buffer only with valid times and due time
                valid_due_time = False
                if valid_times and due_time is not None and not pd.isna(due_time) and isinstance(due_time, (int, float)):
                    # Check if due time is reasonable (not too far in past or future)
                    current_time = int(datetime.now().timestamp())
                    # Only use due dates that are after the job's end time or at most 1 year in the future
                    if job_end <= due_time <= (current_time + 365 * 24 * 3600):
                        buffer_seconds = max(0, due_time - job_end)
                        buffer_hours = buffer_seconds / 3600
                        valid_due_time = True
                    elif due_time < job_end:
                        # Due date is before job end - LATE!
                        # Calculate negative buffer to show how many hours the job exceeds its deadline
                        buffer_seconds = due_time - job_end  # This will be negative
                        buffer_hours = buffer_seconds / 3600  # Negative hours
                        valid_due_time = True
                        logger.warning(f"Chart: Job {unique_job_id} will be LATE! Due at {datetime.fromtimestamp(due_time).strftime('%Y-%m-%d %H:%M')} but ends at {datetime.fromtimestamp(job_end).strftime('%Y-%m-%d %H:%M')}, BAL_HR={buffer_hours:.1f}")
                    else:
                        # Due date too far in future, might be incorrect
                        logger.warning(f"Chart: Due date for {unique_job_id} is too far in future, might be incorrect")
                
                if not valid_due_time:
                    if 'BAL_HR' in job and job['BAL_HR'] is not None and not pd.isna(job['BAL_HR']) and isinstance(job['BAL_HR'], (int, float)):
                        # Use pre-calculated BAL_HR if available
                        buffer_hours = job['BAL_HR']
                    else:
                        # Set a reasonable default buffer for invalid due dates
                        buffer_hours = 24.0
                
                # No capping - return true buffer value regardless of size
                if buffer_hours > 720:  # Just log large values (over 30 days) but don't cap them
                    logger.info(f"Chart: Job {unique_job_id} has a large buffer of {buffer_hours:.1f} hours ({buffer_hours/24:.1f} days)")
                
                # Job family and sequence
                # Use RSC_CODE if available, fall back to legacy ways of determining job code
                job_code = job.get('RSC_CODE', job.get('JOB_CODE', 
                           unique_job_id.split('_', 1)[1].split('-P')[0] if '-P' in unique_job_id.split('_', 1)[1] else unique_job_id))
                
                # Get resource location
                resource_location = job.get('RSC_LOCATION', 'Unknown')
                
                # Get job name
                job_name = job.get('JOB', job_code)
                
                # Validate buffer_hours for buffer status determination
                if buffer_hours is None or pd.isna(buffer_hours) or not isinstance(buffer_hours, (int, float)):
                    buffer_hours = 0.0
                
                # Get buffer status
                buffer_status = ""
                if buffer_hours < 8:
                    buffer_status = "Late"
                elif buffer_hours < 24:
                    buffer_status = "Warning"
                elif buffer_hours < 72:
                    buffer_status = "Caution"
                else:
                    buffer_status = "OK"
                
                # Get quantity information with validation
                job_quantity = job.get('JOB_QUANTITY', 0)
                if job_quantity is None or pd.isna(job_quantity) or not isinstance(job_quantity, (int, float)):
                    job_quantity = 0
                    
                expect_output = job.get('EXPECT_OUTPUT_PER_HOUR', 0)
                if expect_output is None or pd.isna(expect_output) or not isinstance(expect_output, (int, float)):
                    expect_output = 0
                    
                accumulated_output = job.get('ACCUMULATED_DAILY_OUTPUT', 0)
                if accumulated_output is None or pd.isna(accumulated_output) or not isinstance(accumulated_output, (int, float)):
                    accumulated_output = 0
                    
                # Calculate balance quantity with validated values
                try:
                    balance_quantity = job.get('BALANCE_QUANTITY', job_quantity - accumulated_output)
                    if balance_quantity is None or pd.isna(balance_quantity) or not isinstance(balance_quantity, (int, float)):
                        balance_quantity = max(0, job_quantity - accumulated_output)
                except:
                    balance_quantity = 0
                
                # Add to schedule data
                # IMPORTANT: We're keeping job_start_date and end_date in the schedule_data
                # even when START_DATE is specified to ensure they appear in the HTML output
                # Get PLAN_DATE from the job data if available (directly from Excel)
                plan_date = 'N/A'
                if 'PLAN_DATE' in job and job['PLAN_DATE'] is not None and not pd.isna(job['PLAN_DATE']):
                    # If PLAN_DATE is a timestamp, always format it as dd/mm/yy
                    if isinstance(job['PLAN_DATE'], pd.Timestamp):
                        # Format as dd/mm/yy without time component
                        plan_date = job['PLAN_DATE'].strftime('%d/%m/%y')
                    elif isinstance(job['PLAN_DATE'], (int, float)) and job['PLAN_DATE'] > 0:
                        # If it's an epoch timestamp
                        date_obj = datetime.fromtimestamp(job['PLAN_DATE'])
                        # Always format as dd/mm/yy without time component
                        plan_date = date_obj.strftime('%d/%m/%y')
                    else:
                        plan_date = str(job['PLAN_DATE'])
                
                schedule_data.append({
                    'PLAN_DATE': plan_date,
                    'LCD_DATE': due_date,
                    'JOB': job_name,
                    'UNIQUE_JOB_ID': unique_job_id,
                    'RSC_LOCATION': resource_location,
                    'RSC_CODE': job_code,
                    'NUMBER_OPERATOR': job.get('NUMBER_OPERATOR', 1),
                    'JOB_QUANTITY': job_quantity,
                    'EXPECT_OUTPUT_PER_HOUR': expect_output,
                    'PRIORITY': job.get('PRIORITY', 3),
                    'HOURS_NEED': hours_need,  # Using the formula JOB_QUANTITY/EXPECT_OUTPUT_PER_HOUR
                    'SETTING_HOURS': job.get('SETTING_HOURS', job.get('setup_time', 0)),
                    'BREAK_HOURS': job.get('BREAK_HOURS', job.get('break_time', 0)),
                    'NO_PROD': job.get('NO_PROD', job.get('downtime', 0)),
                    'START_DATE': user_start_date,
                    'ACCUMULATED_DAILY_OUTPUT': accumulated_output,
                    'BALANCE_QUANTITY': balance_quantity,
                    'START_TIME': job_start_date,
                    'END_TIME': end_date,
                    'BAL_HR': round(buffer_hours, 1),
                    'BUFFER_STATUS': buffer_status,
                })
        
        # Create DataFrame for easier HTML generation
        df = pd.DataFrame(schedule_data)
        
        # Convert LCD_DATE strings to datetime objects for proper chronological sorting
        # This handles the formatted date strings like '15/04/25 08:00' properly
        df['LCD_DATE_DT'] = pd.to_datetime(df['LCD_DATE'], format='%d/%m/%y %H:%M', errors='coerce')
        
        # Sort by the datetime objects in descending order (newest first)
        df = df.sort_values(by='LCD_DATE_DT', ascending=True)  # True = newest first
        
        # Calculate percentages safely to avoid division by zero
        total_jobs = len(df) if not df.empty else 1  # Avoid division by zero
        critical_percent = len(df[df['BUFFER_STATUS'] == 'Late']) / total_jobs * 100 if not df.empty else 0
        warning_percent = len(df[df['BUFFER_STATUS'] == 'Warning']) / total_jobs * 100 if not df.empty else 0
        caution_percent = len(df[df['BUFFER_STATUS'] == 'Caution']) / total_jobs * 100 if not df.empty else 0
        ok_percent = len(df[df['BUFFER_STATUS'] == 'OK']) / total_jobs * 100 if not df.empty else 0
        
        # Prepare the HTML content
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Production Scheduler</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.10.25/css/dataTables.bootstrap5.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.2.2/css/buttons.bootstrap5.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.2.9/css/responsive.bootstrap5.min.css">
    <style>
        body {{ 
            font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; 
            background-color: #f5f7fa;
            color: #333;
        }}
        .container-fluid {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .dashboard {{
            margin-bottom: 20px; 
            padding: 15px; 
            border-radius: 8px; 
            background: #fff;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border: 1px solid #eaeaea;
        }}
        .dashboard h3 {{
            margin-top: 0;
            font-weight: 600;
            color: #3a3a3a;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
            margin-bottom: 12px;
            font-size: 16px;
        }}
        table.dataTable {{
            border-collapse: separate !important;
            border-spacing: 0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            font-size: 10px;  /* Smaller base font size */
            width: 100%;
            table-layout: fixed;  /* Fixed table layout for better control */
            margin: 0;
            padding: 0;
        }}
        
        table.dataTable th {{
            font-size: 9px;  /* Extra small header font */
            padding: 1px 2px !important;  /* Minimal padding */
            font-weight: 600;
            white-space: nowrap;  /* Prevent wrapping in headers */
            overflow: hidden;
            text-overflow: ellipsis;  /* Show ellipsis for overflow */
        }}
        
        table.dataTable td {{
            font-size: 10px;  /* Extra small cell font */
            padding: 1px 2px !important;  /* Minimal padding */
            overflow: hidden;
            text-overflow: ellipsis;  /* Show ellipsis for overflow */
            white-space: normal;  /* Allow text to wrap */
        }}
        
        /* Set specific column widths based on content - reduced to fit all columns */
        table.dataTable th:nth-child(1), table.dataTable td:nth-child(1) {{ width: 40px; }} /* PLAN_DATE */
        table.dataTable th:nth-child(2), table.dataTable td:nth-child(2) {{ width: 55px; }} /* LCD_DATE */
        table.dataTable th:nth-child(3), table.dataTable td:nth-child(3) {{ width: 60px; }} /* JOB */
        table.dataTable th:nth-child(4), table.dataTable td:nth-child(4) {{ width: 100px; }} /* UNIQUE_JOB_ID */
        table.dataTable th:nth-child(5), table.dataTable td:nth-child(5) {{ width: 40px; }} /* RSC_LOCATION */
        table.dataTable th:nth-child(6), table.dataTable td:nth-child(6) {{ width: 55px; }} /* RSC_CODE */
        table.dataTable th:nth-child(7), table.dataTable td:nth-child(7) {{ width: 35px; }} /* NUMBER_OPERATOR */
        table.dataTable th:nth-child(8), table.dataTable td:nth-child(8) {{ width: 45px; }} /* JOB_QUANTITY */
        table.dataTable th:nth-child(9), table.dataTable td:nth-child(9) {{ width: 45px; }} /* EXPECT_OUTPUT_PER_HOUR */
        table.dataTable th:nth-child(10), table.dataTable td:nth-child(10) {{ width: 25px; }} /* PRIORITY */
        table.dataTable th:nth-child(11), table.dataTable td:nth-child(11) {{ width: 40px; }} /* HOURS_NEED */
        table.dataTable th:nth-child(12), table.dataTable td:nth-child(12) {{ width: 45px; }} /* SETTING_HOURS */
        table.dataTable th:nth-child(13), table.dataTable td:nth-child(13) {{ width: 45px; }} /* BREAK_HOURS */
        table.dataTable th:nth-child(14), table.dataTable td:nth-child(14) {{ width: 35px; }} /* NO_PROD */
        table.dataTable th:nth-child(15), table.dataTable td:nth-child(15) {{ width: 65px; }} /* START_DATE */
        table.dataTable th:nth-child(16), table.dataTable td:nth-child(16) {{ width: 55px; }} /* ACCUMULATED_DAILY_OUTPUT */
        table.dataTable th:nth-child(17), table.dataTable td:nth-child(17) {{ width: 45px; }} /* BALANCE_QUANTITY */
        table.dataTable th:nth-child(18), table.dataTable td:nth-child(18) {{ width: 65px; }} /* START_TIME */
        table.dataTable th:nth-child(19), table.dataTable td:nth-child(19) {{ width: 65px; }} /* END_TIME */
        table.dataTable th:nth-child(20), table.dataTable td:nth-child(20) {{ width: 35px; }} /* BAL_HR */
        table.dataTable th:nth-child(21), table.dataTable td:nth-child(21) {{ width: 50px; }} /* BUFFER_STATUS */
        
        /* Make table container scrollable horizontally */
        .table-container {{
            overflow-x: visible;
            margin-bottom: 30px;
            width: 100%;
        }}
        
        /* Highlight row on hover for better readability */
        table.dataTable tbody tr:hover {{
            background-color: rgba(0, 123, 255, 0.05) !important;
        }}
        
        /* Status styles */
        .status-critical {{ background-color: rgba(255, 0, 0, 0.05); }}
        .status-warning {{ background-color: rgba(255, 190, 0, 0.05); }}
        .status-caution {{ background-color: rgba(128, 0, 128, 0.05); }}
        .status-ok {{ background-color: rgba(0, 128, 0, 0.05); }}
        
        .status-badge {{
            display: inline-block;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: 600;
            text-align: center;
            width: 100%;
            font-size: 10px;
        }}
        
        .badge-critical {{ background-color: #ffcccc; color: #cc0000; border: 1px solid #ff8080; }}
        .badge-warning {{ background-color: #fff2cc; color: #b38600; border: 1px solid #ffdb4d; }}
        .badge-caution {{ background-color: #f0d6f0; color: #800080; border: 1px solid #d699d6; }}
        .badge-ok {{ background-color: #d6f0d6; color: #006600; border: 1px solid #99d699; }}
    </style>
</head>
<body>
    <div class="container-fluid" style="max-width: 100%; padding: 10px;">
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h1 class="mb-0" style="font-size: 20px;"><i class="bi bi-calendar-check me-2"></i>Production Scheduler</h1>
            <div>
                <button class="btn btn-sm btn-primary" onclick="window.print()">
                    <i class="bi bi-printer me-1"></i> Print
                </button>
                <button class="btn btn-sm btn-outline-secondary ms-2" onclick="exportTableToCSV('production_schedule.csv')">
                    <i class="bi bi-download me-1"></i> Export
                </button>
            </div>
        </div>
        
        <div class="dashboard row">
            <div class="col-md-6">
                <h3>Schedule Overview</h3>
                <p style="font-size: 12px;"><strong>Total Jobs:</strong> {len(df)}</p>
                <p style="font-size: 12px;"><strong>Date Range:</strong> {df['START_TIME'].min() if not df.empty else 'N/A'} to {df['END_TIME'].max() if not df.empty else 'N/A'}</p>
                <p style="font-size: 12px;"><strong>Total Duration:</strong> {df['HOURS_NEED'].sum() if not df.empty else 0} hours</p>
            </div>
            <div class="col-md-6">
                <h3>Buffer Status</h3>
                <div class="d-flex flex-column gap-2">
                    <div class="d-flex align-items-center">
                        <div class="status-badge badge-critical me-2" style="width: 80px;">Late</div>
                        <div class="progress flex-grow-1" style="height: 16px;">
                            <div class="progress-bar bg-danger" role="progressbar" 
                                style="width: {critical_percent}%;" 
                                aria-valuenow="{len(df[df['BUFFER_STATUS'] == 'Late']) if not df.empty else 0}" 
                                aria-valuemin="0" aria-valuemax="{len(df) if not df.empty else 0}">
                                {len(df[df['BUFFER_STATUS'] == 'Late']) if not df.empty else 0} jobs
                            </div>
                        </div>
                    </div>
                    <div class="d-flex align-items-center">
                        <div class="status-badge badge-warning me-2" style="width: 80px;">Warning</div>
                        <div class="progress flex-grow-1" style="height: 16px;">
                            <div class="progress-bar bg-warning text-dark" role="progressbar" 
                                style="width: {warning_percent}%;" 
                                aria-valuenow="{len(df[df['BUFFER_STATUS'] == 'Warning']) if not df.empty else 0}" 
                                aria-valuemin="0" aria-valuemax="{len(df) if not df.empty else 0}">
                                {len(df[df['BUFFER_STATUS'] == 'Warning']) if not df.empty else 0} jobs
                            </div>
                        </div>
                    </div>
                    <div class="d-flex align-items-center">
                        <div class="status-badge badge-caution me-2" style="width: 80px;">Caution</div>
                        <div class="progress flex-grow-1" style="height: 16px;">
                            <div class="progress-bar" role="progressbar" style="background-color: purple; width: {caution_percent}%;" 
                                aria-valuenow="{len(df[df['BUFFER_STATUS'] == 'Caution']) if not df.empty else 0}" 
                                aria-valuemin="0" aria-valuemax="{len(df) if not df.empty else 0}">
                                {len(df[df['BUFFER_STATUS'] == 'Caution']) if not df.empty else 0} jobs
                            </div>
                        </div>
                    </div>
                    <div class="d-flex align-items-center">
                        <div class="status-badge badge-ok me-2" style="width: 80px;">OK</div>
                        <div class="progress flex-grow-1" style="height: 16px;">
                            <div class="progress-bar bg-success" role="progressbar" 
                                style="width: {ok_percent}%;" 
                                aria-valuenow="{len(df[df['BUFFER_STATUS'] == 'OK']) if not df.empty else 0}" 
                                aria-valuemin="0" aria-valuemax="{len(df) if not df.empty else 0}">
                                {len(df[df['BUFFER_STATUS'] == 'OK']) if not df.empty else 0} jobs
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="table-container">
            <h2 style="font-size: 16px; margin-bottom: 10px;">Production Jobs</h2>
            <table id="scheduleTable" class="table table-striped table-bordered">
                <thead>
                    <tr>
                        <th>PLAN_DATE</th>
                        <th>LCD_DATE</th>
                        <th>JOB</th>
                        <th>UNIQUE_JOB_ID</th>
                        <th>RSC_LOCATION</th>
                        <th>RSC_CODE</th>
                        <th>NUMBER_OPERATOR</th>
                        <th>JOB_QUANTITY</th>
                        <th>EXPECT_OUTPUT_PER_HOUR</th>
                        <th>PRIORITY</th>
                        <th>HOURS_NEED</th>
                        <th>SETTING_HOURS</th>
                        <th>BREAK_HOURS</th>
                        <th>NO_PROD</th>
                        <th>START_DATE</th>
                        <th>ACCUMULATED_DAILY_OUTPUT</th>
                        <th>BALANCE_QUANTITY</th>
                        <th>START_TIME</th>
                        <th>END_TIME</th>
                        <th>BAL_HR</th>
                        <th>BUFFER_STATUS</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add rows for each job
        for _, row in df.iterrows():
            buffer_class = ""
            if row['BUFFER_STATUS'] == 'Late':
                buffer_class = "status-critical"
            elif row['BUFFER_STATUS'] == 'Warning':
                buffer_class = "status-warning"
            elif row['BUFFER_STATUS'] == 'Caution':
                buffer_class = "status-caution"
            elif row['BUFFER_STATUS'] == 'OK':
                buffer_class = "status-ok"
                
            buffer_badge_class = ""
            if row['BUFFER_STATUS'] == 'Late':
                buffer_badge_class = "badge-critical"
            elif row['BUFFER_STATUS'] == 'Warning':
                buffer_badge_class = "badge-warning"
            elif row['BUFFER_STATUS'] == 'Caution':
                buffer_badge_class = "badge-caution"
            elif row['BUFFER_STATUS'] == 'OK':
                buffer_badge_class = "badge-ok"
                
            # Format numeric values with proper handling for NaN
            hours_need_fmt = f"{row['HOURS_NEED']:.1f}" if pd.notna(row['HOURS_NEED']) else "0.0"
            setting_hours_fmt = f"{row['SETTING_HOURS']:.1f}" if pd.notna(row['SETTING_HOURS']) else "0.0"
            break_hours_fmt = f"{row['BREAK_HOURS']:.1f}" if pd.notna(row['BREAK_HOURS']) else "0.0"
            no_prod_fmt = f"{row['NO_PROD']:.1f}" if pd.notna(row['NO_PROD']) else "0.0"
            bal_hr_fmt = f"{row['BAL_HR']:.1f}" if pd.notna(row['BAL_HR']) else "0.0"
            
            # Format other possibly NaN values
            # Use the exact LCD_DATE format from Excel without any transformations
            # This ensures the date format (DD/MM/YY HH:MM) or (DD/MM/YY HH:MM:SS) and exact times are preserved
            lcd_date = row['LCD_DATE'] if pd.notna(row['LCD_DATE']) else "N/A"
            # If lcd_date is a timestamp object, format it to match Excel's format exactly
            if isinstance(lcd_date, pd.Timestamp):
                # Check if the timestamp has seconds component
                if lcd_date.second != 0:
                    lcd_date = lcd_date.strftime('%d/%m/%y %H:%M:%S')
                else:
                    lcd_date = lcd_date.strftime('%d/%m/%y %H:%M')
            job = row['JOB'] if pd.notna(row['JOB']) else ""
            unique_job_id = row['UNIQUE_JOB_ID'] if pd.notna(row['UNIQUE_JOB_ID']) else ""
            rsc_location = row['RSC_LOCATION'] if pd.notna(row['RSC_LOCATION']) else ""
            rsc_code = row['RSC_CODE'] if pd.notna(row['RSC_CODE']) else ""
            number_operator = row['NUMBER_OPERATOR'] if pd.notna(row['NUMBER_OPERATOR']) else 1
            job_quantity = row['JOB_QUANTITY'] if pd.notna(row['JOB_QUANTITY']) else 0
            expect_output = row['EXPECT_OUTPUT_PER_HOUR'] if pd.notna(row['EXPECT_OUTPUT_PER_HOUR']) else 0
            priority = row['PRIORITY'] if pd.notna(row['PRIORITY']) else 3
            start_date = row['START_DATE'] if pd.notna(row['START_DATE']) else "N/A"
            accumulated = row['ACCUMULATED_DAILY_OUTPUT'] if pd.notna(row['ACCUMULATED_DAILY_OUTPUT']) else 0
            balance = row['BALANCE_QUANTITY'] if pd.notna(row['BALANCE_QUANTITY']) else 0
            
            # IMPORTANT: Ensure START_TIME and END_TIME are always properly displayed
            # This is critical to show calculated values even for jobs without START_DATE constraints
            start_time = row['START_TIME'] 
            end_time = row['END_TIME']
            
            # Enhanced debugging for START_TIME and END_TIME values
            if pd.isna(start_time) or start_time == "" or start_time is None:
                logger.warning(f"Missing START_TIME for job {unique_job_id}")
                start_time = "N/A"
            if pd.isna(end_time) or end_time == "" or end_time is None:
                logger.warning(f"Missing END_TIME for job {unique_job_id}")
                end_time = "N/A"
            
            # Get PLAN_DATE from the row data and ensure proper formatting
            plan_date = row['PLAN_DATE'] if pd.notna(row['PLAN_DATE']) else "N/A"
            # Format PLAN_DATE consistently if it's still a timestamp
            if isinstance(plan_date, pd.Timestamp):
                # Always format as dd/mm/yy without time component
                plan_date = plan_date.strftime('%d/%m/%y')
            
            html_content += f"""
                    <tr class="{buffer_class}">
                        <td title="{plan_date}">{plan_date}</td>
                        <td title="{lcd_date}">{lcd_date}</td>
                        <td title="{job}">{job}</td>
                        <td title="{unique_job_id}">{unique_job_id}</td>
                        <td title="{rsc_location}">{rsc_location}</td>
                        <td title="{rsc_code}">{rsc_code}</td>
                        <td title="{number_operator}">{number_operator}</td>
                        <td title="{job_quantity}">{job_quantity}</td>
                        <td title="{expect_output}">{expect_output}</td>
                        <td title="{priority}">{priority}</td>
                        <td title="{hours_need_fmt}">{hours_need_fmt}</td>
                        <td title="{setting_hours_fmt}">{setting_hours_fmt}</td>
                        <td title="{break_hours_fmt}">{break_hours_fmt}</td>
                        <td title="{no_prod_fmt}">{no_prod_fmt}</td>
                        <td title="{start_date}">{start_date}</td>
                        <td title="{accumulated}">{accumulated}</td>
                        <td title="{balance}">{balance}</td>
                        <td title="{start_time}">{start_time}</td>
                        <td title="{end_time}">{end_time}</td>
                        <td title="{bal_hr_fmt}">{bal_hr_fmt}</td>
                        <td><div class="status-badge {buffer_badge_class}">{row['BUFFER_STATUS']}</div></td>
                    </tr>"""
        
        # Complete the HTML
        html_content += """
                </tbody>
            </table>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.10.25/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.10.25/js/dataTables.bootstrap5.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.2.2/js/dataTables.buttons.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.2.2/js/buttons.bootstrap5.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.2.2/js/buttons.html5.min.js"></script>
    <script src="https://cdn.datatables.net/responsive/2.2.9/js/dataTables.responsive.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.1.3/jszip.min.js"></script>
    <script>
        $(document).ready(function() {
            // Initialize DataTables with improved options
            $('#scheduleTable').DataTable({
                order: [[1, 'asc']], // Sort by LCD_DATE by default (column index 1), showing latest dates first
                pageLength: 25,
                lengthMenu: [10, 25, 50, 100, 200],
                responsive: true,
                scrollX: true,  // Enable horizontal scrolling
                scrollCollapse: true,
                autoWidth: false, // Don't auto-calculate widths
                columnDefs: [
                    // Set column visibility and width options
                    { "width": "90px", "targets": 0 }, // LCD_DATE
                    { "width": "95px", "targets": 1 }, // JOB
                    { "width": "120px", "targets": 2 }, // UNIQUE_JOB_ID
                    { "width": "65px", "targets": 3 }, // RSC_LOCATION
                    { "width": "80px", "targets": 4 }, // RSC_CODE
                    { "width": "60px", "targets": 5 }, // NUMBER_OPERATOR
                    { "width": "70px", "targets": 6 }, // JOB_QUANTITY
                    { "width": "75px", "targets": 7 }, // EXPECT_OUTPUT_PER_HOUR
                    { "width": "50px", "targets": 8 }, // PRIORITY
                    { "width": "60px", "targets": 9 }, // HOURS_NEED
                    { "width": "65px", "targets": 10 }, // SETTING_HOURS
                    { "width": "65px", "targets": 11 }, // BREAK_HOURS
                    { "width": "50px", "targets": 12 }, // NO_PROD
                    { "width": "90px", "targets": 13 }, // START_DATE
                    { "width": "80px", "targets": 14 }, // ACCUMULATED_DAILY_OUTPUT
                    { "width": "70px", "targets": 15 }, // BALANCE_QUANTITY
                    { "width": "90px", "targets": 16 }, // START_TIME
                    { "width": "90px", "targets": 17 }, // END_TIME
                    { "width": "50px", "targets": 18 }, // BAL_HR
                    { "width": "70px", "targets": 19 }  // BUFFER_STATUS
                ],
                dom: 'Bfrtip',
                buttons: [
                    {
                        extend: 'colvis',
                        text: '<i class="bi bi-eye me-1"></i> Show/Hide Columns',
                        className: 'btn btn-sm btn-secondary mb-2',
                        // Show this button first
                        postfixButtons: ['colvisRestore']
                    },
                    {
                        extend: 'excel',
                        text: '<i class="bi bi-file-earmark-excel me-1"></i> Excel',
                        className: 'btn btn-sm btn-success',
                        exportOptions: {
                            columns: ':visible'
                        }
                    },
                    {
                        extend: 'csv',
                        text: '<i class="bi bi-file-earmark-text me-1"></i> CSV',
                        className: 'btn btn-sm btn-info',
                        exportOptions: {
                            columns: ':visible'
                        }
                    },
                    {
                        extend: 'pdf',
                        text: '<i class="bi bi-file-earmark-pdf me-1"></i> PDF',
                        className: 'btn btn-sm btn-danger',
                        exportOptions: {
                            columns: ':visible'
                        }
                    }
                ],
                initComplete: function () {
                    // Add a legend at the top
                    var legendHtml = '<div class="mb-3 p-2 bg-white rounded shadow-sm" style="font-size: 11px;">' +
                        '<h5 class="border-bottom pb-2 mb-2" style="font-size: 14px;"><i class="bi bi-info-circle me-2"></i>Buffer Status Legend:</h5>' +
                        '<div class="d-flex flex-wrap gap-2">' +
                        '<div class="status-badge badge-critical">Late (<8h)</div>' +
                        '<div class="status-badge badge-warning">Warning (<24h)</div>' +
                        '<div class="status-badge badge-caution">Caution (<72h)</div>' +
                        '<div class="status-badge badge-ok">OK (>72h)</div>' +
                        '</div></div>';
                    $('.dataTables_wrapper').prepend(legendHtml);
                }
            });
        });
        
        function exportTableToCSV(filename) {
            var csv = [];
            var rows = document.querySelectorAll("table tr");
            
            for (var i = 0; i < rows.length; i++) {
                var row = [], cols = rows[i].querySelectorAll("td, th");
                
                for (var j = 0; j < cols.length; j++) {
                    // Get the text content, handling the case of the status badge
                    var text = cols[j].innerText;
                    if (cols[j].querySelector('.status-badge')) {
                        text = cols[j].querySelector('.status-badge').innerText;
                    }
                    row.push('"' + text.replace(/"/g, '""') + '"');
                }
                
                csv.push(row.join(","));
            }
            
            // Download CSV file
            downloadCSV(csv.join("\\n"), filename);
        }
        
        function downloadCSV(csv, filename) {
            var csvFile = new Blob([csv], {type: "text/csv"});
            var downloadLink = document.createElement("a");
            
            // File name
            downloadLink.download = filename;
            
            // Create a link to the file
            downloadLink.href = window.URL.createObjectURL(csvFile);
            
            // Hide download link
            downloadLink.style.display = "none";
            
            // Add the link to DOM
            document.body.appendChild(downloadLink);
            
            // Click download link
            downloadLink.click();
            
            // Remove the link
            document.body.removeChild(downloadLink);
        }
    </script>
</body>
</html>"""
        
        # Write the HTML file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML schedule view saved to: {os.path.abspath(output_file)}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating HTML schedule view: {e}")
        return False

if __name__ == "__main__":
    import sys
    from ingest_data import load_jobs_planning_data
    from greedy import greedy_schedule
    
    # Load environment variables
    load_dotenv()
    file_path = os.getenv('file_path')
    
    if not file_path:
        logger.error("No file_path found in environment variables.")
        exit(1)
        
    output_file = "schedule_view.html"
    
    try:
        # Load job data from environment variable path
        jobs, machines, setup_times = load_jobs_planning_data(file_path)
        logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines from {file_path}")
        
        # Create schedule
        schedule = greedy_schedule(jobs, machines, setup_times, enforce_sequence=True)
        logger.info(f"Generated schedule with {sum(len(jobs_list) for jobs_list in schedule.values())} scheduled tasks")
        
        # Export HTML
        result = export_schedule_html(jobs, schedule, output_file)
        if result:
            logger.info(f"Schedule exported successfully to: {os.path.abspath(output_file)}")
        else:
            logger.error("Failed to export schedule to HTML")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")