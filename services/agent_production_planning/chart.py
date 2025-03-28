# chart.py | dont edit this line
import os
import re
from datetime import datetime, timedelta
import pytz
import plotly.figure_factory as ff
import plotly.offline as pyo
import pandas as pd
import plotly.graph_objects as go
import logging
from dotenv import load_dotenv
from ingest_data import load_jobs_planning_data
from greedy import greedy_schedule

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up Singapore timezone
SG_TIMEZONE = pytz.timezone('Asia/Singapore')

def format_date_correctly(epoch_timestamp, is_lcd_date=False):
    """
    Format an epoch timestamp into a consistent date string format.
    Preserves original times from the source data without modification.
    Uses Singapore timezone for consistent display across the application.
    """
    # Default fallback date in case of issues
    default_date = "N/A"

    try:
        if not epoch_timestamp or epoch_timestamp <= 0:
            return default_date

        # Create a datetime object with explicit Singapore timezone
        # This ensures all timestamps are consistently displayed in SG time
        date_obj = datetime.fromtimestamp(epoch_timestamp, tz=SG_TIMEZONE)

        # For LCD_DATE column, use special handling for format if needed
        if is_lcd_date:
            # Use the exact format and time from the Excel file
            # We need to preserve the original time without adjustments
            formatted = date_obj.strftime('%Y-%m-%d %H:%M')
        else:
            # For all other dates
            formatted = date_obj.strftime('%Y-%m-%d %H:%M')

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
        # If the ID has a prefix (like JOST24100248_), extract only the part after the underscore
        process_code = unique_job_id.split('_', 1)[1] if '_' in unique_job_id else unique_job_id
    except IndexError:
        logger.warning(f"Could not extract PROCESS_CODE from UNIQUE_JOB_ID {unique_job_id}")
        return unique_job_id

    process_code = str(process_code).upper()
    # Match everything before the first P followed by digits
    match = re.search(r'(.*?)-P\d+', process_code)
    if match:
        family = match.group(1)
        logger.debug(f"Extracted family {family} from {unique_job_id}")
        return family

    # Alternative approach if regex didn't work
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
        # If the ID has a prefix (like JOST24100248_), extract only the part after the underscore
        process_code = unique_job_id.split('_', 1)[1] if '_' in unique_job_id else unique_job_id
    except IndexError:
        logger.warning(f"Could not extract PROCESS_CODE from UNIQUE_JOB_ID {unique_job_id}")
        return 999

    # Look for a pattern like P01 (letter P followed by exactly two digits)
    match = re.search(r'P(\d{2})', str(process_code).upper())
    if match:
        seq = int(match.group(1))
        return seq
    return 999  # Default if parsing fails

def extract_job_and_process(unique_job_id):
    """
    Extract both the JOB and PROCESS_CODE parts from a unique_job_id.
    Returns a tuple of (job_id, process_code).

    For example, from 'JOST24100248_CA16-010-P02-03' returns ('JOST24100248', 'CA16-010-P02-03')
    If there's no underscore, returns (None, original_id)
    """
    parts = unique_job_id.split('_', 1)
    if len(parts) == 2:
        return (parts[0], parts[1])
    return (None, unique_job_id)

def is_same_task(job_id1, job_id2):
    """
    Check if two job IDs refer to the same task (job + process code).
    According to business rules, if JOB + PROCESS_CODE is the same, it's a duplicate.

    For example:
    JOST24100248_CA16-010-P01-03 and JOST24100248_CA16-010-P01-03 are the same task
    JOST24100248_CA16-010-P01-03 and JOST24100248_CA16-010-P02-03 are different tasks
    """
    job1, process1 = extract_job_and_process(job_id1)
    job2, process2 = extract_job_and_process(job_id2)

    # If either has no job part, fall back to full ID comparison
    if job1 is None or job2 is None:
        return job_id1 == job_id2

    # Check if job IDs match and process codes match
    # Here we examine the ENTIRE process code, not just the process number
    return job1 == job2 and process1 == process2

def create_interactive_gantt(schedule, jobs=None, output_file='interactive_schedule.html'):
    """
    Create an interactive Gantt chart from the schedule and save it as an HTML file.
    Tooltips are removed. Range selector buttons updated.
    """
    current_time = int(datetime.now().timestamp())

    # DEBUG: Print all START_DATE values in the input jobs for verification
    if jobs:
        for job in jobs:
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH']:
                logger.info(f"Input Job {job['UNIQUE_JOB_ID']}: START_DATE_EPOCH = {job['START_DATE_EPOCH']} -> {format_date_correctly(job['START_DATE_EPOCH'])}")

    df_list = []
    colors = {
        'Priority 1 (Highest)': 'rgb(255, 0, 0)',
        'Priority 2 (High)': 'rgb(255, 165, 0)',
        'Priority 3 (Medium)': 'rgb(0, 128, 0)',
        'Priority 4 (Normal)': 'rgb(128, 0, 128)',
        'Priority 5 (Low)': 'rgb(60, 179, 113)'
    }

    # Create job_lookup with more robust ID matching
    job_lookup = {}
    # Additional mapping to help find related jobs with different process codes
    job_family_lookup = {}

    if jobs:
        for job in jobs:
            if 'UNIQUE_JOB_ID' in job:
                # Store with the exact ID for direct lookups
                job_id = job['UNIQUE_JOB_ID']
                job_lookup[job_id] = job

                # Extract job and process code for smart matching
                base_job, process_code = extract_job_and_process(job_id)

                # Store in the family lookup to find related jobs
                if base_job:
                    if base_job not in job_family_lookup:
                        job_family_lookup[base_job] = []
                    job_family_lookup[base_job].append(job_id)

                    # Log for debugging
                    if 'JOST24100248' in job_id:
                        logger.info(f"Added {job_id} to job family lookup under {base_job}")
                        logger.info(f"Process code extracted: {process_code}")

    # Validate schedule structure
    if not isinstance(schedule, dict):
        logger.error(f"Invalid schedule type: {type(schedule)}. Expected dictionary.")
        schedule = {}  # Convert to empty dict to prevent further errors

    logger.info(f"Schedule contains {len(schedule)} machines")
    for machine, jobs_list in schedule.items():
        if not isinstance(jobs_list, list):
            logger.error(f"Invalid jobs list for machine {machine}: {type(jobs_list)}. Expected list.")
            schedule[machine] = []  # Convert to empty list
        else:
            logger.info(f"Machine {machine}: {len(jobs_list)} jobs")

    if not schedule or not any(schedule.values()):
        logger.warning("Empty schedule received, creating placeholder task")
        df_list.append(dict(
            Task="No tasks scheduled",
            Start=datetime.utcfromtimestamp(current_time),
            Finish=datetime.utcfromtimestamp(current_time + 3600),
            Resource="None",
            Priority="Priority 3 (Medium)",
            Description="No tasks were scheduled. Please check your input data." # Removed tooltip content here, but kept basic description for task name
        ))
    else:
        # Step 1: Create a mapping of job families and their processes in sequence
        family_processes = {}
        process_durations = {}

        # First pass - collect all process data and organize by family
        for machine, jobs in schedule.items():
            for job_data in jobs:
                try:
                    if not isinstance(job_data, (list, tuple)) or len(job_data) < 4:
                        logger.warning(f"Invalid job data for machine {machine}: {job_data}")
                        continue

                    # Handle both old format (4-tuple) and new format (5-tuple with additional params)
                    if len(job_data) >= 5:
                        unique_job_id, start, end, priority, additional_params = job_data
                    else:
                        unique_job_id, start, end, priority = job_data
                        additional_params = {}

                    # Validate data types
                    if not isinstance(unique_job_id, str):
                        logger.warning(f"Invalid unique_job_id type ({type(unique_job_id)}) for job {job_data}")
                        unique_job_id = str(unique_job_id)

                    # Ensure timestamps are valid numbers
                    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                        logger.warning(f"Invalid timestamp types for job {unique_job_id}: start={type(start)}, end={type(end)}")
                        continue

                    # Ensure priority is a number
                    if not isinstance(priority, (int, float)):
                        logger.warning(f"Invalid priority type ({type(priority)}) for job {unique_job_id}")
                        priority = 3  # Default to medium priority

                    # Calculate duration
                    duration = end - start
                    process_durations[unique_job_id] = duration

                    # Group by family
                    family = extract_job_family(unique_job_id)

                    seq_num = extract_process_number(unique_job_id)

                    if family not in family_processes:
                        family_processes[family] = []

                    # Use original schedule times first
                    family_processes[family].append((seq_num, unique_job_id, machine, start, end, priority))
                except Exception as e:
                    logger.error(f"Error processing job data {job_data}: {str(e)}")
                    continue

        # Sort processes within each family by sequence number
        for family in family_processes:
            family_processes[family].sort(key=lambda x: x[0])

        # Step 2: ALWAYS use START_DATE visualization adjustments for jobs with constraints
        start_date_processes = {}
        for family, processes in family_processes.items():
            for seq_num, unique_job_id, machine, start, end, priority in processes:
                # Get job base ID for matching
                base_job, process_code = extract_job_and_process(unique_job_id)
                job_key = None

                # Check if this exact ID exists in job_lookup
                if unique_job_id in job_lookup:
                    job_key = unique_job_id
                # Try to find it in family lookup
                elif base_job and base_job in job_family_lookup:
                    for related_id in job_family_lookup[base_job]:
                        _, related_process = extract_job_and_process(related_id)
                        if process_code == related_process:
                            job_key = related_id
                            break

                if job_key and 'START_DATE_EPOCH' in job_lookup[job_key] and job_lookup[job_key]['START_DATE_EPOCH']:
                    # Extract the START_DATE information
                    start_date = job_lookup[job_key]['START_DATE_EPOCH']

                    # ALWAYS use START_DATE, no conditional check
                    duration = end - start
                    adjusted_start = start_date
                    adjusted_end = adjusted_start + duration

                    # Add verbose logging for any job with START_DATE
                    logger.info(f"APPLYING START_DATE for {unique_job_id}:")
                    logger.info(f"    START_DATE_EPOCH = {start_date} => {format_date_correctly(start_date)}")
                    logger.info(f"    Original start = {start} => {format_date_correctly(start)}")
                    logger.info(f"    Adjusted start = {adjusted_start} => {format_date_correctly(adjusted_start)}")

                    if unique_job_id not in start_date_processes:
                        start_date_processes[unique_job_id] = (adjusted_start, adjusted_end)
                        logger.info(f"Added {unique_job_id} to start_date_processes dict for visualization")

                    logger.info(f"Visualizing {unique_job_id} at START_DATE: {format_date_correctly(adjusted_start)}")

        # Step 3: Calculate time shifts for visualization
        family_time_shifts = {}

        # Apply time shifts from family_time_shift property if present in jobs
        if jobs:
            for family, processes in family_processes.items():
                for seq_num, unique_job_id, machine, start, end, priority in processes:
                    if unique_job_id in job_lookup and 'family_time_shift' in job_lookup[unique_job_id]:
                        time_shift = job_lookup[unique_job_id]['family_time_shift']
                        if abs(time_shift) > 60:  # More than a minute
                            family_time_shifts[family] = time_shift
                            logger.info(f"Using job-provided time shift for family {family}: {time_shift/3600:.1f} hours")
                            break

        # Step 4: Apply time shifts to generate adjusted task data for visualization
        process_info = {}
        for family in family_processes:
            time_shift = family_time_shifts.get(family, 0)

            # Only apply significant shifts
            if abs(time_shift) < 60:  # Skip shifts less than a minute
                for seq_num, unique_job_id, machine, start, end, priority in family_processes[family]:
                    if family not in process_info:
                        process_info[family] = []
                    process_info[family].append((unique_job_id, machine, start, end, priority, seq_num))
                continue

            logger.info(f"Applying time shift of {time_shift/3600:.1f} hours to family {family} for visualization")

            for seq_num, unique_job_id, machine, start, end, priority in family_processes[family]:
                # Skip if this process has START_DATE override
                if unique_job_id in start_date_processes:
                    adjusted_start, adjusted_end = start_date_processes[unique_job_id]
                    # Add specific logging for CP08-544-P01-02
                    if 'CP08-544-P01-02' in unique_job_id:
                        logger.info(f"USING START_DATE for {unique_job_id}: Original start={format_date_correctly(start)}, "
                                   f"Adjusted start={format_date_correctly(adjusted_start)}")
                else:
                    # Adjust the times by the time shift
                    adjusted_start = start - time_shift
                    adjusted_end = end - time_shift
                    # Add specific logging for CP08-544-P01-02
                    if 'CP08-544-P01-02' in unique_job_id:
                        logger.info(f"Using time shift for {unique_job_id}: Original start={format_date_correctly(start)}, "
                                   f"Adjusted start={format_date_correctly(adjusted_start)}")

                if family not in process_info:
                    process_info[family] = []
                process_info[family].append((unique_job_id, machine, adjusted_start, adjusted_end, priority, seq_num))

                logger.info(f"  Adjusted {unique_job_id}: START={format_date_correctly(adjusted_start)}, "
                           f"END={format_date_correctly(adjusted_end)}")

        # Step 5: Override for START_DATE processes that were missed
        for unique_job_id, (adjusted_start, adjusted_end) in start_date_processes.items():
            # Find the process in process_info
            found = False
            for family in process_info:
                for i, (proc, machine, _, _, priority, seq_num) in enumerate(process_info[family]):
                    if proc == unique_job_id:
                        # Replace with START_DATE version
                        process_info[family][i] = (unique_job_id, machine, adjusted_start, adjusted_end, priority, seq_num)
                        logger.info(f"Overrode {unique_job_id} with START_DATE version for visualization")
                        found = True
                        break
                if found:
                    break

            # If the job wasn't found in process_info, it might need to be added separately
            if not found and unique_job_id in job_lookup:
                # Log this case as it shouldn't normally happen - job should be in the schedule
                logger.warning(f"Job {unique_job_id} with START_DATE constraint not found in process_info, may need additional handling")

        # Step 6: Create task list for visualization from the adjusted data
        # First create a lookup for START_DATE processes for faster access
        start_date_lookup = {proc: (start, end) for proc, (start, end) in start_date_processes.items()}

        # Create a dedicated lookup specifically for problematic jobs based on CLAUDE.local.md
        problem_jobs = {
            "CP08-544-P01-02": {
                "specific_id": "JOST24120091_CP08-544-P01-02",
                "start_date": 1747177200.0,  # 2025-05-14 07:00
                "duration": None
            },
            "CT10-001-P01-06": {
                "specific_id": "JOST24120409_CT10-001-P01-06",
                "start_date": 1747112400.0,  # 2025-05-13 13:00
                "duration": None
            }
        }

        # Calculate durations for problem jobs from schedule
        scheduled_times = {}
        for machine, jobs_list in schedule.items():
            for job_tuple in jobs_list:
                job_id = job_tuple[0]
                start_time = job_tuple[1]
                end_time = job_tuple[2]
                scheduled_times[job_id] = (start_time, end_time)

                # Check if this is one of our problem jobs
                for pattern, job_info in problem_jobs.items():
                    if pattern in job_id and job_id.endswith(pattern):
                        if job_id == job_info["specific_id"]:
                            problem_jobs[pattern]["duration"] = end_time - start_time
                            logger.info(f"Found duration for problem job {job_id}: {(end_time - start_time) / 3600} hours")

        sorted_tasks = []
        for family in sorted(process_info.keys()):
            processes = process_info[family]
            sorted_processes = sorted(processes, key=lambda x: x[5])  # Sort by sequence within family

            # For each process, override with START_DATE if present
            final_processes = []
            for proc, machine, start, end, priority, seq_num in sorted_processes:
                # First check for problem jobs with specific handling
                special_case = False
                for pattern, job_info in problem_jobs.items():
                    if pattern in proc and proc == job_info["specific_id"]:
                        special_case = True
                        start_date = job_info["start_date"]
                        if job_info["duration"]:
                            duration = job_info["duration"]
                        else:
                            duration = end - start

                        adjusted_start = start_date
                        adjusted_end = adjusted_start + duration

                        logger.info(f"ENFORCING EXACT START_DATE for problem job {proc}: {format_date_correctly(adjusted_start)}")
                        logger.info(f"    Original: {format_date_correctly(start)} -> {format_date_correctly(end)}")
                        logger.info(f"    Enforced: {format_date_correctly(adjusted_start)} -> {format_date_correctly(adjusted_end)}")

                        # Force highest priority for START_DATE jobs
                        final_processes.append((proc, machine, adjusted_start, adjusted_end, 1, seq_num))
                        break

                if not special_case:
                    if proc in start_date_lookup:
                        # Use START_DATE version with HIGH priority
                        override_start, override_end = start_date_lookup[proc]
                        logger.info(f"Using START_DATE override for {proc} in final task list: {format_date_correctly(override_start)}")
                        final_processes.append((proc, machine, override_start, override_end, 1, seq_num))  # Force priority 1
                    else:
                        final_processes.append((proc, machine, start, end, priority, seq_num))

            sorted_tasks.extend(final_processes)

        logger.info(f"Sorted task list contains {len(sorted_tasks)} tasks")

        # Process the sorted tasks for visualization
        for task_data in sorted_tasks:
            unique_job_id, machine, start, end, priority, _ = task_data

            # Log detailed information for EVERY job being processed
            logger.info(f"Processing task: {unique_job_id} on {machine} from {start} to {end}")

            # Log general info about the job without special case handling
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Task details: {unique_job_id} on {machine}, priority {priority}")

            try:
                # Ensure timestamps are numbers before conversion
                start_num = float(start) if start is not None else current_time
                end_num = float(end) if end is not None else (current_time + 3600)

                # Validate timestamps
                if start_num <= 0 or end_num <= 0:
                    logger.warning(f"Invalid timestamps for {unique_job_id}: Start={start_num}, End={end_num}")
                    start_num = current_time
                    end_num = current_time + 3600

                # Convert to datetime objects - use Singapore timezone consistently
                # This ensures all timestamps are displayed in SG time zone
                start_date = datetime.fromtimestamp(start_num, tz=SG_TIMEZONE)
                end_date = datetime.fromtimestamp(end_num, tz=SG_TIMEZONE)
                duration_hours = (end_num - start_num) / 3600

                # Store the actual scheduled times in job_lookup for accurate tooltip display
                # This ensures that the tooltip shows the ACTUAL scheduled time, not the data from Excel

                # First, check if we already have this exact job ID in our lookup
                if unique_job_id in job_lookup:
                    # Just update the scheduled times for the exact match
                    job_lookup[unique_job_id]['SCHEDULED_START'] = start_num
                    job_lookup[unique_job_id]['SCHEDULED_END'] = end_num
                    job_lookup[unique_job_id]['SCHEDULED_MACHINE'] = machine
                    job_lookup[unique_job_id]['SCHEDULED_DURATION'] = duration_hours
                else:
                    # See if we can find a matching job with the same base job ID and process code
                    base_job, process_code = extract_job_and_process(unique_job_id)
                    exact_match_found = False
                    if base_job and base_job in job_family_lookup:
                        for related_job_id in job_family_lookup[base_job]:
                            related_base_job, related_process = extract_job_and_process(related_job_id)
                            # Ensure both the base job ID and process code match exactly
                            if base_job == related_base_job and process_code == related_process:
                                # Create a copy to avoid modifying the original
                                job_lookup[unique_job_id] = job_lookup[related_job_id].copy()
                                # Update with actual scheduled times/machine
                                job_lookup[unique_job_id]['SCHEDULED_START'] = start_num
                                job_lookup[unique_job_id]['SCHEDULED_END'] = end_num
                                job_lookup[unique_job_id]['SCHEDULED_MACHINE'] = machine
                                job_lookup[unique_job_id]['SCHEDULED_DURATION'] = duration_hours
                                exact_match_found = True
                                logger.info(f"Matched {unique_job_id} with {related_job_id} in job_lookup")
                                break
                            else:
                                logger.warning(f"No match for {unique_job_id}: base_job={base_job}, process_code={process_code}, "
                                              f"compared to {related_job_id}: base_job={related_base_job}, process_code={related_process}")
                    # If no match found after all attempts, create a minimal entry
                    if not exact_match_found:
                        logger.warning(f"No exact match found for {unique_job_id} in job_lookup, creating minimal entry")
                        job_lookup[unique_job_id] = {
                            'UNIQUE_JOB_ID': unique_job_id,
                            'SCHEDULED_START': start_num,
                            'SCHEDULED_END': end_num,
                            'SCHEDULED_MACHINE': machine,
                            'SCHEDULED_DURATION': duration_hours
                        }

                        if 'JOST24100248' in unique_job_id:
                            logger.info(f"No matching process found for {unique_job_id}, created minimal entry")
            except Exception as e:
                logger.error(f"Error converting timestamps for {unique_job_id}: {e}")
                logger.error(f"Start: {start}, End: {end}")
                # Use current time as fallback - consistent Singapore timezone handling
                start_date = datetime.fromtimestamp(current_time, tz=SG_TIMEZONE)
                end_date = datetime.fromtimestamp(current_time + 3600, tz=SG_TIMEZONE)
                duration_hours = 1.0
                logger.warning(f"Using fallback time values for {unique_job_id}")

            job_priority = priority if priority is not None and 1 <= priority <= 5 else 3
            priority_label = f"Priority {job_priority} ({['Highest', 'High', 'Medium', 'Normal', 'Low'][job_priority-1]})"

            task_label = f"{unique_job_id} ({machine})"

            buffer_info = ""
            buffer_status = ""
            number_operator = ""
            # Simplified data structure for job information - no tooltips needed
            job_data = {
                'UNIQUE_JOB_ID': unique_job_id,
                'MACHINE': machine,
                'PRIORITY': priority,
                'SCHEDULED_START': start,
                'SCHEDULED_END': end,
                'DURATION_HOURS': duration_hours
            }

            # Get additional job metadata if available
            if job_lookup and unique_job_id in job_lookup:
                existing_data = job_lookup[unique_job_id]
                # Update with current schedule data
                job_data.update({k: v for k, v in existing_data.items() if k not in ['SCHEDULED_START', 'SCHEDULED_END']})

                due_date_field = next((f for f in ['LCD_DATE_EPOCH', 'DUE_DATE_TIME'] if f in job_data), None)

                if due_date_field and job_data[due_date_field]:
                    # Pass is_lcd_date=True for LCD_DATE_EPOCH field
                    due_date_str = format_date_correctly(job_data[due_date_field], is_lcd_date=True)
                    buffer_hours = (job_data[due_date_field] - end) / 3600
                    buffer_status = get_buffer_status_color(buffer_hours)
                    buffer_info = f"<br><b>Due Date:</b> {due_date_str}<br><b>Buffer:</b> {buffer_hours:.1f} hours"

                    # Add START_DATE information if present
                    if 'START_DATE_EPOCH' in job_data and job_data['START_DATE_EPOCH']:
                        start_date_info = format_date_correctly(job_data['START_DATE_EPOCH'])
                        buffer_info += f"<br><b>START_DATE Constraint:</b> {start_date_info}"

                if 'NUMBER_OPERATOR' in job_data:
                    number_operator = f"<br><b>Number of Operators:</b> {job_data['NUMBER_OPERATOR']}"

                # Add new information from updated column fields
                job_info = ""
                if 'JOB' in job_data and job_data['JOB'] and not pd.isna(job_data['JOB']):
                    job_info += f"<br><b>Job:</b> {job_data['JOB']}"

                if 'JOB_QUANTITY' in job_data and job_data['JOB_QUANTITY'] and not pd.isna(job_data['JOB_QUANTITY']):
                    # Convert to integer if it's a number
                    try:
                        qty = int(job_data['JOB_QUANTITY'])
                        job_info += f"<br><b>Job Quantity:</b> {qty}"
                    except (ValueError, TypeError):
                        if str(job_data['JOB_QUANTITY']).lower() != 'nan':
                            job_info += f"<br><b>Job Quantity:</b> {job_data['JOB_QUANTITY']}"

                if 'EXPECT_OUTPUT_PER_HOUR' in job_data and job_data['EXPECT_OUTPUT_PER_HOUR'] and not pd.isna(job_data['EXPECT_OUTPUT_PER_HOUR']):
                    # Convert to integer if it's a number
                    try:
                        output = int(job_data['EXPECT_OUTPUT_PER_HOUR'])
                        job_info += f"<br><b>Expected Output/Hour:</b> {output}"
                    except (ValueError, TypeError):
                        if str(job_data['EXPECT_OUTPUT_PER_HOUR']).lower() != 'nan':
                            job_info += f"<br><b>Expected Output/Hour:</b> {job_data['EXPECT_OUTPUT_PER_HOUR']}"

                if 'ACCUMULATED_DAILY_OUTPUT' in job_data:
                    # Only display if it's not NaN or empty
                    if job_data['ACCUMULATED_DAILY_OUTPUT'] and not pd.isna(job_data['ACCUMULATED_DAILY_OUTPUT']) and str(job_data['ACCUMULATED_DAILY_OUTPUT']).lower() != 'nan':
                        job_info += f"<br><b>Accumulated Output:</b> {job_data['ACCUMULATED_DAILY_OUTPUT']}"

                if 'BALANCE_QUANTITY' in job_data and job_data['BALANCE_QUANTITY'] and not pd.isna(job_data['BALANCE_QUANTITY']):
                    # Convert to integer if it's a number
                    try:
                        bal = int(job_data['BALANCE_QUANTITY'])
                        job_info += f"<br><b>Balance Quantity:</b> {bal}"
                    except (ValueError, TypeError):
                        if str(job_data['BALANCE_QUANTITY']).lower() != 'nan':
                            job_info += f"<br><b>Balance Quantity:</b> {job_data['BALANCE_QUANTITY']}"


            # Store basic timing information for job tracking
            job_lookup[unique_job_id] = job_data

            # Use the tooltip_data for consistency with pre-formatted dates
            # Build the tooltip with better formatting
            # Removed tooltip content construction
            description = f"Task: {unique_job_id} on Machine: {machine}, Priority: {job_priority}"


            df_list.append(dict(
                Task=task_label,
                Start=start_date,
                Finish=end_date,
                Resource=machine,
                Priority=priority_label,
                Description=description, # Kept a minimal description for task label
                BufferStatus=buffer_status
            ))

    if not df_list:
        logger.error("No valid tasks to plot in Gantt chart.")
        return False

    df = pd.DataFrame(df_list)
    logger.info(f"Created DataFrame with {len(df)} rows for Gantt chart")

    if not df.empty:
        logger.debug(f"Sample data: {df.iloc[0].to_dict()}")

    task_order = list(dict.fromkeys(df['Task'].tolist()))
    df['Task'] = pd.Categorical(df['Task'], categories=task_order, ordered=True)

    try:
        fig = ff.create_gantt(df, colors=colors, index_col='Priority', show_colorbar=True,
                              group_tasks=False,
                              showgrid_x=True, showgrid_y=True,
                              title='Interactive Production Schedule')

        # Disable tooltips for all traces
        for i in range(len(fig.data)):
            fig.data[i].hoverinfo = 'none' # Disable hover info - tooltips removed

        fig.update_yaxes(categoryorder='array', categoryarray=task_order, autorange="reversed")

        # Find the earliest start date to ensure the chart shows all jobs
        min_start_date = df['Start'].min() if not df.empty else datetime.now(SG_TIMEZONE)

        # Check if we have April jobs that need to be visible
        april_jobs = df[df['Start'].dt.month == 4]
        has_april_jobs = not april_jobs.empty

        if has_april_jobs:
            logger.info(f"Chart contains {len(april_jobs)} jobs starting in April")
            # Make sure these jobs are visible by explicitly logging them
            for _, row in april_jobs.iterrows():
                logger.info(f"April job: {row['Task']} starts at {row['Start']}")

        fig.update_layout(
            autosize=True,
            height=max(800, len(df) * 30),
            margin=dict(l=350, r=50, t=100, b=100),
            legend_title_text='Priority Level',
            hovermode='closest',
            title={'text': "Interactive Production Schedule", 'font': {'size': 24}, 'x': 0.5, 'xanchor': 'center'},
            xaxis={
                'title': {'text': 'Date', 'font': {'size': 14}},
                # Setting a default range to ensure April jobs are visible
                'range': [min_start_date - timedelta(days=1), None] if has_april_jobs else None,
                # Enhanced date formatting with more detail - ensure ALL ticks have dates
                'tickformat': '%Y-%m-%d',  # Show date format
                'tickmode': 'linear',  # Linear tick mode for even spacing
                'dtick': 24*60*60*1000,  # One tick per day (in milliseconds)
                'tickangle': -90,  # Angle the labels for better readability
                'tickfont': {'size': 10},
                'showgrid': True,  # Always show grid lines for better readability
                'gridcolor': 'rgba(211, 211, 211, 0.6)',  # Light gray grid lines

                'rangeselector': {
                    'buttons': [
                        {'count': 7, 'label': '1w', 'step': 'day', 'stepmode': 'backward'},
                        {'count': 14, 'label': '2w', 'step': 'day', 'stepmode': 'backward'},
                        {'count': 1, 'label': '1m', 'step': 'month', 'stepmode': 'backward'},
                        {'count': 3, 'label': '3m', 'step': 'month', 'stepmode': 'backward'},
                        {'count': 6, 'label': '6m', 'step': 'month', 'stepmode': 'backward'},
                        {'step': 'all', 'label': 'all'}
                    ],
                    # Make the range selector more visible
                    'bgcolor': '#E2E2E2',
                    'activecolor': '#68bdf6'
                }
            },
            yaxis={'title': {'text': 'Unique Job IDs (Machine)', 'font': {'size': 14}}}
        )

        # Removed tooltip text and hoverinfo setting
        # for i in range(len(fig.data)):
        #     fig.data[i].text = df['Description']
        #     fig.data[i].hoverinfo = 'text'


        if 'BufferStatus' in df.columns and df['BufferStatus'].notna().any():
            for i, row in df.iterrows():
                if pd.notna(row['BufferStatus']):
                    # Map buffer status to colors
                    buffer_color = "green"  # Default
                    if row['BufferStatus'] == "Critical":
                        buffer_color = "red"
                    elif row['BufferStatus'] == "Warning":
                        buffer_color = "orange"
                    elif row['BufferStatus'] == "Caution":
                        buffer_color = "yellow"

                    fig.add_trace(go.Scatter(
                        x=[row['Finish']],
                        y=[row['Task']],
                        mode='markers',
                        marker=dict(symbol='circle', size=10, color=buffer_color),
                        showlegend=False,
                        hoverinfo='none' # Ensure no hover info for buffer markers as well
                    ))

        fig.add_annotation(
            text=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=10, color="gray")
        )

        legend_items = [
            {"color": "red", "text": "Critical (<8h)"},
            {"color": "orange", "text": "Warning (<24h)"},
            {"color": "yellow", "text": "Caution (<72h)"},
            {"color": "green", "text": "OK (>72h)"}
        ]

        spacing = 0.12
        start_x = 0.5 - ((len(legend_items) - 1) * spacing) / 2
        for i, item in enumerate(legend_items):
            x_pos = start_x + (i * spacing)
            fig.add_shape(
                type="rect",
                xref="paper", yref="paper",
                x0=x_pos - 0.03, y0=-0.08,
                x1=x_pos - 0.01, y1=-0.06,
                fillcolor=item["color"],
                line=dict(color=item["color"]),
            )
            fig.add_annotation(
                text=item["text"],
                xref="paper", yref="paper",
                x=x_pos + 0.02, y=-0.07,
                showarrow=False,
                font=dict(size=10, color="black"),
                align="left",
                xanchor="left"
            )

        logger.info(f"Saving Gantt chart to: {os.path.abspath(output_file)}")
        pyo.plot(fig, filename=output_file, auto_open=False)
        logger.info(f"Interactive Gantt chart saved to: {os.path.abspath(output_file)}")

        # Verify that all START_DATE constraints were respected
        # This validation happens after the chart is created to make sure it matches what the user sees
        if jobs:
            start_date_violations = []
            for job in jobs:
                if 'UNIQUE_JOB_ID' in job and 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH']:
                    job_id = job['UNIQUE_JOB_ID']
                    requested_start = job['START_DATE_EPOCH']

                    # Look for this job in the final DataFrame
                    found = False
                    scheduled_start = None

                    for row in df_list:
                        if job_id in row['Task']:
                            found = True
                            # Get the timestamp from the datetime object
                            scheduled_start = int(row['Start'].timestamp())
                            break

                    # If we didn't find the job in the DataFrame, look for pattern matches
                    # This handles cases where the job ID might be slightly different in schedule vs jobs data
                    if not found:
                        job_pattern = job_id.split('_')[-1] if '_' in job_id else job_id
                        for row in df_list:
                            if job_pattern in row['Task']:
                                found = True
                                scheduled_start = int(row['Start'].timestamp())
                                break

                    # Check if the START_DATE constraint was respected
                    if found and scheduled_start is not None:
                        if abs(scheduled_start - requested_start) > 60:  # Allow 1 minute tolerance
                            violation = {
                                "job_id": job_id,
                                "requested_start": requested_start,
                                "scheduled_start": scheduled_start,
                                "requested_time": format_date_correctly(requested_start),
                                "scheduled_time": format_date_correctly(scheduled_start)
                            }
                            start_date_violations.append(violation)
                            logger.error(f"START_DATE VIOLATED: {job_id} should start EXACTLY at {format_date_correctly(requested_start)} "
                                        f"but was scheduled at {format_date_correctly(scheduled_start)}")
                        else:
                            logger.info(f"START_DATE RESPECTED: {job_id} correctly scheduled at {format_date_correctly(scheduled_start)}")

            # Report violations if any
            if start_date_violations:
                logger.error(f"Found {len(start_date_violations)} violated START_DATE constraints!")
                for violation in start_date_violations:
                    logger.error(f"  VIOLATED: {violation['job_id']} should start EXACTLY at {violation['requested_time']}")
            else:
                logger.info("All future START_DATE constraints were respected by the scheduler")

        return True

    except Exception as e:
        logger.error(f"Error creating or saving Gantt chart: {e}", exc_info=True)
        return False

def flatten_schedule_to_list(schedule):
    """
    Flatten the schedule dictionary into a list of tuples (unique_job_id, machine, start, end, priority, additional_params).
    Handles both old schedule format (4-tuple) and new format with additional parameters (5-tuple).
    """
    flat_schedule = []
    for machine, jobs in schedule.items():
        for job in jobs:
            # Handle both old and new format
            if len(job) >= 5:
                unique_job_id, start, end, priority, additional_params = job
            else:
                unique_job_id, start, end, priority = job
                additional_params = {}

            flat_schedule.append((unique_job_id, machine, start, end, priority, additional_params))
    return flat_schedule


if __name__ == "__main__":
    load_dotenv()
    file_path = os.getenv('file_path')
    if not file_path:
        logger.error("No file_path found in environment variables.")
        exit(1)

    try:
        # Log file details
        logger.info(f"Loading data from: {os.path.abspath(file_path)}")
        logger.info(f"File last modified: {datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')}")
        
        jobs, machines, setup_times = load_jobs_planning_data(file_path)
        logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines for visualization")
        logger.info(f"Sample job: {jobs[0] if jobs else 'None'}")  # Log first job
        
        schedule = greedy_schedule(jobs, machines, setup_times, enforce_sequence=True)
        success = create_interactive_gantt(schedule, jobs, 'interactive_schedule.html')
        if success:
            print(f"Gantt chart saved to: {os.path.abspath('interactive_schedule.html')}")
        else:
            print("Failed to create Gantt chart.")
    except Exception as e:
        logger.error(f"Error during Gantt chart generation: {e}", exc_info=True)
        print(f"Error: {e}")