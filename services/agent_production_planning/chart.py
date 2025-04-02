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
    Extract the process sequence number from a unique job ID.
    For example, from 'JOST333333_CP33-333-P01-02' returns 1.
    """
    try:
        # Extract the part after the last P and before the next hyphen
        process_part = unique_job_id.split('P')[-1].split('-')[0]
        return int(process_part)
    except (ValueError, IndexError):
        logger.warning(f"Could not extract process number from {unique_job_id}")
        return 999

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
    # Modern professional color palette for priority levels
    colors = {
        'Priority 1 (Highest)': 'rgb(231, 76, 60)',   # Red - #e74c3c
        'Priority 2 (High)': 'rgb(243, 156, 18)',     # Orange - #f39c12
        'Priority 3 (Medium)': 'rgb(41, 128, 185)',   # Blue - #2980b9
        'Priority 4 (Normal)': 'rgb(46, 204, 113)',   # Green - #2ecc71
        'Priority 5 (Low)': 'rgb(149, 165, 166)'      # Gray - #95a5a6
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
        # Create naive datetime to avoid timezone issues
        start_time = datetime.utcfromtimestamp(current_time)
        end_time = datetime.utcfromtimestamp(current_time + 3600)
        df_list.append(dict(
            Task="No tasks scheduled",
            Start=start_time,
            Finish=end_time,
            Resource="None",
            Priority="Priority 3 (Medium)",
            Description="No tasks were scheduled. Please check your input data."
        ))
    else:
        # Process each job in the schedule directly without reordering
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

                    # Convert timestamps to datetime objects
                    start_date = datetime.fromtimestamp(start, tz=SG_TIMEZONE)
                    end_date = datetime.fromtimestamp(end, tz=SG_TIMEZONE)
                    duration_hours = (end - start) / 3600

                    # Create task entry with original scheduling times
                    family = extract_job_family(unique_job_id)
                    process_num = extract_process_number(unique_job_id)
                    # Create task label that will sort correctly - only show UNIQUE_JOB_ID
                    task_label = unique_job_id
                    
                    job_priority = priority if priority is not None and 1 <= priority <= 5 else 3
                    priority_label = f"Priority {job_priority} ({['Highest', 'High', 'Medium', 'Normal', 'Low'][job_priority-1]})"

                    # Build tooltip
                    tooltip = f"<b>Job ID:</b> {unique_job_id}<br>"
                    tooltip += f"<b>Machine:</b> {machine}<br>"
                    tooltip += f"<b>Start:</b> {format_date_correctly(start)}<br>"
                    tooltip += f"<b>End:</b> {format_date_correctly(end)}<br>"
                    tooltip += f"<b>Duration:</b> {duration_hours:.1f} hours<br>"
                    tooltip += f"<b>Priority:</b> {job_priority}"

                    df_list.append(dict(
                        Task=task_label,
                        Start=start_date.replace(tzinfo=None),
                        Finish=end_date.replace(tzinfo=None),
                        Resource=machine,
                        Priority=priority_label,
                        Description=tooltip,
                        Family=family,
                        ProcessNum=process_num
                    ))

                except Exception as e:
                    logger.error(f"Error processing job {unique_job_id} on {machine}: {str(e)}")
                    continue

    # Create DataFrame and plot
    try:
        df = pd.DataFrame(df_list)
        
        if df.empty:
            logger.error("No valid jobs to display")
            return False

        # Sort DataFrame by family and process number in natural order
        df = df.sort_values(['Family', 'ProcessNum'], ascending=[True, True])

        # Create gantt chart
        fig = ff.create_gantt(
            df,
            colors=colors,
            index_col='Priority',
            show_colorbar=True,
            group_tasks=True,  # Group tasks by family
            showgrid_x=True,
            showgrid_y=True,
            title='Interactive Production Schedule',
            bar_width=0.4,
            height=max(800, len(df) * 30)
        )

        # Update layout with professional UI settings
        fig.update_layout(
            autosize=True,
            height=max(800, len(df) * 30),
            margin=dict(l=350, r=50, t=100, b=100),
            paper_bgcolor='#f8f9fa',
            plot_bgcolor='#ffffff',
            font=dict(family="Arial, sans-serif", size=12),
            hovermode='closest',
            xaxis=dict(
                title='Timeline',
                tickfont=dict(size=12),
                gridcolor='#e0e0e0',
                zerolinecolor='#e0e0e0',
                gridwidth=1,
                griddash='dot'  # Add dotted grid lines
            ),
            yaxis=dict(
                title='Jobs',
                tickfont=dict(size=12),
                gridcolor='#e0e0e0',
                zerolinecolor='#e0e0e0',
                gridwidth=1,
                griddash='dot',  # Add dotted grid lines
                categoryorder='array',  # Force specific category order
                categoryarray=df['Task'].tolist()  # Use sorted task list in natural order
            ),
            title=dict(
                text='Interactive Production Schedule',
                font=dict(size=20, family='Arial, sans-serif'),
                x=0.5,
                y=0.95
            ),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )

        # Save the chart
        logger.info(f"Saving Gantt chart to: {os.path.abspath(output_file)}")
        pyo.plot(fig, filename=output_file, auto_open=False)
        logger.info(f"Interactive Gantt chart saved to: {os.path.abspath(output_file)}")
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