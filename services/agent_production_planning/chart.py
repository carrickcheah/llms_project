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
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Set up Singapore timezone
SG_TIMEZONE = pytz.timezone('Asia/Singapore')

def format_date(timestamp, is_lcd_date=False):
    """Format timestamp to date string in Singapore timezone."""
    if not timestamp or timestamp <= 0:
        return "N/A"
    try:
        date_obj = datetime.fromtimestamp(timestamp, tz=SG_TIMEZONE)
        return date_obj.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return "N/A"

def get_buffer_status(buffer_hours):
    """Get status and color based on buffer hours."""
    if buffer_hours < 0:
        return "Late", "red"
    elif buffer_hours < 8:
        return "Critical", "red"
    elif buffer_hours < 24:
        return "Warning", "orange"
    elif buffer_hours < 72:
        return "Caution", "yellow"
    return "OK", "green"

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
    """Create an interactive Gantt chart from the schedule."""
    current_time = int(datetime.now().timestamp())
    
    # Modern color palette
    colors = {
        'Priority 1 (Highest)': '#e74c3c',  # Red
        'Priority 2 (High)': '#f39c12',     # Orange
        'Priority 3 (Medium)': '#2980b9',   # Blue
        'Priority 4 (Normal)': '#2ecc71',   # Green
        'Priority 5 (Low)': '#95a5a6'       # Gray
    }

    df_list = []
    job_lookup = {job['UNIQUE_JOB_ID']: job for job in (jobs or [])}

    # Process schedule
    for machine, jobs_list in schedule.items():
        for job_data in jobs_list:
            try:
                # Extract job data
                if len(job_data) >= 5:
                    unique_job_id, start, end, priority, _ = job_data
                else:
                    unique_job_id, start, end, priority = job_data

                # Convert timestamps to datetime
                start_date = datetime.fromtimestamp(start, tz=SG_TIMEZONE)
                end_date = datetime.fromtimestamp(end, tz=SG_TIMEZONE)
                duration_hours = (end - start) / 3600

                # Get job priority
                job_priority = min(max(priority, 1), 5)
                priority_label = f"Priority {job_priority} ({['Highest', 'High', 'Medium', 'Normal', 'Low'][job_priority-1]})"

                # Calculate buffer
                buffer_hours = None
                buffer_info = ""
                if unique_job_id in job_lookup:
                    job_info = job_lookup[unique_job_id]
                    due_date = job_info.get('LCD_DATE_EPOCH') or job_info.get('DUE_DATE_TIME')
                    if due_date and isinstance(due_date, (int, float)):
                        buffer_seconds = due_date - end
                        buffer_hours = buffer_seconds / 3600
                        status_text = "LATE by" if buffer_hours < 0 else "Buffer"
                        buffer_info = f"<br><b>Due Date:</b> {format_date(due_date)}<br><b>{status_text}:</b> {abs(buffer_hours):.1f} hours"

                # Create task label
                task_label = unique_job_id
                if buffer_hours and buffer_hours < -24:
                    task_label = f"⚠️ {unique_job_id}"

                # Build tooltip
                tooltip = (
                    f"<b>Job ID:</b> {unique_job_id}<br>"
                    f"<b>Machine:</b> {machine}<br>"
                    f"<b>Start:</b> {format_date(start)}<br>"
                    f"<b>End:</b> {format_date(end)}<br>"
                    f"<b>Duration:</b> {duration_hours:.1f} hours<br>"
                    f"<b>Priority:</b> {job_priority}"
                    f"{buffer_info}"
                )

                buffer_status, _ = get_buffer_status(buffer_hours) if buffer_hours is not None else ("Unknown", None)

                df_list.append({
                    'Task': f"{task_label} ({machine})",
                    'Start': start_date.replace(tzinfo=None),
                    'Finish': end_date.replace(tzinfo=None),
                    'Resource': machine,
                    'Priority': priority_label,
                    'Description': tooltip,
                    'BufferStatus': buffer_status
                })

            except Exception as e:
                logger.error(f"Error processing job {unique_job_id}: {str(e)}")
                continue

    if not df_list:
        logger.warning("No tasks to display")
        return False

    # Create DataFrame and Gantt chart
    df = pd.DataFrame(df_list)
    fig = ff.create_gantt(
        df,
        colors=colors,
        index_col='Priority',
        show_colorbar=True,
        group_tasks=False,
        showgrid_x=True,
        showgrid_y=True,
        title='Production Schedule',
        bar_width=0.4,
        height=max(800, len(df) * 30)
    )

    # Add current time line
    current_date = datetime.now().replace(tzinfo=None)
    fig.add_shape(
        type="line",
        x0=current_date,
        x1=current_date,
        y0=0,
        y1=1,
        line=dict(color="red", width=2, dash="dash"),
        xref="x",
        yref="paper"
    )

    # Update layout
    fig.update_layout(
        autosize=True,
        margin=dict(l=350, r=50, t=100, b=100),
        paper_bgcolor='white',
        plot_bgcolor='white',
        title={
            'text': "Production Schedule",
            'font': {'size': 24, 'color': '#2c3e50'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis={
            'title': 'Date',
            'tickformat': '%Y-%m-%d',
            'tickangle': -45,
            'gridcolor': 'lightgray',
            'rangeselector': {
                'buttons': [
                    {'step': 'all', 'label': 'All'},
                    {'count': 7, 'label': '1W', 'step': 'day', 'stepmode': 'todate'},
                    {'count': 14, 'label': '2W', 'step': 'day', 'stepmode': 'todate'},
                    {'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                ]
            }
        },
        yaxis={
            'title': 'Jobs',
            'gridcolor': 'lightgray'
        }
    )

    # Add buffer status indicators
    for status, color in [("Critical", "red"), ("Warning", "orange"), ("Caution", "yellow"), ("OK", "green")]:
        mask = df['BufferStatus'] == status
        if mask.any():
            fig.add_trace(go.Scatter(
                x=df[mask]['Finish'],
                y=df[mask]['Task'],
                mode='markers',
                marker=dict(symbol='circle', size=10, color=color),
                name=status,
                showlegend=True
            ))

    # Save chart
    try:
        pyo.plot(fig, filename=output_file, auto_open=False)
        logger.info(f"Chart saved to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save chart: {str(e)}")
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
        logger.error("No file_path specified in environment variables.")
        exit(1)

    try:
        jobs, machines, setup_times = load_jobs_planning_data(file_path)
        schedule = greedy_schedule(jobs, machines, setup_times, enforce_sequence=True)
        if create_interactive_gantt(schedule, jobs):
            print("✅ Schedule visualization created successfully!")
        else:
            print("❌ Failed to create schedule visualization.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")