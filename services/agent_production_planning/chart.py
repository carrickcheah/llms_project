import os
import re
from datetime import datetime
import plotly.figure_factory as ff
import plotly.offline as pyo
import pandas as pd
import plotly.graph_objects as go
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_job_family(process_code):
    """Extract the job family (e.g., 'CP08-231B') from the process code."""
    process_code = str(process_code).upper()
    match = re.search(r'(.*?)-P\d+', process_code)
    if match:
        family = match.group(1)
        return family
    parts = process_code.split("-P")
    if len(parts) >= 2:
        return parts[0]
    return process_code  # Fallback to full code if format is unexpected

def extract_process_number(process_code):
    """Extract the process sequence number (e.g., 1 from 'P01-06') or return 999 if not found."""
    print(f"Extracting sequence from: {process_code}")
    match = re.search(r'P(\d{2})-\d+', str(process_code).upper())  # Match exactly two digits after P
    if match:
        seq = int(match.group(1))
        print(f"Extracted sequence: {seq}")
        return seq
    print("No match found, returning 999")
    return 999  # Default for invalid formats

def create_interactive_gantt(schedule, output_file='interactive_schedule.html'):
    """
    Create an interactive Gantt chart from the schedule and save it as an HTML file.
    
    Args:
        schedule (dict): Dictionary of machine schedules {machine: [(process_code, start, end, priority), ...]}
        output_file (str): Path to save the HTML file
    
    Returns:
        bool: True if successful, False otherwise
    """
    current_time = int(datetime.now().timestamp())
    df_list = []
    colors = {
        'Priority 1 (Highest)': 'rgb(255, 0, 0)',
        'Priority 2 (High)': 'rgb(255, 165, 0)',
        'Priority 3 (Medium)': 'rgb(0, 128, 0)',
        'Priority 4 (Normal)': 'rgb(128, 0, 128)',
        'Priority 5 (Low)': 'rgb(60, 179, 113)'
    }

    if not schedule or not any(schedule.values()):
        df_list.append(dict(
            Task="No tasks scheduled",
            Start=datetime.fromtimestamp(current_time),
            Finish=datetime.fromtimestamp(current_time + 3600),
            Resource="None",
            Priority="Priority 3 (Medium)",
            Description="No tasks were scheduled. Please check your input data."
        ))
    else:
        process_info = {}
        for machine, jobs in schedule.items():
            for process_code, start, end, priority in jobs:
                if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                    logger.warning(f"Invalid timestamp for process {process_code} on machine {machine}: start={start}, end={end}")
                    continue
                if end <= start:
                    logger.warning(f"Invalid time range for process {process_code} on machine {machine}: {start} to {end}")
                    continue
                family = extract_job_family(process_code)
                sequence = extract_process_number(process_code)
                if family not in process_info:
                    process_info[family] = []
                process_info[family].append((process_code, machine, start, end, priority, sequence))

        sorted_tasks = []
        for family in sorted(process_info.keys()):
            processes = process_info[family]
            sorted_processes = sorted(processes, key=lambda x: x[5])  # Sort by sequence
            sorted_tasks.extend(sorted_processes)

        for process_code, machine, start, end, priority, _ in sorted_tasks:
            start_date = datetime.fromtimestamp(start)
            end_date = datetime.fromtimestamp(end)
            duration_hours = (end - start) / 3600

            job_priority = priority if priority is not None and 1 <= priority <= 5 else 3
            priority_label = f"Priority {job_priority} ({['Highest', 'High', 'Medium', 'Normal', 'Low'][job_priority-1]})"

            description = (f"<b>Process:</b> {process_code}<br>"
                          f"<b>Machine:</b> {machine}<br>"
                          f"<b>Priority:</b> {job_priority}<br>"
                          f"<b>Start:</b> {start_date.strftime('%Y-%m-%d %H:%M')}<br>"
                          f"<b>End:</b> {end_date.strftime('%Y-%m-%d %H:%M')}<br>"
                          f"<b>Duration:</b> {duration_hours:.1f} hours")

            df_list.append(dict(
                Task=process_code,
                Start=start_date,
                Finish=end_date,
                Resource=machine,
                Priority=priority_label,
                Description=description
            ))

    df = pd.DataFrame(df_list)

    if df.empty:
        logger.error("No valid tasks to plot in Gantt chart.")
        return False

    # Deduplicate task_order to avoid Categorical error
    task_order = list(dict.fromkeys(df['Task'].tolist()))
    df['Task'] = pd.Categorical(df['Task'], categories=task_order, ordered=True)

    fig = ff.create_gantt(df, colors=colors, index_col='Priority', show_colorbar=True,
                          group_tasks=False,
                          showgrid_x=True, showgrid_y=True,
                          title='Interactive Production Schedule')

    # Explicitly reverse the Y-axis to put lower sequences at the bottom
    fig.update_yaxes(categoryorder='array', categoryarray=task_order, autorange="reversed")

    fig.update_layout(
        autosize=True,
        height=max(400, len(df) * 20),
        margin=dict(l=250, r=30, t=80, b=50),
        legend_title_text='Priority Level',
        hovermode='closest',
        title={'text': "Interactive Production Schedule", 'font': {'size': 24}, 'x': 0.5, 'xanchor': 'center'},
        xaxis={'title': {'text': 'Date and Time', 'font': {'size': 14}}},
        yaxis={'title': {'text': 'Process Codes', 'font': {'size': 14}}}
    )

    for i in range(len(fig.data)):
        fig.data[i].text = df['Description']
        fig.data[i].hoverinfo = 'text'

    # Add generation timestamp
    fig.add_annotation(
        text=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        xref="paper", yref="paper",
        x=0.5, y=-0.05,
        showarrow=False,
        font=dict(size=10, color="gray")
    )

    try:
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"Interactive Gantt chart saved to: {os.path.abspath(output_file)}")
        return True
    except Exception as e:
        logger.error(f"Error saving Gantt chart: {e}")
        return False

def flatten_schedule_to_list(schedule):
    """
    Flatten the schedule dictionary into a list of tuples (process_code, machine, start, end, priority).
    """
    flat_schedule = []
    for machine, jobs in schedule.items():
        for job in jobs:
            process_code, start, end, priority = job
            flat_schedule.append((process_code, machine, start, end, priority))
    return flat_schedule

if __name__ == "__main__":
    # Updated example schedule with correct machine format
    example_schedule = {
        'WS01': [
            ('CP08-231B-P01-06', 1710720000, 1710723000, 2),  # 2024-03-17 08:00 to 08:50
            ('CP08-231B-P02-06', 1710723000, 1710726000, 2),  # 2024-03-17 08:50 to 09:40
        ],
        'PP23-060T': [
            ('CP08-231B-P03-06', 1710726000, 1710729000, 2),  # 2024-03-17 09:40 to 10:30
            ('CP08-231B-P04-06', 1710729000, 1710732000, 2),  # 2024-03-17 10:30 to 11:20
        ],
        'PP23': [
            ('CP08-231B-P05-06', 1710732000, 1710735000, 2),  # 2024-03-17 11:20 to 12:10
        ]
    }
    create_interactive_gantt(example_schedule, 'test_schedule.html')