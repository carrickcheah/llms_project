import os
import re
from datetime import datetime, timedelta
import plotly.figure_factory as ff
import plotly.offline as pyo
import pandas as pd
import plotly.graph_objects as go
import logging
from ingest_data import load_jobs_planning_data
from greedy import greedy_schedule

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
    return process_code

def extract_process_number(process_code):
    """Extract the process sequence number (e.g., 1 from 'P01-06') or return 999 if not found."""
    process_code = str(process_code).upper()
    match = re.search(r'P(\d{2})', process_code)  # Match exactly two digits after P
    if match:
        seq = int(match.group(1))
        return seq
    return 999  # Default if parsing fails

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

def create_interactive_gantt(schedule, jobs=None, output_file='interactive_schedule.html'):
    """
    Create an interactive Gantt chart from the schedule and save it as an HTML file.
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

    job_lookup = {}
    if jobs:
        for job in jobs:
            job_lookup[job['PROCESS_CODE']] = job

    logger.info(f"Schedule contains {len(schedule)} machines")
    for machine, jobs_list in schedule.items():
        logger.info(f"Machine {machine}: {len(jobs_list)} jobs")

    if not schedule or not any(schedule.values()):
        logger.warning("Empty schedule received, creating placeholder task")
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
            for job_data in jobs:
                if len(job_data) < 4:
                    logger.warning(f"Invalid job data for machine {machine}: {job_data}")
                    continue
                    
                process_code, start, end, priority = job_data
                
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
                logger.debug(f"Added {process_code} to family {family} with sequence {sequence}")

        logger.info(f"Organized jobs into {len(process_info)} job families")
        for family, processes in process_info.items():
            logger.debug(f"Family {family}: {len(processes)} processes")

        sorted_tasks = []
        for family in sorted(process_info.keys()):
            processes = process_info[family]
            sorted_processes = sorted(processes, key=lambda x: x[5])
            sorted_tasks.extend(sorted_processes)
            
        logger.info(f"Sorted task list contains {len(sorted_tasks)} tasks")

        for task_data in sorted_tasks:
            process_code, machine, start, end, priority, _ = task_data
            
            logger.debug(f"Processing task: {process_code} on {machine} from {start} to {end}")
            
            try:
                start_date = datetime.fromtimestamp(start)
                end_date = datetime.fromtimestamp(end)
                duration_hours = (end - start) / 3600
            except Exception as e:
                logger.error(f"Error converting timestamps for {process_code}: {e}")
                logger.error(f"Start: {start}, End: {end}")
                continue

            job_priority = priority if priority is not None and 1 <= priority <= 5 else 3
            priority_label = f"Priority {job_priority} ({['Highest', 'High', 'Medium', 'Normal', 'Low'][job_priority-1]})"

            task_label = f"{process_code} ({machine})"
            
            buffer_info = ""
            buffer_status = ""
            number_operator = ""
            if job_lookup and process_code in job_lookup:
                job_data = job_lookup[process_code]
                due_date_field = next((f for f in ['LCD_DATE_EPOCH', 'DUE_DATE_TIME'] if f in job_data), None)
                
                if due_date_field and job_data[due_date_field]:
                    due_date = datetime.fromtimestamp(job_data[due_date_field])
                    buffer_hours = (job_data[due_date_field] - end) / 3600
                    buffer_status = get_buffer_status_color(buffer_hours)
                    buffer_info = f"<br><b>Due Date:</b> {due_date.strftime('%Y-%m-%d %H:%M')}<br><b>Buffer:</b> {buffer_hours:.1f} hours"
                    
                    if 'START_DATE_EPOCH' in job_data and job_data['START_DATE_EPOCH'] and job_data['START_DATE_EPOCH'] > current_time:
                        earliest_start = datetime.fromtimestamp(job_data['START_DATE_EPOCH'])
                        buffer_info += f"<br><b>Earliest Start:</b> {earliest_start.strftime('%Y-%m-%d %H:%M')}"
                
                if 'NUMBER_OPERATOR' in job_data:
                    number_operator = f"<br><b>Number of Operators:</b> {job_data['NUMBER_OPERATOR']}"

            description = (f"<b>Process:</b> {process_code}<br>"
                          f"<b>Machine:</b> {machine}<br>"
                          f"<b>Priority:</b> {job_priority}<br>"
                          f"<b>Start:</b> {start_date.strftime('%Y-%m-%d %H:%M')}<br>"
                          f"<b>End:</b> {end_date.strftime('%Y-%m-%d %H:%M')}<br>"
                          f"<b>Duration:</b> {duration_hours:.1f} hours{buffer_info}{number_operator}")

            df_list.append(dict(
                Task=task_label,
                Start=start_date,
                Finish=end_date,
                Resource=machine,
                Priority=priority_label,
                Description=description,
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
        
        fig.update_yaxes(categoryorder='array', categoryarray=task_order, autorange="reversed")

        fig.update_layout(
            autosize=True,
            height=max(800, len(df) * 30),
            margin=dict(l=350, r=50, t=100, b=100),
            legend_title_text='Priority Level',
            hovermode='closest',
            title={'text': "Interactive Production Schedule", 'font': {'size': 24}, 'x': 0.5, 'xanchor': 'center'},
            xaxis={
                'title': {'text': 'Date', 'font': {'size': 1}},
                'rangeselector': {
                    'buttons': [
                        {'count': 7, 'label': '1w', 'step': 'day', 'stepmode': 'backward'},
                        {'count': 1, 'label': '1m', 'step': 'month', 'stepmode': 'backward'},
                        {'count': 6, 'label': '6m', 'step': 'month', 'stepmode': 'backward'},
                        {'count': 1, 'label': 'YTD', 'step': 'year', 'stepmode': 'todate'},
                        {'count': 1, 'label': '1y', 'step': 'year', 'stepmode': 'backward'},
                        {'step': 'all', 'label': 'all'}
                    ]
                }
            },
            yaxis={'title': {'text': 'Process Codes (Machine)', 'font': {'size': 14}}}
        )

        for i in range(len(fig.data)):
            fig.data[i].text = df['Description']
            fig.data[i].hoverinfo = 'text'

        if 'BufferStatus' in df.columns and df['BufferStatus'].notna().any():
            for i, row in df.iterrows():
                if pd.notna(row['BufferStatus']):
                    fig.add_trace(go.Scatter(
                        x=[row['Finish']],
                        y=[row['Task']],
                        mode='markers',
                        marker=dict(symbol='circle', size=10, color=row['BufferStatus']),
                        showlegend=False,
                        hoverinfo='none'
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

        debug_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        debug_output_file = output_file
        if '.' in output_file:
            name, ext = os.path.splitext(output_file)
            debug_output_file = f"{name}_{debug_timestamp}{ext}"

        logger.info(f"Saving Gantt chart to: {os.path.abspath(output_file)}")
        pyo.plot(fig, filename=output_file, auto_open=False)
        logger.info(f"Interactive Gantt chart saved to: {os.path.abspath(output_file)}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating or saving Gantt chart: {e}", exc_info=True)
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
    file_path = "/Users/carrickcheah/llms_project/services/agent_production_planning/mydata.xlsx"
    try:
        jobs, machines, setup_times = load_jobs_planning_data(file_path)
        logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines for visualization")
        schedule = greedy_schedule(jobs, machines, setup_times, enforce_sequence=True)
        success = create_interactive_gantt(schedule, jobs, 'interactive_schedule.html')
        if success:
            print(f"Gantt chart saved to: {os.path.abspath('interactive_schedule.html')}")
        else:
            print("Failed to create Gantt chart.")
    except Exception as e:
        logger.error(f"Error during Gantt chart generation: {e}")
        print(f"Error: {e}")