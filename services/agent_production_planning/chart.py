# chart.py | dont edit this line
import os
import re
from datetime import datetime, timedelta
import plotly.figure_factory as ff
import plotly.offline as pyo
import pandas as pd
import plotly.graph_objects as go
import logging
from ingest_data import load_jobs_planning_data
from greedy import greedy_schedule, extract_job_family, extract_process_number

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        # Step 1: Create a mapping of job families and their processes in sequence
        family_processes = {}
        process_durations = {}
        
        # First pass - collect all process data and organize by family
        for machine, jobs in schedule.items():
            for job_data in jobs:
                if len(job_data) < 4:
                    logger.warning(f"Invalid job data for machine {machine}: {job_data}")
                    continue
                    
                process_code, start, end, priority = job_data
                
                # Calculate duration
                duration = end - start
                process_durations[process_code] = duration
                
                # Group by family
                family = extract_job_family(process_code)
                seq_num = extract_process_number(process_code)
                
                if family not in family_processes:
                    family_processes[family] = []
                
                # Use original schedule times first
                family_processes[family].append((seq_num, process_code, machine, start, end, priority))
        
        # Sort processes within each family by sequence number
        for family in family_processes:
            family_processes[family].sort(key=lambda x: x[0])
        
        # Step 2: Identify families that need time adjustment due to START_DATE constraints
        family_time_shifts = {}
        for family, processes in family_processes.items():
            # Check if any process in this family has START_DATE
            for seq_num, process_code, machine, start, end, priority in processes:
                if process_code in job_lookup and 'START_DATE_EPOCH' in job_lookup[process_code] and job_lookup[process_code]['START_DATE_EPOCH']:
                    # Calculate the time shift needed
                    requested_start = job_lookup[process_code]['START_DATE_EPOCH']
                    time_shift = start - requested_start
                    
                    # Store the time shift for this family
                    if family not in family_time_shifts or abs(time_shift) > abs(family_time_shifts[family]):
                        family_time_shifts[family] = time_shift
                    
                    logger.info(f"Family {family} has START_DATE constraint: shift={time_shift/3600:.1f} hours")
        
        # Step 3: Apply time shifts to generate adjusted task data for visualization
        process_info = {}
        for family in family_processes:
            time_shift = family_time_shifts.get(family, 0)
            
            # Only apply significant shifts
            if abs(time_shift) < 60:  # Skip shifts less than a minute
                for seq_num, process_code, machine, start, end, priority in family_processes[family]:
                    if family not in process_info:
                        process_info[family] = []
                    process_info[family].append((process_code, machine, start, end, priority, seq_num))
                continue
            
            logger.info(f"Applying time shift of {time_shift/3600:.1f} hours to family {family} for visualization")
            
            for seq_num, process_code, machine, start, end, priority in family_processes[family]:
                # Adjust the times by the time shift
                adjusted_start = start - time_shift
                adjusted_end = end - time_shift
                
                if family not in process_info:
                    process_info[family] = []
                process_info[family].append((process_code, machine, adjusted_start, adjusted_end, priority, seq_num))
                
                logger.info(f"  Adjusted {process_code}: START={datetime.fromtimestamp(adjusted_start).strftime('%Y-%m-%d %H:%M')}, "
                           f"END={datetime.fromtimestamp(adjusted_end).strftime('%Y-%m-%d %H:%M')}")
        
        # Step 4: Create task list for visualization from the adjusted data
        sorted_tasks = []
        for family in sorted(process_info.keys()):
            processes = process_info[family]
            sorted_processes = sorted(processes, key=lambda x: x[5])  # Sort by sequence within family
            sorted_tasks.extend(sorted_processes)
            
        logger.info(f"Sorted task list contains {len(sorted_tasks)} tasks")

        # Process the sorted tasks for visualization
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
                    
                    # Add START_DATE information if present
                    if 'START_DATE_EPOCH' in job_data and job_data['START_DATE_EPOCH']:
                        start_date_info = datetime.fromtimestamp(job_data['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
                        buffer_info += f"<br><b>START_DATE Constraint:</b> {start_date_info}"
                
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