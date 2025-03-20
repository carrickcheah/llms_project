# chart_three.py | dont edit this line
import os
import re
from datetime import datetime, timedelta
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
    process_code = str(process_code).upper()
    match = re.search(r'P(\d{2})', process_code)  # Match exactly two digits after P
    if match:
        seq = int(match.group(1))
        return seq
    return 999  # Default for invalid formats

def get_buffer_status_color(buffer_hours):
    """
    Get color for buffer status based on hours remaining.
    
    Args:
        buffer_hours (float): Hours remaining until deadline
        
    Returns:
        str: Color code for the buffer status
    """
    if buffer_hours < 8:
        return "red"  # Critical - less than 8 hours buffer
    elif buffer_hours < 24:
        return "orange"  # Warning - less than 24 hours buffer
    elif buffer_hours < 72:
        return "yellow"  # Caution - less than 3 days buffer
    else:
        return "green"  # OK - more than 3 days buffer

def create_interactive_gantt(schedule, jobs=None, output_file='interactive_schedule.html'):
    """
    Create an interactive Gantt chart from the schedule and save it as an HTML file.
    Modified to show Machine ID in the y-axis labels alongside Process Code.
    
    Args:
        schedule (dict): Dictionary of machine schedules {machine: [(process_code, start, end, priority), ...]}
        jobs (list): Optional list of job dictionaries with additional metadata like buffer times
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

    # Create a lookup dictionary for job data if jobs list is provided
    job_lookup = {}
    if jobs:
        for job in jobs:
            job_lookup[job['PROCESS_CODE']] = job

    # Debug: Print schedule keys and size
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
                # Ensure we have at least 4 elements in the tuple
                if len(job_data) < 4:
                    logger.warning(f"Invalid job data for machine {machine}: {job_data}")
                    continue
                    
                process_code, start, end, priority = job_data
                
                # Validate timestamps
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

        # Debug: Check process_info
        logger.info(f"Organized jobs into {len(process_info)} job families")
        for family, processes in process_info.items():
            logger.debug(f"Family {family}: {len(processes)} processes")

        sorted_tasks = []
        for family in sorted(process_info.keys()):
            processes = process_info[family]
            sorted_processes = sorted(processes, key=lambda x: x[5])  # Sort by sequence
            sorted_tasks.extend(sorted_processes)
            
        # Debug: Check sorted_tasks
        logger.info(f"Sorted task list contains {len(sorted_tasks)} tasks")

        for task_data in sorted_tasks:
            process_code, machine, start, end, priority, _ = task_data
            
            # Debug: Print task info
            logger.debug(f"Processing task: {process_code} on {machine} from {start} to {end}")
            
            # Ensure timestamps are valid, convert to datetime
            try:
                start_date = datetime.fromtimestamp(start)
                end_date = datetime.fromtimestamp(end)
                duration_hours = (end - start) / 3600
            except Exception as e:
                logger.error(f"Error converting timestamps for {process_code}: {e}")
                logger.error(f"Start: {start}, End: {end}")
                continue

            # Ensure priority is valid
            job_priority = priority if priority is not None and 1 <= priority <= 5 else 3
            priority_label = f"Priority {job_priority} ({['Highest', 'High', 'Medium', 'Normal', 'Low'][job_priority-1]})"

            # Create task label that includes both process code and machine ID
            task_label = f"{process_code} ({machine})"
            
            # Add buffer information if available
            buffer_info = ""
            buffer_status = ""
            if job_lookup and process_code in job_lookup:
                job_data = job_lookup[process_code]
                # Look for LCD_DATE_EPOCH first, then fall back to DUE_DATE_TIME
                due_date_field = next((f for f in ['LCD_DATE_EPOCH', 'DUE_DATE_TIME'] if f in job_data), None)
                
                if due_date_field and job_data[due_date_field]:
                    due_date = datetime.fromtimestamp(job_data[due_date_field])
                    buffer_hours = (job_data[due_date_field] - end) / 3600
                    buffer_status = get_buffer_status_color(buffer_hours)
                    buffer_info = f"<br><b>Due Date:</b> {due_date.strftime('%Y-%m-%d %H:%M')}<br><b>Buffer:</b> {buffer_hours:.1f} hours"
                    
                    # Add information about CUT_Q if it exists
                    cut_q_field = next((f for f in ['CUT_Q_EPOCH', 'EARLIEST_START_TIME'] if f in job_data), None)
                    if cut_q_field and job_data[cut_q_field] and job_data[cut_q_field] > current_time:
                        earliest_start = datetime.fromtimestamp(job_data[cut_q_field])
                        buffer_info += f"<br><b>Earliest Start:</b> {earliest_start.strftime('%Y-%m-%d %H:%M')}"

            description = (f"<b>Process:</b> {process_code}<br>"
                          f"<b>Machine:</b> {machine}<br>"
                          f"<b>Priority:</b> {job_priority}<br>"
                          f"<b>Start:</b> {start_date.strftime('%Y-%m-%d %H:%M')}<br>"
                          f"<b>End:</b> {end_date.strftime('%Y-%m-%d %H:%M')}<br>"
                          f"<b>Duration:</b> {duration_hours:.1f} hours{buffer_info}")

            df_list.append(dict(
                Task=task_label,  # Use the combined task label
                Start=start_date,
                Finish=end_date,
                Resource=machine,
                Priority=priority_label,
                Description=description,
                BufferStatus=buffer_status
            ))

    # Check if we have any tasks
    if not df_list:
        logger.error("No valid tasks to plot in Gantt chart.")
        return False
        
    # Create DataFrame
    df = pd.DataFrame(df_list)
    logger.info(f"Created DataFrame with {len(df)} rows for Gantt chart")
    
    # Debug: Print sample data
    if not df.empty:
        logger.debug(f"Sample data: {df.iloc[0].to_dict()}")

    # Deduplicate task_order to avoid Categorical error
    task_order = list(dict.fromkeys(df['Task'].tolist()))
    df['Task'] = pd.Categorical(df['Task'], categories=task_order, ordered=True)

    try:
        # Create Gantt chart
        fig = ff.create_gantt(df, colors=colors, index_col='Priority', show_colorbar=True,
                              group_tasks=False,
                              showgrid_x=True, showgrid_y=True, 
                              title='Interactive Production Schedule')
        
        # Explicitly reverse the Y-axis to put lower sequences at the bottom
        fig.update_yaxes(categoryorder='array', categoryarray=task_order, autorange="reversed")

        # Update layout for better visualization
        fig.update_layout(
            autosize=True,
            height=max(800, len(df) * 30),  # Increased height for better visibility
            margin=dict(l=350, r=50, t=100, b=100),  # Increased left margin for labels
            legend_title_text='Priority Level',
            hovermode='closest',
            title={'text': "Interactive Production Schedule", 'font': {'size': 24}, 'x': 0.5, 'xanchor': 'center'},
            xaxis={
                'title': {'text': 'Date and Time', 'font': {'size': 14}},
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

        # Add hover text
        for i in range(len(fig.data)):
            fig.data[i].text = df['Description']
            fig.data[i].hoverinfo = 'text'

        # Add buffer status indicators if available
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

        # Add generation timestamp
        fig.add_annotation(
            text=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=10, color="gray")
        )
        
        # Add buffer status legend without title text
        legend_items = [
            {"color": "red", "text": "Critical (<8h)"},
            {"color": "orange", "text": "Warning (<24h)"},
            {"color": "yellow", "text": "Caution (<72h)"},
            {"color": "green", "text": "OK (>72h)"}
        ]
        
        # Position them horizontally with proper spacing
        spacing = 0.12
        start_x = 0.5 - ((len(legend_items) - 1) * spacing) / 2
        for i, item in enumerate(legend_items):
            x_pos = start_x + (i * spacing)

            # Add colored square
            fig.add_shape(
                type="rect",
                xref="paper", yref="paper",
                x0=x_pos - 0.03, y0=-0.08,
                x1=x_pos - 0.01, y1=-0.06,
                fillcolor=item["color"],
                line=dict(color=item["color"]),
            )

            # Add text label
            fig.add_annotation(
                text=item["text"],
                xref="paper", yref="paper",
                x=x_pos + 0.02, y=-0.07,
                showarrow=False,
                font=dict(size=10, color="black"),
                align="left",
                xanchor="left"
            )

        # For debug purposes, add a timestamp to the filename
        debug_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        debug_output_file = output_file
        if '.' in output_file:
            name, ext = os.path.splitext(output_file)
            debug_output_file = f"{name}_{debug_timestamp}{ext}"

        # Save the figure
        logger.info(f"Saving Gantt chart to: {os.path.abspath(output_file)}")
        pyo.plot(fig, filename=output_file, auto_open=False)
        logger.info(f"Interactive Gantt chart saved to: {os.path.abspath(output_file)}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating or saving Gantt chart: {e}", exc_info=True)
        return False

def export_schedule_html(jobs, schedule, output_file='schedule_view.html'):
    """
    Export the schedule as an HTML file with a detailed view.
    
    Args:
        jobs (list): List of job dictionaries with metadata
        schedule (dict): The schedule dictionary (machine: [(process_code, start, end, priority), ...])
        output_file (str): Output HTML file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a lookup for job data
        job_lookup = {job['PROCESS_CODE']: job for job in jobs}
        
        # Flatten the schedule
        flat_schedule = []
        for machine, tasks in schedule.items():
            for task in tasks:
                process_code, start, end, priority = task
                flat_schedule.append({
                    'process_code': process_code,
                    'machine': machine,
                    'start': start,
                    'end': end,
                    'priority': priority
                })
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(flat_schedule)
        if df.empty:
            logger.warning("Empty schedule, cannot create HTML view")
            return False
            
        # Add human-readable dates
        df['start_time'] = df['start'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M'))
        df['end_time'] = df['end'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M'))
        df['duration_hours'] = (df['end'] - df['start']) / 3600
        
        # Add job family and sequence for sorting
        df['job_family'] = df['process_code'].apply(extract_job_family)
        df['sequence'] = df['process_code'].apply(extract_process_number)
        
        # Add metadata from jobs
        df['job_code'] = df['process_code'].apply(lambda x: job_lookup.get(x, {}).get('JOB_CODE', 'Unknown'))
        
        # Add buffer time if LCD_DATE_EPOCH exists
        df['due_date'] = df['process_code'].apply(
            lambda x: datetime.fromtimestamp(job_lookup.get(x, {}).get('LCD_DATE_EPOCH', 0)).strftime('%Y-%m-%d %H:%M') 
            if job_lookup.get(x, {}).get('LCD_DATE_EPOCH', 0) > 0 else 'Not Set'
        )
        
        df['buffer_hours'] = df.apply(
            lambda row: (job_lookup.get(row['process_code'], {}).get('LCD_DATE_EPOCH', 0) - row['end']) / 3600 
            if job_lookup.get(row['process_code'], {}).get('LCD_DATE_EPOCH', 0) > 0 else 0,
            axis=1
        )
        
        df['buffer_status'] = df['buffer_hours'].apply(get_buffer_status_color)
        
        # Sort by job family and sequence
        df = df.sort_values(['job_family', 'sequence'])
        
        # Create HTML tables by job family
        html_parts = ["""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Production Schedule</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { padding: 8px; border: 1px solid #ddd; }
                th { background-color: #f2f2f2; text-align: left; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .priority-1 { background-color: #ffdddd; }
                .priority-2 { background-color: #ffeecc; }
                .priority-3 { background-color: #ddffdd; }
                .priority-4 { background-color: #eeddff; }
                .priority-5 { background-color: #ddffee; }
                .buffer-red { color: white; background-color: red; padding: 2px 5px; border-radius: 3px; }
                .buffer-orange { color: white; background-color: orange; padding: 2px 5px; border-radius: 3px; }
                .buffer-yellow { color: black; background-color: yellow; padding: 2px 5px; border-radius: 3px; }
                .buffer-green { color: white; background-color: green; padding: 2px 5px; border-radius: 3px; }
                .summary { margin-bottom: 30px; }
                .timestamp { color: #999; font-size: 0.8em; margin-top: 30px; text-align: center; }
            </style>
        </head>
        <body>
            <h1>Production Schedule</h1>
        """]
        
        # Add summary statistics
        html_parts.append("<div class='summary'>")
        html_parts.append(f"<p><strong>Total Jobs:</strong> {len(df)}</p>")
        html_parts.append(f"<p><strong>Machines Used:</strong> {df['machine'].nunique()}</p>")
        html_parts.append(f"<p><strong>Total Production Hours:</strong> {df['duration_hours'].sum():.1f}</p>")
        
        # Critical jobs summary
        critical_jobs = df[df['buffer_hours'] < 8]
        if not critical_jobs.empty:
            html_parts.append(f"<p><strong>Critical Jobs (buffer &lt; 8h):</strong> {len(critical_jobs)}</p>")
        
        html_parts.append("</div>")
        
        # Add legend
        html_parts.append("""
        <div class="legend" style="margin-bottom: 20px;">
            <h3>Priority Legend:</h3>
            <div style="display: flex; gap: 15px;">
                <div style="background-color: #ffdddd; padding: 5px;">Priority 1 (Highest)</div>
                <div style="background-color: #ffeecc; padding: 5px;">Priority 2 (High)</div>
                <div style="background-color: #ddffdd; padding: 5px;">Priority 3 (Medium)</div>
                <div style="background-color: #eeddff; padding: 5px;">Priority 4 (Normal)</div>
                <div style="background-color: #ddffee; padding: 5px;">Priority 5 (Low)</div>
            </div>
            <h3>Buffer Legend:</h3>
            <div style="display: flex; gap: 15px;">
                <div class="buffer-red">Critical (&lt;8h)</div>
                <div class="buffer-orange">Warning (&lt;24h)</div>
                <div class="buffer-yellow">Caution (&lt;72h)</div>
                <div class="buffer-green">OK (&gt;72h)</div>
            </div>
        </div>
        """)
        
        # Group by job family
        job_families = df['job_family'].unique()
        
        for family in job_families:
            family_df = df[df['job_family'] == family]
            html_parts.append(f"<h2>Job Family: {family}</h2>")
            
            html_parts.append("<table>")
            html_parts.append("""
            <tr>
                <th>Process Code</th>
                <th>Job Code</th>
                <th>Machine</th>
                <th>Start Time</th>
                <th>End Time</th>
                <th>Duration (hours)</th>
                <th>Priority</th>
                <th>Due Date</th>
                <th>Buffer</th>
            </tr>
            """)
            
            for _, row in family_df.iterrows():
                priority_class = f"priority-{int(row['priority'])}"
                
                buffer_class = ""
                buffer_text = ""
                if row['buffer_hours'] > 0:
                    buffer_class = f"buffer-{row['buffer_status']}"
                    buffer_text = f"{row['buffer_hours']:.1f} hours"
                
                html_parts.append(f"""
                <tr class="{priority_class}">
                    <td>{row['process_code']}</td>
                    <td>{row['job_code']}</td>
                    <td>{row['machine']}</td>
                    <td>{row['start_time']}</td>
                    <td>{row['end_time']}</td>
                    <td>{row['duration_hours']:.1f}</td>
                    <td>{row['priority']}</td>
                    <td>{row['due_date']}</td>
                    <td><span class="{buffer_class}">{buffer_text}</span></td>
                </tr>
                """)
            
            html_parts.append("</table>")
        
        # Add timestamp
        html_parts.append(f"""
        <div class='timestamp'>
            Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        </body>
        </html>
        """)
        
        # Write HTML file
        with open(output_file, 'w') as f:
            f.write(''.join(html_parts))
            
        logger.info(f"HTML schedule view saved to: {os.path.abspath(output_file)}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating HTML view: {e}", exc_info=True)
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
    
    # Example of using the functions
    print("Creating interactive Gantt chart...")
    create_interactive_gantt(example_schedule, None, 'test_schedule.html')
    
    # Example jobs data for export_schedule_html
    example_jobs = [
        {'PROCESS_CODE': 'CP08-231B-P01-06', 'JOB_CODE': 'JOB123', 'LCD_DATE_EPOCH': 1710750000, 'PRIORITY': 2},
        {'PROCESS_CODE': 'CP08-231B-P02-06', 'JOB_CODE': 'JOB123', 'LCD_DATE_EPOCH': 1710750000, 'PRIORITY': 2},
        {'PROCESS_CODE': 'CP08-231B-P03-06', 'JOB_CODE': 'JOB123', 'LCD_DATE_EPOCH': 1710750000, 'PRIORITY': 2},
        {'PROCESS_CODE': 'CP08-231B-P04-06', 'JOB_CODE': 'JOB123', 'LCD_DATE_EPOCH': 1710750000, 'PRIORITY': 2},
        {'PROCESS_CODE': 'CP08-231B-P05-06', 'JOB_CODE': 'JOB123', 'LCD_DATE_EPOCH': 1710750000, 'PRIORITY': 2},
    ]
    
    print("Creating HTML schedule view...")
    export_schedule_html(example_jobs, example_schedule, 'test_schedule_view.html')
    
    print("Done!")