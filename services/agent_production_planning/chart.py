# chart.py
import plotly.figure_factory as ff
import plotly.offline as pyo
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_interactive_gantt(schedule, output_file='interactive_schedule.html'):
    """
    Create an interactive Gantt chart visualization of the production schedule.
    
    Args:
        schedule: Dictionary mapping machine IDs to lists of scheduled jobs
                 Each job is a tuple of (job_name, start_time, end_time, priority, [planned_start], [planned_end])
        output_file: Path to save the HTML output
        
    Returns:
        Boolean indicating success
    """
    current_time = int(datetime.now().timestamp())
    df_list = []
    # Priority colors from highest (red) to lowest (green)
    colors = {
        'Priority 1 (Highest)': 'rgb(255, 0, 0)',      # Bright red (unchanged)
        'Priority 2 (High)': 'rgb(255, 165, 0)',       # Traffic light orange
        'Priority 3 (Medium)': 'rgb(0, 128, 0)',       # Green
        'Priority 4 (Normal)': 'rgb(128, 0, 128)',     # Purple (unchanged)
        'Priority 5 (Low)': 'rgb(0, 0, 255)'        # Blue (unchanged)
    }
    
    # Check if schedule is empty
    if not schedule:
        logger.warning("Empty schedule provided to create_interactive_gantt. Creating minimal chart.")
        # Create a dummy entry
        current_time = int(datetime.now().timestamp())
        df_list.append(dict(
            Task="No tasks scheduled",
            Start=datetime.fromtimestamp(current_time),
            Finish=datetime.fromtimestamp(current_time + 3600),
            Resource="None",
            Priority="Priority 3 (Medium)",
            Description="No tasks were scheduled. Please check your input data."
        ))
    else:
        # First group jobs by machine for better visualization
        machine_groups = {}
        for machine, tasks in schedule.items():
            if machine not in machine_groups:
                machine_groups[machine] = []
            
            for task in tasks:
                # Handle different formats from scheduler
                # Format 1: (job_id, start, end, priority)
                # Format 2: (job_id, start, end, priority, planned_start, planned_end)
                if len(task) >= 6:
                    job_id, start, end, priority, planned_start, planned_end = task
                elif len(task) >= 4:
                    job_id, start, end, priority = task
                    planned_start = start
                    planned_end = end
                else:
                    logger.warning(f"Invalid task format: {task}")
                    continue
                
                machine_groups[machine].append((job_id, start, end, priority, planned_start, planned_end))
        
        # Process all jobs
        flat_schedule = []
        for machine, jobs in machine_groups.items():
            for job_id, start, end, priority, planned_start, planned_end in jobs:
                flat_schedule.append((job_id, machine, start, end, priority, planned_start, planned_end))
        
        # Sort by start time
        flat_schedule.sort(key=lambda x: x[2])
        
        for job_id, machine, start, end, priority, planned_start, planned_end in flat_schedule:
            # Validate timestamps
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                logger.warning(f"Invalid timestamp for job {job_id} on machine {machine}")
                continue
            
            # Skip invalid times
            if end <= start:
                logger.warning(f"Job {job_id} has invalid time range: {start} to {end}")
                continue
            
            # Use extracted priority if available, otherwise default to 3 (medium)
            job_priority = priority if priority is not None and 1 <= priority <= 5 else 3
            priority_label = f"Priority {job_priority} ({['Highest', 'High', 'Medium', 'Normal', 'Low'][job_priority-1]})"
            
            try:
                start_date = datetime.fromtimestamp(start)
                end_date = datetime.fromtimestamp(end)
                planned_start_date = datetime.fromtimestamp(planned_start) if planned_start else None
                planned_end_date = datetime.fromtimestamp(planned_end) if planned_end else None
                duration_hours = (end-start)/3600
                
                # Machine name with shortened display for better visualization
                machine_display = machine
                if len(str(machine)) > 15:
                    machine_display = str(machine)[:15] + "..."
                
                # Create description with original and planned times
                description = (f"<b>Job:</b> {job_id}<br>"
                              f"<b>Machine:</b> {machine}<br>"
                              f"<b>Priority:</b> {job_priority} ({['Highest', 'High', 'Medium', 'Normal', 'Low'][job_priority-1]})<br>"
                              f"<b>Start:</b> {start_date.strftime('%Y-%m-%d %H:%M')}<br>"
                              f"<b>End:</b> {end_date.strftime('%Y-%m-%d %H:%M')}<br>"
                              f"<b>Duration:</b> {duration_hours:.1f} hours<br>")
                
                # Add planned times if different from actual times
                if planned_start_date and abs(planned_start - start) > 3600:  # More than 1 hour difference
                    description += f"<b>Planned Start:</b> {planned_start_date.strftime('%Y-%m-%d %H:%M')}<br>"
                
                if planned_end_date and abs(planned_end - end) > 3600:  # More than 1 hour difference
                    description += f"<b>Planned End:</b> {planned_end_date.strftime('%Y-%m-%d %H:%M')}<br>"
                
                description += f"<b>Days from now:</b> {(start-current_time)/(24*3600):.1f} days"
                
                df_list.append(dict(
                    Task=str(machine_display),
                    Start=start_date,
                    Finish=end_date,
                    Resource=str(job_id),
                    Priority=priority_label,
                    Description=description
                ))
            except (ValueError, OverflowError) as e:
                logger.error(f"Error processing job {job_id}: {e}")
    
    df = pd.DataFrame(df_list)
    
    try:
        # Create the Gantt chart
        fig = ff.create_gantt(df, colors=colors, index_col='Priority', show_colorbar=True, 
                            group_tasks=True, showgrid_x=True, showgrid_y=True, 
                            title='Interactive Production Schedule')
        
        # Improve layout
        fig.update_layout(
            autosize=True, 
            height=900,  # Taller chart
            margin=dict(l=150, r=30, t=80, b=50),
            legend_title_text='Priority Level',
            hovermode='closest',
            title={
                'text': "Interactive Production Schedule",
                'font': {'size': 24},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis={
                'title': {'text': 'Date and Time', 'font': {'size': 14}},
            },
            yaxis={
                'title': {'text': 'Machines', 'font': {'size': 14}},
            }
        )
        
        # Improve tooltips
        for i in range(len(fig.data)):
            fig.data[i].text = df['Description']
            fig.data[i].hoverinfo = 'text'
        
        # Add a timestamp to show when this was generated
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.add_annotation(
            text=f"Generated on: {current_time_str}",
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=10, color="gray")
        )
        
        # Add a vertical line for the current time
        fig.add_shape(
            type="line",
            x0=datetime.fromtimestamp(current_time),
            y0=0,
            x1=datetime.fromtimestamp(current_time),
            y1=1,
            yref="paper",
            line=dict(
                color="red",
                width=2,
                dash="dash",
            ),
        )
        
        fig.add_annotation(
            x=datetime.fromtimestamp(current_time),
            y=1.01,
            yref="paper",
            text="Current Time",
            showarrow=False,
            font=dict(color="red"),
        )
        
        # Save the chart
        pyo.plot(fig, filename=output_file, auto_open=False)
        logger.info(f"Interactive Gantt chart saved to: {os.path.abspath(output_file)}")
        return True
    except Exception as e:
        logger.error(f"Error creating Gantt chart: {e}")
        return False