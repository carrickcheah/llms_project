# chart.py (simplified for PROCESS_CODE)
import plotly.figure_factory as ff
import plotly.offline as pyo
import pandas as pd
from datetime import datetime
import os

def create_interactive_gantt(schedule, output_file='interactive_schedule.html'):
    """
    Create an interactive Gantt chart visualization of the production schedule.
    
    Args:
        schedule: Dictionary mapping machine IDs to lists of scheduled jobs (process_code, start_time, end_time, priority)
                 Note: First element is now PROCESS_CODE, not job name
        output_file: Path to save the HTML output file
        
    Returns:
        Boolean indicating success or failure
    """
    current_time = int(datetime.now().timestamp())
    df_list = []
    # Priority colors from highest (red) to lowest (green)
    colors = {
        'Priority 1 (Highest)': 'rgb(255, 0, 0)',      # Bright red
        'Priority 2 (High)': 'rgb(255, 165, 0)',       # Traffic light orange
        'Priority 3 (Medium)': 'rgb(0, 128, 0)',       # Green
        'Priority 4 (Normal)': 'rgb(128, 0, 128)',     # Purple
        'Priority 5 (Low)': 'rgb(60, 179, 113)'        # Medium sea green
    }
    
    # Check if schedule is empty
    if not schedule:
        print("Warning: Empty schedule provided to create_interactive_gantt. Creating minimal chart.")
        # Create a dummy entry
        df_list.append(dict(
            Task="No tasks scheduled",
            Start=datetime.fromtimestamp(current_time),
            Finish=datetime.fromtimestamp(current_time + 3600),
            Resource="None",
            Priority="Priority 3 (Medium)",
            Description="No tasks were scheduled. Please check your input data."
        ))
    else:
        # Process all jobs from all machines
        flat_schedule = []
        for machine, jobs in schedule.items():
            for job_details in jobs:
                # Extract values safely regardless of tuple length
                process_code = job_details[0] if len(job_details) > 0 else "Unknown"  # This is now PROCESS_CODE
                start = job_details[1] if len(job_details) > 1 else current_time
                end = job_details[2] if len(job_details) > 2 else current_time + 3600
                priority = job_details[3] if len(job_details) > 3 else 3
                
                flat_schedule.append((process_code, machine, start, end, priority))
        
        # Now create the data for the chart
        for process_code, machine, start, end, priority in flat_schedule:
            # Validate timestamps
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                print(f"Warning: Invalid timestamp for process {process_code} on machine {machine}")
                continue
            
            # Skip invalid times
            if end <= start:
                print(f"Warning: Process {process_code} has invalid time range: {start} to {end}")
                continue
            
            # Use extracted priority if available, otherwise default to 3 (medium)
            job_priority = priority if priority is not None and 1 <= priority <= 5 else 3
            priority_label = f"Priority {job_priority} ({['Highest', 'High', 'Medium', 'Normal', 'Low'][job_priority-1]})"
            
            try:
                start_date = datetime.fromtimestamp(start)
                end_date = datetime.fromtimestamp(end)
                duration_hours = (end-start)/3600
                
                # Use PROCESS_CODE as the Task (Y-axis label)
                df_list.append(dict(
                    Task=str(process_code),  # PROCESS_CODE on Y-axis
                    Start=start_date,
                    Finish=end_date,
                    Resource=str(machine),  # Machine is displayed as Resource
                    Priority=priority_label,
                    Description=(f"<b>Process:</b> {process_code}<br>"
                                f"<b>Machine:</b> {machine}<br>"
                                f"<b>Priority:</b> {job_priority} ({['Highest', 'High', 'Medium', 'Normal', 'Low'][job_priority-1]})<br>"
                                f"<b>Start:</b> {start_date.strftime('%Y-%m-%d %H:%M')}<br>"
                                f"<b>End:</b> {end_date.strftime('%Y-%m-%d %H:%M')}<br>"
                                f"<b>Duration:</b> {duration_hours:.1f} hours<br>"
                                f"<b>Days from now:</b> {(start-current_time)/(24*3600):.1f} days")
                ))
            except (ValueError, OverflowError) as e:
                print(f"Error processing process {process_code}: {e}")
    
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
            margin=dict(l=200, r=30, t=80, b=50),  # Increased left margin for longer process codes
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
                'title': {'text': 'Process Codes', 'font': {'size': 14}},
            }
        )
        
        # Improve tooltips
        for i in range(len(fig.data)):
            fig.data[i].text = df['Description']
            fig.data[i].hoverinfo = 'text'
        
        # Add a timestamp to show when this was generated
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.add_annotation(
            text=f"Generated on: {current_time}",
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=10, color="gray")
        )
        
        # Save the chart
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"Interactive Gantt chart saved to: {os.path.abspath(output_file)}")
        return True
    except Exception as e:
        print(f"Error creating Gantt chart: {e}")
        return False