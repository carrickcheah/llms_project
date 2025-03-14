# chart.py
import plotly.figure_factory as ff
import plotly.offline as pyo
import pandas as pd
from datetime import datetime
import os

def create_interactive_gantt(schedule, output_file='interactive_schedule.html'):
    df_list = []
    # Priority colors from highest (red) to lowest (green)
    colors = {
        'Priority 1 (Highest)': 'rgb(178, 34, 34)',    # Dark red
        'Priority 2 (High)': 'rgb(255, 69, 0)',        # Orange-red
        'Priority 3 (Medium)': 'rgb(255, 165, 0)',     # Orange
        'Priority 4 (Normal)': 'rgb(30, 144, 255)',    # Blue
        'Priority 5 (Low)': 'rgb(60, 179, 113)'        # Green
    }
    
    # Check if schedule is empty
    if not schedule:
        print("Warning: Empty schedule provided to create_interactive_gantt. Creating minimal chart.")
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
            
            for job_id, start, end in tasks:
                # Parse job ID to extract priority if available
                job_parts = str(job_id).split('_')
                priority = None
                
                # Try to extract priority from job_id if it has a format like "JOXX12345_P1"
                for part in job_parts:
                    if part.startswith('P') and len(part) == 2 and part[1].isdigit():
                        priority = int(part[1])
                        break
                
                machine_groups[machine].append((job_id, start, end, priority))
        
        # Process all jobs
        flat_schedule = []
        for machine, jobs in machine_groups.items():
            for job_id, start, end, priority in jobs:
                flat_schedule.append((job_id, machine, start, end, priority))
        
        # Sort by start time
        flat_schedule.sort(key=lambda x: x[2])
        
        for job_id, machine, start, end, priority in flat_schedule:
            # Validate timestamps
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                print(f"Warning: Invalid timestamp for job {job_id} on machine {machine}")
                continue
            
            # Skip invalid times
            if end <= start:
                print(f"Warning: Job {job_id} has invalid time range: {start} to {end}")
                continue
            
            # Use extracted priority if available, otherwise default to 3 (medium)
            job_priority = priority if priority is not None and 1 <= priority <= 5 else 3
            priority_label = f"Priority {job_priority} ({'Highest' if job_priority == 1 else 'High' if job_priority == 2 else 'Medium' if job_priority == 3 else 'Normal' if job_priority == 4 else 'Low'})"
            
            try:
                start_date = datetime.fromtimestamp(start)
                end_date = datetime.fromtimestamp(end)
                duration_hours = (end-start)/3600
                
                # Machine name with shortened display for better visualization
                machine_display = machine
                if len(str(machine)) > 15:
                    machine_display = str(machine)[:15] + "..."
                
                df_list.append(dict(
                    Task=str(machine_display),
                    Start=start_date,
                    Finish=end_date,
                    Resource=str(job_id),
                    Priority=priority_label,
                    Description=(f"<b>Job:</b> {job_id}<br>"
                                f"<b>Machine:</b> {machine}<br>"
                                f"<b>Priority:</b> {job_priority}<br>"
                                f"<b>Start:</b> {start_date.strftime('%Y-%m-%d %H:%M')}<br>"
                                f"<b>End:</b> {end_date.strftime('%Y-%m-%d %H:%M')}<br>"
                                f"<b>Duration:</b> {duration_hours:.1f} hours")
                ))
            except (ValueError, OverflowError) as e:
                print(f"Error processing job {job_id}: {e}")
    
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