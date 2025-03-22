# app.py - Production Schedule Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import re
import logging
from dotenv import load_dotenv

# Import from existing modules
from ingest_data import load_jobs_planning_data, extract_job_family, extract_process_number
from greedy import greedy_schedule
from main import add_schedule_times_and_buffer
from chart import get_buffer_status_color, create_interactive_gantt
from chart_two import export_schedule_html

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Production Schedule Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {padding-top: 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px;
        padding: 10px 20px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5 !important;
        color: white !important;
    }
    div[data-testid="stSidebarContent"] > div:first-child {padding-top: 2rem;}
    div[data-testid="stSidebarContent"] hr {margin: 1rem 0;}
    .buffer-critical {background-color: rgba(255, 0, 0, 0.15);}
    .buffer-warning {background-color: rgba(255, 165, 0, 0.15);}
    .buffer-caution {background-color: rgba(128, 0, 128, 0.15);}
    .buffer-ok {background-color: rgba(0, 128, 0, 0.15);}
    .status-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        text-align: center;
    }
    .badge-critical {background-color: #ffcccc; color: #cc0000; border: 1px solid #ff8080;}
    .badge-warning {background-color: #fff2cc; color: #b38600; border: 1px solid #ffdb4d;}
    .badge-caution {background-color: #f0d6f0; color: #800080; border: 1px solid #d699d6;}
    .badge-ok {background-color: #d6f0d6; color: #006600; border: 1px solid #99d699;}
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1rem;
        color: #555;
    }
    .metric-card h2 {
        margin: 0.5rem 0 0 0;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .metric-card.critical h2 {color: #cc0000;}
    .metric-card.warning h2 {color: #b38600;}
    .metric-card.caution h2 {color: #800080;}
    .metric-card.ok h2 {color: #006600;}
</style>
""", unsafe_allow_html=True)

#-------------------------------------------------------------
# Helper Functions
#-------------------------------------------------------------

def get_buffer_status(buffer_hours):
    """Get status category based on buffer hours."""
    if buffer_hours < 8:
        return "Critical"
    elif buffer_hours < 24:
        return "Warning"
    elif buffer_hours < 72:
        return "Caution"
    else:
        return "OK"

def get_buffer_color(buffer_status):
    """Get color for buffer status."""
    status_colors = {
        "Critical": "#cc0000",
        "Warning": "#fd7e14",
        "Caution": "#6f42c1",
        "OK": "#28a745"
    }
    return status_colors.get(buffer_status, "#28a745")

def format_time(seconds):
    """Format time in seconds to a more human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"

def prepare_gantt_data(jobs, schedule):
    """Prepare data for Gantt chart visualization."""
    df_list = []
    current_time = int(datetime.now().timestamp())
    
    for machine, jobs_list in schedule.items():
        for job_data in jobs_list:
            process_code, start, end, priority = job_data
            
            # Find the job details
            job_entry = next((j for j in jobs if j['PROCESS_CODE'] == process_code), None)
            if not job_entry:
                continue
                
            # Get START_TIME and END_TIME from the job if they exist (time-shifted)
            # Otherwise use the original schedule times
            start_time = job_entry.get('START_TIME', start)
            end_time = job_entry.get('END_TIME', end)
                
            start_date = datetime.fromtimestamp(start_time)
            end_date = datetime.fromtimestamp(end_time)
            duration_hours = (end_time - start_time) / 3600
            
            # Format priority
            job_priority = priority if priority is not None and 1 <= priority <= 5 else 3
            priority_label = f"Priority {job_priority} ({['Highest', 'High', 'Medium', 'Normal', 'Low'][job_priority-1]})"
            
            # Process information
            task_label = f"{process_code} ({machine})"
            
            # Extra job information
            buffer_info = ""
            buffer_status = ""
            buffer_status_text = ""
            buffer_hours = 0
            due_date = ""
            number_operator = ""
            
            if 'LCD_DATE_EPOCH' in job_entry and job_entry['LCD_DATE_EPOCH']:
                lcd_time = job_entry['LCD_DATE_EPOCH']
                due_date = datetime.fromtimestamp(lcd_time).strftime('%Y-%m-%d %H:%M')
                buffer_hours = (lcd_time - end_time) / 3600
                buffer_status_text = get_buffer_status(buffer_hours)
                buffer_status = get_buffer_color(buffer_status_text)
                
                # Add START_DATE information if present
                if 'START_DATE_EPOCH' in job_entry and job_entry['START_DATE_EPOCH']:
                    start_date_info = datetime.fromtimestamp(job_entry['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
                    buffer_info = f"<br><b>START_DATE Constraint:</b> {start_date_info}"
            
            if 'NUMBER_OPERATOR' in job_entry:
                number_operator = f"<br><b>Number of Operators:</b> {job_entry['NUMBER_OPERATOR']}"
            
            # Build description for tooltip
            job_name = job_entry.get('JOB', job_entry.get('JOB_CODE', process_code))
            job_quantity = job_entry.get('JOB_QUANTITY', 0)
            expect_output = job_entry.get('EXPECT_OUTPUT_PER_HOUR', 0)
            accumulated_output = job_entry.get('ACCUMULATED_DAILY_OUTPUT', 0)
            balance_quantity = job_entry.get('BALANCE_QUANTITY', job_quantity - accumulated_output if job_quantity else 0)
            
            description = (
                f"<b>Process:</b> {process_code}<br>"
                f"<b>Job:</b> {job_name}<br>"
                f"<b>Machine:</b> {machine}<br>"
                f"<b>Priority:</b> {job_priority}<br>"
                f"<b>Start:</b> {start_date.strftime('%Y-%m-%d %H:%M')}<br>"
                f"<b>End:</b> {end_date.strftime('%Y-%m-%d %H:%M')}<br>"
                f"<b>Duration:</b> {duration_hours:.1f} hours"
            )
            
            if due_date:
                description += f"<br><b>Due Date:</b> {due_date}<br><b>Buffer:</b> {buffer_hours:.1f} hours"
            
            description += buffer_info + number_operator
            
            if job_quantity:
                description += f"<br><b>Job Quantity:</b> {job_quantity}"
            if expect_output:
                description += f"<br><b>Expected Output/Hour:</b> {expect_output}"
            if accumulated_output:
                description += f"<br><b>Accumulated Output:</b> {accumulated_output}"
            if balance_quantity:
                description += f"<br><b>Balance Quantity:</b> {balance_quantity}"
            
            # Add to the data list
            df_list.append({
                'Task': task_label,
                'Start': start_date,
                'Finish': end_date,
                'Resource': machine,
                'Priority': priority_label,
                'Description': description,
                'BufferStatus': buffer_status,
                'BufferStatusText': buffer_status_text,
                'BufferHours': buffer_hours,
                'DueDate': due_date
            })
    
    return pd.DataFrame(df_list)

def create_streamlit_gantt_chart(df, colors_dict=None, height=600):
    """Create a Gantt chart specifically for Streamlit display."""
    if colors_dict is None:
        colors_dict = {
            'Priority 1 (Highest)': 'rgb(255, 0, 0)',
            'Priority 2 (High)': 'rgb(255, 165, 0)',
            'Priority 3 (Medium)': 'rgb(0, 128, 0)',
            'Priority 4 (Normal)': 'rgb(128, 0, 128)',
            'Priority 5 (Low)': 'rgb(60, 179, 113)'
        }
    
    # Make a copy to avoid modifying the original
    df_gantt = df.copy()
    
    # Sort by task name to maintain consistent ordering
    task_order = list(dict.fromkeys(df_gantt['Task'].tolist()))
    df_gantt['Task'] = pd.Categorical(df_gantt['Task'], categories=task_order, ordered=True)
    
    fig = ff.create_gantt(
        df_gantt, 
        colors=colors_dict, 
        index_col='Priority', 
        show_colorbar=True,
        group_tasks=False,
        showgrid_x=True, 
        showgrid_y=True,
        title='Interactive Production Schedule'
    )
    
    # Update layout for better readability
    fig.update_layout(
        autosize=True,
        height=height,
        margin=dict(l=350, r=50, t=100, b=100),
        legend_title_text='Priority Level',
        hovermode='closest',
        title={'text': "Production Schedule", 'font': {'size': 24}, 'x': 0.5, 'xanchor': 'center'},
        xaxis={
            'title': {'text': 'Date', 'font': {'size': 14}},
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
    
    # Update y-axis to reverse the order
    fig.update_yaxes(categoryorder='array', categoryarray=task_order, autorange="reversed")
    
    # Add hover text to improve information display
    for i in range(len(fig.data)):
        fig.data[i].text = df_gantt['Description']
        fig.data[i].hoverinfo = 'text'
    
    # Add buffer status indicators at the end of each task
    if 'BufferStatus' in df_gantt.columns and df_gantt['BufferStatus'].notna().any():
        for i, row in df_gantt.iterrows():
            if pd.notna(row['BufferStatus']):
                fig.add_trace(go.Scatter(
                    x=[row['Finish']],
                    y=[row['Task']],
                    mode='markers',
                    marker=dict(
                        symbol='circle', 
                        size=10, 
                        color=row['BufferStatus'],
                        line=dict(width=1, color='black')
                    ),
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=f"Buffer: {row['BufferHours']:.1f}h<br>Status: {row['BufferStatusText']}"
                ))
    
    # Add a legend for buffer status
    legend_items = [
        {"color": "#cc0000", "text": "Critical (<8h)"},
        {"color": "#fd7e14", "text": "Warning (<24h)"},
        {"color": "#6f42c1", "text": "Caution (<72h)"},
        {"color": "#28a745", "text": "OK (>72h)"}
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
    
    # Add current time line
    now = datetime.now()
    fig.add_shape(
        type="line",
        x0=now,
        y0=0,
        x1=now,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=now,
        y=1.01,
        yref="paper",
        text="Current Time",
        showarrow=False,
        font=dict(color="red", size=12),
    )
    
    return fig

def create_buffer_distribution_chart(jobs_df):
    """Create a histogram showing distribution of buffer hours."""
    fig = px.histogram(
        jobs_df, 
        x='BAL_HR',
        color='BUFFER_STATUS',
        title='Buffer Time Distribution',
        labels={'BAL_HR': 'Buffer Hours', 'count': 'Number of Jobs'},
        color_discrete_map={
            'Critical': '#cc0000',
            'Warning': '#fd7e14',
            'Caution': '#6f42c1',
            'OK': '#28a745'
        },
        category_orders={'BUFFER_STATUS': ['Critical', 'Warning', 'Caution', 'OK']},
        nbins=30,
        opacity=0.8
    )
    
    fig.update_layout(
        xaxis_title="Buffer Hours",
        yaxis_title="Number of Jobs",
        legend_title="Buffer Status",
        bargap=0.2,
        height=400
    )
    
    # Add mean and median lines
    mean_buffer = jobs_df['BAL_HR'].mean()
    median_buffer = jobs_df['BAL_HR'].median()
    
    fig.add_vline(x=mean_buffer, line_dash="dash", line_color="#000000",
                 annotation_text=f"Mean: {mean_buffer:.1f}h",
                 annotation_position="top right")
    
    fig.add_vline(x=median_buffer, line_dash="dot", line_color="#000000",
                 annotation_text=f"Median: {median_buffer:.1f}h",
                 annotation_position="top left")
    
    return fig

def create_machine_load_chart(schedule, jobs):
    """Create a chart showing machine loading."""
    machine_data = []
    current_time = int(datetime.now().timestamp())
    
    for machine, tasks in schedule.items():
        total_time = sum(end - start for _, start, end, _ in tasks)
        num_jobs = len(tasks)
        
        # Get earliest start and latest end for machine
        if tasks:
            earliest_start = min(start for _, start, _, _ in tasks)
            latest_end = max(end for _, _, end, _ in tasks)
            timespan = latest_end - earliest_start
            utilization = (total_time / timespan * 100) if timespan > 0 else 0
            
            # Get job details
            job_list = []
            critical_jobs = 0
            for process_code, start, end, _ in tasks:
                job_entry = next((j for j in jobs if j['PROCESS_CODE'] == process_code), None)
                if job_entry:
                    job_list.append(job_entry['PROCESS_CODE'])
                    if job_entry.get('BUFFER_STATUS') == 'Critical':
                        critical_jobs += 1
            
            machine_data.append({
                'Machine': machine,
                'Jobs': num_jobs,
                'Critical Jobs': critical_jobs,
                'Total Hours': total_time / 3600,
                'Utilization': utilization,
                'Start Time': datetime.fromtimestamp(earliest_start).strftime('%Y-%m-%d %H:%M'),
                'End Time': datetime.fromtimestamp(latest_end).strftime('%Y-%m-%d %H:%M'),
                'Job List': ", ".join(job_list[:5]) + ("..." if len(job_list) > 5 else "")
            })
    
    if not machine_data:
        return None
    
    machine_df = pd.DataFrame(machine_data)
    
    # Sort by total hours
    machine_df = machine_df.sort_values('Total Hours', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        machine_df,
        x='Machine',
        y='Total Hours',
        color='Utilization',
        text='Jobs',
        hover_data=['Job List', 'Critical Jobs', 'Start Time', 'End Time'],
        labels={'Total Hours': 'Total Processing Hours'},
        title='Machine Load Distribution',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_traces(texttemplate='%{text} jobs', textposition='outside')
    
    fig.update_layout(
        xaxis_title="Machine",
        yaxis_title="Total Hours",
        height=500,
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig

def create_process_flow_chart(jobs):
    """Create a directed graph showing process flows between job families."""
    # Create family-level process flows
    family_flows = {}
    
    for job in jobs:
        process_code = job['PROCESS_CODE']
        family = extract_job_family(process_code)
        seq_num = extract_process_number(process_code)
        
        if family not in family_flows:
            family_flows[family] = []
        
        machine_id = job.get('RSC_LOCATION', job.get('MACHINE_ID', 'Unknown'))
        buffer_hours = job.get('BAL_HR', 0)
        buffer_status = job.get('BUFFER_STATUS', 'Unknown')
        
        family_flows[family].append({
            'family': family,
            'process_code': process_code,
            'seq_num': seq_num,
            'machine': machine_id,
            'buffer_hours': buffer_hours,
            'buffer_status': buffer_status
        })
    
    # Sort processes within each family
    for family in family_flows:
        family_flows[family].sort(key=lambda x: x['seq_num'])
    
    # Create nodes and edges for the graph
    nodes = []
    edges = []
    
    for family, processes in family_flows.items():
        if len(processes) < 2:
            continue
            
        # Create edges between sequential processes
        for i in range(len(processes) - 1):
            source = processes[i]['process_code']
            target = processes[i + 1]['process_code']
            
            # Add nodes with their details
            nodes.append({
                'id': source,
                'label': source,
                'family': family,
                'machine': processes[i]['machine'],
                'buffer_status': processes[i]['buffer_status'],
                'buffer_hours': processes[i]['buffer_hours']
            })
            
            if i == len(processes) - 2:
                nodes.append({
                    'id': target,
                    'label': target,
                    'family': family,
                    'machine': processes[i+1]['machine'],
                    'buffer_status': processes[i+1]['buffer_status'],
                    'buffer_hours': processes[i+1]['buffer_hours']
                })
            
            # Add edge
            edges.append({
                'source': source,
                'target': target,
                'family': family
            })
    
    # Create a Plotly diagram if we have enough data
    if len(nodes) > 0 and len(edges) > 0:
        # Convert to dataframes
        nodes_df = pd.DataFrame(nodes).drop_duplicates(subset=['id'])
        edges_df = pd.DataFrame(edges)
        
        # Generate node positions
        pos = {}
        families = sorted(list(set(nodes_df['family'])))
        
        for i, family in enumerate(families):
            family_nodes = nodes_df[nodes_df['family'] == family].sort_values('id')
            for j, row in enumerate(family_nodes.iterrows()):
                _, node = row
                pos[node['id']] = (j, i)
        
        # Create edge traces
        edge_traces = []
        for _, edge in edges_df.iterrows():
            x0, y0 = pos[edge['source']]
            x1, y1 = pos[edge['target']]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            
            edge_traces.append(edge_trace)
        
        # Create node trace with fixed colorbar configuration
        node_trace = go.Scatter(
            x=[pos[node][0] for node in pos],
            y=[pos[node][1] for node in pos],
            mode='markers+text',
            text=nodes_df['label'],
            textposition='bottom center',
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                reversescale=True,
                color=[],
                size=15,
                line_width=2,
                line=dict(color='black'),
                colorbar=dict(
                    thickness=15,
                    title=dict(
                        text='Buffer Hours',
                        side='right'  # Using 'side' instead of 'titleside'
                    ),
                    xanchor='left'
                )
            ),
            hoverinfo='text',
            hovertext=[]
        )
        
        # Add color and hover text to nodes
        buffer_hours = []
        hover_texts = []
        
        for _, node in nodes_df.iterrows():
            buffer_hours.append(node['buffer_hours'])
            hover_texts.append(
                f"Process: {node['id']}<br>"
                f"Family: {node['family']}<br>"
                f"Machine: {node['machine']}<br>"
                f"Buffer: {node['buffer_hours']:.1f} hours<br>"
                f"Status: {node['buffer_status']}"
            )
        
        node_trace.marker.color = buffer_hours
        node_trace.hovertext = hover_texts
        
        # Create the figure
        fig = go.Figure(data=edge_traces + [node_trace],
                        layout=go.Layout(
                            title='Process Flow Diagram',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=600
                        ))
        
        return fig
    
    return None

def run_scheduler(file_path, max_jobs, force_greedy, cp_sat_only, enforce_sequence, time_limit):
    """Run the scheduler with the given parameters and return the results."""
    try:
        # Load job data
        start_time = time.time()
        jobs, machines, setup_times = load_jobs_planning_data(file_path)
        
        # Sort jobs by process code to maintain sequential processing
        jobs.sort(key=lambda job: (extract_job_family(job['PROCESS_CODE']), extract_process_number(job['PROCESS_CODE'])))
        
        # Limit to max_jobs
        if len(jobs) > max_jobs:
            jobs = jobs[:max_jobs]
            st.warning(f"Limited to {max_jobs} jobs (out of {len(jobs)} total)")
        
        load_time = time.time() - start_time
        st.success(f"Loaded {len(jobs)} jobs and {len(machines)} machines in {load_time:.2f} seconds")
        
        # Check for START_DATE values and display them
        current_time = int(datetime.now().timestamp())
        start_date_jobs = [job for job in jobs if job.get('START_DATE_EPOCH', current_time) > current_time]
        if start_date_jobs:
            st.info(f"Found {len(start_date_jobs)} jobs with future START_DATE constraints")
            
            # Ensure START_DATE constraints are strictly enforced
            for job in jobs:
                if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None:
                    # Only enforce constraints that are in the future
                    if job['START_DATE_EPOCH'] > current_time:
                        logger.info(f"ENFORCING START_DATE {datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')} for {job['PROCESS_CODE']}")
                else:
                    # If START_DATE wasn't provided in input, remove it if present
                    if 'START_DATE_EPOCH' in job:
                        del job['START_DATE_EPOCH']
        
        # Schedule jobs
        schedule_start_time = time.time()
        schedule = None
        
        if not force_greedy and not cp_sat_only:
            # Try CP-SAT first, fall back to greedy
            st.info("Attempting to create schedule with CP-SAT solver...")
            try:
                from sch_jobs import schedule_jobs
                schedule = schedule_jobs(jobs, machines, setup_times, 
                                       enforce_sequence=enforce_sequence,
                                       time_limit_seconds=time_limit)
                if not schedule or not any(schedule.values()):
                    st.warning("CP-SAT solver returned an empty schedule. Falling back to greedy algorithm.")
                    schedule = None
                else:
                    st.success("CP-SAT scheduler successful!")
            except Exception as e:
                st.error(f"CP-SAT solver failed: {str(e)}")
                schedule = None
        
        if (force_greedy or not schedule or not any(schedule.values())) and not cp_sat_only:
            st.info("Using greedy scheduler...")
            try:
                schedule = greedy_schedule(jobs, machines, setup_times, enforce_sequence=enforce_sequence)
                st.success("Greedy scheduler successful!")
            except Exception as e:
                st.error(f"Greedy scheduler failed: {str(e)}")
                schedule = None
        
        if cp_sat_only:
            try:
                from sch_jobs import schedule_jobs
                schedule = schedule_jobs(jobs, machines, setup_times, 
                                       enforce_sequence=enforce_sequence,
                                       time_limit_seconds=time_limit)
                if not schedule or not any(schedule.values()):
                    st.error("CP-SAT solver returned an empty schedule.")
                    schedule = None
                else:
                    st.success("CP-SAT scheduler successful!")
            except Exception as e:
                st.error(f"CP-SAT solver failed: {str(e)}")
                schedule = None
        
        if not schedule or not any(schedule.values()):
            st.error("Failed to create a valid schedule.")
            return None, None, None
        
        schedule_time = time.time() - schedule_start_time
        st.success(f"Schedule generated in {schedule_time:.2f} seconds")
        
        # Add schedule times and calculate buffers for each job
        jobs = add_schedule_times_and_buffer(jobs, schedule)
        
        # Convert jobs list to DataFrame for easier manipulation
        jobs_df = pd.DataFrame(jobs)
        
        # Prepare data for Gantt chart
        gantt_data = prepare_gantt_data(jobs, schedule)
        
        return jobs, schedule, gantt_data
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

#-------------------------------------------------------------
# Main Dashboard
#-------------------------------------------------------------

def main():
    """Main function to run the Streamlit dashboard."""
    # Dashboard header with logo and title
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <div style="font-size: 2.5rem; font-weight: 700; margin-right: 1rem;">📊</div>
        <div>
            <h1 style="margin: 0; padding: 0;">Production Schedule Dashboard</h1>
            <p style="margin: 0; padding: 0; color: #666;">Interactive scheduling and monitoring system</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    #-------------------------------------------------------------
    # Sidebar - Settings and File Upload
    #-------------------------------------------------------------
    with st.sidebar:
        st.header("Settings")
        
        # File upload option
        uploaded_file = st.file_uploader("Upload Schedule Data (Excel)", type=["xlsx"])
        
        if uploaded_file:
            file_path = "temp_upload.xlsx"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File uploaded successfully!")
        else:
            # Load environment variables for default file path
            load_dotenv()
            file_path = os.getenv('file_path')
            if not file_path:
                st.error("No file path provided. Please upload a file or set 'file_path' in .env file.")
                return
            
            st.info(f"Using default data file: {os.path.basename(file_path)}")
        
        # Scheduling parameters
        st.subheader("Scheduling Parameters")
        
        max_jobs = st.slider("Maximum jobs to schedule", min_value=10, max_value=500, value=250, step=10)
        
        scheduler_type = st.radio(
            "Scheduler Algorithm",
            options=["Auto (CP-SAT with Greedy fallback)", "Force Greedy", "CP-SAT Only"],
            index=0
        )
        
        force_greedy = scheduler_type == "Force Greedy"
        cp_sat_only = scheduler_type == "CP-SAT Only"
        
        enforce_sequence = st.checkbox("Enforce process sequence dependencies", value=True)
        
        time_limit = st.slider("Solver time limit (seconds)", min_value=10, max_value=600, value=300, step=10)
        
        # Visualization settings
        st.subheader("Visualization Settings")
        
        chart_height = st.slider("Chart height (px)", min_value=400, max_value=1200, value=600, step=50)
        
        show_tooltips = st.checkbox("Show detailed tooltips", value=True)
        
        st.divider()
        
        if st.button("Run Scheduler", type="primary", use_container_width=True):
            with st.spinner("Loading job data..."):
                st.session_state.run_scheduler = True
                st.session_state.file_path = file_path
                st.session_state.max_jobs = max_jobs
                st.session_state.force_greedy = force_greedy
                st.session_state.cp_sat_only = cp_sat_only
                st.session_state.enforce_sequence = enforce_sequence
                st.session_state.time_limit = time_limit
                st.session_state.chart_height = chart_height
                st.session_state.show_tooltips = show_tooltips
        
        st.divider()
        
        # About section
        st.subheader("About")
        st.markdown("""
        This dashboard provides interactive visualization and analysis of production schedules, 
        helping production planners optimize resource utilization and meet delivery deadlines.
        
        **Features:**
        - Optimized scheduling with CP-SAT or Greedy algorithms
        - Interactive Gantt chart visualization
        - Production metrics and KPIs
        - Buffer analysis and critical path identification
        """)
        
    #-------------------------------------------------------------
    # Main Content - Dashboard
    #-------------------------------------------------------------
    
    # Initialize session state for scheduler results
    if 'run_scheduler' not in st.session_state:
        st.session_state.run_scheduler = False
    
    if not st.session_state.run_scheduler:
        # Show welcome screen with instructions
        st.info("👈 Configure scheduling parameters in the sidebar and click 'Run Scheduler' to generate a schedule.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jobs", "0", "Not loaded")
        with col2:
            st.metric("Machines", "0", "Not loaded")
        with col3:
            st.metric("Schedule", "Not generated", "")
        
        st.markdown("""
        ### Getting Started
        
        1. **Configure Settings**: Adjust the scheduling parameters in the sidebar.
        2. **Upload Data**: You can upload your own Excel file or use the default data.
        3. **Run Scheduler**: Click the 'Run Scheduler' button to generate the schedule.
        4. **Analyze Results**: Explore the interactive visualizations and production metrics.
        
        ### About the Scheduler
        
        The scheduler uses two algorithms:
        
        - **CP-SAT**: An advanced constraint programming solver that finds optimal solutions.
        - **Greedy**: A faster heuristic algorithm that provides good solutions quickly.
        
        By default, the system attempts to use CP-SAT first, and falls back to Greedy if 
        CP-SAT cannot find a solution within the time limit.
        """)
        
    else:
        # Run the scheduler and display results
        try:
            # Get parameters from session state
            file_path = st.session_state.file_path
            max_jobs = st.session_state.max_jobs
            force_greedy = st.session_state.force_greedy
            cp_sat_only = st.session_state.cp_sat_only
            enforce_sequence = st.session_state.enforce_sequence
            time_limit = st.session_state.time_limit
            chart_height = st.session_state.chart_height
            show_tooltips = st.session_state.show_tooltips
            
            # Run the scheduler
            with st.spinner("Running scheduler..."):
                jobs, schedule, gantt_data = run_scheduler(
                    file_path, max_jobs, force_greedy, cp_sat_only, enforce_sequence, time_limit
                )
            
            if jobs is None or schedule is None:
                st.error("Failed to generate schedule. Please adjust parameters and try again.")
                return
            
            # Convert jobs list to DataFrame
            jobs_df = pd.DataFrame(jobs)
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Schedule Overview", 
                "📈 Production Metrics", 
                "🔄 Process Flow", 
                "📋 Job Details"
            ])
            
            with tab1:
                st.header("Production Schedule")
                
                # Schedule statistics
                total_jobs = sum(len(tasks) for tasks in schedule.values())
                machines_used = len([m for m, tasks in schedule.items() if tasks])
                total_machines = len(schedule.keys())
                total_duration = sum((end - start) for machine, tasks in schedule.items() for _, start, end, _ in tasks)
                schedule_span = max((end for machine, tasks in schedule.items() for _, _, end, _ in tasks), default=int(datetime.now().timestamp())) - \
                                min((start for machine, tasks in schedule.items() for _, start, _, _ in tasks), default=int(datetime.now().timestamp()))
                
                # KPI metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Jobs", f"{total_jobs}")
                with col2:
                    st.metric("Machines Used", f"{machines_used}/{total_machines}", f"{(machines_used/total_machines*100):.1f}%")
                with col3:
                    st.metric("Total Duration", f"{(total_duration/3600):.1f} hours")
                with col4:
                    st.metric("Schedule Span", f"{(schedule_span/3600):.1f} hours")
                
                # Buffer status metrics
                if 'BAL_HR' in jobs_df.columns:
                    critical_jobs = jobs_df[jobs_df['BAL_HR'] < 8]
                    warning_jobs = jobs_df[(jobs_df['BAL_HR'] >= 8) & (jobs_df['BAL_HR'] < 24)]
                    caution_jobs = jobs_df[(jobs_df['BAL_HR'] >= 24) & (jobs_df['BAL_HR'] < 72)]
                    ok_jobs = jobs_df[jobs_df['BAL_HR'] >= 72]
                    
                    st.markdown("### Buffer Status")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card critical">
                            <h3>Critical (&lt;8h)</h3>
                            <h2>{len(critical_jobs)} jobs</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card warning">
                            <h3>Warning (&lt;24h)</h3>
                            <h2>{len(warning_jobs)} jobs</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card caution">
                            <h3>Caution (&lt;72h)</h3>
                            <h2>{len(caution_jobs)} jobs</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card ok">
                            <h3>OK (&gt;72h)</h3>
                            <h2>{len(ok_jobs)} jobs</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Gantt chart visualization
                st.subheader("Interactive Gantt Chart")
                
                # Filter controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Machine filter
                    all_machines = sorted(gantt_data['Resource'].unique())
                    selected_machines = st.multiselect(
                        "Filter by Machine",
                        options=all_machines,
                        default=all_machines[:5] if len(all_machines) > 5 else all_machines
                    )
                with col2:
                    # Priority filter
                    all_priorities = sorted(gantt_data['Priority'].unique())
                    selected_priorities = st.multiselect(
                        "Filter by Priority",
                        options=all_priorities,
                        default=all_priorities
                    )
                with col3:
                    # Buffer status filter
                    buffer_options = ["Critical", "Warning", "Caution", "OK"]
                    selected_buffer = st.multiselect(
                        "Filter by Buffer Status",
                        options=buffer_options,
                        default=buffer_options
                    )
                
                # Apply filters
                filtered_data = gantt_data
                if selected_machines:
                    filtered_data = filtered_data[filtered_data['Resource'].isin(selected_machines)]
                if selected_priorities:
                    filtered_data = filtered_data[filtered_data['Priority'].isin(selected_priorities)]
                if selected_buffer:
                    filtered_data = filtered_data[filtered_data['BufferStatusText'].isin(selected_buffer)]
                
                # Create and display the Gantt chart
                if not filtered_data.empty:
                    # Use the wrapper function specific for Streamlit display
                    gantt_fig = create_streamlit_gantt_chart(filtered_data, height=chart_height)
                    st.plotly_chart(gantt_fig, use_container_width=True)
                    
                    # Export options
                    col1, col2 = st.columns(2)
                    with col1:
                        export_html = st.button("Export as Interactive HTML", type="secondary")
                    with col2:
                        export_csv = st.button("Export Job Data as CSV", type="secondary")
                    
                    if export_html:
                        try:
                            output_file = "interactive_schedule.html"
                            html_output = os.path.splitext(output_file)[0] + "_view.html"
                            
                            # Create interactive HTML using original chart.py function
                            create_interactive_gantt(schedule, jobs, output_file)
                            
                            # Create user-friendly HTML view
                            export_schedule_html(jobs, schedule, html_output)
                            
                            # Provide download links
                            with open(output_file, "rb") as file:
                                btn = st.download_button(
                                    label="Download Interactive Gantt Chart",
                                    data=file,
                                    file_name=output_file,
                                    mime="text/html"
                                )
                            
                            with open(html_output, "rb") as file:
                                btn2 = st.download_button(
                                    label="Download Schedule View",
                                    data=file,
                                    file_name=html_output,
                                    mime="text/html"
                                )
                        except Exception as e:
                            st.error(f"Error exporting HTML: {str(e)}")
                    
                    if export_csv:
                        csv = jobs_df.to_csv(index=False)
                        st.download_button(
                            label="Download Job Data CSV",
                            data=csv,
                            file_name="production_schedule.csv",
                            mime="text/csv",
                        )
                else:
                    st.warning("No data to display with the current filters.")
            
            with tab2:
                st.header("Production Metrics")
                
                # Machine load chart
                st.subheader("Machine Utilization")
                machine_fig = create_machine_load_chart(schedule, jobs)
                if machine_fig:
                    st.plotly_chart(machine_fig, use_container_width=True)
                
                # Buffer distribution
                st.subheader("Buffer Time Distribution")
                if 'BAL_HR' in jobs_df.columns:
                    buffer_fig = create_buffer_distribution_chart(jobs_df)
                    st.plotly_chart(buffer_fig, use_container_width=True)
                
                # Critical jobs list
                st.subheader("Critical Jobs (Limited Buffer)")
                if 'BAL_HR' in jobs_df.columns:
                    critical_df = jobs_df[jobs_df['BAL_HR'] < 24].sort_values('BAL_HR')
                    if not critical_df.empty:
                        critical_cols = ['PROCESS_CODE', 'JOB', 'RSC_LOCATION', 'BAL_HR', 'BUFFER_STATUS']
                        display_cols = [col for col in critical_cols if col in critical_df.columns]
                        
                        # Format the display
                        display_df = critical_df[display_cols].copy()
                        if 'BAL_HR' in display_df.columns:
                            display_df['BAL_HR'] = display_df['BAL_HR'].round(1)
                            
                        # Add buffer status styling
                        def style_buffer_status(val):
                            if val == 'Critical':
                                return 'background-color: rgba(255, 0, 0, 0.15); font-weight: bold; color: #cc0000'
                            elif val == 'Warning':
                                return 'background-color: rgba(255, 165, 0, 0.15); font-weight: bold; color: #b38600'
                            return ''
                        
                        if 'BUFFER_STATUS' in display_df.columns:
                            styled_df = display_df.style.applymap(style_buffer_status, subset=['BUFFER_STATUS'])
                            st.dataframe(styled_df, use_container_width=True)
                        else:
                            st.dataframe(display_df, use_container_width=True)
                    else:
                        st.info("No critical jobs found.")
            
            with tab3:
                st.header("Process Flow Analysis")
                
                # Process flow diagram
                st.subheader("Process Flow Diagram")
                process_fig = create_process_flow_chart(jobs)
                if process_fig:
                    st.plotly_chart(process_fig, use_container_width=True)
                else:
                    st.info("Not enough process sequence data to generate a flow diagram.")
                
                # Job family analysis
                st.subheader("Job Family Analysis")
                
                # Extract job families and their details
                family_data = {}
                current_time = int(datetime.now().timestamp())
                for job in jobs:
                    family = extract_job_family(job['PROCESS_CODE'])
                    if family not in family_data:
                        family_data[family] = {
                            'processes': [],
                            'machines': set(),
                            'total_time': 0,
                            'min_buffer': float('inf'),
                            'has_critical': False,
                            'has_start_date': False
                        }
                    
                    family_data[family]['processes'].append(job['PROCESS_CODE'])
                    
                    machine = job.get('RSC_LOCATION', job.get('MACHINE_ID', 'Unknown'))
                    family_data[family]['machines'].add(machine)
                    
                    if 'processing_time' in job:
                        family_data[family]['total_time'] += job['processing_time']
                    
                    if 'BAL_HR' in job:
                        family_data[family]['min_buffer'] = min(family_data[family]['min_buffer'], job['BAL_HR'])
                        if job['BAL_HR'] < 8:
                            family_data[family]['has_critical'] = True
                    
                    if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] and job['START_DATE_EPOCH'] > current_time:
                        family_data[family]['has_start_date'] = True
                
                # Convert to DataFrame
                family_rows = []
                for family, data in family_data.items():
                    family_rows.append({
                        'Family': family,
                        'Process Count': len(data['processes']),
                        'Machines Used': len(data['machines']),
                        'Total Duration (h)': data['total_time'] / 3600,
                        'Min Buffer (h)': data['min_buffer'] if data['min_buffer'] != float('inf') else 'N/A',
                        'Has Critical': data['has_critical'],
                        'Has START_DATE': data['has_start_date']
                    })
                
                family_df = pd.DataFrame(family_rows)
                
                # Add buffer status styling
                def highlight_critical(row):
                    style = [''] * len(row)
                    if row['Has Critical']:
                        idx = list(row.index).index('Has Critical')
                        style[idx] = 'background-color: rgba(255, 0, 0, 0.15)'
                        idx = list(row.index).index('Min Buffer (h)')
                        style[idx] = 'background-color: rgba(255, 0, 0, 0.15)'
                    if row['Has START_DATE']:
                        idx = list(row.index).index('Has START_DATE')
                        style[idx] = 'background-color: rgba(0, 0, 255, 0.15)'
                    return style
                
                # Format and display
                if not family_df.empty:
                    family_df = family_df.sort_values('Min Buffer (h)')
                    
                    # Replace inf with N/A
                    if 'Min Buffer (h)' in family_df.columns:
                        family_df['Min Buffer (h)'] = family_df['Min Buffer (h)'].apply(
                            lambda x: f"{x:.1f}" if isinstance(x, (int, float)) and x != float('inf') else 'N/A'
                        )
                    
                    # Display with styling
                    st.dataframe(family_df.style.apply(highlight_critical, axis=1), use_container_width=True)
                else:
                    st.info("No family data to display.")
            
            with tab4:
                st.header("Job Details")
                
                # Search and filter controls
                col1, col2 = st.columns(2)
                with col1:
                    search_term = st.text_input("Search by Process Code or Job Name", "")
                with col2:
                    sort_by = st.selectbox(
                        "Sort by",
                        options=["Process Code", "Due Date", "Buffer (Ascending)", "Buffer (Descending)", "Priority"],
                        index=2
                    )
                
                # Convert jobs to DataFrame for display
                if isinstance(jobs_df, pd.DataFrame) and not jobs_df.empty:
                    display_df = jobs_df.copy()
                    
                    # Apply search filter
                    if search_term:
                        if 'PROCESS_CODE' in display_df.columns:
                            mask1 = display_df['PROCESS_CODE'].astype(str).str.contains(search_term, case=False)
                        else:
                            mask1 = pd.Series([False] * len(display_df))
                            
                        if 'JOB' in display_df.columns:
                            mask2 = display_df['JOB'].astype(str).str.contains(search_term, case=False)
                        else:
                            mask2 = pd.Series([False] * len(display_df))
                            
                        display_df = display_df[mask1 | mask2]
                    
                    # Apply sorting
                    if sort_by == "Process Code" and 'PROCESS_CODE' in display_df.columns:
                        display_df = display_df.sort_values('PROCESS_CODE')
                    elif sort_by == "Due Date" and 'LCD_DATE_EPOCH' in display_df.columns:
                        display_df = display_df.sort_values('LCD_DATE_EPOCH')
                    elif sort_by == "Buffer (Ascending)" and 'BAL_HR' in display_df.columns:
                        display_df = display_df.sort_values('BAL_HR')
                    elif sort_by == "Buffer (Descending)" and 'BAL_HR' in display_df.columns:
                        display_df = display_df.sort_values('BAL_HR', ascending=False)
                    elif sort_by == "Priority" and 'PRIORITY' in display_df.columns:
                        display_df = display_df.sort_values('PRIORITY')
                    
                    # Select and format columns for display
                    display_columns = [
                        'PROCESS_CODE', 'JOB', 'RSC_LOCATION', 'PRIORITY', 
                        'JOB_QUANTITY', 'EXPECT_OUTPUT_PER_HOUR', 'HOURS_NEED',
                        'START_TIME', 'END_TIME', 'BAL_HR', 'BUFFER_STATUS'
                    ]
                    
                    # Keep only columns that exist
                    display_columns = [col for col in display_columns if col in display_df.columns]
                    
                    # Format timestamps to dates
                    for col in ['START_TIME', 'END_TIME']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(
                                lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M') if pd.notna(x) else ''
                            )
                    
                    # Format numeric columns
                    for col in ['BAL_HR', 'HOURS_NEED']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].round(1)
                    
                    # Define styling function
                    def style_dataframe(df):
                        # Style the entire row based on BUFFER_STATUS
                        if 'BUFFER_STATUS' in df.columns:
                            return df.style.apply(
                                lambda row: [
                                    'background-color: rgba(255, 0, 0, 0.15)' if row['BUFFER_STATUS'] == 'Critical' else
                                    'background-color: rgba(255, 165, 0, 0.15)' if row['BUFFER_STATUS'] == 'Warning' else
                                    'background-color: rgba(128, 0, 128, 0.15)' if row['BUFFER_STATUS'] == 'Caution' else
                                    'background-color: rgba(0, 128, 0, 0.15)' if row['BUFFER_STATUS'] == 'OK' else
                                    '' for _ in row.index
                                ],
                                axis=1
                            )
                        return df.style
                    
                    # Display data with styling
                    styled_df = style_dataframe(display_df[display_columns])
                    st.dataframe(styled_df, use_container_width=True, height=500)
                    
                    # Download button
                    csv = display_df[display_columns].to_csv(index=False)
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv,
                        file_name="production_schedule.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No job data available for display.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()