import streamlit as st
import pandas as pd
import os
import tempfile
import plotly.express as px
import streamlit.components.v1 as components
from chart import create_interactive_gantt, load_jobs_planning_data, greedy_schedule
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Production Scheduling Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #3B82F6;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
    }
    .stButton button {
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="main-header">Production Scheduling Dashboard</div>', unsafe_allow_html=True)
st.markdown("Optimize your production timeline with interactive scheduling tools")

# Sidebar
with st.sidebar:
    st.markdown('<div class="sub-header">Configuration</div>', unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Production Data", type=["xlsx", "csv"], 
                                    help="Upload an Excel or CSV file containing your production data")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filters section
    st.markdown('<div class="sub-header">Visualization Filters</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Date range filter (initialized but will be updated when data is loaded)
    date_range = st.date_input(
        "Schedule Date Range",
        value=[datetime.now().date(), (datetime.now() + timedelta(days=30)).date()],
        help="Filter the Gantt chart to show tasks within this date range"
    )
    
    # Priority filter
    priority_options = ["All Priorities", "Priority 1 (Highest)", "Priority 2 (High)", 
                        "Priority 3 (Medium)", "Priority 4 (Normal)", "Priority 5 (Low)"]
    selected_priorities = st.multiselect(
        "Filter by Priority",
        options=priority_options,
        default=["All Priorities"],
        help="Select which priority levels to display"
    )
    
    # Machine filter (will be populated with actual machines when data is loaded)
    machine_filter = st.multiselect(
        "Filter by Machine",
        options=["All Machines"],
        default=["All Machines"],
        help="Select which machines to display"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Export options
    st.markdown('<div class="sub-header">Export Options</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    export_format = st.radio(
        "Export Format",
        options=["HTML", "PDF", "PNG"],
        help="Choose the format to export your schedule"
    )
    
    export_button = st.button("Export Schedule", disabled=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # About section
    st.markdown('---')
    st.markdown('##### About')
    st.info('This dashboard helps you visualize and optimize your production scheduling. Upload your data to generate an interactive Gantt chart.')

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Gantt Chart", "Schedule Statistics", "Process Flow"])

with tab1:
    if uploaded_file is not None:
        try:
            with st.spinner("Processing data and generating schedule..."):
                # Create a temporary file to store the uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx' if uploaded_file.name.endswith('.xlsx') else '.csv') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # Load data using the temporary file path
                jobs, machines, setup_times = load_jobs_planning_data(tmp_file_path)
                
                # Update machine filter options based on loaded data
                available_machines = ["All Machines"]
                # Check if machines is a dictionary (keys accessible) or a list
                if isinstance(machines, dict):
                    available_machines += list(machines.keys())
                elif isinstance(machines, list):
                    available_machines += machines
                
                machine_filter = st.sidebar.multiselect(
                    "Filter by Machine",
                    options=available_machines,
                    default=["All Machines"],
                    key="machine_filter_updated",
                    help="Select which machines to display"
                )
                
                # Generate schedule
                schedule = greedy_schedule(jobs, machines, setup_times)
                
                # Enable export button
                st.sidebar.button("Export Schedule", key="export_enabled", disabled=False)
                
                # Display scheduling metrics in card format
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-value">{}</div>'.format(len(jobs)), unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Total Jobs</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-value">{}</div>'.format(len(machines)), unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Available Machines</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    # Calculate earliest start and latest end times
                    earliest = datetime.now()
                    latest = datetime.now()
                    if any(schedule.values()):
                        all_starts = []
                        all_ends = []
                        for machine, jobs_list in schedule.items():
                            for job in jobs_list:
                                if len(job) >= 3:
                                    _, start, end, _ = job
                                    all_starts.append(start)
                                    all_ends.append(end)
                        if all_starts and all_ends:
                            earliest = datetime.fromtimestamp(min(all_starts))
                            latest = datetime.fromtimestamp(max(all_ends))
                    
                    schedule_days = (latest - earliest).days + 1
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-value">{}</div>'.format(schedule_days), unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Schedule Days</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    # Calculate average utilization
                    total_machine_time = 0
                    total_job_time = 0
                    for machine, jobs_list in schedule.items():
                        for job in jobs_list:
                            if len(job) >= 3:
                                _, start, end, _ = job
                                total_job_time += (end - start)
                        total_machine_time += (latest.timestamp() - earliest.timestamp())
                    
                    utilization = 0
                    if total_machine_time > 0:
                        utilization = (total_job_time / total_machine_time) * 100
                    
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-value">{:.1f}%</div>'.format(utilization), unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Avg. Machine Utilization</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Generate interactive Gantt chart
                html_file = os.path.join(tempfile.gettempdir(), 'gantt_chart.html')
                try:
                    create_interactive_gantt(schedule, jobs, output_file=html_file)
                    
                    if os.path.exists(html_file):
                        with open(html_file, 'r') as f:
                            html_content = f.read()
                        
                        # Create Gantt chart display container with professional styling
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown('<div class="sub-header">Production Schedule Gantt Chart</div>', unsafe_allow_html=True)
                        components.html(html_content, height=800, scrolling=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error('Failed to generate Gantt chart file')
                except Exception as e:
                    st.error(f'Gantt chart generation failed: {str(e)}')

        except Exception as e:
            st.error(f"Error processing data: {e}")
            # Clean up the temporary file in case of an error
            if 'tmp_file_path' in locals():
                try:
                    os.remove(tmp_file_path)
                except Exception as e:
                    st.warning(f"Failed to clean up temporary file: {e}")
            st.stop()
        finally:
            # Clean up the temporary file after successful loading
            if 'tmp_file_path' in locals():
                try:
                    os.remove(tmp_file_path)
                except Exception as e:
                    st.warning(f"Failed to clean up temporary file: {e}")
    else:
        # Display instructions when no file is uploaded
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Getting Started")
            st.markdown("""
            1. Upload your production data file (Excel or CSV) using the sidebar
            2. The system will process your data and generate an optimized schedule
            3. Use the filters to customize your view
            4. Export the schedule in your preferred format
            """)
        
        with col2:
            st.markdown("### Sample Data Format")
            st.markdown("""
            Your data file should include:
            - Job/Process information
            - Machine availability
            - Processing times
            - Due dates
            - Priority levels
            
            For a sample template, click the button below:
            """)
            st.download_button(
                label="Download Sample Template",
                data=open("mydata.xlsx", "rb").read() if os.path.exists("mydata.xlsx") else b"",
                file_name="production_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                disabled=not os.path.exists("mydata.xlsx")
            )
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    if 'schedule' in locals() and 'jobs' in locals():
        # Create statistics tab content
        st.markdown('<div class="sub-header">Schedule Analysis</div>', unsafe_allow_html=True)
        
        # Priority distribution chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Job Priority Distribution")
        
        # Extract priority data from schedule
        priority_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for machine, jobs_list in schedule.items():
            for job in jobs_list:
                if len(job) >= 4:
                    _, _, _, priority = job
                    if priority in priority_counts:
                        priority_counts[priority] += 1
        
        priority_df = pd.DataFrame({
            'Priority': ['Priority 1\n(Highest)', 'Priority 2\n(High)', 'Priority 3\n(Medium)', 'Priority 4\n(Normal)', 'Priority 5\n(Low)'],
            'Count': list(priority_counts.values())
        })
        
        # Create a color map matching the Gantt chart colors
        color_map = {
            'Priority 1\n(Highest)': 'rgb(255, 0, 0)',
            'Priority 2\n(High)': 'rgb(255, 165, 0)',
            'Priority 3\n(Medium)': 'rgb(0, 128, 0)',
            'Priority 4\n(Normal)': 'rgb(128, 0, 128)',
            'Priority 5\n(Low)': 'rgb(60, 179, 113)'
        }
        
        fig = px.bar(priority_df, x='Priority', y='Count', 
                    color='Priority', color_discrete_map=color_map,
                    title='Job Distribution by Priority Level')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Machine utilization chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Machine Utilization")
        
        # Calculate utilization by machine
        machine_utilization = {}
        earliest_time = float('inf')
        latest_time = 0
        
        for machine, jobs_list in schedule.items():
            if jobs_list:
                machine_job_time = 0
                for job in jobs_list:
                    if len(job) >= 3:
                        _, start, end, _ = job
                        machine_job_time += (end - start)
                        earliest_time = min(earliest_time, start)
                        latest_time = max(latest_time, end)
                
                if latest_time > earliest_time:
                    total_time = latest_time - earliest_time
                    machine_utilization[machine] = (machine_job_time / total_time) * 100
        
        if machine_utilization:
            util_df = pd.DataFrame({
                'Machine': list(machine_utilization.keys()),
                'Utilization (%)': list(machine_utilization.values())
            })
            
            util_fig = px.bar(util_df, x='Machine', y='Utilization (%)', 
                            color='Utilization (%)', color_continuous_scale='Viridis',
                            title='Machine Utilization Percentage')
            util_fig.update_layout(height=400)
            st.plotly_chart(util_fig, use_container_width=True)
        else:
            st.info("No utilization data available yet. Upload a file to view machine utilization.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Upload your production data to view schedule statistics")

with tab3:
    if 'schedule' in locals() and 'jobs' in locals():
        st.markdown('<div class="sub-header">Process Flow Visualization</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Group jobs by family to visualize process flow
        job_families = {}
        for job in jobs:
            if 'PROCESS_CODE' in job:
                process_code = job['PROCESS_CODE']
                family = process_code.split('-')[0] if '-' in process_code else process_code
                
                if family not in job_families:
                    job_families[family] = []
                job_families[family].append(job)
        
        # Family selector
        if job_families:
            selected_family = st.selectbox("Select Job Family to View Process Flow", 
                                         options=list(job_families.keys()))
            
            if selected_family and selected_family in job_families:
                # Sort jobs in this family by process sequence
                family_jobs = sorted(job_families[selected_family], 
                                    key=lambda x: int(x['PROCESS_CODE'].split('-')[1]) 
                                    if '-' in x['PROCESS_CODE'] and x['PROCESS_CODE'].split('-')[1].isdigit() 
                                    else 0)
                
                # Create a process flow diagram
                process_df = pd.DataFrame([{
                    'Process': job.get('PROCESS_CODE', 'Unknown'),
                    'Machine': job.get('MACHINE_CODE', 'Unknown'),
                    'Duration (hrs)': job.get('PROCESSING_TIME', 0) / 3600 if 'PROCESSING_TIME' in job else 0,
                    'Operators': job.get('NUMBER_OPERATOR', 1)
                } for job in family_jobs])
                
                if not process_df.empty:
                    # Create a horizontal bar chart to represent process flow
                    flow_fig = px.bar(process_df, y='Process', x='Duration (hrs)', 
                                    hover_data=['Machine', 'Operators'],
                                    orientation='h', height=400,
                                    color='Machine', title=f'Process Flow for {selected_family}')
                    flow_fig.update_layout(yaxis={'categoryorder': 'array', 
                                                'categoryarray': process_df['Process'].tolist()})
                    st.plotly_chart(flow_fig, use_container_width=True)
                    
                    # Display detailed process information in a table
                    st.markdown("#### Process Details")
                    st.dataframe(process_df, hide_index=True, use_container_width=True)
                else:
                    st.info(f"No process information available for {selected_family}")
        else:
            st.info("No job family information available to display process flow")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Upload your production data to view process flow visualization")

# Footer
st.markdown("""
---
<div style="text-align: center; color: #6B7280; padding: 20px;">
    Production Scheduling Dashboard © 2025
</div>
""", unsafe_allow_html=True)