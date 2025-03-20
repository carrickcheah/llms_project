import os
import pandas as pd
from datetime import datetime
import logging
import re

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

def export_schedule_html(jobs, schedule, output_file='schedule_view.html'):
    """
    Export the schedule to an interactive HTML file.
    
    Args:
        jobs (list): List of job dictionaries with buffer information
        schedule (dict): Schedule as {machine: [(process_code, start, end, priority), ...]}
        output_file (str): Path to save the HTML file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Step 1: Create a mapping of job families and their processes in sequence
        family_processes = {}
        for job in jobs:
            process_code = job['PROCESS_CODE']
            family = extract_job_family(process_code)
            seq_num = extract_process_number(process_code)
            
            if family not in family_processes:
                family_processes[family] = []
            
            family_processes[family].append((seq_num, process_code, job))
        
        # Sort processes within each family by sequence number
        for family in family_processes:
            family_processes[family].sort(key=lambda x: x[0])
        
        # Create a lookup of original end and start times from the schedule
        end_times = {}
        start_times = {}
        for machine, tasks in schedule.items():
            for task in tasks:
                process_code, start, end, priority = task
                end_times[process_code] = end
                start_times[process_code] = start
        
        # Step 2: Apply time shifts from jobs to their visualization times
        adjusted_times = {}
        current_time = int(datetime.now().timestamp())
        
        for family, processes in family_processes.items():
            # Check if any job in this family has a time shift
            family_time_shift = 0
            for seq_num, process_code, job in processes:
                if 'family_time_shift' in job and abs(job['family_time_shift']) > 60:  # More than a minute
                    family_time_shift = job['family_time_shift']
                    break
            
            # If family has time shift, apply to all processes in family
            if family_time_shift != 0:
                logger.info(f"Applying time shift of {family_time_shift/3600:.1f} hours to family {family} for visualization")
                for seq_num, process_code, job in processes:
                    if process_code in start_times and process_code in end_times:
                        original_start = start_times[process_code]
                        original_end = end_times[process_code]
                        
                        # Apply the time shift
                        adjusted_start = original_start - family_time_shift
                        adjusted_end = original_end - family_time_shift
                        
                        # Store the adjusted times
                        adjusted_times[process_code] = (adjusted_start, adjusted_end)
                        
                        logger.info(f"Adjusted {process_code}: START={datetime.fromtimestamp(adjusted_start).strftime('%Y-%m-%d %H:%M')}, "
                                  f"END={datetime.fromtimestamp(adjusted_end).strftime('%Y-%m-%d %H:%M')}")
        
        # Step 3: Process each job and create schedule data
        schedule_data = []
        
        for job in jobs:
            process_code = job['PROCESS_CODE']
            if process_code in end_times:
                # Get original scheduled times
                original_start = start_times[process_code]
                original_end = end_times[process_code]
                due_time = job.get('LCD_DATE_EPOCH', 0)
                
                # Use adjusted times if available, otherwise use original times
                if process_code in adjusted_times:
                    job_start, job_end = adjusted_times[process_code]
                else:
                    job_start, job_end = original_start, original_end
                
                # Override with exact START_DATE if specified
                if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH']:
                    # For display purposes
                    user_start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
                    
                    # Find the family this process belongs to
                    family = extract_job_family(process_code)
                    seq_num = extract_process_number(process_code)
                    
                    # Check if this is the first process in the family
                    is_first_process = False
                    if family in family_processes and len(family_processes[family]) > 0:
                        is_first_process = (family_processes[family][0][1] == process_code)
                    
                    # If this is the first process with START_DATE, use the exact date
                    if is_first_process:
                        job_start = job['START_DATE_EPOCH']
                        # Adjust end time to maintain the same duration
                        original_duration = original_end - original_start
                        job_end = job_start + original_duration
                
                # Format the dates for display
                job_start_date = datetime.fromtimestamp(job_start).strftime('%Y-%m-%d %H:%M')
                end_date = datetime.fromtimestamp(job_end).strftime('%Y-%m-%d %H:%M')
                due_date = datetime.fromtimestamp(due_time).strftime('%Y-%m-%d %H:%M')
                
                # Get START_DATE for display
                user_start_date = ""
                if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH']:
                    user_start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
                
                # Calculate duration and buffer
                duration_seconds = job_end - job_start
                duration_hours = duration_seconds / 3600
                buffer_seconds = max(0, due_time - job_end)
                buffer_hours = buffer_seconds / 3600
                
                # Job family and sequence
                job_code = job.get('JOB_CODE', process_code.split('-P')[0] if '-P' in process_code else process_code)
                
                # Get buffer status
                buffer_status = ""
                if buffer_hours < 8:
                    buffer_status = "Critical"
                elif buffer_hours < 24:
                    buffer_status = "Warning"
                elif buffer_hours < 72:
                    buffer_status = "Caution"
                else:
                    buffer_status = "OK"
                
                # Add to schedule data
                schedule_data.append({
                    'LCD_DATE': due_date,
                    'START_DATE': user_start_date,
                    'PROCESS_CODE': process_code,
                    'JOB_CODE': job_code,
                    'MACHINE_ID': job.get('MACHINE_ID', 'Unknown'),
                    'PRIORTY': job.get('PRIORITY', 3),
                    'START_TIME': job_start_date,
                    'END_TIME': end_date,
                    'DURATION_HOURS': round(duration_hours, 1),
                    'BUFFER_HOURS': round(buffer_hours, 1),
                    'BUFFER_STATUS': buffer_status,
                })
        
        # Create DataFrame for easier HTML generation
        df = pd.DataFrame(schedule_data)
        
        # Sort by LCD_DATE by default
        df = df.sort_values(by='LCD_DATE')
        
        # Prepare the HTML content
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Production Schedule</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.10.25/css/dataTables.bootstrap5.min.css">
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        .dashboard {{ margin-bottom: 30px; padding: 15px; border-radius: 5px; background: #f8f9fa; }}
        .dashboard h2 {{ margin-top: 0; }}
        .status-critical {{ background-color: rgba(255, 0, 0, 0.1); }}
        .status-warning {{ background-color: rgba(255, 165, 0, 0.1); }}
        .status-caution {{ background-color: rgba(255, 255, 0, 0.1); }}
        .status-ok {{ background-color: rgba(0, 128, 0, 0.1); }}
        .table-container {{ margin-top: 30px; }}
        .filter-section {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1 class="mb-4">Production Schedule</h1>
        
        <div class="dashboard row">
            <div class="col-md-6">
                <h3>Schedule Overview</h3>
                <p><strong>Total Jobs:</strong> {len(df)}</p>
                <p><strong>Date Range:</strong> {df['START_TIME'].min() if not df.empty else 'N/A'} to {df['END_TIME'].max() if not df.empty else 'N/A'}</p>
                <p><strong>Total Duration:</strong> {df['DURATION_HOURS'].sum() if not df.empty else 0} hours</p>
            </div>
            <div class="col-md-6">
                <h3>Buffer Status</h3>
                <p><strong>Critical (&lt;8h):</strong> {len(df[df['BUFFER_STATUS'] == 'Critical']) if not df.empty else 0} jobs</p>
                <p><strong>Warning (&lt;24h):</strong> {len(df[df['BUFFER_STATUS'] == 'Warning']) if not df.empty else 0} jobs</p>
                <p><strong>Caution (&lt;72h):</strong> {len(df[df['BUFFER_STATUS'] == 'Caution']) if not df.empty else 0} jobs</p>
                <p><strong>OK (&gt;72h):</strong> {len(df[df['BUFFER_STATUS'] == 'OK']) if not df.empty else 0} jobs</p>
            </div>
        </div>
        
        <div class="table-container">
            <h2>Production Jobs</h2>
            <table id="scheduleTable" class="table table-striped table-bordered">
                <thead>
                    <tr>
                        <th>LCD_DATE</th>
                        <th>START_DATE</th>
                        <th>PROCESS_CODE</th>
                        <th>JOB_CODE</th>
                        <th>MACHINE_ID</th>
                        <th>PRIORTY</th>
                        <th>START_TIME</th>
                        <th>END_TIME</th>
                        <th>DURATION (h)</th>
                        <th>BUFFER (h)</th>
                        <th>BUFFER_STATUS</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add rows for each job
        for _, row in df.iterrows():
            buffer_class = ""
            if row['BUFFER_STATUS'] == 'Critical':
                buffer_class = "status-critical"
            elif row['BUFFER_STATUS'] == 'Warning':
                buffer_class = "status-warning"
            elif row['BUFFER_STATUS'] == 'Caution':
                buffer_class = "status-caution"
            elif row['BUFFER_STATUS'] == 'OK':
                buffer_class = "status-ok"
            
            html_content += f"""
                    <tr class="{buffer_class}">
                        <td>{row['LCD_DATE']}</td>
                        <td>{row['START_DATE']}</td>
                        <td>{row['PROCESS_CODE']}</td>
                        <td>{row['JOB_CODE']}</td>
                        <td>{row['MACHINE_ID']}</td>
                        <td>{row['PRIORTY']}</td>
                        <td>{row['START_TIME']}</td>
                        <td>{row['END_TIME']}</td>
                        <td>{row['DURATION_HOURS']}</td>
                        <td>{row['BUFFER_HOURS']}</td>
                        <td>{row['BUFFER_STATUS']}</td>
                    </tr>"""
        
        # Complete the HTML
        html_content += """
                </tbody>
            </table>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.10.25/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.10.25/js/dataTables.bootstrap5.min.js"></script>
    <script>
        $(document).ready(function() {
            $('#scheduleTable').DataTable({
                order: [[0, 'asc']], // Sort by LCD_DATE by default
                pageLength: 25,
                lengthMenu: [10, 25, 50, 100, 200],
                responsive: true
            });
        });
    </script>
</body>
</html>
"""
        
        # Write the HTML file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML schedule view saved to: {os.path.abspath(output_file)}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating HTML schedule view: {e}")
        return False

if __name__ == "__main__":
    import sys
    from ingest_data import load_jobs_planning_data
    from greedy import greedy_schedule
    
    # Use real data path
    file_path = "/Users/carrickcheah/llms_project/services/agent_production_planning/mydata.xlsx"
    output_file = "schedule_view.html"
    
    try:
        # Load job data from real data path
        jobs, machines, setup_times = load_jobs_planning_data(file_path)
        logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines from {file_path}")
        
        # Create schedule
        schedule = greedy_schedule(jobs, machines, setup_times, enforce_sequence=True)
        logger.info(f"Generated schedule with {sum(len(jobs_list) for jobs_list in schedule.values())} scheduled tasks")
        
        # Export HTML
        result = export_schedule_html(jobs, schedule, output_file)
        if result:
            logger.info(f"Schedule exported successfully to: {os.path.abspath(output_file)}")
        else:
            logger.error("Failed to export schedule to HTML")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")