import os
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        # Create a flattened list of schedule entries
        schedule_data = []
        
        # Create a lookup of end times from the schedule
        end_times = {}
        start_times = {}
        for machine, tasks in schedule.items():
            for task in tasks:
                process_code, start, end, priority = task
                end_times[process_code] = end
                start_times[process_code] = start
        
        # Process each job with buffer information
        for job in jobs:
            process_code = job['PROCESS_CODE']
            if process_code in end_times:
                job_end = end_times[process_code]
                due_time = job.get('LCD_DATE_EPOCH', 0)
                
                # UPDATED: Handle START_DATE and START_TIME properly
                # If START_DATE_EPOCH exists, use it for display as START_DATE
                user_start_date = ""
                if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH']:
                    user_start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
                    
                    # For START_TIME, use START_DATE_EPOCH if it exists and is in the future
                    # This ensures START_TIME matches START_DATE for future dates
                    if job['START_DATE_EPOCH'] > int(datetime.now().timestamp()):
                        job_start = job['START_DATE_EPOCH']
                    else:
                        job_start = start_times[process_code]
                else:
                    # No START_DATE specified, use the scheduled start time
                    job_start = start_times[process_code]
                
                # Format the start and end dates for display
                job_start_date = datetime.fromtimestamp(job_start).strftime('%Y-%m-%d %H:%M')
                end_date = datetime.fromtimestamp(job_end).strftime('%Y-%m-%d %H:%M')
                due_date = datetime.fromtimestamp(due_time).strftime('%Y-%m-%d %H:%M')
                
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