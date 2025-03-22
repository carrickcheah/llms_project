# chart_two.py | dont edit this line
import os
import pandas as pd
from datetime import datetime
import logging
import re
from dotenv import load_dotenv

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
        
        # FIRST PRIORITY: EXPLICIT START_DATE OVERRIDES
        for job in jobs:
            process_code = job['PROCESS_CODE']
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] and process_code in start_times:
                # For START_DATE jobs, always use exactly the requested START_DATE for visualization
                original_start = start_times[process_code]
                original_end = end_times[process_code]
                duration = original_end - original_start
                
                # Set the visualization start time exactly to START_DATE
                requested_start = job['START_DATE_EPOCH']
                adjusted_start = requested_start  # Use exactly the START_DATE
                adjusted_end = adjusted_start + duration
                
                # Store the adjusted times
                adjusted_times[process_code] = (adjusted_start, adjusted_end)
                
                start_date_str = datetime.fromtimestamp(adjusted_start).strftime('%Y-%m-%d %H:%M')
                logger.info(f"Job {process_code} visualization using exact START_DATE: {start_date_str}")
        
        # SECOND PRIORITY: FAMILY-WIDE TIME SHIFTS        
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
                    # Skip if already processed by START_DATE override
                    if process_code in adjusted_times:
                        continue
                        
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
                    
                    # For jobs with START_DATE constraints, prioritize using the exact date
                    if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] > current_time:
                        job_start = job['START_DATE_EPOCH']
                        # Adjust end time to maintain the same duration
                        original_duration = original_end - original_start
                        job_end = job_start + original_duration
                        logger.info(f"Using START_DATE={user_start_date} for {process_code} in visualization")
                
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
                # Use RSC_CODE if available, fall back to legacy ways of determining job code
                job_code = job.get('RSC_CODE', job.get('JOB_CODE', 
                           process_code.split('-P')[0] if '-P' in process_code else process_code))
                
                # Get resource location
                resource_location = job.get('RSC_LOCATION', job.get('MACHINE_ID', 'Unknown'))
                
                # Get job name
                job_name = job.get('JOB', job_code)
                
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
                
                # Get quantity information
                job_quantity = job.get('JOB_QUANTITY', 0)
                expect_output = job.get('EXPECT_OUTPUT_PER_HOUR', 0)
                accumulated_output = job.get('ACCUMULATED_DAILY_OUTPUT', 0)
                balance_quantity = job.get('BALANCE_QUANTITY', job_quantity - accumulated_output)
                
                # Add to schedule data
                schedule_data.append({
                    'LCD_DATE': due_date,
                    'JOB': job_name,
                    'PROCESS_CODE': process_code,
                    'RSC_LOCATION': resource_location,
                    'RSC_CODE': job_code,
                    'NUMBER_OPERATOR': job.get('NUMBER_OPERATOR', 1),
                    'JOB_QUANTITY': job_quantity,
                    'EXPECT_OUTPUT_PER_HOUR': expect_output,
                    'PRIORITY': job.get('PRIORITY', 3),
                    'HOURS_NEED': duration_hours,  # Using the calculated duration
                    'SETTING_HOURS': job.get('setup_time', 0) / 3600,
                    'BREAK_HOURS': job.get('break_time', 0) / 3600 if 'break_time' in job else 0,
                    'NO_PROD': job.get('downtime', 0) / 3600,
                    'START_DATE': user_start_date,
                    'ACCUMULATED_DAILY_OUTPUT': accumulated_output,
                    'BALANCE_QUANTITY': balance_quantity,
                    'START_TIME': job_start_date,
                    'END_TIME': end_date,
                    'BAL_HR': round(buffer_hours, 1),
                    'BUFFER_STATUS': buffer_status,
                })
        
        # Create DataFrame for easier HTML generation
        df = pd.DataFrame(schedule_data)
        
        # Sort by LCD_DATE by default
        df = df.sort_values(by='LCD_DATE')
        
        # Calculate percentages safely to avoid division by zero
        total_jobs = len(df) if not df.empty else 1  # Avoid division by zero
        critical_percent = len(df[df['BUFFER_STATUS'] == 'Critical']) / total_jobs * 100 if not df.empty else 0
        warning_percent = len(df[df['BUFFER_STATUS'] == 'Warning']) / total_jobs * 100 if not df.empty else 0
        caution_percent = len(df[df['BUFFER_STATUS'] == 'Caution']) / total_jobs * 100 if not df.empty else 0
        ok_percent = len(df[df['BUFFER_STATUS'] == 'OK']) / total_jobs * 100 if not df.empty else 0
        
        # Prepare the HTML content
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Production Scheduler</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.10.25/css/dataTables.bootstrap5.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.2.2/css/buttons.bootstrap5.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.2.9/css/responsive.bootstrap5.min.css">
    <style>
        body {{ 
            font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; 
            background-color: #f5f7fa;
            color: #333;
        }}
        .container-fluid {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .dashboard {{
            margin-bottom: 30px; 
            padding: 20px; 
            border-radius: 8px; 
            background: #fff;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border: 1px solid #eaeaea;
        }}
        .dashboard h3 {{
            margin-top: 0;
            font-weight: 600;
            color: #3a3a3a;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        table.dataTable {{
            border-collapse: separate !important;
            border-spacing: 0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }}
        .status-critical {{ background-color: rgba(255, 0, 0, 0.15); }}
        .status-warning {{ background-color: rgba(255, 190, 0, 0.15); }}
        .status-caution {{ background-color: rgba(128, 0, 128, 0.15); }}
        .status-ok {{ background-color: rgba(0, 128, 0, 0.15); }}
        
        .status-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: 600;
            text-align: center;
            width: 100%;
        }}
        .badge-critical {{ background-color: #ffcccc; color: #cc0000; border: 1px solid #ff8080; }}
        .badge-warning {{ background-color: #fff2cc; color: #b38600; border: 1px solid #ffdb4d; }}
        .badge-caution {{ background-color: #f0d6f0; color: #800080; border: 1px solid #d699d6; }}
        .badge-ok {{ background-color: #d6f0d6; color: #006600; border: 1px solid #99d699; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h1 class="mb-0"><i class="bi bi-calendar-check me-2"></i>Production Scheduler</h1>
            <div>
                <button class="btn btn-primary" onclick="window.print()">
                    <i class="bi bi-printer me-1"></i> Print
                </button>
                <button class="btn btn-outline-secondary ms-2" onclick="exportTableToCSV('production_schedule.csv')">
                    <i class="bi bi-download me-1"></i> Export
                </button>
            </div>
        </div>
        
        <div class="dashboard row">
            <div class="col-md-6">
                <h3>Schedule Overview</h3>
                <p><strong>Total Jobs:</strong> {len(df)}</p>
                <p><strong>Date Range:</strong> {df['START_TIME'].min() if not df.empty else 'N/A'} to {df['END_TIME'].max() if not df.empty else 'N/A'}</p>
                <p><strong>Total Duration:</strong> {df['HOURS_NEED'].sum() if not df.empty else 0} hours</p>
            </div>
            <div class="col-md-6">
                <h3>Buffer Status</h3>
                <div class="d-flex flex-column gap-2">
                    <div class="d-flex align-items-center">
                        <div class="status-badge badge-critical me-2" style="width: 120px;">Critical</div>
                        <div class="progress flex-grow-1" style="height: 20px;">
                            <div class="progress-bar bg-danger" role="progressbar" 
                                style="width: {critical_percent}%;" 
                                aria-valuenow="{len(df[df['BUFFER_STATUS'] == 'Critical']) if not df.empty else 0}" 
                                aria-valuemin="0" aria-valuemax="{len(df) if not df.empty else 0}">
                                {len(df[df['BUFFER_STATUS'] == 'Critical']) if not df.empty else 0} jobs
                            </div>
                        </div>
                    </div>
                    <div class="d-flex align-items-center">
                        <div class="status-badge badge-warning me-2" style="width: 120px;">Warning</div>
                        <div class="progress flex-grow-1" style="height: 20px;">
                            <div class="progress-bar bg-warning text-dark" role="progressbar" 
                                style="width: {warning_percent}%;" 
                                aria-valuenow="{len(df[df['BUFFER_STATUS'] == 'Warning']) if not df.empty else 0}" 
                                aria-valuemin="0" aria-valuemax="{len(df) if not df.empty else 0}">
                                {len(df[df['BUFFER_STATUS'] == 'Warning']) if not df.empty else 0} jobs
                            </div>
                        </div>
                    </div>
                    <div class="d-flex align-items-center">
                        <div class="status-badge badge-caution me-2" style="width: 120px;">Caution</div>
                        <div class="progress flex-grow-1" style="height: 20px;">
                            <div class="progress-bar" role="progressbar" style="background-color: purple; width: {caution_percent}%;" 
                                aria-valuenow="{len(df[df['BUFFER_STATUS'] == 'Caution']) if not df.empty else 0}" 
                                aria-valuemin="0" aria-valuemax="{len(df) if not df.empty else 0}">
                                {len(df[df['BUFFER_STATUS'] == 'Caution']) if not df.empty else 0} jobs
                            </div>
                        </div>
                    </div>
                    <div class="d-flex align-items-center">
                        <div class="status-badge badge-ok me-2" style="width: 120px;">OK</div>
                        <div class="progress flex-grow-1" style="height: 20px;">
                            <div class="progress-bar bg-success" role="progressbar" 
                                style="width: {ok_percent}%;" 
                                aria-valuenow="{len(df[df['BUFFER_STATUS'] == 'OK']) if not df.empty else 0}" 
                                aria-valuemin="0" aria-valuemax="{len(df) if not df.empty else 0}">
                                {len(df[df['BUFFER_STATUS'] == 'OK']) if not df.empty else 0} jobs
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="table-container">
            <h2>Production Jobs</h2>
            <table id="scheduleTable" class="table table-striped table-bordered">
                <thead>
                    <tr>
                        <th>LCD_DATE</th>
                        <th>JOB</th>
                        <th>PROCESS_CODE</th>
                        <th>RSC_LOCATION</th>
                        <th>RSC_CODE</th>
                        <th>NUMBER_OPERATOR</th>
                        <th>JOB_QUANTITY</th>
                        <th>EXPECT_OUTPUT_PER_HOUR</th>
                        <th>PRIORITY</th>
                        <th>HOURS_NEED</th>
                        <th>SETTING_HOURS</th>
                        <th>BREAK_HOURS</th>
                        <th>NO_PROD</th>
                        <th>START_DATE</th>
                        <th>ACCUMULATED_DAILY_OUTPUT</th>
                        <th>BALANCE_QUANTITY</th>
                        <th>START_TIME</th>
                        <th>END_TIME</th>
                        <th>BAL_HR</th>
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
                
            buffer_badge_class = ""
            if row['BUFFER_STATUS'] == 'Critical':
                buffer_badge_class = "badge-critical"
            elif row['BUFFER_STATUS'] == 'Warning':
                buffer_badge_class = "badge-warning"
            elif row['BUFFER_STATUS'] == 'Caution':
                buffer_badge_class = "badge-caution"
            elif row['BUFFER_STATUS'] == 'OK':
                buffer_badge_class = "badge-ok"
                
            html_content += f"""
                    <tr class="{buffer_class}">
                        <td>{row['LCD_DATE']}</td>
                        <td>{row['JOB']}</td>
                        <td>{row['PROCESS_CODE']}</td>
                        <td>{row['RSC_LOCATION']}</td>
                        <td>{row['RSC_CODE']}</td>
                        <td>{row['NUMBER_OPERATOR']}</td>
                        <td>{row['JOB_QUANTITY']}</td>
                        <td>{row['EXPECT_OUTPUT_PER_HOUR']}</td>
                        <td>{row['PRIORITY']}</td>
                        <td>{row['HOURS_NEED']:.1f}</td>
                        <td>{row['SETTING_HOURS']:.1f}</td>
                        <td>{row['BREAK_HOURS']:.1f}</td>
                        <td>{row['NO_PROD']:.1f}</td>
                        <td>{row['START_DATE']}</td>
                        <td>{row['ACCUMULATED_DAILY_OUTPUT']}</td>
                        <td>{row['BALANCE_QUANTITY']}</td>
                        <td>{row['START_TIME']}</td>
                        <td>{row['END_TIME']}</td>
                        <td>{row['BAL_HR']:.1f}</td>
                        <td><div class="status-badge {buffer_badge_class}">{row['BUFFER_STATUS']}</div></td>
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
    <script src="https://cdn.datatables.net/buttons/2.2.2/js/dataTables.buttons.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.2.2/js/buttons.bootstrap5.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.2.2/js/buttons.html5.min.js"></script>
    <script src="https://cdn.datatables.net/responsive/2.2.9/js/dataTables.responsive.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.1.3/jszip.min.js"></script>
    <script>
        $(document).ready(function() {
            $('#scheduleTable').DataTable({
                order: [[0, 'asc']], // Sort by LCD_DATE by default
                pageLength: 25,
                lengthMenu: [10, 25, 50, 100, 200],
                responsive: true,
                dom: 'Bfrtip',
                buttons: [
                    {
                        extend: 'excel',
                        text: '<i class="bi bi-file-earmark-excel me-1"></i> Excel',
                        className: 'btn btn-sm btn-success',
                        exportOptions: {
                            columns: ':visible'
                        }
                    },
                    {
                        extend: 'csv',
                        text: '<i class="bi bi-file-earmark-text me-1"></i> CSV',
                        className: 'btn btn-sm btn-info',
                        exportOptions: {
                            columns: ':visible'
                        }
                    },
                    {
                        extend: 'pdf',
                        text: '<i class="bi bi-file-earmark-pdf me-1"></i> PDF',
                        className: 'btn btn-sm btn-danger',
                        exportOptions: {
                            columns: ':visible'
                        }
                    },
                    {
                        extend: 'colvis',
                        text: '<i class="bi bi-eye me-1"></i> Columns',
                        className: 'btn btn-sm btn-secondary'
                    }
                ],
                initComplete: function () {
                    // Add a legend at the top
                    var legendHtml = '<div class="mb-3 p-3 bg-white rounded shadow-sm">' +
                        '<h5 class="border-bottom pb-2 mb-3"><i class="bi bi-info-circle me-2"></i>Buffer Status Legend:</h5>' +
                        '<div class="d-flex flex-wrap gap-3">' +
                        '<div class="status-badge badge-critical">Critical (&lt;8h)</div>' +
                        '<div class="status-badge badge-warning">Warning (&lt;24h)</div>' +
                        '<div class="status-badge badge-caution">Caution (&lt;72h)</div>' +
                        '<div class="status-badge badge-ok">OK (&gt;72h)</div>' +
                        '</div></div>';
                    $('.dataTables_wrapper').prepend(legendHtml);
                }
            });
        });
        
        function exportTableToCSV(filename) {
            var csv = [];
            var rows = document.querySelectorAll("table tr");
            
            for (var i = 0; i < rows.length; i++) {
                var row = [], cols = rows[i].querySelectorAll("td, th");
                
                for (var j = 0; j < cols.length; j++) {
                    // Get the text content, handling the case of the status badge
                    var text = cols[j].innerText;
                    if (cols[j].querySelector('.status-badge')) {
                        text = cols[j].querySelector('.status-badge').innerText;
                    }
                    row.push('"' + text.replace(/"/g, '""') + '"');
                }
                
                csv.push(row.join(","));
            }
            
            // Download CSV file
            downloadCSV(csv.join("\\n"), filename);
        }
        
        function downloadCSV(csv, filename) {
            var csvFile = new Blob([csv], {type: "text/csv"});
            var downloadLink = document.createElement("a");
            
            // File name
            downloadLink.download = filename;
            
            // Create a link to the file
            downloadLink.href = window.URL.createObjectURL(csvFile);
            
            // Hide download link
            downloadLink.style.display = "none";
            
            // Add the link to DOM
            document.body.appendChild(downloadLink);
            
            // Click download link
            downloadLink.click();
            
            // Remove the link
            document.body.removeChild(downloadLink);
        }
    </script>
</body>
</html>"""
        
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
    
    # Load environment variables
    load_dotenv()
    file_path = os.getenv('file_path')
    
    if not file_path:
        logger.error("No file_path found in environment variables.")
        exit(1)
        
    output_file = "schedule_view.html"
    
    try:
        # Load job data from environment variable path
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