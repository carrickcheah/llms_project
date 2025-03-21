# ui.py
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Create templates folder if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Save the HTML to the templates folder
with open("templates/index.html", "w") as f:
    f.write("""
    <<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Production Scheduler</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 20px;
            background-color: #f8f9fa;
        }
        .tab-content {
            padding: 20px;
            border: 1px solid #dee2e6;
            border-top: none;
            background-color: white;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .card {
            margin-bottom: 20px;
        }
        .table-responsive {
            margin-top: 15px;
        }
        .progress {
            height: 25px;
        }
        .priority-1 {
            background-color: #ffcccc; /* Light red */
        }
        .priority-2 {
            background-color: #ffebcc; /* Light orange */
        }
        .priority-3 {
            background-color: #c6efce; /* Light green */
        }
        .buffer-critical {
            background-color: #ff9696; /* Red */
        }
        .buffer-warning {
            background-color: #ffc864; /* Orange */
        }
        .buffer-caution {
            background-color: #ffff96; /* Yellow */
        }
        .buffer-ok {
            background-color: #96ff96; /* Green */
        }
        .status-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 8px 20px;
            background-color: #f8f9fa;
            border-top: 1px solid #dee2e6;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1 class="mb-4">Production Scheduler</h1>
        
        <!-- Tabs -->
        <ul class="nav nav-tabs" id="schedulerTabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="setup-tab" data-bs-toggle="tab" data-bs-target="#setup" type="button" role="tab">Setup & Run</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link disabled" id="results-tab" data-bs-toggle="tab" data-bs-target="#results" type="button" role="tab">Schedule Results</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link disabled" id="analysis-tab" data-bs-toggle="tab" data-bs-target="#analysis" type="button" role="tab">Analysis</button>
            </li>
        </ul>
        
        <!-- Tab Content -->
        <div class="tab-content" id="schedulerTabContent">
            <!-- Setup Tab -->
            <div class="tab-pane fade show active" id="setup" role="tabpanel" aria-labelledby="setup-tab">
                <!-- File Selection -->
                <div class="card">
                    <div class="card-header">
                        <h5>Data Source</h5>
                    </div>
                    <div class="card-body">
                        <div class="row align-items-center">
                            <div class="col-md-3">
                                <label for="fileInput" class="form-label">Production Data File:</label>
                            </div>
                            <div class="col-md-7">
                                <div id="filePathLabel" class="form-control-plaintext">No file selected</div>
                            </div>
                            <div class="col-md-2">
                                <input type="file" id="fileInput" class="form-control d-none" accept=".xlsx,.xls,.csv">
                                <button class="btn btn-outline-primary" id="browseButton">Browse...</button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Scheduling Options -->
                <div class="card">
                    <div class="card-header">
                        <h5>Scheduling Options</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="form-check mb-3">
                                    <input class="form-check-input" type="checkbox" id="enforceSequence" checked>
                                    <label class="form-check-label" for="enforceSequence" data-bs-toggle="tooltip" title="Ensure job processes are scheduled in the correct sequence">
                                        Enforce Process Sequence
                                    </label>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="form-check mb-3">
                                    <input class="form-check-input" type="checkbox" id="optimizeOperators">
                                    <label class="form-check-label" for="optimizeOperators" data-bs-toggle="tooltip" title="Consider operator requirements when scheduling jobs">
                                        Optimize Operator Assignments
                                    </label>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="considerSetup" checked>
                                    <label class="form-check-label" for="considerSetup" data-bs-toggle="tooltip" title="Include machine setup times between different job types">
                                        Consider Setup Times
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Planning Horizon -->
                <div class="card">
                    <div class="card-header">
                        <h5>Planning Horizon</h5>
                    </div>
                    <div class="card-body">
                        <div class="row mb-3">
                            <div class="col-md-3">
                                <label for="startDate" class="form-label">Start Date:</label>
                            </div>
                            <div class="col-md-9">
                                <input type="datetime-local" class="form-control" id="startDate">
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-3">
                                <label for="endDate" class="form-label">End Date:</label>
                            </div>
                            <div class="col-md-9">
                                <input type="datetime-local" class="form-control" id="endDate">
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Run Button and Progress Bar -->
                <div class="row mt-4">
                    <div class="col-md-3">
                        <button class="btn btn-primary btn-lg" id="runButton" disabled>Run Scheduler</button>
                    </div>
                    <div class="col-md-9">
                        <div class="progress">
                            <div class="progress-bar" id="progressBar" role="progressbar" style="width: 0%;" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">0%</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Results Tab -->
            <div class="tab-pane fade" id="results" role="tabpanel" aria-labelledby="results-tab">
                <!-- Summary Section -->
                <div class="card">
                    <div class="card-header">
                        <h5>Schedule Summary</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <p id="totalJobs"><strong>Total Jobs:</strong> 0</p>
                            </div>
                            <div class="col-md-3">
                                <p id="totalMachines"><strong>Total Machines:</strong> 0</p>
                            </div>
                            <div class="col-md-3">
                                <p id="makespan"><strong>Makespan:</strong> 0 hours</p>
                            </div>
                            <div class="col-md-3">
                                <p id="utilization"><strong>Average Machine Utilization:</strong> 0%</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Schedule Table -->
                <div class="card">
                    <div class="card-header">
                        <h5>Schedule Details</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-striped table-hover" id="scheduleTable">
                                <thead>
                                    <tr>
                                        <th>Process Code</th>
                                        <th>Machine</th>
                                        <th>Start</th>
                                        <th>End</th>
                                        <th>Duration (hrs)</th>
                                        <th>Priority</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <!-- Schedule data will be populated here -->
                                </tbody>
                            </table>
                        </div>
                        
                        <!-- Buttons -->
                        <div class="row mt-3">
                            <div class="col-md-6">
                                <button class="btn btn-primary" id="viewGanttButton">View Gantt Chart</button>
                            </div>
                            <div class="col-md-6">
                                <button class="btn btn-success" id="exportButton">Export Schedule</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Analysis Tab -->
            <div class="tab-pane fade" id="analysis" role="tabpanel" aria-labelledby="analysis-tab">
                <!-- Machine Utilization -->
                <div class="card">
                    <div class="card-header">
                        <h5>Machine Utilization</h5>
                    </div>
                    <div class="card-body">
                        <div class="row mb-3">
                            <div class="col-md-3">
                                <label for="machineFilter" class="form-label">Filter by Machine:</label>
                            </div>
                            <div class="col-md-9">
                                <select class="form-select" id="machineFilter">
                                    <option selected>All Machines</option>
                                </select>
                            </div>
                        </div>
                        <div class="table-responsive">
                            <table class="table table-striped table-hover" id="utilizationTable">
                                <thead>
                                    <tr>
                                        <th>Machine</th>
                                        <th>Utilization (%)</th>
                                        <th>Total Hours</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <!-- Utilization data will be populated here -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <!-- Due Date Performance -->
                <div class="card">
                    <div class="card-header">
                        <h5>Due Date Performance</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-striped table-hover" id="dueDateTable">
                                <thead>
                                    <tr>
                                        <th>Process Code</th>
                                        <th>Due Date</th>
                                        <th>Completion</th>
                                        <th>Buffer (hrs)</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <!-- Due date data will be populated here -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Status Bar -->
    <div class="status-bar" id="statusBar">Ready to schedule production</div>
    
    <!-- Bootstrap JS and Dependencies -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/js/bootstrap.bundle.min.js"></script>
    
    <script>
        // Set current date as default for date pickers
        document.addEventListener('DOMContentLoaded', function() {
            const now = new Date();
            const twoWeeksLater = new Date(now);
            twoWeeksLater.setDate(now.getDate() + 14);
            
            // Format for datetime-local input
            const formatDate = (date) => {
                return date.toISOString().slice(0, 16);
            };
            
            document.getElementById('startDate').value = formatDate(now);
            document.getElementById('endDate').value = formatDate(twoWeeksLater);
            
            // Initialize tooltips
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
            
            // File selection
            document.getElementById('browseButton').addEventListener('click', function() {
                document.getElementById('fileInput').click();
            });
            
            document.getElementById('fileInput').addEventListener('change', function(e) {
                if (e.target.files.length > 0) {
                    const fileName = e.target.files[0].name;
                    document.getElementById('filePathLabel').textContent = fileName;
                    document.getElementById('runButton').disabled = false;
                    document.getElementById('statusBar').textContent = `Selected file: ${fileName}`;
                }
            });
            
            // Run button click simulation
            document.getElementById('runButton').addEventListener('click', function() {
                const filePath = document.getElementById('filePathLabel').textContent;
                
                if (filePath === 'No file selected') {
                    alert('Please select a valid data file first.');
                    return;
                }
                
                // Disable button during "calculation"
                this.disabled = true;
                document.getElementById('statusBar').textContent = 'Running scheduler...';
                
                // Simulate progress
                simulateProgress();
            });
            
            // View Gantt button (just for demo)
            document.getElementById('viewGanttButton').addEventListener('click', function() {
                alert('This would open the Gantt chart in a new window.');
            });
            
            // Export button (just for demo)
            document.getElementById('exportButton').addEventListener('click', function() {
                alert('This would export the schedule to Excel.');
            });
        });
        
        // Function to simulate scheduler progress
        function simulateProgress() {
            const progressBar = document.getElementById('progressBar');
            let progress = 0;
            
            const interval = setInterval(() => {
                progress += 5;
                progressBar.style.width = `${progress}%`;
                progressBar.textContent = `${progress}%`;
                progressBar.setAttribute('aria-valuenow', progress);
                
                if (progress >= 100) {
                    clearInterval(interval);
                    simulateScheduleComplete();
                }
            }, 100);
        }
        
        // Function to simulate schedule completion
        function simulateScheduleComplete() {
            // Enable other tabs
            document.getElementById('results-tab').classList.remove('disabled');
            document.getElementById('analysis-tab').classList.remove('disabled');
            
            // Switch to results tab
            const resultsTab = new bootstrap.Tab(document.getElementById('results-tab'));
            resultsTab.show();
            
            // Update status
            document.getElementById('statusBar').textContent = 'Scheduling completed successfully';
            document.getElementById('runButton').disabled = false;
            
            // Populate with sample data
            populateSampleData();
        }
        
        // Function to populate sample data
        function populateSampleData() {
            // Summary stats
            document.getElementById('totalJobs').innerHTML = '<strong>Total Jobs:</strong> 35';
            document.getElementById('totalMachines').innerHTML = '<strong>Total Machines:</strong> 6';
            document.getElementById('makespan').innerHTML = '<strong>Makespan:</strong> 640.1 hours';
            document.getElementById('utilization').innerHTML = '<strong>Average Machine Utilization:</strong> 38.6%';
            
            // Sample schedule data
            const sampleSchedule = [
                ['CP08-145-P01-06', 'ST-3', '2025-03-15 08:00', '2025-03-16 00:40', 16.7, 2],
                ['CP08-145-P02-06', 'ST-3', '2025-03-16 00:40', '2025-03-16 17:20', 16.7, 2],
                ['CA25-001-P01-04', 'AL', '2025-03-15 08:00', '2025-03-17 19:01', 59.0, 2],
                ['CT10-026A-A-P01-07', 'TP', '2025-04-16 08:00', '2025-04-16 12:30', 4.5, 2],
                ['CT10-026A-A-P06-07', 'TP', '2025-04-17 01:06', '2025-04-17 03:30', 2.4, 2]
            ];
            
            const scheduleTable = document.getElementById('scheduleTable').getElementsByTagName('tbody')[0];
            scheduleTable.innerHTML = '';
            
            sampleSchedule.forEach(row => {
                const tr = document.createElement('tr');
                
                for (let i = 0; i < row.length; i++) {
                    const td = document.createElement('td');
                    td.textContent = row[i];
                    
                    // Apply styling for priority column
                    if (i === 5) {
                        td.classList.add(`priority-${row[i]}`);
                    }
                    
                    tr.appendChild(td);
                }
                
                scheduleTable.appendChild(tr);
            });
            
            // Sample machine utilization data
            const sampleUtilization = [
                ['AL', '25.1', '59.0'],
                ['SM', '9.5', '0.4'],
                ['ST', '4.2', '1.3'],
                ['ST-3', '92.5', '126.2'],
                ['ST-4', '20.3', '14.8'],
                ['TP', '40.5', '19.5']
            ];
            
            const utilizationTable = document.getElementById('utilizationTable').getElementsByTagName('tbody')[0];
            utilizationTable.innerHTML = '';
            
            sampleUtilization.forEach(row => {
                const tr = document.createElement('tr');
                
                row.forEach(cell => {
                    const td = document.createElement('td');
                    td.textContent = cell;
                    tr.appendChild(td);
                });
                
                utilizationTable.appendChild(tr);
            });
            
            // Sample due date data
            const sampleDueDate = [
                ['CT10-026A-A-P06-07', '2025-04-17 08:00', '2025-04-17 03:30', 4.5, 'critical'],
                ['CT10-026A-A-P05-07', '2025-04-17 08:00', '2025-04-17 01:06', 6.9, 'critical'],
                ['CP08-154-P03-04', '2025-03-24 12:00', '2025-03-22 02:27', 57.5, 'ok'],
                ['CA25-001-P03-04', '2025-03-25 08:00', '2025-03-18 05:39', 170.4, 'ok'],
                ['CP08-145-P05-06', '2025-04-01 12:00', '2025-03-18 19:20', 328.7, 'ok']
            ];
            
            const dueDateTable = document.getElementById('dueDateTable').getElementsByTagName('tbody')[0];
            dueDateTable.innerHTML = '';
            
            sampleDueDate.forEach(row => {
                const tr = document.createElement('tr');
                
                for (let i = 0; i < 4; i++) {
                    const td = document.createElement('td');
                    td.textContent = row[i];
                    
                    // Apply buffer status class
                    if (i === 3) {
                        td.classList.add(`buffer-${row[4]}`);
                    }
                    
                    tr.appendChild(td);
                }
                
                dueDateTable.appendChild(tr);
            });
            
            // Populate machine filter dropdown
            const machineFilter = document.getElementById('machineFilter');
            machineFilter.innerHTML = '<option selected>All Machines</option>';
            
            sampleUtilization.forEach(row => {
                const option = document.createElement('option');
                option.textContent = row[0];
                machineFilter.appendChild(option);
            });
        }
    </script>
</body>
</html>>
    """)

@app.route('/')
def index():
    return render_template('index.html')

# Add API endpoints for real scheduler functionality
@app.route('/api/run-scheduler', methods=['POST'])
def run_scheduler():
    # Here you can call your real Python scheduling code
    # and return results as JSON
    return jsonify({"status": "success", "message": "Schedule completed"})

if __name__ == '__main__':
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True)