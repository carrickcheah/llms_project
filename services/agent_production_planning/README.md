# Production Planning Scheduler

A tool for scheduling production jobs across multiple machines with priority-based planning and interactive visualization.

## Features

- Load job data from Excel files
- Optimize scheduling using constraint programming (CP-SAT solver)
- Fallback to an enhanced greedy algorithm when necessary
- Account for machine capacities and setup times
- Prioritize jobs based on due dates and priority levels
- Generate interactive Gantt chart visualizations
- Command-line interface with multiple options

## Requirements

- Python 3.13+
- Required Python packages:
  - matplotlib
  - openpyxl
  - ortools
  - pandas
  - plotly

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
# or using uv
uv pip install -e .
```

## Usage

### Command Line Options

```bash
python main.py [options]
```

Available options:

- `--file`, `-f`: Path to Excel file with job data (default: local mydata.xlsx)
- `--jobs`, `-j`: Maximum number of jobs to schedule (default: 100)
- `--force-greedy`: Force using greedy scheduler instead of CP-SAT solver
- `--output`, `-o`: Output file for interactive Gantt chart (default: interactive_schedule.html)

### Using Make

The project includes a Makefile with helpful shortcuts:

```bash
# Run with default settings
make

# Force using greedy scheduler
make greedy

# Run with fewer jobs (50)
make small

# Run with a custom file
make custom FILE=path/to/your/excel_file.xlsx

# Run a sample configuration
make sample

# Show help
make help

# Clean generated HTML files
make clean
```

## Input Data Format

The input Excel file should contain job information with the following columns:

- `LCD_DATE`: Latest completion date
- `JOBCODE`: Unique job identifier
- `PROCESS_CODE`: Process identifier
- `MACHINE_ID`: Machine to execute the job
- `NUMBER_OPERATOR`: Number of operators required
- `OUTPUT_PER_HOUR`: Production rate
- `HOURS NEEDED`: Processing time in hours
- `PLANNED_START`: Planned start date/time
- `PLANNED_END`: Planned end date/time
- `PRIORITY`: Job priority (1-5, where 1 is highest)

## System Architecture

The production planning system is composed of several key modules:

### 1. main.py

The main entry point for the production planning system:

- Command Line Interface:
  - Accepts arguments for file path, max jobs, and scheduling options
  - Provides verbose logging and force-greedy options

- Data Processing:
  - Loads job data and machine information
  - Handles START_DATE constraints for exact start times
  - Enforces process sequence dependencies

- Scheduling Workflow:
  - Tries CP-SAT solver first (with time limits based on job count)
  - Falls back to greedy scheduler if CP-SAT fails
  - Handles setup times and machine constraints

- Output Generation:
  - Creates interactive Gantt charts
  - Generates HTML schedule views
  - Provides detailed logging of the scheduling process

The main orchestrates the entire scheduling process, from data loading to visualization,
with a focus on handling constraints and fallback scenarios.

### 2. ingest_data.py

Handles data loading and preprocessing for the production planning system:

- Core Functions:
  - `get_column_time()`: Manages time configurations
  - `detect_date_format()`: Identifies date formats
  - `convert_column_to_dates()`: Converts dates to timestamps
  - `load_jobs_planning_data()`: Main data loading function

- Key Features:
  - Processes Excel files with flexible header locations
  - Handles multiple date formats (European, US, ISO)
  - Converts all dates to Singapore timezone (UTC+8)
  - Creates unique job IDs from JOB + PROCESS_CODE
  - Manages setup times, breaks, and non-production hours
  - Handles operator efficiency adjustments

- Data Validation:
  - Checks for required columns (JOB, PROCESS_CODE, etc.)
  - Validates date columns and timestamps
  - Manages duplicates and missing values
  - Ensures proper time zone handling

- Output:
  - Returns job data, machine list, and setup times
  - Creates standardized epoch timestamps
  - Provides detailed logging of data processing

The module ensures data is properly formatted and validated before being used by the
scheduling algorithms.

### 3. greedy.py

A production scheduling algorithm that:

1. Takes job data (processing times, machines, priorities) and creates an optimized manufacturing schedule
2. Handles special constraints like START_DATE (exact start times) for specific jobs
3. Enforces correct process sequencing (P01 → P02 → P03)
4. Functions as a fallback scheduler when the more complex CP-SAT solver can't find solutions
5. Provides detailed logging of scheduling decisions and constraint handling

Key functions:

- `extract_process_number()`: Gets sequence numbers from job IDs
- `extract_job_family()`: Groups related jobs by family
- `greedy_schedule()`: The main scheduling algorithm

The scheduler outputs a complete production plan where jobs are assigned to machines with
specific start/end times while minimizing late jobs as much as possible.

### 4. setup_times.py

Manages timing aspects of the production planning system:

1. Standardizes access to START_DATE fields with consistent helper functions
2. Handles setup times between consecutive manufacturing processes
3. Calculates buffer times between job completion and deadlines
4. Categorizes jobs by buffer status (Late, Critical, Warning, Caution, OK)
5. Adds schedule times to jobs based on the optimization solution

Key functions:

- `get_start_date_epoch()`: Consistent access to START_DATE constraints
- `extract_job_family()`: Parse job details
- `extract_process_number()`: Parse job details
- `add_schedule_times_and_buffer()`: Main function that applies timing info
- `get_buffer_status()`: Categorizes jobs based on time cushion

It ensures that jobs have proper timing data for visualization and handling setup times between processes.

### 5. chart.py

Creates interactive visualizations of the production schedule:

- Core Functions:
  - `format_date_correctly()`: Formats timestamps with Singapore timezone
  - `get_buffer_status_color()`: Colors for buffer status visualization
  - `create_interactive_gantt()`: Main Gantt chart creation
  - `flatten_schedule_to_list()`: Converts schedule to list format

- Key Features:
  - Interactive Gantt charts with tooltips
  - Priority-based color coding (5 levels)
  - Time zone handling (Singapore)
  - Buffer status visualization (Late, Critical, Warning, Caution, OK)
  - Job family and process sequence tracking

- Visualization Components:
  - Modern color scheme for priorities
  - Range selector buttons
  - Consistent date formatting
  - Tooltips for detailed job information

The module provides interactive visualizations of the production schedule, helping users understand timing, priorities, and buffer status.

### 6. chart_two.py

Creates HTML-based visualizations of the production schedule:

- Core Functions:
  - `format_date_correctly()`: Formats timestamps with Singapore timezone
  - `get_buffer_status_color()`: Colors for buffer status visualization
  - `export_schedule_html()`: Main HTML visualization function

- Key Features:
  - HTML-based schedule visualization
  - START_DATE constraint visualization
  - Family-wide time shifts
  - Process sequence visualization
  - Buffer status indicators

- Visualization Components:
  - Time zone handling (Singapore)
  - Family process tracking
  - START_DATE exact time visualization
  - Time shift adjustments
  - Buffer status color coding

The module provides HTML-based visualizations of the production schedule, with a focus on START_DATE constraints and family-wide time shifts.

### 7. server.py

A simple HTTP server that provides a web interface for the production planning system:

- Server Features:
  - Serves static files (HTML, CSS, JS)
  - Handles file uploads and processing
  - Provides web-based interface for scheduling

- Key Components:
  - `ProductionPlanningHandler`: Handles HTTP requests
  - `run_server()`: Starts the HTTP server

- Main Functionality:
  - GET requests: Serves static files (upload page, HTML, CSS, JS)
  - POST requests: Processes uploaded Excel files
  - Runs main.py with scheduling parameters

- File Processing:
  - Accepts Excel files (.xlsx, .xls)
  - Uses temporary files for processing
  - Generates Gantt charts and schedule views

- Scheduling Parameters:
  - Max jobs (default: 500)
  - Force greedy scheduler
  - Enforce sequence dependencies

The server provides a web interface for users to upload Excel files and
generate production schedules, with visualizations available through the browser.

### 8. sch_jobs.py

Contains job scheduling logic and optimization algorithms.

## Scheduling Algorithms

### CP-SAT Solver

The primary scheduler uses Google OR-Tools' CP-SAT solver to find an optimal solution considering:

- Machine capacity constraints
- Setup times between processes
- Due dates with priority-based penalties
- Makespan minimization

### Enhanced Greedy Scheduler

The fallback scheduler prioritizes jobs based on:

- Job priority (highest first)
- Due dates (urgent jobs first)
- Setup time considerations
- Machine availability

## Visualization

The scheduler generates an interactive HTML Gantt chart with:

- Color-coded jobs by priority
- Machine timeline visualization
- Detailed tooltips with job information
- Interactive zooming and panning
