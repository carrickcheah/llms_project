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

- Python 3.11+
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
