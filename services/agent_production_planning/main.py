# main.py
import os
import argparse
import logging
from datetime import datetime
from ingest_data import load_jobs_planning_data
from chart import create_interactive_gantt
from sch_jobs import schedule_jobs
from greedy import greedy_schedule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('production_scheduler.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Production Planning Scheduler")
    parser.add_argument("--file", "-f", 
                        default="mydata.xlsx",
                        help="Path to Excel file with job data")
    parser.add_argument("--jobs", "-j", type=int, default=100,
                        help="Maximum number of jobs to schedule")
    parser.add_argument("--force-greedy", action="store_true",
                        help="Force using greedy scheduler instead of CP-SAT solver")
    parser.add_argument("--output", "-o", default="interactive_schedule.html",
                        help="Output file for interactive Gantt chart")
    parser.add_argument("--log-level", "-l", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    return parser.parse_args()

def export_schedule_to_excel(schedule, output_file="schedule_export.xlsx"):
    """Export the schedule to Excel for further analysis or sharing."""
    try:
        import pandas as pd
        
        # Create a flattened structure for the schedule
        rows = []
        for machine, jobs in schedule.items():
            for job in jobs:
                # Depending on the job tuple structure - original schedule or enhanced with planned times
                if len(job) >= 6:  # Enhanced format with planned times
                    job_name, start_time, end_time, priority, planned_start, planned_end = job
                else:  # Original format
                    job_name, start_time, end_time, priority = job
                    planned_start = None
                    planned_end = None
                
                # Convert timestamps to datetime
                start_datetime = datetime.fromtimestamp(start_time)
                end_datetime = datetime.fromtimestamp(end_time)
                planned_start_datetime = datetime.fromtimestamp(planned_start) if planned_start else None
                planned_end_datetime = datetime.fromtimestamp(planned_end) if planned_end else None
                
                duration_hours = (end_time - start_time) / 3600
                
                # Determine status based on current time
                current_time = datetime.now().timestamp()
                if end_time < current_time:
                    status = "Completed"
                elif start_time <= current_time < end_time:
                    status = "In Progress"
                else:
                    status = "Scheduled"
                
                rows.append({
                    "Machine": machine,
                    "Job": job_name,
                    "Priority": priority,
                    "Start Time": start_datetime,
                    "End Time": end_datetime,
                    "Duration (hours)": round(duration_hours, 2),
                    "Planned Start": planned_start_datetime,
                    "Planned End": planned_end_datetime,
                    "Status": status
                })
        
        # Create DataFrame and export to Excel
        df = pd.DataFrame(rows)
        
        # Sort by machine and start time
        df = df.sort_values(by=["Machine", "Start Time"])
        
        # Export to Excel with formatting
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Production Schedule")
            
            # Auto-adjust column widths
            worksheet = writer.sheets["Production Schedule"]
            for i, col in enumerate(df.columns):
                max_length = df[col].astype(str).map(len).max()
                max_length = max(max_length, len(col) + 2)
                worksheet.column_dimensions[chr(65 + i)].width = max_length
        
        logger.info(f"Schedule exported to Excel: {os.path.abspath(output_file)}")
        return True
    except Exception as e:
        logger.error(f"Error exporting schedule to Excel: {e}")
        return False

def main():
    # Parse command line arguments
    args = parse_arguments()
    file_path = args.file
    max_jobs = args.jobs
    force_greedy = args.force_greedy
    output_file = args.output
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("Production Planning Scheduler")
    logger.info(f"Configuration: file={file_path}, max_jobs={max_jobs}, force_greedy={force_greedy}, output={output_file}")
    
    try:
        # Record start time for performance tracking
        start_time = datetime.now()
        logger.info(f"Loading job data from {file_path}...")
        
        all_jobs, all_machines, setup_times = load_jobs_planning_data(file_path)
        
        # Limit to a reasonable number of jobs for demonstration, but allow customization
        if len(all_jobs) > max_jobs:
            logger.info(f"Limited to {max_jobs} jobs out of {len(all_jobs)} for scheduling efficiency")
            jobs = all_jobs[:max_jobs]
        else:
            jobs = all_jobs
            
        # Only use machines that have jobs assigned to them
        used_machines = set(job[2] for job in jobs)
        machines = [m for m in all_machines if m[1] in used_machines]
        
        logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines")
        
        if jobs and machines:
            # Determine which scheduling algorithm to use
            if force_greedy:
                logger.info("Using greedy scheduler as requested...")
                schedule = greedy_schedule(jobs, machines, setup_times)
            else:
                # First try with CP-SAT solver
                try:
                    logger.info("Attempting to create schedule with CP-SAT solver...")
                    schedule = schedule_jobs(jobs, machines, setup_times)
                except Exception as e:
                    logger.error(f"Error in CP-SAT solver: {e}")
                    logger.info("Falling back to greedy scheduler...")
                    schedule = greedy_schedule(jobs, machines, setup_times)
                
            if schedule:
                # Export the schedule to Excel for further analysis
                excel_output = os.path.splitext(output_file)[0] + "_data.xlsx"
                export_schedule_to_excel(schedule, excel_output)
                
                # Create an interactive visualization
                success = create_interactive_gantt(schedule, output_file)
                if success:
                    logger.info(f"Interactive chart saved to: {os.path.abspath(output_file)}")
                    
                    # Print some statistics about the schedule
                    total_jobs = sum(len(jobs) for jobs in schedule.values())
                    machines_used = len(schedule)
                    logger.info(f"Schedule statistics:")
                    logger.info(f"- Total jobs scheduled: {total_jobs}")
                    logger.info(f"- Machines utilized: {machines_used}/{len(machines)} ({machines_used/len(machines)*100:.1f}%)")
                    
                    # Analyze machine utilization
                    machine_loads = {}
                    for machine, jobs_list in schedule.items():
                        total_duration = sum(job[2] - job[1] for job in jobs_list)
                        machine_loads[machine] = total_duration
                    
                    if machine_loads:
                        avg_load = sum(machine_loads.values()) / len(machine_loads)
                        logger.info(f"- Average machine load: {avg_load/3600:.1f} hours")
                        most_loaded = max(machine_loads.items(), key=lambda x: x[1])
                        logger.info(f"- Most loaded machine: {most_loaded[0]} ({most_loaded[1]/3600:.1f} hours)")
                        
                    # Calculate completion statistics
                    completion_time = max(job[2] for machine, jobs_list in schedule.items() for job in jobs_list)
                    completion_date = datetime.fromtimestamp(completion_time)
                    logger.info(f"- All jobs will complete by: {completion_date}")
                    
                    # Calculate total production time
                    total_production_time = sum(sum(job[2] - job[1] for job in jobs_list) for machine, jobs_list in schedule.items())
                    logger.info(f"- Total production time: {total_production_time/3600:.1f} hours")
                    
                    # Calculate total elapsed time
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    logger.info(f"Scheduling completed in {elapsed_time:.2f} seconds")
                else:
                    logger.error("Failed to create Gantt chart, but schedule was generated.")
            else:
                logger.error("Failed to generate schedule.")
        else:
            logger.error("No jobs or machines found in the input data.")
    except Exception as e:
        logger.error(f"Error in production planning: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)