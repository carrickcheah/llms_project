# main.py (modified to prioritize sequence dependencies)
import os
import argparse
from ingest_data import load_jobs_planning_data
from chart import create_interactive_gantt
from sch_jobs import schedule_jobs
from greedy import greedy_schedule
from datetime import datetime
import time

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
    parser.add_argument("--enforce-sequence", action="store_true", default=True,
                        help="Enforce process code sequence dependencies (P01->P02->P03)")
    return parser.parse_args()

def enforce_process_sequence(jobs):
    """
    Sort jobs so that lower process numbers come first within each job family.
    Example: P01 before P02 before P03
    """
    import re
    
    def extract_job_family(process_code):
        """Extract job family code (e.g., CP08-231B from CP08-231B-P01-06 or cp08-231 from cp08-231-P01)"""
        # Convert to uppercase for consistency
        process_code = process_code.upper()
        
        match = re.search(r'(.*?)-P\d+', process_code)
        if match:
            return match.group(1)
        parts = process_code.split("-P")
        if len(parts) >= 2:
            return parts[0]
        return process_code
        
    def extract_process_number(process_code):
        """Extract process number (e.g., 1 from CP08-231B-P01-06 or cp08-231-P01)"""
        # Convert to uppercase for consistency
        process_code = process_code.upper()
        
        # Try to find P followed by 2 digits (like "P01", "P02")
        match = re.search(r'P(\d{2})', process_code)
        if match:
            try:
                # Extract and convert to int - "01" becomes 1, "02" becomes 2, etc.
                sequence_str = match.group(1)
                return int(sequence_str)
            except ValueError:
                return 999
        return 999
    
    # Group jobs by family
    job_families = {}
    for job in jobs:
        if len(job) < 3:  # Need at least process_code and machine
            continue
        process_code = job[1]
        family = extract_job_family(process_code)
        
        if family not in job_families:
            job_families[family] = []
        job_families[family].append(job)
    
    # Sort each family by process number
    sorted_jobs = []
    for family, family_jobs in job_families.items():
        family_jobs.sort(key=lambda job: extract_process_number(job[1]))
        sorted_jobs.extend(family_jobs)
    
    print(f"Sorted {len(sorted_jobs)} jobs by process code sequence")
    return sorted_jobs

if __name__ == "__main__":
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    file_path = args.file
    max_jobs = args.jobs
    force_greedy = args.force_greedy
    output_file = args.output
    enforce_sequence = args.enforce_sequence
    
    print(f"Production Planning Scheduler")
    print(f"Configuration: file={file_path}, max_jobs={max_jobs}, force_greedy={force_greedy}, " +
          f"output={output_file}, enforce_sequence={enforce_sequence}")
    
    try:
        print(f"Loading job data from {file_path}...")
        all_jobs, all_machines, setup_times = load_jobs_planning_data(file_path)
        
        # Apply sequence enforcement if requested
        if enforce_sequence:
            print("Enforcing process sequence dependencies (P01->P02->P03)...")
            all_jobs = enforce_process_sequence(all_jobs)
        
        # Limit to a reasonable number of jobs for demonstration, but allow customization
        if len(all_jobs) > max_jobs:
            print(f"Limited to {max_jobs} jobs out of {len(all_jobs)} for scheduling efficiency")
            jobs = all_jobs[:max_jobs]
        else:
            jobs = all_jobs
            
        # Only use machines that have jobs assigned to them
        used_machines = set(job[2] for job in jobs)
        machines = [m for m in all_machines if m[1] in used_machines]
        
        print(f"Loaded {len(jobs)} jobs and {len(machines)} machines")
        if jobs and machines:
            # Determine which scheduling algorithm to use
            if force_greedy:
                print("Using greedy scheduler as requested...")
                schedule = greedy_schedule(jobs, machines, setup_times)
            else:
                # First try with CP-SAT solver
                try:
                    print("Attempting to create schedule with CP-SAT solver...")
                    schedule = schedule_jobs(jobs, machines, setup_times)
                    if not schedule:  # If schedule is empty, fall back to greedy
                        print("CP-SAT solver failed to produce a valid schedule")
                        print("Falling back to greedy scheduler...")
                        schedule = greedy_schedule(jobs, machines, setup_times)
                except Exception as e:
                    print(f"Error in CP-SAT solver: {e}")
                    print("Falling back to greedy scheduler...")
                    schedule = greedy_schedule(jobs, machines, setup_times)
                
            if schedule:
                # Create an interactive visualization (only)
                success = create_interactive_gantt(schedule, output_file)
                if success:
                    # Print some statistics about the schedule
                    total_jobs = sum(len(jobs) for jobs in schedule.values())
                    machines_used = len(schedule)
                    print(f"Schedule statistics:")
                    print(f"- Total jobs scheduled: {total_jobs}")
                    print(f"- Machines utilized: {machines_used}/{len(machines)} ({machines_used/len(machines)*100:.1f}%)")
                    
                    # Analyze machine utilization
                    machine_loads = {}
                    for machine, jobs_list in schedule.items():
                        total_duration = sum(job[2] - job[1] for job in jobs_list)
                        machine_loads[machine] = total_duration
                    
                    if machine_loads:
                        avg_load = sum(machine_loads.values()) / len(machine_loads)
                        print(f"- Average machine load: {avg_load/3600:.1f} hours")
                        most_loaded = max(machine_loads.items(), key=lambda x: x[1])
                        print(f"- Most loaded machine: {most_loaded[0]} ({most_loaded[1]/3600:.1f} hours)")
                        
                        # Find completion date
                        latest_end = max(job[2] for machine_jobs in schedule.values() for job in machine_jobs)
                        print(f"- All jobs will complete by: {datetime.fromtimestamp(latest_end)}")
                        
                        # Total production time
                        total_hours = sum(machine_loads.values()) / 3600
                        print(f"- Total production time: {total_hours:.1f} hours")
                else:
                    print("Failed to create Gantt chart.")
            else:
                print("Failed to generate schedule.")
        else:
            print("No jobs or machines found in the input data.")
            
        # Print execution time
        elapsed_time = time.time() - start_time
        print(f"Scheduling completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        print(f"Error in production planning: {e}")
        import traceback
        traceback.print_exc()