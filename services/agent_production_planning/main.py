# main.py
from ortools.sat.python import cp_model
from datetime import datetime
import pandas as pd
import os
from ingest_data import (
    clean_excel_data, convert_to_epoch, load_jobs_planning_data
)
from chart import create_interactive_gantt

def schedule_jobs(jobs, machines, setup_times):
    if not jobs or not machines:
        print("No jobs or machines available for scheduling!")
        return None
    
    model = cp_model.CpModel()
    current_time = int(datetime.now().timestamp())
    
    # Deduplicate jobs by combining identical jobs with same machine and process code
    job_dict = {}
    for job in jobs:
        job_name, process_code, machine, duration, due_time, priority, start_time = job
        key = (job_name, process_code, machine)
        
        if key in job_dict:
            # Keep job with highest priority (lowest number)
            if priority < job_dict[key][5]:
                job_dict[key] = job
        else:
            job_dict[key] = job
    
    processed_jobs = []
    for job in job_dict.values():
        job_name, process_code, machine, duration, due_time, priority, start_time = job
        duration_seconds = int(duration)  # Already in seconds from processing_time
        due_seconds = max(current_time, int(due_time))
        start_seconds = max(current_time, int(start_time)) if pd.notna(start_time) else current_time
        priority = max(1, min(5, int(priority))) if pd.notna(priority) else 3
        
        if duration_seconds > 0:
            processed_jobs.append((job_name, process_code, machine, duration_seconds, due_seconds, priority, start_seconds))
            # Print duration in hours for clarity alongside epoch timestamps
            duration_hours = duration_seconds / 3600
            print(f"Job {job_name}: start={start_seconds} ({datetime.fromtimestamp(start_seconds)}), "
                  f"duration={duration_seconds} sec ({duration_hours:.2f} hours), "
                  f"due={due_seconds} ({datetime.fromtimestamp(due_seconds)}), priority={priority}")
        else:
            print(f"Warning: Job {job_name} has zero or negative duration. Skipping.")
    
    if not processed_jobs:
        print("No valid jobs to schedule!")
        return None
    
    jobs_by_machine = {machine[1]: [job for job in processed_jobs if job[2] == machine[1]] for machine in machines}
    
    total_duration = sum(job[3] for job in processed_jobs)
    max_duration = max(job[3] for job in processed_jobs)
    
    # Calculate a more appropriate horizon based on job durations and due dates
    max_due_time = max(job[4] for job in processed_jobs)
    time_span = max_due_time - current_time
    
    horizon = min(
        max(
            total_duration // max(1, len(machines)),
            max_duration * 2,
            time_span
        ),
        30 * 24 * 3600  # Cap at 30 days
    )
    print(f"Planning horizon: {horizon / (24 * 3600):.1f} days ({horizon} seconds)")
    
    starts = {}
    ends = {}
    intervals = {}
    for job_id, job in enumerate(processed_jobs):
        job_name, process_code, machine, duration_seconds, due_time, priority, start_time = job
        
        # Ensure start time window is valid
        start_min = max(current_time, start_time)
        start_max = min(due_time - duration_seconds, horizon + current_time)
        
        # If the window is impossible, adjust it to allow at least minimal scheduling
        if start_max < start_min:
            start_max = horizon + current_time - duration_seconds
        
        start = model.NewIntVar(start_min, start_max, f'start_{job_id}')
        end = model.NewIntVar(start_min + duration_seconds, horizon + current_time, f'end_{job_id}')
        interval = model.NewIntervalVar(start, duration_seconds, end, f'interval_{job_id}')
        
        starts[job_id] = start
        ends[job_id] = end
        intervals[job_id] = interval
        
        # Only enforce strict deadlines for highest priority jobs
        if priority == 1:
            # Use a soft constraint instead of a hard one
            model.Add(end <= due_time).OnlyEnforceIf(model.NewBoolVar(f'meet_deadline_{job_id}'))
    
    machine_intervals = {m[1]: [intervals[job_id] for job_id, job in enumerate(processed_jobs) if job[2] == m[1]] 
                        for m in machines}
    for intervals_list in machine_intervals.values():
        if len(intervals_list) > 1:
            model.AddNoOverlap(intervals_list)
    
    makespan = model.NewIntVar(0, horizon + current_time, 'makespan')
    total_tardiness = model.NewIntVar(0, horizon * len(processed_jobs), 'total_tardiness')
    
    for job_id in ends:
        model.Add(makespan >= ends[job_id])
    
    tardiness_vars = []
    for job_id, job in enumerate(processed_jobs):
        if job_id in ends:
            due_time = job[4]
            priority = job[5]
            tardiness = model.NewIntVar(0, horizon, f'tardiness_{job_id}')
            time_diff = model.NewIntVar(-horizon, horizon, f'time_diff_{job_id}')
            model.Add(time_diff == ends[job_id] - due_time)
            model.Add(tardiness >= 0)
            model.Add(tardiness >= time_diff)
            weighted_tardiness = model.NewIntVar(0, horizon * 5, f'weighted_tardiness_{job_id}')
            model.Add(weighted_tardiness == tardiness * (6 - priority))
            tardiness_vars.append(weighted_tardiness)
    
    if tardiness_vars:
        model.Add(total_tardiness == sum(tardiness_vars))
    else:
        model.Add(total_tardiness == 0)
    
    objective = model.NewIntVar(0, horizon * len(processed_jobs) * 10, 'objective')
    model.Add(objective == makespan + total_tardiness)
    model.Minimize(objective)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = True
    
    print("Solving scheduling problem...")
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        schedule = {}
        for job_id in starts:
            job = processed_jobs[job_id]
            job_name = job[0]
            machine = job[2]
            start = solver.Value(starts[job_id])
            end = solver.Value(ends[job_id])
            if machine not in schedule:
                schedule[machine] = []
            schedule[machine].append((job_name, start, end))
        print(f"Schedule created with {sum(len(jobs) for jobs in schedule.values())} jobs")
        return schedule
    else:
        print("No solution found! Trying simplified approach...")
        return greedy_schedule(processed_jobs, machines)

def greedy_schedule(jobs, machines):
    print("Using greedy scheduling algorithm as fallback")
    sorted_jobs = sorted(jobs, key=lambda job: (-job[5], job[4]))
    machine_end_times = {m[1]: int(datetime.now().timestamp()) for m in machines}
    schedule = {}
    
    for job in sorted_jobs[:1000]:
        job_name, _, machine, duration, _, _, _ = job
        if duration <= 0 or machine not in machine_end_times:
            continue
        current_end_time = machine_end_times[machine]
        new_end_time = current_end_time + duration
        if machine not in schedule:
            schedule[machine] = []
        schedule[machine].append((job_name, current_end_time, new_end_time))
        machine_end_times[machine] = new_end_time
    
    print(f"Greedy schedule created with {sum(len(jobs) for jobs in schedule.values())} jobs")
    return schedule

if __name__ == "__main__":
    # Allow for command line specified file or use default
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "/Users/carrickcheah/llms_project/services/agent_production_planning/mydata.xlsx"
    
    try:
        print(f"Loading job data from {file_path}...")
        all_jobs, all_machines, setup_times = load_jobs_planning_data(file_path)
        
        # Limit to a reasonable number of jobs for demonstration, but allow customization
        max_jobs = 100  # Increased from 30 to get a more representative schedule
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
            # First try with CP-SAT solver
            try:
                schedule = schedule_jobs(jobs, machines, setup_times)
            except Exception as e:
                print(f"Error in CP-SAT solver: {e}")
                print("Falling back to greedy scheduler...")
                schedule = greedy_schedule(jobs, machines)
                
            if schedule:
                # Create an interactive visualization
                success = create_interactive_gantt(schedule)
                if success:
                    print(f"Interactive chart saved to: {os.path.abspath('interactive_schedule.html')}")
                else:
                    print("Failed to create Gantt chart, but schedule was generated.")
            else:
                print("Failed to generate schedule.")
        else:
            print("No jobs or machines found in the input data.")
    except Exception as e:
        print(f"Error in production planning: {e}")
        import traceback
        traceback.print_exc()