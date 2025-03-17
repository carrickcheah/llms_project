# Import necessary modules for the scheduler
from ortools.sat.python import cp_model
from datetime import datetime
import pandas as pd
import logging
from greedy import greedy_schedule

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def schedule_jobs(jobs, machines, setup_times):
    """
    Create an optimized production schedule using CP-SAT solver.
    
    Args:
        jobs: List of tuples (job_name, process_code, machine, duration, due_time, priority, start_time, 
              [num_operators], [output_rate], [planned_end])
        machines: List of tuples (id, machine_name, capacity)
        setup_times: Dictionary mapping (process_code1, process_code2) to setup duration
    
    Returns:
        Dictionary mapping machine IDs to lists of scheduled jobs 
        (job_name, start_time, end_time, priority, planned_start, planned_end)
    """
    if not jobs or not machines:
        logger.error("No jobs or machines available for scheduling!")
        return None
    
    model = cp_model.CpModel()
    current_time = int(datetime.now().timestamp())
    
    # 1. JOB HANDLING WITH PLANNED_START/END SUPPORT
    # Deduplicate jobs and validate job data
    job_dict = {}
    has_extended_info = False  # Flag to check if jobs have num_operators and output_rate
    has_planned_end = False    # Flag to check if jobs have planned_end
    
    for job in jobs:
        # Check job format based on length
        if len(job) >= 10:  # With planned_end
            has_extended_info = True
            has_planned_end = True
            job_name, process_code, machine, duration, due_time, priority, start_time, num_operators, output_rate, planned_end = job
        elif len(job) >= 9:  # With extended info but no planned_end
            has_extended_info = True
            job_name, process_code, machine, duration, due_time, priority, start_time, num_operators, output_rate = job
            planned_end = None  # No planned end time provided
        elif len(job) == 7:  # Basic info
            job_name, process_code, machine, duration, due_time, priority, start_time = job
            # Default values for missing fields
            num_operators = 1
            output_rate = 100.0
            planned_end = None
        else:
            logger.warning(f"Warning: Skipping malformed job {job}")
            continue
            
        # Basic validation
        if not isinstance(duration, (int, float)) or duration <= 0:
            logger.warning(f"Warning: Job {job_name} has invalid duration ({duration}). Skipping.")
            continue
            
        if not isinstance(machine, str) or not machine:
            logger.warning(f"Warning: Job {job_name} has invalid machine ({machine}). Skipping.")
            continue
        
        # Create a unique key for deduplication
        key = (job_name, process_code, machine)
        
        if key in job_dict:
            # Keep job with highest priority (lowest number)
            if priority < job_dict[key][5]:
                logger.info(f"Duplicate job {job_name} found. Keeping higher priority version.")
                if has_planned_end:
                    job_dict[key] = (job_name, process_code, machine, duration, due_time, priority, 
                                     start_time, num_operators, output_rate, planned_end)
                elif has_extended_info:
                    job_dict[key] = (job_name, process_code, machine, duration, due_time, priority, 
                                     start_time, num_operators, output_rate)
                else:
                    job_dict[key] = job
        else:
            if has_planned_end:
                job_dict[key] = (job_name, process_code, machine, duration, due_time, priority, 
                                 start_time, num_operators, output_rate, planned_end)
            elif has_extended_info:
                job_dict[key] = (job_name, process_code, machine, duration, due_time, priority, 
                                 start_time, num_operators, output_rate)
            else:
                job_dict[key] = job
    
    # 2. MACHINE VALIDATION
    # Validate machines exist and build machine capacity map
    machine_ids = {m[1] for m in machines if isinstance(m[1], str) and m[1]}
    machine_capacity = {m[1]: m[2] if len(m) > 2 and isinstance(m[2], int) and m[2] > 0 else 1 
                      for m in machines if isinstance(m[1], str) and m[1]}
    
    # Process valid jobs
    processed_jobs = []
    
    # Count total number of operators needed
    total_operators_needed = 0
    operator_utilization = {}  # To track operator usage over time
    
    for job in job_dict.values():
        if has_planned_end and len(job) >= 10:
            job_name, process_code, machine, duration, due_time, priority, start_time, num_operators, output_rate, planned_end = job
        elif has_extended_info and len(job) >= 9:
            job_name, process_code, machine, duration, due_time, priority, start_time, num_operators, output_rate = job
            planned_end = None
        else:
            job_name, process_code, machine, duration, due_time, priority, start_time = job
            num_operators = 1
            output_rate = 100.0
            planned_end = None
        
        # Skip jobs for invalid machines
        if machine not in machine_ids:
            logger.warning(f"Warning: Job {job_name} assigned to non-existent machine {machine}. Skipping.")
            continue
            
        # Ensure numeric types
        duration_seconds = int(float(duration))
        due_seconds = max(current_time, int(float(due_time)))
        start_seconds = max(current_time, int(float(start_time)) if pd.notna(start_time) else current_time)
        planned_end_seconds = int(float(planned_end)) if pd.notna(planned_end) else start_seconds + duration_seconds
        priority = max(1, min(5, int(float(priority)))) if pd.notna(priority) else 3
        num_operators = max(1, min(10, int(float(num_operators)) if pd.notna(num_operators) else 1))
        output_rate = max(0.1, float(output_rate) if pd.notna(output_rate) else 100.0)
        
        # Track operator needs
        total_operators_needed = max(total_operators_needed, num_operators)
        
        if duration_seconds > 0:
            processed_jobs.append((job_name, process_code, machine, duration_seconds, 
                                 due_seconds, priority, start_seconds, num_operators, output_rate, planned_end_seconds))
            # Print duration in hours for clarity alongside epoch timestamps
            duration_hours = duration_seconds / 3600
            logger.info(f"Job {job_name}: start={start_seconds} ({datetime.fromtimestamp(start_seconds)}), "
                      f"duration={duration_seconds} sec ({duration_hours:.2f} hours), "
                      f"due={due_seconds} ({datetime.fromtimestamp(due_seconds)}), "
                      f"priority={priority}, operators={num_operators}, output_rate={output_rate}/hr, "
                      f"planned_end={planned_end_seconds} ({datetime.fromtimestamp(planned_end_seconds)})")
        else:
            logger.warning(f"Warning: Job {job_name} has zero or negative duration. Skipping.")
    
    if not processed_jobs:
        logger.error("No valid jobs to schedule!")
        return None
    
    # Display job counts by priority
    priority_counts = {}
    for job in processed_jobs:
        priority = job[5]
        if priority not in priority_counts:
            priority_counts[priority] = 0
        priority_counts[priority] += 1
    
    logger.info("Jobs by priority:")
    for priority in sorted(priority_counts.keys()):
        logger.info(f"- Priority {priority}: {priority_counts[priority]} jobs")
    
    # Group jobs by machine for analysis
    jobs_by_machine = {}
    for job in processed_jobs:
        machine = job[2]
        if machine not in jobs_by_machine:
            jobs_by_machine[machine] = []
        jobs_by_machine[machine].append(job)
    
    # Calculate scheduling statistics
    total_duration = sum(job[3] for job in processed_jobs)
    max_duration = max(job[3] for job in processed_jobs)
    avg_duration = total_duration / len(processed_jobs)
    
    # 3. SETUP TIME HANDLING
    # Check if setup times are properly defined
    valid_setup_times = True
    for job in processed_jobs:
        process_code = job[1]
        if process_code not in setup_times:
            logger.warning(f"Warning: Process code {process_code} not found in setup_times.")
            valid_setup_times = False
            break
        for other_job in processed_jobs:
            other_process = other_job[1]
            if other_process not in setup_times.get(process_code, {}):
                logger.warning(f"Warning: Setup time from {process_code} to {other_process} not defined.")
                valid_setup_times = False
                break
    
    # Calculate a more appropriate horizon based on multiple factors
    max_due_time = max(job[4] for job in processed_jobs)
    time_span = max_due_time - current_time
    
    # More intelligent horizon calculation
    machine_load = {}
    for machine, jobs in jobs_by_machine.items():
        machine_load[machine] = sum(job[3] for job in jobs)
    
    max_machine_load = max(machine_load.values()) if machine_load else 0
    
    # Calculate horizon considering machine load and setup times
    setup_time_factor = 1.2  # Allow 20% extra time for setups
    if valid_setup_times:
        # Estimate total setup time based on average setup time
        avg_setup = sum(sum(times.values()) for times in setup_times.values()) / (len(setup_times) ** 2)
        setup_time_factor = 1 + (avg_setup * (len(processed_jobs) - len(machine_ids))) / total_duration
    
    horizon = min(
        max(
            max_machine_load * setup_time_factor,  # Machine load with setup times
            max_duration * 3,                      # Longest job might need flexibility
            time_span,                             # Time until latest due date
            avg_duration * len(processed_jobs) / len(machine_ids) * 1.5  # Average case with buffer
        ),
        30 * 24 * 3600  # Cap at 30 days
    )
    logger.info(f"Planning horizon: {horizon / (24 * 3600):.1f} days ({horizon} seconds)")
    logger.info(f"Maximum operators needed at any time: {total_operators_needed}")
    
    # Define CP-SAT variables
    starts = {}
    ends = {}
    intervals = {}
    presence_vars = {}
    
    for job_id, job in enumerate(processed_jobs):
        job_name, process_code, machine, duration_seconds, due_time, priority, start_time, num_operators, output_rate, planned_end = job
        
        # Add weight to respecting planned start/end times if they're specified
        planned_start_weight = 1 if start_time > current_time else 0
        planned_end_weight = 1 if planned_end > start_time + duration_seconds else 0
        
        # Ensure start time window is valid
        start_min = max(current_time, start_time if planned_start_weight > 0 else current_time)
        start_max = min(due_time - duration_seconds, horizon + current_time)
        
        # If the window is impossible, adjust it to allow scheduling
        if start_max < start_min:
            logger.warning(f"Warning: Job {job_name} has impossible time window. Adjusting.")
            start_max = horizon + current_time - duration_seconds
        
        start = model.NewIntVar(start_min, start_max, f'start_{job_id}')
        end = model.NewIntVar(start_min + duration_seconds, horizon + current_time, f'end_{job_id}')
        interval = model.NewIntervalVar(start, duration_seconds, end, f'interval_{job_id}')
        
        # Add optional presence variable for soft constraints
        presence = model.NewBoolVar(f'presence_{job_id}')
        
        starts[job_id] = start
        ends[job_id] = end
        intervals[job_id] = interval
        presence_vars[job_id] = presence
        
        # Add deadline constraints with penalties based on priority
        deadline_penalty = 10 ** (6 - priority)  # Higher priority = higher penalty
        deadline_var = model.NewBoolVar(f'meet_deadline_{job_id}')
        model.Add(end <= due_time).OnlyEnforceIf(deadline_var)
        model.Add(end > due_time).OnlyEnforceIf(deadline_var.Not())
        model.Add(presence == 1)  # All jobs must be scheduled
        
        # Add soft constraints for planned start time if specified
        if planned_start_weight > 0:
            planned_start_var = model.NewBoolVar(f'meet_planned_start_{job_id}')
            model.Add(start == start_time).OnlyEnforceIf(planned_start_var)
            model.Add(start != start_time).OnlyEnforceIf(planned_start_var.Not())
        
        # Add soft constraints for planned end time if specified
        if planned_end_weight > 0:
            planned_end_var = model.NewBoolVar(f'meet_planned_end_{job_id}')
            model.Add(end == planned_end).OnlyEnforceIf(planned_end_var)
            model.Add(end != planned_end).OnlyEnforceIf(planned_end_var.Not())
    
    # Machine capacity constraints
    machine_intervals = {}
    for m_id in machine_ids:
        machine_intervals[m_id] = []
    
    for job_id, job in enumerate(processed_jobs):
        machine = job[2]
        if job_id in intervals and machine in machine_intervals:
            machine_intervals[machine].append(intervals[job_id])
    
    # No-overlap constraints for each machine
    for machine, intervals_list in machine_intervals.items():
        if len(intervals_list) > 1:
            model.AddNoOverlap(intervals_list)
    
    # OPERATOR CONSTRAINTS
    # Only add if we have jobs that need multiple operators
    if total_operators_needed > 1:
        # Track operator usage over time
        max_operators = sum(job[7] for job in processed_jobs)  # Maximum possible operator demand
        operator_usage = model.NewIntVar(0, max_operators, 'operator_usage')
        
        # Create operator demands for each interval
        operator_intervals = []
        for job_id, job in enumerate(processed_jobs):
            num_operators = job[7]
            if num_operators > 0:  # Create for all jobs with operator needs
                interval_var = intervals[job_id]
                operator_intervals.append((interval_var, num_operators))
        
        # Add cumulative constraint for operators - ensure we don't exceed available operators
        if operator_intervals:
            model.AddCumulative([interval for interval, _ in operator_intervals],
                               [demand for _, demand in operator_intervals],
                               operator_usage)
    
    # SETUP TIME CONSTRAINTS (if valid)
    if valid_setup_times:
        for machine, jobs_on_machine in jobs_by_machine.items():
            if len(jobs_on_machine) <= 1:
                continue
                
            # Create variables for job sequence on this machine
            job_indices = [i for i, job in enumerate(processed_jobs) if job[2] == machine]
            for i in range(len(job_indices)):
                for j in range(len(job_indices)):
                    if i != j:
                        job_i = processed_jobs[job_indices[i]]
                        job_j = processed_jobs[job_indices[j]]
                        
                        # Check if setup time exists
                        if (job_i[1] in setup_times and 
                            job_j[1] in setup_times[job_i[1]]):
                            
                            setup_duration = setup_times[job_i[1]][job_j[1]]
                            
                            # If job i is before job j, enforce setup time
                            before_var = model.NewBoolVar(f'before_{job_indices[i]}_{job_indices[j]}')
                            
                            # Either job i is before job j with setup time, or job j is before job i
                            model.Add(ends[job_indices[i]] + setup_duration <= starts[job_indices[j]]).OnlyEnforceIf(before_var)
                            model.Add(ends[job_indices[j]] <= starts[job_indices[i]]).OnlyEnforceIf(before_var.Not())
    
    # Objective function components
    makespan = model.NewIntVar(0, horizon + current_time, 'makespan')
    total_tardiness = model.NewIntVar(0, horizon * len(processed_jobs) * 10, 'total_tardiness')
    
    # Makespan constraint
    for job_id in ends:
        model.Add(makespan >= ends[job_id])
    
    # Tardiness calculation with priority weighting and output rate consideration
    tardiness_vars = []
    output_bonuses = []
    planned_deviation_vars = []  # Track deviation from planned times
    
    for job_id, job in enumerate(processed_jobs):
        if job_id in ends:
            due_time = job[4]
            priority = job[5]
            output_rate = job[8]
            duration = job[3]
            planned_start = job[6]
            planned_end = job[9]
            
            # Calculate tardiness
            tardiness = model.NewIntVar(0, horizon, f'tardiness_{job_id}')
            time_diff = model.NewIntVar(-horizon, horizon, f'time_diff_{job_id}')
            model.Add(time_diff == ends[job_id] - due_time)
            model.Add(tardiness >= 0)
            model.Add(tardiness >= time_diff)
            
            # Weight by priority (exponential weighting for clearer differentiation)
            priority_weight = 2 ** (6 - priority)  # Priority 1 = weight 32, Priority 5 = weight 2
            weighted_tardiness = model.NewIntVar(0, horizon * priority_weight, f'weighted_tardiness_{job_id}')
            model.Add(weighted_tardiness == tardiness * priority_weight)
            tardiness_vars.append(weighted_tardiness)
            
            # Consider output value - give small bonus to high-output jobs
            total_output = output_rate * (duration / 3600)  # Total output of the job
            output_bonus = model.NewIntVar(0, 1000, f'output_bonus_{job_id}')
            model.Add(output_bonus == min(1000, int(total_output / 50)))  # Scale down
            output_bonuses.append(output_bonus)
            
            # Calculate deviation from planned times
            if planned_start > current_time:
                start_deviation = model.NewIntVar(0, horizon, f'start_dev_{job_id}')
                start_diff = model.NewIntVar(-horizon, horizon, f'start_diff_{job_id}')
                model.Add(start_diff == starts[job_id] - planned_start)
                model.Add(start_deviation >= start_diff)
                model.Add(start_deviation >= -start_diff)  # Absolute value
                planned_deviation_vars.append(start_deviation)
            
            if planned_end > planned_start + duration:
                end_deviation = model.NewIntVar(0, horizon, f'end_dev_{job_id}')
                end_diff = model.NewIntVar(-horizon, horizon, f'end_diff_{job_id}')
                model.Add(end_diff == ends[job_id] - planned_end)
                model.Add(end_deviation >= end_diff)
                model.Add(end_deviation >= -end_diff)  # Absolute value
                planned_deviation_vars.append(end_deviation)
    
    if tardiness_vars:
        model.Add(total_tardiness == sum(tardiness_vars))
    else:
        model.Add(total_tardiness == 0)
    
    # Output bonus component
    total_output_bonus = model.NewIntVar(0, 100000, 'total_output_bonus')
    if output_bonuses:
        model.Add(total_output_bonus == sum(output_bonuses))
    else:
        model.Add(total_output_bonus == 0)
    
    # Planned time deviation component
    total_planned_deviation = model.NewIntVar(0, horizon * len(processed_jobs), 'total_planned_deviation')
    if planned_deviation_vars:
        model.Add(total_planned_deviation == sum(planned_deviation_vars))
    else:
        model.Add(total_planned_deviation == 0)
    
    # Combined objective with balanced weights
    makespan_weight = 1
    tardiness_weight = 10   # Higher weight on tardiness to prioritize due dates
    output_weight = -1      # Negative weight to maximize output (minimize negative)
    planned_dev_weight = 5  # Weight for minimizing deviation from planned times
    
    # FIXED APPROACH: Avoid division by using pre-computed weights instead
    # This removes the infeasible division constraint that was causing the error
    scaling_factor = max(1, len(machine_ids))
    scaled_makespan_weight = makespan_weight / scaling_factor
    
    normalized_planned_dev_weight = 0
    if planned_deviation_vars:
        normalized_planned_dev_weight = planned_dev_weight / len(planned_deviation_vars)
    
    # Use scaled weights directly instead of normalizing with division
    objective = model.NewIntVar(0, horizon * len(processed_jobs) * 100, 'objective')
    model.Add(objective == scaled_makespan_weight * makespan + 
                           tardiness_weight * total_tardiness + 
                           output_weight * total_output_bonus +
                           normalized_planned_dev_weight * total_planned_deviation)
    model.Minimize(objective)
    
    # Solver configuration
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0  # 5-minute time limit
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = True
    
    # Try disabling presolve if we're still having issues
    # Comment out next line to use default presolve
    # solver.parameters.cp_model_presolve = False
    
    # Optional: Add hints to guide the solver
    for job_id, job in enumerate(processed_jobs):
        priority = job[5]
        if priority == 1:  # Give hints for high-priority jobs
            model.AddHint(starts[job_id], job[6])  # Suggest starting at earliest time
    
    logger.info("Solving scheduling problem...")
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Build the schedule
        schedule = {}
        total_output = 0
        
        for job_id in starts:
            job = processed_jobs[job_id]
            job_name = job[0]
            machine = job[2]
            priority = job[5]
            output_rate = job[8]
            
            # Get solved values
            start = solver.Value(starts[job_id])
            end = solver.Value(ends[job_id])
            
            # Get the original planned times
            planned_start = job[6]
            planned_end = job[9]
            
            duration_hours = (end - start) / 3600
            job_output = output_rate * duration_hours
            total_output += job_output
            
            if machine not in schedule:
                schedule[machine] = []
            
            # Include planned times in the job data
            schedule[machine].append((job_name, start, end, priority, planned_start, planned_end))
        
        # Sort jobs by start time for each machine
        for machine in schedule:
            schedule[machine].sort(key=lambda x: x[1])
        
        total_jobs = sum(len(jobs) for jobs in schedule.values())
        logger.info(f"Schedule created with {total_jobs} jobs")
        logger.info(f"Total estimated output: {total_output:.1f} units")
        
        # Calculate schedule statistics
        earliest_start = min(job[1] for machine_jobs in schedule.values() for job in machine_jobs)
        latest_end = max(job[2] for machine_jobs in schedule.values() for job in machine_jobs)
        total_span = latest_end - earliest_start
        logger.info(f"Schedule span: {total_span/3600:.2f} hours ({datetime.fromtimestamp(earliest_start)} to {datetime.fromtimestamp(latest_end)})")
        
        # Check for tardiness
        late_jobs = []
        for job_id, job in enumerate(processed_jobs):
            if job_id in ends:
                job_name = job[0]
                due_time = job[4]
                end_time = solver.Value(ends[job_id])
                if end_time > due_time:
                    late_jobs.append((job_name, (end_time - due_time)/3600, job[5]))
        
        if late_jobs:
            logger.warning(f"Warning: {len(late_jobs)} jobs will be late:")
            for job_name, hours_late, priority in sorted(late_jobs, key=lambda x: (x[2], -x[1]))[:5]:  # Show 5 most important late jobs
                logger.warning(f"  - {job_name}: {hours_late:.1f} hours late (Priority {priority})")
            if len(late_jobs) > 5:
                logger.warning(f"  - ... and {len(late_jobs) - 5} more")
        
        # Calculate deviation from planned times
        plan_deviations = []
        for job_id, job in enumerate(processed_jobs):
            if job_id in starts and job_id in ends:
                job_name = job[0]
                planned_start = job[6]
                planned_end = job[9]
                actual_start = solver.Value(starts[job_id])
                actual_end = solver.Value(ends[job_id])
                
                start_dev = abs(actual_start - planned_start) / 3600 if planned_start > current_time else 0
                end_dev = abs(actual_end - planned_end) / 3600 if planned_end > planned_start else 0
                
                if start_dev > 0 or end_dev > 0:
                    plan_deviations.append((job_name, start_dev, end_dev, job[5]))
        
        if plan_deviations:
            logger.info(f"{len(plan_deviations)} jobs deviate from planned times:")
            for job_name, start_dev, end_dev, priority in sorted(plan_deviations, key=lambda x: (x[3], -(x[1] + x[2])))[:5]:
                logger.info(f"  - {job_name} (Priority {priority}): Start: {start_dev:.1f}h, End: {end_dev:.1f}h")
            if len(plan_deviations) > 5:
                logger.info(f"  - ... and {len(plan_deviations) - 5} more")
                
        return schedule
    else:
        logger.warning("No solution found with CP-SAT! Trying simplified approach...")
        return greedy_schedule(processed_jobs, machines)