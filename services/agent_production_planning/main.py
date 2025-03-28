# main.py | dont edit this line
import logging
import time
import argparse
import os
from datetime import datetime
import pandas as pd
import re
from dotenv import load_dotenv
from ingest_data import load_jobs_planning_data
from sch_jobs import schedule_jobs
from greedy import greedy_schedule
from chart import create_interactive_gantt
from chart_two import export_schedule_html
from setup_times import add_schedule_times_and_buffer
from loguru import logger





def main():
    print("Production Planning Scheduler")

    load_dotenv()
    parser = argparse.ArgumentParser(description="Production Planning Scheduler")
    parser.add_argument("--file", default=os.getenv('file_path'), 
                       help="Path to the Excel file with job data")
    parser.add_argument("--max-jobs", type=int, default=500, help="Maximum number of jobs to schedule (default: 250)")
    parser.add_argument("--force-greedy", action="store_true", help="Force the use of the greedy scheduler")
    parser.add_argument("--output", default="interactive_schedule.html", help="Output file for the Gantt chart")
    parser.add_argument("--enforce-sequence", action="store_true", default=True, help="Enforce process sequence dependencies (default: True)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (show debug messages)")
    args = parser.parse_args()

    # Set up logging based on whether --verbose was used
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    if not args.file:
        logger.error("No file path provided. Set --file argument or file_path in .env file.")
        return

    print(f"Configuration: file={args.file}, max_jobs={args.max_jobs}, force_greedy={args.force_greedy}, "
          f"output={args.output}, enforce_sequence={args.enforce_sequence}")

    current_time = int(datetime.now().timestamp())
    logger.info(f"Loading job data from {args.file}...")
    try:
        jobs, machines, setup_times = load_jobs_planning_data(args.file)
    except Exception as e:
        logger.error(f"Failed to load job data: {e}")
        return

    for job in jobs:
        job['UNIQUE_JOB_ID'] = f"{job['JOB']}_{job['PROCESS_CODE']}"

    jobs.sort(key=lambda job: job.get('LCD_DATE_EPOCH', 0))
    logger.info(f"Sorted {len(jobs)} jobs by LCD_DATE (First In, First Out)")

    if len(jobs) > args.max_jobs:
        logger.warning(f"Number of jobs ({len(jobs)}) exceeds max_jobs ({args.max_jobs}). Limiting to {args.max_jobs} jobs.")
        jobs = jobs[:args.max_jobs]

    logger.info(f"Loaded {len(jobs)} jobs and {len(machines)} machines")

    start_date_jobs = [job for job in jobs if 
                      ('START_DATE_EPOCH' in job and job.get('START_DATE_EPOCH') is not None and not pd.isna(job.get('START_DATE_EPOCH')) and job.get('START_DATE_EPOCH') > current_time) or 
                      ('START_DATE _EPOCH' in job and job.get('START_DATE _EPOCH') is not None and not pd.isna(job.get('START_DATE _EPOCH')) and job.get('START_DATE _EPOCH') > current_time)]
    if start_date_jobs:
        logger.info(f"Found {len(start_date_jobs)} jobs with START_DATE constraints:")
        for job in start_date_jobs:
            start_date_epoch = None
            if 'START_DATE_EPOCH' in job and job.get('START_DATE_EPOCH') is not None and not pd.isna(job.get('START_DATE_EPOCH')):
                start_date_epoch = job.get('START_DATE_EPOCH')
            elif 'START_DATE _EPOCH' in job and job.get('START_DATE _EPOCH') is not None and not pd.isna(job.get('START_DATE _EPOCH')):
                start_date_epoch = job.get('START_DATE _EPOCH')
                
            start_date = datetime.fromtimestamp(start_date_epoch).strftime('%Y-%m-%d %H:%M') if start_date_epoch is not None else 'INVALID DATE'
            resource_location = job.get('RSC_CODE')
            logger.info(f"  Job {job['UNIQUE_JOB_ID']} (Resource: {resource_location}): MUST start EXACTLY at {start_date}")
        
        logger.info("Ensuring START_DATE constraints are enforced...")
        for job in jobs:
            start_date_epoch = None
            if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and not pd.isna(job['START_DATE_EPOCH']):
                start_date_epoch = job['START_DATE_EPOCH']
            elif 'START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None and not pd.isna(job['START_DATE _EPOCH']):
                start_date_epoch = job['START_DATE _EPOCH']
                job['START_DATE_EPOCH'] = start_date_epoch
            
            if start_date_epoch is not None:
                if start_date_epoch > current_time:
                    logger.info(f"ENFORCING START_DATE {datetime.fromtimestamp(start_date_epoch).strftime('%Y-%m-%d %H:%M')} for {job['UNIQUE_JOB_ID']}")
            else:
                if 'START_DATE_EPOCH' in job:
                    del job['START_DATE_EPOCH']
                    logger.debug(f"Removed empty START_DATE_EPOCH for {job['UNIQUE_JOB_ID']}")
                if 'START_DATE _EPOCH' in job:
                    del job['START_DATE _EPOCH']
                    logger.debug(f"Removed empty START_DATE _EPOCH for {job['UNIQUE_JOB_ID']}")

    start_time = time.perf_counter()
    schedule = None

    if not args.force_greedy:
        logger.info("Attempting to create schedule with CP-SAT solver...")
        cp_sat_start_time = time.perf_counter()
        try:
            if len(jobs) > 200:
                time_limit = 900
            elif len(jobs) > 100:
                time_limit = 600
            else:
                time_limit = 300
                
            logger.info(f"Using {time_limit} seconds time limit for CP-SAT solver with {len(jobs)} jobs")
            
            valid_jobs = []
            for job in jobs:
                if 'UNIQUE_JOB_ID' not in job:
                    logger.warning(f"Job missing UNIQUE_JOB_ID field, skipping: {job}")
                    continue
                    
                if not job.get('RSC_CODE'):
                    logger.warning(f"Job {job['UNIQUE_JOB_ID']} has no machine assignment, skipping")
                    continue
                    
                if job.get('processing_time') is None or not isinstance(job.get('processing_time'), (int, float)) or job.get('processing_time') <= 0:
                    logger.warning(f"Job {job['UNIQUE_JOB_ID']} has missing or invalid processing time (HOURS_NEED)")
                    job['processing_time'] = 3600  # Still provide a default for the scheduler
                    
                valid_jobs.append(job)
                
            if len(valid_jobs) < len(jobs):
                logger.warning(f"Filtered out {len(jobs) - len(valid_jobs)} invalid jobs before scheduling")
                
            schedule = schedule_jobs(valid_jobs, machines, setup_times, enforce_sequence=args.enforce_sequence, time_limit_seconds=time_limit)
            
            if not schedule:
                logger.warning("CP-SAT solver returned None instead of a schedule dictionary")
                schedule = None
            elif not any(schedule.values()):
                logger.warning("CP-SAT solver returned an empty schedule (no jobs scheduled)")
                schedule = None
            else:
                total_jobs = sum(len(jobs_list) for jobs_list in schedule.values())
                cp_sat_time = max(0, time.perf_counter() - cp_sat_start_time)
                logger.info(f"CP-SAT solver successfully scheduled {total_jobs} jobs in {cp_sat_time:.2f} seconds")
        except Exception as e:
            logger.error(f"CP-SAT solver failed: {e}")
            logger.error(f"Stack trace:", exc_info=True)
            schedule = None

    if not schedule or not any(schedule.values()):
        logger.info("Falling back to greedy scheduler...")
        greedy_start_time = time.perf_counter()
        try:
            start_date_jobs = [job for job in jobs if 
                              ('START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and not pd.isna(job['START_DATE_EPOCH']) and job['START_DATE_EPOCH'] > current_time) or
                              ('START_DATE _EPOCH' in job and job['START_DATE _EPOCH'] is not None and not pd.isna(job['START_DATE _EPOCH']) and job['START_DATE _EPOCH'] > current_time)]
            
            if start_date_jobs:
                logger.info(f"Passing {len(start_date_jobs)} START_DATE constraints to greedy scheduler:")
                for job in start_date_jobs:
                    start_date_epoch = job.get('START_DATE_EPOCH')
                    if start_date_epoch is None or pd.isna(start_date_epoch):
                        start_date_epoch = job.get('START_DATE _EPOCH')
                        
                    if start_date_epoch is not None and not pd.isna(start_date_epoch):
                        start_date = datetime.fromtimestamp(start_date_epoch).strftime('%Y-%m-%d %H:%M')
                        logger.info(f"  Constraint: {job['UNIQUE_JOB_ID']} must start EXACTLY at {start_date}")
            
            valid_jobs = []
            for job in jobs:
                if 'UNIQUE_JOB_ID' not in job:
                    logger.warning(f"Job missing UNIQUE_JOB_ID field, skipping: {job}")
                    continue
                    
                if not job.get('RSC_CODE'):
                    logger.warning(f"Job {job['UNIQUE_JOB_ID']} has no machine assignment, skipping")
                    continue
                    
                valid_jobs.append(job)
                
            schedule = greedy_schedule(valid_jobs, machines, setup_times, enforce_sequence=args.enforce_sequence)
            
            if schedule and any(schedule.values()):
                total_jobs = sum(len(jobs_list) for jobs_list in schedule.values())
                greedy_time = max(0, time.perf_counter() - greedy_start_time)
                logger.info(f"Greedy scheduler successfully scheduled {total_jobs} jobs in {greedy_time:.2f} seconds")
            else:
                logger.error("Greedy scheduler returned an empty schedule")
        except Exception as e:
            logger.error(f"Greedy scheduler failed: {e}")
            logger.error(f"Stack trace:", exc_info=True)
            schedule = None

    if not schedule or not any(schedule.values()):
        logger.error("Failed to create a valid schedule.")
        return

    jobs = add_schedule_times_and_buffer(jobs, schedule)
    
    try:
        html_output = os.path.splitext(args.output)[0] + "_view.html"
        html_success = export_schedule_html(jobs, schedule, html_output)
        if not html_success:
            logger.error("Failed to generate HTML schedule view.")
            return
        logger.info(f"HTML schedule view saved to: {os.path.abspath(html_output)}")
    except Exception as e:
        logger.error(f"Error generating HTML schedule view: {e}")
        return

    flat_schedule = []
    for machine, tasks in schedule.items():
        for task in tasks:
            unique_job_id, start, end, priority = task
            
            job_entry = next((j for j in jobs if j['UNIQUE_JOB_ID'] == unique_job_id), None)
            start_time = job_entry['START_TIME'] if job_entry and 'START_TIME' in job_entry else start
            end_time = job_entry['END_TIME'] if job_entry and 'END_TIME' in job_entry else end
            
            resource_location = job_entry.get('RSC_CODE', machine)
            
            flat_schedule.append({
                'UNIQUE_JOB_ID': unique_job_id,
                'RSC_CODE': resource_location,
                'START_TIME': start_time,
                'END_TIME': end_time,
                'PRIORITY': priority
            })

    df_schedule = pd.DataFrame(flat_schedule)
    latest_completion = df_schedule.groupby('UNIQUE_JOB_ID')['END_TIME'].max().to_dict()
    for entry in flat_schedule:
        entry['LATEST_COMPLETION_TIME'] = latest_completion[entry['UNIQUE_JOB_ID']]

    logger.info("Computed schedule with START_TIME, END_TIME, and LATEST_COMPLETION_TIME:")
    logger.info(df_schedule.to_string())

    try:
        success = create_interactive_gantt(schedule, jobs, args.output)
        if not success:
            logger.error("Failed to generate Gantt chart.")
            return
    except Exception as e:
        logger.error(f"Error generating Gantt chart: {e}")
        return

    total_jobs = sum(len(tasks) for tasks in schedule.values())
    machines_used = len([m for m, tasks in schedule.items() if tasks])
    total_duration = sum((end - start) for machine, tasks in schedule.items() for _, start, end, _ in tasks)
    avg_machine_load = total_duration / len(machines) if machines else 0
    most_loaded_machine = max(schedule.items(), key=lambda x: sum(end - start for _, start, end, _ in x[1]), default=(None, []))[0]
    most_loaded_time = sum(end - start for _, start, end, _ in schedule.get(most_loaded_machine, []))
    schedule_span = max((end for machine, tasks in schedule.items() for _, _, end, _ in tasks), default=current_time) - \
                    min((start for machine, tasks in schedule.items() for _, start, _, _ in tasks), default=current_time)
    
    jobs_per_machine = {machine: len(tasks) for machine, tasks in schedule.items() if tasks}

    jobs_with_buffer = [job for job in jobs if 'BAL_HR' in job]
    if jobs_with_buffer:
        avg_buffer = sum(job['BAL_HR'] for job in jobs_with_buffer) / len(jobs_with_buffer)
        min_buffer = min(job['BAL_HR'] for job in jobs_with_buffer)
        max_buffer = max(job['BAL_HR'] for job in jobs_with_buffer)
        critical_jobs = [job for job in jobs_with_buffer if job['BAL_HR'] < 8]
        warning_jobs = [job for job in jobs_with_buffer if 8 <= job['BAL_HR'] < 24]
        
        print("\nBuffer statistics:")
        print(f"- Average buffer time: {avg_buffer:.1f} hours")
        print(f"- Min buffer time: {min_buffer:.1f} hours")
        print(f"- Max buffer time: {max_buffer:.1f} hours")
        print(f"- Critical jobs (<8h buffer): {len(critical_jobs)}")
        print(f"- Warning jobs (<24h buffer): {len(warning_jobs)}")
        
        if critical_jobs:
            print("\nCritical jobs with minimal buffer:")
            for job in sorted(critical_jobs, key=lambda x: x['BAL_HR'])[:5]:
                unique_job_id = job['UNIQUE_JOB_ID']
                resource_location = job.get('RSC_CODE', 'Unknown')
                buffer = job['BAL_HR']
                lcd_epoch = job.get('LCD_DATE_EPOCH', 0)
                dt = datetime.fromtimestamp(lcd_epoch)
                due_date = dt.strftime('%Y-%m-%d %H:%M')
                orig_date = ""
                for j in jobs:
                    if j['UNIQUE_JOB_ID'] == unique_job_id:
                        if 'LCD_DATE' in j and pd.notna(j['LCD_DATE']):
                            if isinstance(j['LCD_DATE'], str):
                                orig_date = f" (original={j['LCD_DATE']})"
                            else:
                                orig_date = f" (original={j['LCD_DATE'].strftime('%Y-%m-%d %H:%M')})"
                print(f"  Debug - Job {unique_job_id}: LCD_DATE_EPOCH={lcd_epoch}, formatted={due_date}{orig_date}")
                print(f"  {unique_job_id} on {resource_location}: {buffer:.1f} hours buffer, due {due_date}")

    print("\nSchedule statistics:")
    print(f"- Total jobs scheduled: {total_jobs}")
    print(f"- Machines utilized: {machines_used}/{len(machines)} ({machines_used/len(machines)*100:.1f}%)")
    print(f"- Average machine load: {avg_machine_load/3600:.1f} hours")
    print(f"- Most loaded machine: {most_loaded_machine} ({most_loaded_time/3600:.1f} hours)")
    print(f"- Jobs per machine: {jobs_per_machine}")
    print(f"- All jobs will complete by: {datetime.fromtimestamp(max((end for machine, tasks in schedule.items() for _, _, end, _ in tasks), default=current_time)).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"- Total production time: {total_duration/3600:.1f} hours")
    print(f"- Schedule span: {schedule_span/3600:.1f} hours")
    elapsed_time = max(0, time.perf_counter() - start_time)  # Ensure non-negative
    print(f"Scheduling completed in {elapsed_time:.2f} seconds")
    
    start_date_jobs = [job for job in jobs if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and not pd.isna(job['START_DATE_EPOCH'])]
    future_date_jobs = [job for job in start_date_jobs if job['START_DATE_EPOCH'] > current_time]
    
    if start_date_jobs:
        print("\nSTART_DATE constraints:")
        for job in start_date_jobs:
            unique_job_id = job['UNIQUE_JOB_ID']
            resource_location = job.get('RSC_CODE', 'Unknown')
            start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
            is_future = job['START_DATE_EPOCH'] > current_time
            
            scheduled_start = None
            for tasks in schedule.values():
                for proc_id, start, _, _ in tasks:
                    if proc_id == unique_job_id:
                        scheduled_start = start
                        break
            
            if scheduled_start:
                scheduled_date = datetime.fromtimestamp(scheduled_start).strftime('%Y-%m-%d %H:%M')
                start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
                
                if is_future:
                    if scheduled_start >= job['START_DATE_EPOCH']:
                        impact = "RESPECTED"
                    else:
                        impact = "VIOLATED"
                    
                    start_time_matches = job.get('START_TIME', 0) == job['START_DATE_EPOCH']
                    
                    print(f"  {unique_job_id} on {resource_location}: START_DATE={start_date}, Scheduled={scheduled_date} - {impact}")
                    if not start_time_matches:
                        print(f"    START_TIME doesn't match START_DATE for {unique_job_id}")
        
        if future_date_jobs:
            violated_constraints = []
            for job in future_date_jobs:
                unique_job_id = job['UNIQUE_JOB_ID']
                for tasks in schedule.values():
                    for proc_id, start, _, _ in tasks:
                        if proc_id == unique_job_id and start < job['START_DATE_EPOCH']:
                            violated_constraints.append(job)
                            break
            
            if violated_constraints:
                logger.error(f"Found {len(violated_constraints)} violated START_DATE constraints!")
                for job in violated_constraints:
                    unique_job_id = job['UNIQUE_JOB_ID']
                    start_date = datetime.fromtimestamp(job['START_DATE_EPOCH']).strftime('%Y-%m-%d %H:%M')
                    logger.error(f"  VIOLATED: {unique_job_id} should start EXACTLY at {start_date}")
            else:
                logger.info("All future START_DATE constraints were respected by the scheduler")

    print(f"\nResults saved to:")
    print(f"- Gantt chart: {os.path.abspath(args.output)}")
    print(f"- HTML Schedule View: {os.path.abspath(html_output)}")

if __name__ == "__main__":
    main()