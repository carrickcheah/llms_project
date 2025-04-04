# urgent.py
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reduce_nonproductive_time(jobs, reduction_percent=50):
    """
    Reduce setting_hours, break_hours, and no_prod by the specified percentage
    for jobs that are late (negative buffer hours).
    
    Args:
        jobs (list): List of job dictionaries
        reduction_percent (int): Percentage to reduce non-productive times (50 or 100)
        
    Returns:
        list: Updated jobs with reduced non-productive times for late jobs
        bool: Whether any jobs were modified
    """
    logger.info(f"Checking for late jobs and reducing non-productive time by {reduction_percent}%...")
    late_jobs_count = 0
    modified = False
    reduction_factor = 1 - (reduction_percent / 100)
    
    for job in jobs:
        # Check if job is late (negative buffer) or has the late status
        is_late = job.get('BAL_HR', 0) < 0 or job.get('BUFFER_STATUS') == 'Late'
        
        if is_late:
            late_jobs_count += 1
            modified = True
            
            # Keep track of original values for reporting
            original_processing_time = job.get('processing_time', 0)
            
            # Reduce SETTING_HOURS by specified percentage if present
            if 'SETTING_HOURS' in job and job['SETTING_HOURS'] is not None:
                job['SETTING_HOURS'] = job['SETTING_HOURS'] * reduction_factor
                # Also reduce the seconds equivalent if present
                if 'setup_time' in job and job['setup_time'] is not None:
                    job['setup_time'] = job['setup_time'] * reduction_factor
            
            # Reduce BREAK_HOURS by specified percentage if present
            if 'BREAK_HOURS' in job and job['BREAK_HOURS'] is not None:
                job['BREAK_HOURS'] = job['BREAK_HOURS'] * reduction_factor
                # Also reduce the seconds equivalent if present
                if 'break_time' in job and job['break_time'] is not None:
                    job['break_time'] = job['break_time'] * reduction_factor
            
            # Reduce NO_PROD by specified percentage if present
            if 'NO_PROD' in job and job['NO_PROD'] is not None:
                job['NO_PROD'] = job['NO_PROD'] * reduction_factor
                # Also reduce the seconds equivalent if present
                if 'no_prod_time' in job and job['no_prod_time'] is not None:
                    job['no_prod_time'] = job['no_prod_time'] * reduction_factor
            
            # Recalculate processing_time
            hours_need = job.get('HOURS_NEED', 0) * 3600
            setup_time = job.get('setup_time', 0)
            break_time = job.get('break_time', 0)
            no_prod_time = job.get('no_prod_time', 0)
            
            # Update processing_time with the reduced non-productive times
            job['processing_time'] = hours_need + setup_time + break_time + no_prod_time
            
            # Check if processing time was actually reduced
            if job['processing_time'] < original_processing_time:
                job['URGENT_APPLIED'] = True
                job['URGENT_REDUCTION'] = f"{reduction_percent}%"
                logger.info(f"Reduced non-productive time for job {job.get('UNIQUE_JOB_ID', 'unknown')}: " 
                          f"Processing time from {original_processing_time/3600:.2f}h to {job['processing_time']/3600:.2f}h")
    
    if late_jobs_count > 0:
        logger.info(f"Reduced non-productive time for {late_jobs_count} late jobs by {reduction_percent}%")
    else:
        logger.info("No late jobs found, no changes made to non-productive times")
    
    return jobs, modified

def should_reschedule(jobs, reduction_percent):
    """
    Determine if rescheduling is necessary based on how many jobs were modified
    and how significant the reduction is.
    
    Args:
        jobs (list): List of job dictionaries
        reduction_percent (int): Percentage used for reduction
        
    Returns:
        bool: True if rescheduling is recommended
    """
    late_jobs = [job for job in jobs if job.get('BAL_HR', 0) < 0 or job.get('BUFFER_STATUS') == 'Late']
    
    # If more than 10% of jobs are late and reduction is significant, recommend rescheduling
    if len(late_jobs) > 0.1 * len(jobs) and reduction_percent >= 50:
        return True
    
    # If any job has significant non-productive time (>20% of total), recommend rescheduling
    for job in late_jobs:
        total_time = job.get('processing_time', 0)
        if total_time <= 0:
            continue
            
        nonprod_time = (job.get('setup_time', 0) + job.get('break_time', 0) + job.get('no_prod_time', 0))
        if nonprod_time / total_time > 0.2:
            return True
    
    return False
