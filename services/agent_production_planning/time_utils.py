"""
Time utilities for the production planning system.
This implements a relative time approach to reduce the issue of large epoch numbers.
"""

import logging
import math
from datetime import datetime, timezone, timedelta
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

# Global reference time (scheduling start time)
REFERENCE_TIME = None
SINGAPORE_TZ = timezone(timedelta(hours=8))

def initialize_reference_time():
    """Initialize the reference time for relative calculations to midnight today."""
    now = datetime.now(SINGAPORE_TZ)
    reference = datetime(now.year, now.month, now.day, tzinfo=SINGAPORE_TZ)
    logging.info(f"Reference time initialized to {reference.isoformat()}")
    return reference

REFERENCE_TIME = initialize_reference_time()
REFERENCE_EPOCH = REFERENCE_TIME.timestamp()

def get_reference_time():
    """Get the reference time, initializing it if necessary."""
    global REFERENCE_TIME
    if REFERENCE_TIME is None:
        initialize_reference_time()
    return REFERENCE_TIME

def datetime_to_epoch(dt):
    """Convert a datetime object to epoch timestamp."""
    if dt is None:
        return None
    try:
        return dt.timestamp()
    except (AttributeError, TypeError):
        logging.warning(f"Could not convert to epoch: {dt}")
        return None

def epoch_to_datetime(epoch):
    """Convert an epoch timestamp to a datetime object."""
    if epoch is None or pd.isna(epoch):
        return None
    try:
        return datetime.fromtimestamp(epoch, SINGAPORE_TZ)
    except (ValueError, TypeError, OSError) as e:
        logging.warning(f"Invalid epoch value: {epoch}, error: {e}")
        return None

def epoch_to_relative_hours(epoch):
    """Convert an epoch timestamp to hours since reference time."""
    if epoch is None or pd.isna(epoch):
        return 0
    try:
        return (epoch - REFERENCE_EPOCH) / 3600
    except (TypeError, ValueError):
        logging.warning(f"Could not convert epoch to relative hours: {epoch}")
        return 0

def relative_hours_to_epoch(hours):
    """Convert hours since reference time to epoch timestamp."""
    if hours is None or pd.isna(hours):
        return REFERENCE_EPOCH
    try:
        return REFERENCE_EPOCH + (hours * 3600)
    except (TypeError, ValueError):
        logging.warning(f"Could not convert relative hours to epoch: {hours}")
        return REFERENCE_EPOCH

def iso_to_datetime(iso_string):
    """Convert an ISO format string to a datetime object."""
    if not iso_string or pd.isna(iso_string):
        return None
    try:
        return datetime.fromisoformat(iso_string)
    except (ValueError, TypeError):
        logging.warning(f"Could not parse ISO datetime: {iso_string}")
        return None

def datetime_to_iso(dt):
    """Convert a datetime object to ISO format string."""
    if dt is None:
        return None
    try:
        return dt.isoformat()
    except (AttributeError, TypeError):
        logging.warning(f"Could not convert to ISO: {dt}")
        return None

def format_datetime_for_display(dt):
    """Format a datetime object for display."""
    if dt is None:
        return "N/A"
    try:
        return dt.strftime('%Y-%m-%d %H:%M')
    except (AttributeError, TypeError):
        return "Invalid Date"

def convert_job_times_to_relative(job):
    """Convert job time fields from epoch to relative time."""
    if not job:
        return
        
    # Store original values
    for field in ['LCD_DATE_EPOCH', 'START_DATE_EPOCH', 'START_TIME', 'END_TIME']:
        if field in job and job[field] is not None:
            job[f"{field}_ORIGINAL"] = job[field]
    
    # Handle LCD_DATE_EPOCH (due date)
    if 'LCD_DATE_EPOCH' in job and job['LCD_DATE_EPOCH'] is not None and not pd.isna(job['LCD_DATE_EPOCH']):
        epoch_value = job['LCD_DATE_EPOCH']
        dt = epoch_to_datetime(epoch_value)
        if dt:
            job['LCD_DATE_ISO'] = datetime_to_iso(dt)
            job['LCD_DATE_REL_HOURS'] = epoch_to_relative_hours(epoch_value)
    
    # Handle START_DATE_EPOCH (required start date)
    if 'START_DATE_EPOCH' in job and job['START_DATE_EPOCH'] is not None and not pd.isna(job['START_DATE_EPOCH']):
        epoch_value = job['START_DATE_EPOCH']
        dt = epoch_to_datetime(epoch_value)
        if dt:
            job['START_DATE_ISO'] = datetime_to_iso(dt)
            job['START_DATE_REL_HOURS'] = epoch_to_relative_hours(epoch_value)
    
    # Handle START_TIME (scheduled start)
    if 'START_TIME' in job and job['START_TIME'] is not None and not pd.isna(job['START_TIME']):
        epoch_value = job['START_TIME']
        dt = epoch_to_datetime(epoch_value)
        if dt:
            job['START_TIME_ISO'] = datetime_to_iso(dt)
            job['START_TIME_REL_HOURS'] = epoch_to_relative_hours(epoch_value)
    
    # Handle END_TIME (scheduled end)
    if 'END_TIME' in job and job['END_TIME'] is not None and not pd.isna(job['END_TIME']):
        epoch_value = job['END_TIME']
        dt = epoch_to_datetime(epoch_value)
        if dt:
            job['END_TIME_ISO'] = datetime_to_iso(dt)
            job['END_TIME_REL_HOURS'] = epoch_to_relative_hours(epoch_value)

def convert_job_times_to_epoch(job):
    """Convert job time fields from relative time back to epoch."""
    if not job:
        return
    
    # Handle LCD_DATE_REL_HOURS
    if 'LCD_DATE_REL_HOURS' in job and job['LCD_DATE_REL_HOURS'] is not None and not pd.isna(job['LCD_DATE_REL_HOURS']):
        rel_hours = job['LCD_DATE_REL_HOURS']
        job['LCD_DATE_EPOCH'] = relative_hours_to_epoch(rel_hours)
    
    # Handle START_DATE_REL_HOURS
    if 'START_DATE_REL_HOURS' in job and job['START_DATE_REL_HOURS'] is not None and not pd.isna(job['START_DATE_REL_HOURS']):
        rel_hours = job['START_DATE_REL_HOURS']
        job['START_DATE_EPOCH'] = relative_hours_to_epoch(rel_hours)
    
    # Handle START_TIME_REL_HOURS
    if 'START_TIME_REL_HOURS' in job and job['START_TIME_REL_HOURS'] is not None and not pd.isna(job['START_TIME_REL_HOURS']):
        rel_hours = job['START_TIME_REL_HOURS']
        job['START_TIME'] = relative_hours_to_epoch(rel_hours)
    
    # Handle END_TIME_REL_HOURS
    if 'END_TIME_REL_HOURS' in job and job['END_TIME_REL_HOURS'] is not None and not pd.isna(job['END_TIME_REL_HOURS']):
        rel_hours = job['END_TIME_REL_HOURS']
        job['END_TIME'] = relative_hours_to_epoch(rel_hours) 