#!/usr/bin/env python3
"""
Log Cleanup Script for Production Scheduler

This script automatically cleans the production_scheduler.log file,
keeping only the most recent entries to prevent excessive file growth.
"""

import os
import re
import sys
from datetime import datetime
import shutil

LOG_FILE_PATH = "../production_scheduler.log"
BACKUP_DIR = "../log_backups"
MAX_LINES_TO_KEEP = 10000  # Adjust based on your needs
MAX_SIZE_MB = 50  # Maximum size in MB before cleanup

def get_file_size_mb(file_path):
    """Get the size of a file in megabytes"""
    return os.path.getsize(file_path) / (1024 * 1024)

def create_backup(file_path):
    """Create a backup of the log file before cleaning"""
    # Ensure backup directory exists
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"production_scheduler_{timestamp}.log.bak")
    
    # Copy the file
    shutil.copy2(file_path, backup_path)
    print(f"Created backup at {backup_path}")
    
    # Only keep the last 5 backups
    backups = sorted([os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR)])
    while len(backups) > 5:
        os.remove(backups[0])  # Remove oldest backup
        backups.pop(0)
    
    return backup_path

def clean_log_file(file_path, max_lines):
    """Clean the log file, keeping only the most recent entries"""
    if not os.path.exists(file_path):
        print(f"Log file {file_path} not found.")
        return False
    
    # Check if file needs cleaning
    file_size_mb = get_file_size_mb(file_path)
    if file_size_mb < MAX_SIZE_MB:
        print(f"Log file size ({file_size_mb:.2f} MB) is below threshold, no cleanup needed.")
        return False
    
    # Create backup
    create_backup(file_path)
    
    # Read the last N lines from the file
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
        # If file is smaller than max_lines, no need to truncate
        if len(lines) <= max_lines:
            print(f"Log file has {len(lines)} lines, below threshold of {max_lines}. No cleanup needed.")
            return False
            
        # Keep only the most recent lines
        recent_lines = lines[-max_lines:]
        
        # Find the first full log entry (starts with date)
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        for i, line in enumerate(recent_lines):
            if re.match(date_pattern, line):
                recent_lines = recent_lines[i:]
                break
        
        # Write back the truncated content
        with open(file_path, 'w') as file:
            file.writelines(recent_lines)
            
        print(f"Log file cleaned. Reduced from {len(lines)} to {len(recent_lines)} lines.")
        return True
        
    except Exception as e:
        print(f"Error cleaning log file: {str(e)}")
        return False

if __name__ == "__main__":
    print(f"Starting log cleanup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    result = clean_log_file(LOG_FILE_PATH, MAX_LINES_TO_KEEP)
    print(f"Log cleanup {'completed successfully' if result else 'not needed'}.") 