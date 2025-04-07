#!/bin/bash
# Weekly log cleanup cron job setup script

# Get the absolute path of the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_CLEANUP_SCRIPT="$SCRIPT_DIR/log_cleanup.py"

# Make the script executable
chmod +x "$LOG_CLEANUP_SCRIPT"

# Create a temporary file for the cron job
CRON_FILE=$(mktemp)

# Get existing crontab
crontab -l > "$CRON_FILE" 2>/dev/null || echo "# New crontab" > "$CRON_FILE"

# Check if our job is already in crontab
if ! grep -q "log_cleanup.py" "$CRON_FILE"; then
    # Add job to run every Sunday at 1:00 AM
    echo "0 1 * * 0 cd $SCRIPT_DIR && /usr/bin/env python3 $LOG_CLEANUP_SCRIPT >> $SCRIPT_DIR/../log_cleanup.log 2>&1" >> "$CRON_FILE"
    
    # Install the new crontab
    crontab "$CRON_FILE"
    echo "Weekly log cleanup job has been scheduled to run every Sunday at 1:00 AM"
else
    echo "Cleanup job already exists in crontab"
fi

# Clean up
rm "$CRON_FILE"

echo "You can also run the cleanup script manually with:"
echo "  cd $SCRIPT_DIR && python3 log_cleanup.py" 