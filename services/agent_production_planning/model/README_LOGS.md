# Log Management System

This directory contains tools to manage the `production_scheduler.log` file, preventing it from growing too large over time.

## Components

1. **`log_cleanup.py`** - Python script that:
   - Trims the log file to a reasonable size
   - Keeps only the most recent log entries
   - Creates backups before deletion
   - Only runs if the log exceeds the size threshold (10MB by default)

2. **`weekly_cleanup.sh`** - Shell script that:
   - Sets up a weekly cron job to run the cleanup automatically
   - Runs every Sunday at 1:00 AM

## Usage

### Setup Automatic Weekly Cleanup

Run the setup script to create the cron job:

```bash
cd services/agent_production_planning/model
./weekly_cleanup.sh
```

### Manual Cleanup

You can also run the cleanup script manually:

```bash
cd services/agent_production_planning/model
python3 log_cleanup.py
```

## Configuration

You can modify these settings in `log_cleanup.py`:

- `MAX_LINES_TO_KEEP = 10000` - Number of most recent log lines to retain
- `MAX_SIZE_MB = 50` - Minimum file size before cleanup is triggered
- `BACKUP_DIR = "../log_backups"` - Where backup files are stored

## Backup Management

- Backups are stored in the `../log_backups` directory
- Only the 5 most recent backups are kept
- Backups use the naming format `production_scheduler_YYYYMMDD_HHMMSS.log.bak` 