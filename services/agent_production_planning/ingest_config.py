"""
Configuration for data ingestion from Excel.
This allows for configurable date column handling without hardcoding.
"""

# Define date columns that need specific time handling
DATE_COLUMNS_CONFIG = {
    # Column name (without trailing spaces) : (hour, minute)
    'START_DATE': (13, 0),  # Set START_DATE to 13:00 (1 PM)
    'PLANNED_END': (16, 0),  # Set PLANNED_END to 16:00 (4 PM)
    'LATEST_COMPLETION_DATE': (16, 0),  # Set LATEST_COMPLETION_DATE to 16:00 (4 PM)
    # Removed hardcoded time for LCD_DATE to use the original time from Excel
    # Add more columns as needed
}

# Default time for date columns not specifically configured
DEFAULT_TIME = (0, 0)  # Default to midnight

def get_column_time(column_name):
    """
    Get the configured time for a column name, handling trailing spaces.
    
    Args:
        column_name (str): The column name, possibly with trailing spaces
        
    Returns:
        tuple: (hour, minute) tuple from configuration or default
    """
    # Strip any trailing spaces to match with our configuration
    clean_name = column_name.strip() if isinstance(column_name, str) else column_name
    
    # Return the configured time or default
    return DATE_COLUMNS_CONFIG.get(clean_name, DEFAULT_TIME)
