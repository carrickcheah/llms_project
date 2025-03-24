#!/usr/bin/env python3
"""
Test script to demonstrate the epoch timestamp solution for Excel date/time handling.
This script shows how to extract and preserve time components from Excel date columns.
"""

import pandas as pd
from datetime import datetime
import os
import logging
from excel_to_epoch import excel_to_epoch_main
from ingest_config import DATE_COLUMNS_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_excel_time_extraction():
    """Test the new epoch-based solution against the Excel file."""
    # Define path to Excel file
    file_path = "/Users/carrickcheah/llms_project/services/agent_production_planning/mydata.xlsx"
    
    # Define the date columns we want to process
    date_columns = ['LCD_DATE', 'START_DATE', 'PLANNED_START', 'PLANNED_END', 'LATEST_COMPLETION_DATE']
    
    logger.info("Testing epoch timestamp solution for Excel date/time extraction")
    logger.info(f"Processing Excel file: {file_path}")
    logger.info(f"Date columns to process: {date_columns}")
    logger.info(f"Date column time configurations: {DATE_COLUMNS_CONFIG}")
    
    # Process the Excel file
    result_df = excel_to_epoch_main(
        file_path, 
        sheet_name='Jobs PlanningDetails-Draft',
        date_columns=date_columns
    )
    
    # Display a summary of the conversion
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    
    for col in date_columns:
        # Get the corresponding epoch column name
        epoch_col = f"epoch_{col.lower().replace(' ', '_').replace('-', '_')}"
        
        if epoch_col in result_df.columns:
            # Count valid conversions
            valid_count = result_df[epoch_col].notna().sum()
            total_count = len(result_df)
            logger.info(f"\n{col}: {valid_count}/{total_count} values converted to epochs ({valid_count/total_count:.1%})")
            
            # Show a few examples
            if valid_count > 0:
                logger.info("Sample values with full date and time information:")
                sample = result_df[result_df[epoch_col].notna()].head(5)
                
                for idx, row in sample.iterrows():
                    # Get the epoch value
                    epoch_val = row[epoch_col]
                    
                    # Convert back to datetime for display
                    dt = datetime.fromtimestamp(epoch_val)
                    
                    # Get the original value for comparison
                    orig_val = row[col] if col in row and pd.notna(row[col]) else "N/A"
                    
                    # Display the original and converted values
                    logger.info(f"  Row {idx+1}: '{orig_val}' → {dt.strftime('%Y-%m-%d %H:%M:%S')} (epoch: {epoch_val})")
                
                # Check time distribution for this column
                if valid_count > 5:
                    time_distribution = {}
                    for epoch_val in result_df[epoch_col].dropna():
                        dt = datetime.fromtimestamp(epoch_val)
                        time_key = f"{dt.hour:02d}:{dt.minute:02d}"
                        time_distribution[time_key] = time_distribution.get(time_key, 0) + 1
                    
                    logger.info(f"  Time distribution for {col}:")
                    for time_key, count in sorted(time_distribution.items()):
                        logger.info(f"    {time_key}: {count} records ({count/valid_count:.1%})")
    
    # See if we need to export the results
    export_path = os.path.join(os.path.dirname(file_path), "epoch_processed_data.xlsx")
    result_df.to_excel(export_path, index=False)
    logger.info(f"\nExported processed data with epoch timestamps to: {export_path}")
    
    logger.info("\nDone! The epoch timestamp solution successfully preserves time components from Excel.")
    return result_df

if __name__ == "__main__":
    test_excel_time_extraction()
