#!/usr/bin/env python3
# server.py - FastAPI server for processing production planning files

import os
import json
import tempfile
import logging
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Store the most recent Excel file path (to avoid relying on .env)
LAST_PROCESSED_FILE = None

# Create FastAPI app
app = FastAPI(
    title="Production Planning API",
    description="API for production planning and scheduling",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=SCRIPT_DIR), name="static")

# Define response models
class ProcessResponse(BaseModel):
    success: bool
    message: str
    gantt_chart: Optional[str] = None
    schedule_view: Optional[str] = None
    resource_view: Optional[str] = None
    report_view: Optional[str] = None
    error: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def get_root():
    """Serve upload page as default"""
    try:
        with open(os.path.join(SCRIPT_DIR, "upload.html"), "r") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error serving upload.html: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/static_gantt", response_class=JSONResponse)
async def get_static_gantt():
    """Deprecated endpoint - retained for backward compatibility"""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "This endpoint has been removed"
        }
    )

@app.get("/{path:path}")
async def get_file(path: str):
    """Serve static files"""
    # Get the full file path
    file_path = os.path.join(SCRIPT_DIR, path)
    
    # Check if file exists
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine content type based on file extension
    content_type = None  # FastAPI will guess based on extension
    
    # Return the file
    return FileResponse(file_path, media_type=content_type)

async def update_report_timestamp(report_file: str):
    """Update the timestamp in the generated report HTML file"""
    try:
        if os.path.exists(report_file):
            # Read the existing report file
            with open(report_file, 'r') as f:
                content = f.read()
            
            # Get current timestamp
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Update the report date and time - more aggressive pattern matching
            updated_content = content
            
            # Update date in meta-info section
            import re
            
            # Replace the date value directly
            date_pattern = r'<div class="meta-label">Report Date</div>\s*<div class="meta-value">[^<]*</div>'
            date_replacement = f'<div class="meta-label">Report Date</div>\n                <div class="meta-value">{current_date}</div>'
            updated_content = re.sub(date_pattern, date_replacement, updated_content)
            
            # Replace the time value directly
            time_pattern = r'<div class="meta-label">Report Time</div>\s*<div class="meta-value">[^<]*</div>'
            time_replacement = f'<div class="meta-label">Report Time</div>\n                <div class="meta-value">{current_time}</div>'
            updated_content = re.sub(time_pattern, time_replacement, updated_content)
            
            # Update the generated timestamp at the end if it exists 
            if '<p><i>Report generated on' in updated_content:
                timestamp_pattern = r'<p><i>Report generated on .*?</i></p>'
                updated_content = re.sub(
                    timestamp_pattern,
                    f'<p><i>Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</i></p>',
                    updated_content
                )
            
            # Write the updated content back to the file
            with open(report_file, 'w') as f:
                f.write(updated_content)
            
            logger.info(f"Updated timestamp in report file: {report_file}")
    except Exception as e:
        logger.error(f"Error updating report timestamp: {e}")

@app.post("/process", response_model=ProcessResponse)
async def process_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_jobs: str = Form("500"),
    force_greedy: str = Form("false"),
    enforce_sequence: str = Form("true"),
    max_operators: str = Form("500"),
    urgent50: str = Form("false"),
    urgent100: str = Form("false"),
    generate_report: str = Form("false")
):
    """Process the uploaded Excel file and generate a production schedule"""
    try:
        # Check if file is an Excel file
        if not file.filename.endswith(('.xlsx', '.xls')):
            return ProcessResponse(
                success=False,
                message="File upload failed",
                error="File must be an Excel file (.xlsx or .xls)"
            )

        # Create a temporary file to store the uploaded Excel
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(await file.read())
        
        # Convert form strings to proper types
        force_greedy_bool = force_greedy.lower() == 'true'
        enforce_sequence_bool = enforce_sequence.lower() == 'true'
        urgent50_bool = urgent50.lower() == 'true'
        urgent100_bool = urgent100.lower() == 'true'
        generate_report_bool = generate_report.lower() == 'true'
        
        # Output file paths
        output_html = os.path.join(SCRIPT_DIR, 'interactive_schedule.html')
        output_view_html = os.path.join(SCRIPT_DIR, 'interactive_schedule_view.html')
        output_report_html = os.path.join(SCRIPT_DIR, 'production_report.html')
        
        # Build command to run the main.py script
        cmd = [
            'python', 
            os.path.join(SCRIPT_DIR, 'main.py'),
            '--file', temp_file_path,
            '--max-jobs', max_jobs,
            '--max-operators', max_operators,
            '--output', output_html
        ]
        
        if force_greedy_bool:
            cmd.append('--force-greedy')
        
        if enforce_sequence_bool:
            cmd.append('--enforce-sequence')
        
        # Add urgent options
        if urgent50_bool:
            cmd.append('--urgent50')
        
        if urgent100_bool:
            cmd.append('--urgent100')
        
        # Add report generation option
        if generate_report_bool:
            cmd.append('--generate-report')
            cmd.append('--report-output')
            cmd.append(output_report_html)
        
        # Run the command
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if the process was successful
        if process.returncode == 0:
            # Store the file path for later use
            global LAST_PROCESSED_FILE
            LAST_PROCESSED_FILE = temp_file_path
            
            # If report was generated, update its timestamp directly instead of using background tasks
            if generate_report_bool and os.path.exists(output_report_html):
                # Update timestamps directly rather than in background for more reliable results
                await update_report_timestamp(output_report_html)
            
            output_resource_html = os.path.join(SCRIPT_DIR, 'interactive_schedule_r.html')
            
            response = ProcessResponse(
                success=True,
                message="Schedule generated successfully",
                gantt_chart=f'/interactive_schedule.html?t={os.path.getmtime(output_html)}',
                schedule_view=f'/interactive_schedule_view.html?t={os.path.getmtime(output_view_html)}',
                resource_view=f'/interactive_schedule_r.html?t={os.path.getmtime(output_resource_html) if os.path.exists(output_resource_html) else ""}'
            )
            
            # Add report view to response if report was generated
            if generate_report_bool and os.path.exists(output_report_html):
                # No need to sleep since we waited for the update to complete
                response.report_view = f'/production_report.html?t={os.path.getmtime(output_report_html)}'
            
            return response
        else:
            # Error response
            error_message = process.stderr or "Unknown error occurred"
            logger.error(f"Error running main.py: {error_message}")
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            return ProcessResponse(
                success=False,
                message="Processing failed",
                error=f"Processing error: {error_message}"
            )
            
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return ProcessResponse(
            success=False, 
            message="Server error",
            error=f"Server error: {str(e)}"
        )

# Check if any model/report_generator.py is being used for report generation
async def check_report_generator():
    report_generator_path = os.path.join(SCRIPT_DIR, 'model', 'report_generator.py')
    if os.path.exists(report_generator_path):
        try:
            # Import and modify the report_generator module to use current timestamps
            import importlib.util
            spec = importlib.util.spec_from_file_location("report_generator", report_generator_path)
            report_generator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(report_generator)
            
            # Monkey-patch the generate_html_report function to use current time
            original_generate_html_report = report_generator.generate_html_report
            
            def patched_generate_html_report(*args, **kwargs):
                html_report = original_generate_html_report(*args, **kwargs)
                # Replace the timestamps with current ones
                current_date = datetime.now().strftime("%Y-%m-%d")
                current_time = datetime.now().strftime("%H:%M:%S")
                html_report = html_report.replace(
                    f'<div class="meta-value">{current_date[:4]}-{current_date[5:7]}-{current_date[8:10]}</div>',
                    f'<div class="meta-value">{current_date}</div>'
                )
                html_report = html_report.replace(
                    '<div class="meta-value">01:12:18</div>',
                    f'<div class="meta-value">{current_time}</div>'
                )
                return html_report
            
            # Apply the monkey patch
            report_generator.generate_html_report = patched_generate_html_report
            
            logger.info("Successfully patched report_generator.py to use current timestamps")
        except Exception as e:
            logger.error(f"Error patching report_generator: {e}")

@app.on_event("startup")
async def startup_event():
    # Check and patch report generator on startup
    await check_report_generator()
    logger.info("FastAPI server started with timestamp fix for reports")

# Entry point for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the FastAPI server for production planning')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    
    args = parser.parse_args()
    
    logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
    logger.info(f"Open http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port} in your browser")
    
    uvicorn.run("server:app", host=args.host, port=args.port, reload=True)