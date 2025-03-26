# app2.py - Web interface for production planning with TypeScript-based Gantt chart

import os
import json
import tempfile
from datetime import datetime
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import logging
from dotenv import load_dotenv

# Import job scheduling functionality
from ingest_data import load_jobs_planning_data
from main import run_scheduler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store data
current_schedule = {}
current_jobs = []

# Allowed file extensions
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for job data"""
    global current_jobs
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load job data
            jobs = load_jobs_planning_data(filepath)
            current_jobs = jobs
            
            return jsonify({
                'message': f'Successfully loaded {len(jobs)} jobs from {filename}',
                'job_count': len(jobs)
            })
        except Exception as e:
            logger.error(f"Error loading job data: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/schedule', methods=['POST'])
def schedule_jobs():
    """Generate a schedule based on uploaded job data and parameters"""
    global current_schedule, current_jobs
    
    if not current_jobs:
        return jsonify({'error': 'No job data loaded. Please upload a file first.'}), 400
    
    try:
        # Extract scheduling parameters from request
        data = request.json
        
        # Default parameters
        parameters = {
            'prioritize_by_lcd': data.get('prioritize_by_lcd', True),
            'consider_setup_times': data.get('consider_setup_times', True),
            'optimize_machine_utilization': data.get('optimize_machine_utilization', True),
            'time_limit_seconds': int(data.get('time_limit_seconds', 60))
        }
        
        # Run the scheduler
        schedule = run_scheduler(current_jobs, **parameters)
        
        if not schedule:
            return jsonify({'error': 'Scheduler failed to generate a valid schedule'}), 500
        
        # Store the generated schedule
        current_schedule = schedule
        
        # Count total jobs scheduled
        job_count = sum(len(jobs) for jobs in schedule.values())
        
        return jsonify({
            'message': f'Successfully scheduled {job_count} jobs across {len(schedule)} machines',
            'machine_count': len(schedule),
            'job_count': job_count
        })
    except Exception as e:
        logger.error(f"Error in scheduling: {str(e)}")
        return jsonify({'error': f'Error generating schedule: {str(e)}'}), 500

@app.route('/get_schedule_data')
def get_schedule_data():
    """Provide the current schedule data in JSON format"""
    global current_schedule, current_jobs
    
    # Convert jobs list to a lookup dictionary
    job_lookup = {}
    for job in current_jobs:
        if 'PROCESS_CODE' in job:
            job_lookup[job['PROCESS_CODE']] = job
    
    return jsonify({
        'schedule': current_schedule,
        'jobs': job_lookup
    })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs(os.path.join('static', 'dist'), exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5001)