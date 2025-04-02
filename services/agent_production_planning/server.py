#!/usr/bin/env python3
# server.py - Simple HTTP server for processing production planning files

import os
import json
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
import cgi
import logging
import subprocess
from urllib.parse import parse_qs, urlparse
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Store the most recent Excel file path (to avoid relying on .env)
LAST_PROCESSED_FILE = None

class ProductionPlanningHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests - serve static files"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Serve the upload page as default
        if path == '/':
            path = '/upload.html'
        
        # Handle static Gantt chart download
        if path.startswith('/static_gantt'):
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'success': False,
                'error': 'This endpoint has been removed'
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Get the file path
        file_path = os.path.join(SCRIPT_DIR, path.lstrip('/'))
        
        # Check if the file exists
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'File not found')
            return
        
        # Determine content type based on file extension
        content_type = 'text/plain'
        if file_path.endswith('.html'):
            content_type = 'text/html'
        elif file_path.endswith('.css'):
            content_type = 'text/css'
        elif file_path.endswith('.js'):
            content_type = 'application/javascript'
        elif file_path.endswith('.png'):
            content_type = 'image/png'
        elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
            content_type = 'image/jpeg'
        
        # Serve the file
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Content-length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.error(f"Error serving file {file_path}: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Server error: {str(e)}".encode())
    
    def do_POST(self):
        """Handle POST requests - process uploaded files"""
        if self.path == '/process':
            try:
                # Parse form data
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST',
                             'CONTENT_TYPE': self.headers['Content-Type']}
                )
                
                # Get the uploaded file
                if 'file' not in form:
                    self.send_error_response("No file uploaded")
                    return
                
                fileitem = form['file']
                
                # Check if it's an Excel file
                if not fileitem.filename.endswith(('.xlsx', '.xls')):
                    self.send_error_response("File must be an Excel file (.xlsx or .xls)")
                    return
                
                # Create a temporary file to store the uploaded Excel
                with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    temp_file.write(fileitem.file.read())
                
                # Get scheduling parameters
                max_jobs = form.getvalue('max_jobs', '500')
                force_greedy = form.getvalue('force_greedy', 'false').lower() == 'true'
                enforce_sequence = form.getvalue('enforce_sequence', 'true').lower() == 'true'
                max_operators = form.getvalue('max_operators', '500')
                
                # Output file paths
                output_html = os.path.join(SCRIPT_DIR, 'interactive_schedule.html')
                output_view_html = os.path.join(SCRIPT_DIR, 'interactive_schedule_view.html')
                
                # Build command to run the main.py script
                cmd = [
                    'python', 
                    os.path.join(SCRIPT_DIR, 'main.py'),
                    '--file', temp_file_path,
                    '--max-jobs', max_jobs,
                    '--max-operators', max_operators,
                    '--output', output_html
                ]
                
                if force_greedy:
                    cmd.append('--force-greedy')
                
                if enforce_sequence:
                    cmd.append('--enforce-sequence')
                
                # Run the command
                logger.info(f"Running command: {' '.join(cmd)}")
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                # Check if the process was successful
                if process.returncode == 0:
                    # Store the file path for later use
                    global LAST_PROCESSED_FILE
                    LAST_PROCESSED_FILE = temp_file_path
                    
                    # Send success response
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    response = {
                        'success': True,
                        'gantt_chart': f'/interactive_schedule.html?t={os.path.getmtime(output_html)}',
                        'schedule_view': f'/interactive_schedule_view.html?t={os.path.getmtime(output_view_html)}',
                        'message': 'Schedule generated successfully'
                    }
                    self.wfile.write(json.dumps(response).encode())
                else:
                    # Send error response
                    error_message = process.stderr or "Unknown error occurred"
                    logger.error(f"Error running main.py: {error_message}")
                    self.send_error_response(f"Processing error: {error_message}")
                    
                    # Clean up the temporary file
                    os.unlink(temp_file_path)
            except Exception as e:
                logger.error(f"Error processing upload: {e}")
                self.send_error_response(f"Server error: {str(e)}")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')
    
    def send_error_response(self, message):
        """Send an error response as JSON"""
        self.send_response(400)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {
            'success': False,
            'error': message
        }
        self.wfile.write(json.dumps(response).encode())


def run_server(port=8000):
    """Run the HTTP server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, ProductionPlanningHandler)
    logger.info(f"Starting server on port {port}")
    logger.info(f"Open http://localhost:{port} in your browser")
    httpd.serve_forever()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a simple HTTP server for production planning')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    
    args = parser.parse_args()
    
    run_server(args.port)