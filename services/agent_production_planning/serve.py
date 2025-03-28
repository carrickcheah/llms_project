#!/usr/bin/env python
# serve.py - Script to run scheduler and then serve results
import os
import subprocess
import sys
import http.server
import socketserver
from concurrent.futures import ThreadPoolExecutor

def run_scheduler():
    print("Running production planning scheduler...")
    # Run the main.py script to generate schedules and charts
    result = subprocess.run(["python", "main.py"], capture_output=False)
    
    if result.returncode != 0:
        print("ERROR: Scheduler process failed!")
        return False
    
    return True

def start_http_server():
    PORT = 8000
    DIRECTORY = "/app"
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=DIRECTORY, **kwargs)
            
        def log_message(self, format, *args):
            # Customize logging to be more informative
            sys.stderr.write("[HTTP Server] %s - %s\n" % 
                             (self.address_string(), format % args))

    httpd = socketserver.TCPServer(("", PORT), Handler)

    print(f"=============================================================")
    print(f"🌐 HTTP Server started at http://localhost:{PORT}/")
    print(f"📊 Access visualization at http://localhost:{PORT}/interactive_schedule.html")
    print(f"📋 Access report at http://localhost:{PORT}/interactive_schedule_view.html") 
    print(f"=============================================================")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()

if __name__ == "__main__":
    # Run the scheduler first
    success = run_scheduler()
    
    # Start the HTTP server to serve the results
    if success or os.path.exists("/app/interactive_schedule.html"):
        start_http_server()
    else:
        print("No visualization files found. Exiting.")
        sys.exit(1)
